import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union, Callable, Tuple
from PIL import Image, ImageOps
from einops import rearrange
from tqdm import tqdm
from diffusers import DiffusionPipeline


def sinusoidal_embedding_1d(dim, position):
    """1D sinusoidal positional embedding for timesteps."""
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def _build_rope_3d(rope_module, f, h, w, device):
    """
    Build 3D RoPE (cos, sin) for a given (f, h, w) grid using the
    WanRotaryPosEmbed module's precomputed buffers.

    Returns:
        (freqs_cos, freqs_sin) each of shape [1, f*h*w, 1, head_dim]
    """
    split_sizes = [rope_module.t_dim, rope_module.h_dim, rope_module.w_dim]
    cos_parts = rope_module.freqs_cos.split(split_sizes, dim=1)
    sin_parts = rope_module.freqs_sin.split(split_sizes, dim=1)

    cos_f = cos_parts[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1)
    cos_h = cos_parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)
    cos_w = cos_parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)

    sin_f = sin_parts[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1)
    sin_h = sin_parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)
    sin_w = sin_parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)

    freqs_cos = torch.cat([cos_f, cos_h, cos_w], dim=-1).reshape(1, f * h * w, 1, -1).to(device)
    freqs_sin = torch.cat([sin_f, sin_h, sin_w], dim=-1).reshape(1, f * h * w, 1, -1).to(device)
    return freqs_cos, freqs_sin


class KiwiEditPipeline(DiffusionPipeline):
    """
    Pipeline for reference-guided video and image editing using KiwiEdit.

    This pipeline uses a Qwen2.5-VL multimodal LLM encoder for understanding
    editing instructions with source visual context, a WanTransformer3DModel
    for diffusion, and AutoencoderKLWan for VAE encoding/decoding.

    Args:
        transformer: WanTransformer3DModel - DiT backbone for denoising.
        vae: AutoencoderKLWan - 3D causal VAE.
        scheduler: FlowMatchEulerDiscreteScheduler or compatible scheduler.
        mllm_encoder: MLLMEncoder - Qwen2.5-VL MLLM with learnable queries.
        processor: AutoProcessor - Qwen2.5-VL processor/tokenizer bundle.
        source_embedder: ConditionalEmbedder - VAE source conditioning.
        ref_embedder: ConditionalEmbedder - VAE reference conditioning.
    """

    model_cpu_offload_seq = "mllm_encoder->source_embedder->ref_embedder->transformer->vae"

    def __init__(
        self,
        transformer,
        vae,
        scheduler,
        mllm_encoder,
        source_embedder,
        ref_embedder,
        processor=None,
    ):
        super().__init__()
        if isinstance(processor, (list, tuple)):
            # Diffusers may pass the raw model_index spec; let MLLMEncoder resolve it later.
            processor = None
        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            mllm_encoder=mllm_encoder,
            processor=processor,
            source_embedder=source_embedder,
            ref_embedder=ref_embedder,
        )
        if processor is not None:
            self.mllm_encoder.processor = processor

    # ------------------------------------------------------------------ #
    #                        Helper utilities                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_resize(height, width, num_frames, h_div=16, w_div=16, t_div=4, t_rem=1):
        """Round height/width/num_frames to valid values."""
        if height % h_div != 0:
            height = (height + h_div - 1) // h_div * h_div
        if width % w_div != 0:
            width = (width + w_div - 1) // w_div * w_div
        if num_frames % t_div != t_rem:
            num_frames = (num_frames + t_div - 1) // t_div * t_div + t_rem
        return height, width, num_frames

    @staticmethod
    def _preprocess_image(image: Image.Image, dtype, device):
        """Convert PIL Image to tensor in [-1, 1]."""
        arr = np.array(image, dtype=np.float32)
        tensor = torch.from_numpy(arr).to(dtype=dtype, device=device)
        tensor = tensor / 127.5 - 1.0  # [0, 255] -> [-1, 1]
        tensor = tensor.permute(2, 0, 1)  # H W C -> C H W
        return tensor

    def _preprocess_video(self, frames: List[Image.Image], dtype, device):
        """Convert list of PIL Images to tensor [1, C, T, H, W] in [-1, 1]."""
        tensors = [self._preprocess_image(f, dtype, device) for f in frames]
        video = torch.stack(tensors, dim=1)  # C T H W
        return video.unsqueeze(0)  # 1 C T H W

    @staticmethod
    def _vae_output_to_video(vae_output):
        """Convert VAE output tensor to list of PIL Images."""
        # vae_output shape: [B, C, T, H, W] or [T, H, W, C]
        if vae_output.dim() == 5:
            vae_output = vae_output.squeeze(0).permute(1, 2, 3, 0)  # T H W C
        frames = []
        for t in range(vae_output.shape[0]):
            frame = ((vae_output[t] + 1.0) * 127.5).clamp(0, 255)
            frame = frame.to(device="cpu", dtype=torch.uint8).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    # ------------------------------------------------------------------ #
    #                   Custom Flow Match Scheduler                       #
    # ------------------------------------------------------------------ #

    def _setup_scheduler(self, num_inference_steps, denoising_strength=1.0, shift=5.0):
        """
        Set up flow-match sigmas and timesteps matching the original diffsynth
        FlowMatchScheduler with extra_one_step=True and shift.
        """
        sigma_min = 0.003 / 1.002
        sigma_max = 1.0
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        # extra_one_step: generate N+1 points, drop last
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        # Apply shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * 1000  # num_train_timesteps = 1000
        return sigmas, timesteps

    def _scheduler_step(self, model_output, sigmas, step_index, sample):
        """Euler step for flow matching."""
        sigma = sigmas[step_index]
        if step_index + 1 >= len(sigmas):
            sigma_next = 0.0
        else:
            sigma_next = sigmas[step_index + 1]
        return sample + model_output * (sigma_next - sigma)

    def _scheduler_add_noise(self, original_samples, noise, sigmas, step_index):
        """Add noise at given timestep for img2img / video2video."""
        sigma = sigmas[step_index]
        return (1 - sigma) * original_samples + sigma * noise

    def _scheduler_get_sigma(self, timestep, sigmas, timesteps):
        """Get sigma for a given timestep."""
        timestep_id = torch.argmin((timesteps - timestep).abs())
        return sigmas[timestep_id]

    # ------------------------------------------------------------------ #
    #                    Transformer forward helpers                      #
    # ------------------------------------------------------------------ #

    def _model_forward(
        self,
        latents,
        timestep,
        context,
        vae_source_input=None,
        vae_ref_image=None,
        sigmas=None,
        timesteps_schedule=None,
    ):
        """
        Custom DiT forward pass that handles source/ref conditioning.
        Mirrors model_fn_wan_video from the original diffsynth pipeline.
        """
        device = latents.device
        dtype = latents.dtype
        t = self.transformer

        # --- Timestep embedding ---
        timestep_emb = sinusoidal_embedding_1d(
            t.config.freq_dim, timestep
        ).to(dtype)
        time_emb = t.condition_embedder.time_embedder(timestep_emb)
        # diffusers time_proj = Linear only (SiLU is applied separately)
        t_mod = t.condition_embedder.time_proj(F.silu(time_emb)).unflatten(
            1, (6, t.config.num_attention_heads * t.config.attention_head_dim)
        )

        # --- Text/context embedding ---
        # NOTE: Do NOT apply text_embedder here. The MLLM encoder's connector
        # already projects to dit_dim. text_embedder is for raw text encoder
        # output (text_dim → dim), which doesn't apply to MLLM output.

        # --- Patchify latents ---
        x = latents
        if vae_source_input is not None:
            vae_source_cond = self.source_embedder(vae_source_input)
            x = t.patch_embedding(x)
            # Get sigma for this timestep
            sigma = self._scheduler_get_sigma(timestep, sigmas, timesteps_schedule)
            x = x + vae_source_cond * sigma
        else:
            x = t.patch_embedding(x)

        f, h, w = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        # --- 3D RoPE frequencies (real-valued cos/sin format) ---
        rotary_emb = _build_rope_3d(t.rope, f, h, w, device)

        # --- Reference image conditioning ---
        vae_ref_input_length = 0
        if vae_ref_image is not None:
            if len(vae_ref_image) > 1:
                vae_ref = torch.cat(vae_ref_image, dim=2)  # concat along temporal
            else:
                vae_ref = vae_ref_image[0]

            vae_ref = self.ref_embedder(vae_ref)
            ref_f, ref_h, ref_w = vae_ref.shape[2:]
            vae_ref = rearrange(vae_ref, "b c f h w -> b (f h w) c").contiguous()

            # Recompute RoPE for extended sequence (main + ref tokens)
            total_f = f + ref_f
            rotary_emb = _build_rope_3d(t.rope, total_f, h, w, device)

            vae_ref_input_length = vae_ref.shape[1]

            if self.ref_embedder.config.ref_pad_first:
                x = torch.cat([vae_ref, x], dim=1)
            else:
                x = torch.cat([x, vae_ref], dim=1)

        # --- Transformer blocks ---
        for block in t.blocks:
            x = block(x, context, t_mod, rotary_emb)

        # --- Output head ---
        # Match diffusers' FP32 norm + modulation + projection
        table = t.scale_shift_table
        shift, scale = (
            table.to(device=device) + time_emb.unsqueeze(1)
        ).chunk(2, dim=1)
        shift = shift.to(device=x.device)
        scale = scale.to(device=x.device)
        x = (t.norm_out(x.float()) * (1 + scale) + shift).type_as(x)
        x = t.proj_out(x)

        # --- Remove ref tokens from output ---
        if vae_ref_image is not None and vae_ref_input_length > 0:
            if self.ref_embedder.config.ref_pad_first:
                x = x[:, vae_ref_input_length:, :]
            else:
                x = x[:, :-vae_ref_input_length, :]

        # --- Unpatchify ---
        patch_size = t.config.patch_size
        x = rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f, h=h, w=w,
            x=patch_size[0], y=patch_size[1], z=patch_size[2],
        )
        return x

    # ------------------------------------------------------------------ #
    #                          Main __call__                              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        source_video: Optional[List[Image.Image]] = None,
        source_input: Optional[List[Image.Image]] = None,
        ref_image: Optional[List[Image.Image]] = None,
        negative_prompt: Optional[str] = "",
        input_video: Optional[List[Image.Image]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        sigma_shift: float = 5.0,
        denoising_strength: float = 1.0,
        seed: Optional[int] = None,
        tiled: bool = True,
        tile_size: Tuple[int, int] = (30, 52),
        tile_stride: Tuple[int, int] = (15, 26),
        output_type: str = "pil",
        progress_bar: Callable = tqdm,
    ) -> List[Image.Image]:
        """
        Run KiwiEdit inference.

        Args:
            prompt: Editing instruction text.
            source_video: Source video/image frames for MLLM context (also used as
                source_input if source_input is not provided).
            source_input: Source frames for VAE conditioning. If None but source_video
                is provided, source_video is used.
            ref_image: Optional reference image(s) for guided editing.
            negative_prompt: Negative prompt for CFG.
            input_video: Optional input video for video-to-video (adds noise then denoises).
            height: Output height in pixels.
            width: Output width in pixels.
            num_frames: Number of output frames (1 for image editing).
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            sigma_shift: Flow matching shift parameter.
            denoising_strength: How much noise to add (1.0 = full noise).
            seed: Random seed for reproducibility.
            tiled: Whether to use tiled VAE encoding/decoding.
            tile_size: VAE tile size.
            tile_stride: VAE tile stride.
            output_type: "pil" for PIL Images, "latent" for raw latents.
            progress_bar: Progress bar callable (e.g., tqdm).

        Returns:
            List of PIL Images (video frames).
        """
        device = self._execution_device
        dtype = torch.bfloat16
        # --- 1. Shape check ---
        # VAE spatial factor is 16, transformer patch spatial is 2,
        # so pixel dims must be multiples of 32.
        height, width, num_frames = self._check_resize(
            height, width, num_frames, h_div=32, w_div=32
        )
        
        # --- 2. Determine VAE parameters ---
        z_dim = self.vae.config.z_dim
        # Compute upsampling factor from VAE config
        dim_mult = self.vae.config.get("dim_mult", [1, 2, 4, 4])
        temporal_downsample = self.vae.config.get("temperal_downsample", [False, True, True])
        # Wan VideoVAE spatial factor is 2^(len(dim_mult)) due to extra
        # downsampling in the encoder beyond the level transitions.
        spatial_factor = 2 ** len(dim_mult)  # 16 for 4 levels
        temporal_factor = 2 ** sum(temporal_downsample)  # 4 for [F, T, T]

        # --- 3. MLLM encoding ---
        context = None
        src_video_for_mllm = source_video
        if src_video_for_mllm is not None:
            self.mllm_encoder._ensure_qwen_loaded()
            if ref_image is not None:
                # Ref mode always uses the video path (even for a single frame)
                context = self.mllm_encoder(
                    prompt, src_video=src_video_for_mllm, ref_image=ref_image
                )
            elif len(src_video_for_mllm) == 1:
                context = self.mllm_encoder(
                    prompt, src_image=src_video_for_mllm
                )
            else:
                context = self.mllm_encoder(
                    prompt, src_video=src_video_for_mllm
                )
        # For negative prompt: use zero context
        context_nega = None

        # --- 4. Setup scheduler ---
        sigmas, timesteps = self._setup_scheduler(
            num_inference_steps, denoising_strength, sigma_shift
        )
        sigmas = sigmas.to(device)
        timesteps = timesteps.to(device)

        # --- 5. Initialize noise ---
        latent_length = (num_frames - 1) // temporal_factor + 1
        latent_h = height // spatial_factor
        latent_w = width // spatial_factor
        shape = (1, z_dim, latent_length, latent_h, latent_w)

        generator = None if seed is None else torch.Generator("cpu").manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32)
        noise = noise.to(dtype=dtype, device=device)

        # --- 6. Encode source input ---
        vae_source_input = None
        # Fall back to source_video if source_input not provided
        src_for_vae = source_input if source_input is not None else source_video
        if src_for_vae is not None:
            src_frames = [src_for_vae[i] for i in range(min(num_frames, len(src_for_vae)))]
            # Resize source frames to match the (possibly adjusted) target dimensions
            src_frames = [f.resize((width, height), Image.LANCZOS) for f in src_frames]
            src_tensor = self._preprocess_video(src_frames, dtype=torch.float32, device=device)
            vae_source_input = self.vae.encode(src_tensor).latent_dist.sample()
            vae_source_input = vae_source_input.to(dtype=dtype)

        # --- 7. Encode reference images ---
        vae_ref_image = None
        if ref_image is not None:
            vae_ref_image = []
            for item in ref_image:
                target_size = (width, height)
                item = ImageOps.pad(item, target_size, color="white", centering=(0.5, 0.5))
                ref_tensor = self._preprocess_video([item], dtype=torch.float32, device=device)
                ref_latent = self.vae.encode(ref_tensor).latent_dist.sample()
                vae_ref_image.append(ref_latent.to(dtype=dtype))

        # --- 8. Handle input_video (video-to-video) ---
        if input_video is not None:
            input_tensor = self._preprocess_video(input_video, dtype=torch.float32, device=device)
            input_latents = self.vae.encode(input_tensor).latent_dist.sample()
            input_latents = input_latents.to(dtype=dtype)
            latents = self._scheduler_add_noise(input_latents, noise, sigmas, 0)
        else:
            latents = noise

        # --- 9. Denoising loop ---
        for step_idx, timestep_val in enumerate(progress_bar(timesteps)):
            timestep = timestep_val.unsqueeze(0).to(dtype=dtype, device=device)

            # Positive prediction
            noise_pred = self._model_forward(
                latents=latents,
                timestep=timestep,
                context=context,
                vae_source_input=vae_source_input,
                vae_ref_image=vae_ref_image,
                sigmas=sigmas,
                timesteps_schedule=timesteps,
            )

            # CFG
            # if guidance_scale != 1.0:
            #     noise_pred_nega = self._model_forward(
            #         latents=latents,
            #         timestep=timestep,
            #         context=context_nega,
            #         vae_source_input=vae_source_input,
            #         vae_ref_image=vae_ref_image,
            #         sigmas=sigmas,
            #         timesteps_schedule=timesteps,
            #     )
            #     noise_pred = noise_pred_nega + guidance_scale * (
            #         noise_pred_posi - noise_pred_nega
            #     )
            # else:
            #     noise_pred = noise_pred_posi

            # Scheduler step
            latents = self._scheduler_step(noise_pred, sigmas, step_idx, latents)

        # --- 10. Decode ---
        if output_type == "latent":
            return latents

        video = self.vae.decode(latents).sample
        video = self._vae_output_to_video(video)
        return video
