# KiwiEdit - Diffusers Integration

Convert models trained with the refvie framework into HuggingFace diffusers format, upload to HuggingFace Hub, and run inference with a simple API.

## Prerequisites

```bash
pip install torch diffusers transformers safetensors huggingface_hub einops pillow tqdm
```

## Step 1: Convert Checkpoint

Convert a refvie-trained checkpoint (with LoRA weights) into diffusers format.

### Minimal usage (auto-downloads base model and VAE from HuggingFace):

```bash
python convert_weights.py \
    --ckpt_path path_to_ckpt \
    --base_vae_path path_to_vae \
    --output_dir path_to_output \
    --mllm_model Qwen/Qwen2.5-VL-3B-Instruct
```

### Full options:

```bash
python convert_weights.py \
    --ckpt_path path_to_ckpt \
    --base_model_id Wan-AI/Wan2.2-TI2V-5B \
    --base_vae_path path_to_vae \
    --output_dir path_to_output \
    --variant 5B \
    --mllm_model Qwen/Qwen2.5-VL-3B-Instruct \
    --lora_alpha 1.0 \
    --num_image_queries 256 \
    --num_video_queries 512 \
    --num_ref_queries 768 \
    --ref_pad_first
```

| Argument | Description |
|---|---|
| `--ckpt_path` | Path to refvie LoRA checkpoint (`.safetensors` or `.pth`) |
| `--base_model_id` | HuggingFace model ID for base Wan model |
| `--base_dit_path` | Local path to base DiT weights (skips download) |
| `--base_vae_path` | Local path to base VAE weights or directory |
| `--output_dir` | Output directory for converted model |
| `--variant` | `5B` or `14B` (auto-detected from ckpt_path if omitted) |
| `--mllm_model` | HuggingFace model ID for Qwen VL encoder |
| `--lora_alpha` | LoRA merge strength (default: 1.0) |
| `--ref_pad_first` | Prepend ref tokens instead of appending |

The conversion script will:
1. Load and merge LoRA weights into the base DiT
2. Extract MLLM encoder weights (queries + connectors)
3. Extract source/ref conditional embedder weights
4. Copy VAE weights
5. Save scheduler config
6. Copy all pipeline code files for `trust_remote_code` loading

### Output structure:

```
kiwi-edit-5b-diffusers/
  model_index.json
  __init__.py
  pipeline_kiwi_edit.py
  kiwi_mllm_encoder.py
  kiwi_conditional_embedder.py
  modeling_qwen2_5_vl.py
  qwen_vl_utils.py
  transformer/
    config.json
    diffusion_pytorch_model.safetensors
  vae/
    config.json
    diffusion_pytorch_model.safetensors
  mllm_encoder/
    config.json
    diffusion_pytorch_model.safetensors
  source_embedder/
    config.json
    diffusion_pytorch_model.safetensors
  ref_embedder/
    config.json
    diffusion_pytorch_model.safetensors
  scheduler/
    scheduler_config.json
```

## Step 2: Upload to HuggingFace Hub

### Option A: Using huggingface-cli

```bash
# Login first
huggingface-cli login

# Upload the entire directory
huggingface-cli upload your-username/kiwi-edit-5b ./kiwi-edit-5b-diffusers .
```

### Option B: Using Python API

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo("your-username/kiwi-edit-5b", exist_ok=True)
api.upload_folder(
    folder_path="./kiwi-edit-5b-diffusers",
    repo_id="your-username/kiwi-edit-5b",
    repo_type="model",
)
```

## Step 3: Test the Model

### Load from local directory:

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "./kiwi-edit-5b-diffusers",
    trust_remote_code=True,
    torch_dtype="auto",
)
pipe.to("cuda")

# Load the Qwen VL backbone (loaded lazily on first call)
pipe.mllm_encoder.load_qwen_model(device="cuda")
```

### Load from HuggingFace Hub:

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "your-username/kiwi-edit-5b",
    trust_remote_code=True,
    torch_dtype="auto",
)
pipe.to("cuda")
pipe.mllm_encoder.load_qwen_model(device="cuda")
```

### Video editing (instruction-only):

```python
from PIL import Image

# Load source video frames
src_frames = [Image.open(f"frame_{i:04d}.png") for i in range(81)]

result = pipe(
    prompt="Make the sky sunset orange",
    source_video=src_frames,
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=50,
    seed=42,
)

# Save output frames
for i, frame in enumerate(result):
    frame.save(f"output_{i:04d}.png")
```

### Image editing:

```python
src_image = Image.open("source.png").resize((832, 480))

result = pipe(
    prompt="Add sunglasses to the person",
    source_video=[src_image],
    height=480,
    width=832,
    num_frames=1,
    num_inference_steps=50,
    seed=42,
)

result[0].save("edited.png")
```

### Reference-guided video editing:

```python
src_frames = [Image.open(f"frame_{i:04d}.png") for i in range(81)]
ref_img = Image.open("reference_style.png")

result = pipe(
    prompt="Apply the style from the reference image",
    source_video=src_frames,
    ref_image=[ref_img],
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=50,
    seed=42,
)
```

## Pipeline Parameters

| Parameter | Default | Description |
|---|---|---|
| `prompt` | (required) | Editing instruction |
| `source_video` | `None` | Source frames for MLLM context + VAE conditioning |
| `source_input` | `None` | Separate source for VAE conditioning (falls back to `source_video`) |
| `ref_image` | `None` | Reference image(s) for guided editing |
| `input_video` | `None` | Input video for video-to-video mode |
| `height` | `480` | Output height (rounded to multiple of 16) |
| `width` | `832` | Output width (rounded to multiple of 16) |
| `num_frames` | `81` | Number of output frames |
| `num_inference_steps` | `50` | Denoising steps |
| `sigma_shift` | `5.0` | Flow matching shift |
| `denoising_strength` | `1.0` | Noise level (1.0 = full, <1.0 for img2img/vid2vid) |
| `seed` | `None` | Random seed |
| `output_type` | `"pil"` | `"pil"` for images, `"latent"` for raw latents |

## Architecture

KiwiEdit uses a custom pipeline built on top of diffusers components:

- **Transformer**: `WanTransformer3DModel` (diffusers native) - DiT backbone
- **VAE**: `AutoencoderKLWan` (diffusers native) - 3D causal VAE
- **MLLM Encoder**: `KiwiEditMLLMEncoder` (custom) - Qwen2.5-VL with learnable queries
- **Source Embedder**: `KiwiEditConditionalEmbedder` (custom) - Conv3d for source conditioning
- **Ref Embedder**: `KiwiEditConditionalEmbedder` (custom) - Conv3d for reference conditioning
- **Scheduler**: `FlowMatchEulerDiscreteScheduler` (diffusers native, with custom step logic)
