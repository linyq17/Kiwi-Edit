#!/usr/bin/env python3
"""
Convert KiwiEdit weights from diffsynth format to HuggingFace diffusers format.

Usage:
    python convert_weights.py \
        --ckpt_path /path/to/kiwi_edit_lora.safetensors \
        --base_model_id Wan-AI/Wan2.2-TI2V-5B \
        --output_dir ./kiwi-edit-5b-diffusers \
        --mllm_model Qwen/Qwen2.5-VL-3B-Instruct

        
python convert_weights.py \
        --ckpt_path /home/svu/e1374517/workspace/refvie/models/ckpt/Wan2.2-TI2V-5B_stage3_mix_train_720p_81f_ref_pad_last/final.safetensors \
        --base_model_id Wan-AI/Wan2.2-TI2V-5B \
        --output_dir ./kiwi-edit-5b-diffusers \
        --mllm_model Qwen/Qwen2.5-VL-3B-Instruct
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

from wan_video_vae import VideoVAE_, VideoVAE38_

REQUIRED_PROCESSOR_ASSETS = [
    "added_tokens.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "qwen_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]

OPTIONAL_PROCESSOR_ASSETS = [
    "merges.txt",
]

WEIGHT_FILE_NAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "generation_config.json",
}

WEIGHT_FILE_PREFIXES = ("model-", "pytorch_model-")
WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin")


# ─────────────────────────────────────────────────────────────────────────────
# DiT State Dict Key Mapping: diffsynth → diffusers
# ─────────────────────────────────────────────────────────────────────────────

# Block-level patterns (use .format(i=block_idx))
BLOCK_KEY_MAP = {
    # Self-attention
    "blocks.{i}.self_attn.q.weight": "blocks.{i}.attn1.to_q.weight",
    "blocks.{i}.self_attn.q.bias": "blocks.{i}.attn1.to_q.bias",
    "blocks.{i}.self_attn.k.weight": "blocks.{i}.attn1.to_k.weight",
    "blocks.{i}.self_attn.k.bias": "blocks.{i}.attn1.to_k.bias",
    "blocks.{i}.self_attn.v.weight": "blocks.{i}.attn1.to_v.weight",
    "blocks.{i}.self_attn.v.bias": "blocks.{i}.attn1.to_v.bias",
    "blocks.{i}.self_attn.o.weight": "blocks.{i}.attn1.to_out.0.weight",
    "blocks.{i}.self_attn.o.bias": "blocks.{i}.attn1.to_out.0.bias",
    "blocks.{i}.self_attn.norm_q.weight": "blocks.{i}.attn1.norm_q.weight",
    "blocks.{i}.self_attn.norm_k.weight": "blocks.{i}.attn1.norm_k.weight",
    # Cross-attention
    "blocks.{i}.cross_attn.q.weight": "blocks.{i}.attn2.to_q.weight",
    "blocks.{i}.cross_attn.q.bias": "blocks.{i}.attn2.to_q.bias",
    "blocks.{i}.cross_attn.k.weight": "blocks.{i}.attn2.to_k.weight",
    "blocks.{i}.cross_attn.k.bias": "blocks.{i}.attn2.to_k.bias",
    "blocks.{i}.cross_attn.v.weight": "blocks.{i}.attn2.to_v.weight",
    "blocks.{i}.cross_attn.v.bias": "blocks.{i}.attn2.to_v.bias",
    "blocks.{i}.cross_attn.o.weight": "blocks.{i}.attn2.to_out.0.weight",
    "blocks.{i}.cross_attn.o.bias": "blocks.{i}.attn2.to_out.0.bias",
    "blocks.{i}.cross_attn.norm_q.weight": "blocks.{i}.attn2.norm_q.weight",
    "blocks.{i}.cross_attn.norm_k.weight": "blocks.{i}.attn2.norm_k.weight",
    # Image cross-attention (only present in I2V models)
    "blocks.{i}.cross_attn.k_img.weight": "blocks.{i}.attn2.add_k_proj.weight",
    "blocks.{i}.cross_attn.k_img.bias": "blocks.{i}.attn2.add_k_proj.bias",
    "blocks.{i}.cross_attn.v_img.weight": "blocks.{i}.attn2.add_v_proj.weight",
    "blocks.{i}.cross_attn.v_img.bias": "blocks.{i}.attn2.add_v_proj.bias",
    "blocks.{i}.cross_attn.norm_k_img.weight": "blocks.{i}.attn2.norm_added_k.weight",
    # FFN
    "blocks.{i}.ffn.0.weight": "blocks.{i}.ffn.net.0.proj.weight",
    "blocks.{i}.ffn.0.bias": "blocks.{i}.ffn.net.0.proj.bias",
    "blocks.{i}.ffn.2.weight": "blocks.{i}.ffn.net.2.weight",
    "blocks.{i}.ffn.2.bias": "blocks.{i}.ffn.net.2.bias",
    # Norms
    # diffsynth norm1 (self-attn) and norm2 (FFN) have elementwise_affine=False → no params
    # diffsynth norm3 (cross-attn, has params) maps to diffusers norm2
    "blocks.{i}.norm3.weight": "blocks.{i}.norm2.weight",
    "blocks.{i}.norm3.bias": "blocks.{i}.norm2.bias",
    # Modulation / scale_shift_table
    "blocks.{i}.modulation": "blocks.{i}.scale_shift_table",
}

# Top-level key mapping
TOP_LEVEL_KEY_MAP = {
    "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
    "text_embedding.0.bias": "condition_embedder.text_embedder.linear_1.bias",
    "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight",
    "text_embedding.2.bias": "condition_embedder.text_embedder.linear_2.bias",
    "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
    "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
    "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
    "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
    "time_projection.1.weight": "condition_embedder.time_proj.weight",
    "time_projection.1.bias": "condition_embedder.time_proj.bias",
    # Image embedder (for has_image_input models)
    "img_emb.proj.0.weight": "condition_embedder.image_embedder.norm1.weight",
    "img_emb.proj.0.bias": "condition_embedder.image_embedder.norm1.bias",
    "img_emb.proj.1.weight": "condition_embedder.image_embedder.ff.net.0.proj.weight",
    "img_emb.proj.1.bias": "condition_embedder.image_embedder.ff.net.0.proj.bias",
    "img_emb.proj.3.weight": "condition_embedder.image_embedder.ff.net.2.weight",
    "img_emb.proj.3.bias": "condition_embedder.image_embedder.ff.net.2.bias",
    "img_emb.proj.4.weight": "condition_embedder.image_embedder.norm2.weight",
    "img_emb.proj.4.bias": "condition_embedder.image_embedder.norm2.bias",
    # Patch embedding
    "patch_embedding.weight": "patch_embedding.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    # Head / output
    "head.modulation": "scale_shift_table",
    "head.head.weight": "proj_out.weight",
    "head.head.bias": "proj_out.bias",
}


def convert_dit_key(key_diffsynth: str) -> str:
    """Convert a single diffsynth DiT key to diffusers format."""
    # Try top-level mapping first
    if key_diffsynth in TOP_LEVEL_KEY_MAP:
        return TOP_LEVEL_KEY_MAP[key_diffsynth]

    # Try block-level mapping: extract block index
    parts = key_diffsynth.split(".")
    if parts[0] == "blocks" and len(parts) > 1:
        block_idx = parts[1]
        # Create template key with index 0
        template_key = ".".join(["blocks", "{i}"] + parts[2:])
        if template_key in BLOCK_KEY_MAP:
            return BLOCK_KEY_MAP[template_key].format(i=block_idx)

    return None  # Key not found in mapping


def merge_mllm_lora_into_qwen(qwen_sd, ckpt_sd, alpha=1.0):
    """
    Merge MLLM LoRA weights from checkpoint into Qwen model state dict.
    LoRA keys are expected as: mllm.<qwen_target>.lora_B.default.weight
    """
    lora_pairs = {}
    for key in ckpt_sd:
        if ".lora_B." in key and key.startswith("mllm."):
            parts = key.split(".")
            lora_b_idx = parts.index("lora_B")
            # Target = everything between "mllm." and ".lora_B"
            target_name = ".".join(parts[1:lora_b_idx])
            lora_a_key = key.replace(".lora_B.", ".lora_A.")
            lora_pairs[target_name] = (key, lora_a_key)
    merged_count = 0
    skipped = []
    for target_name, (lora_b_key, lora_a_key) in lora_pairs.items():
        target_name = target_name.replace("model.model", "model")
        weight_key = target_name + ".weight"
        
        if weight_key not in qwen_sd:
            raise ValueError(f"Target weight {weight_key} not found in Qwen model for LoRA merge")

        weight_up = ckpt_sd[lora_b_key].float()
        weight_down = ckpt_sd[lora_a_key].float()

        if len(weight_up.shape) == 4:
            weight_up = weight_up.squeeze(3).squeeze(2)
            weight_down = weight_down.squeeze(3).squeeze(2)
            weight_lora = (alpha * torch.mm(weight_up, weight_down)).unsqueeze(2).unsqueeze(3)
        else:
            weight_lora = alpha * torch.mm(weight_up, weight_down)

        qwen_sd[weight_key] = qwen_sd[weight_key].float() + weight_lora
        merged_count += 1

    print(f"  Merged {merged_count} LoRA tensors into Qwen model")
    if skipped:
        print(f"  Skipped {len(skipped)} LoRA targets not found in Qwen: {skipped[:5]}...")

    return qwen_sd


def _is_weight_file(fname: str) -> bool:
    if fname in WEIGHT_FILE_NAMES:
        return True
    if fname.startswith(WEIGHT_FILE_PREFIXES) and fname.endswith(WEIGHT_FILE_SUFFIXES):
        return True
    if fname.endswith(".safetensors") and fname.startswith("model"):
        return True
    if fname.endswith(".bin") and fname.startswith("pytorch_model"):
        return True
    return False


def validate_qwen_weights(mllm_dir: str):
    """Ensure mllm_encoder directory contains Qwen weight files."""
    if not os.path.isdir(mllm_dir):
        raise RuntimeError(f"Missing mllm_encoder directory at {mllm_dir}")
    files = [f for f in os.listdir(mllm_dir) if os.path.isfile(os.path.join(mllm_dir, f))]
    weight_shards = [
        f for f in files
        if f.startswith(("model", "pytorch_model"))
        and (f.endswith(".safetensors") or f.endswith(".bin"))
    ]
    if not weight_shards:
        raise RuntimeError(
            "Missing Qwen weight shards in mllm_encoder. Expected model.safetensors "
            "or model-*.safetensors (or pytorch_model*.bin)."
        )


def validate_processor_assets(processor_dir: str):
    """Ensure processor directory contains required tokenizer/processor assets."""
    missing = [
        f for f in REQUIRED_PROCESSOR_ASSETS
        if not os.path.isfile(os.path.join(processor_dir, f))
    ]
    if missing:
        raise RuntimeError(
            "Missing required processor assets in processor/: "
            + ", ".join(missing)
            + ". Re-run conversion and make sure transformers/huggingface_hub can fully save the Qwen processor."
        )


def convert_dit_state_dict(diffsynth_sd):
    """Convert diffsynth DiT state dict to diffusers format."""
    diffusers_sd = {}
    unmapped = []
    for key, value in diffsynth_sd.items():
        new_key = convert_dit_key(key)
        if new_key is not None:
            diffusers_sd[new_key] = value
        else:
            unmapped.append(key)
    if unmapped:
        print(f"  Warning: {len(unmapped)} unmapped DiT keys: {unmapped[:10]}...")
    return diffusers_sd


def _extract_lora_pairs(ckpt_sd):
    """Extract LoRA (B, A) pairs keyed by target module name."""
    lora_pairs = {}
    for key in ckpt_sd:
        if ".lora_B." not in key:
            continue
        parts = key.split(".")
        if "lora_B" not in parts:
            continue
        lora_b_idx = parts.index("lora_B")
        # Remove "default" after lora_B if present
        if len(parts) > lora_b_idx + 2:
            parts.pop(lora_b_idx + 1)
        parts.pop(lora_b_idx)
        if parts and parts[0] == "diffusion_model":
            parts.pop(0)
        if parts and parts[-1] == "weight":
            parts.pop(-1)
        target_name = ".".join(parts)
        lora_a_key = key.replace(".lora_B.", ".lora_A.")
        if lora_a_key not in ckpt_sd:
            continue
        lora_pairs[target_name] = (key, lora_a_key)
    return lora_pairs


def merge_dit_lora_into_state_dict(dit_sd, ckpt_sd, alpha=1.0, base_is_diffusers=False):
    """Merge DiT LoRA weights from checkpoint into a DiT state dict."""
    lora_pairs = _extract_lora_pairs(ckpt_sd)
    merged_count = 0
    skipped = []
    for target_name, (lora_b_key, lora_a_key) in lora_pairs.items():
        if target_name.startswith(("mllm.", "vae_condition.", "ref_vae_condition.")):
            continue
        weight_key = target_name + ".weight"
        if base_is_diffusers:
            weight_key = convert_dit_key(weight_key)
        if not weight_key or weight_key not in dit_sd:
            skipped.append(weight_key or target_name)
            continue

        weight_up = ckpt_sd[lora_b_key].float()
        weight_down = ckpt_sd[lora_a_key].float()

        if len(weight_up.shape) == 4:
            weight_up = weight_up.squeeze(3).squeeze(2)
            weight_down = weight_down.squeeze(3).squeeze(2)
            weight_lora = (alpha * torch.mm(weight_up, weight_down)).unsqueeze(2).unsqueeze(3)
        else:
            weight_lora = alpha * torch.mm(weight_up, weight_down)

        base_weight = dit_sd[weight_key]
        dit_sd[weight_key] = base_weight + weight_lora.to(dtype=base_weight.dtype)
        merged_count += 1

    if merged_count:
        print(f"  Merged {merged_count} DiT LoRA tensors into base weights")
    if skipped:
        print(f"  Skipped {len(skipped)} DiT LoRA targets not found in base weights: {skipped[:5]}...")
    return dit_sd


def extract_mllm_weights(full_sd):
    """Extract MLLM custom weights from the full checkpoint."""
    mllm_sd = {}
    # Map diffsynth keys → diffusers keys
    mllm_key_map = {
        "mllm.image_queries.q": "image_queries",
        "mllm.video_queries.q": "video_queries",
        "mllm.ref_queries.q": "ref_queries",
        # Handle both the typo ("connetor") and correct ("connector") spellings
        "mllm.connetor.0.weight": "connector.0.weight",
        "mllm.connetor.0.bias": "connector.0.bias",
        "mllm.connetor.2.weight": "connector.2.weight",
        "mllm.connetor.2.bias": "connector.2.bias",
        "mllm.connector.0.weight": "connector.0.weight",
        "mllm.connector.0.bias": "connector.0.bias",
        "mllm.connector.2.weight": "connector.2.weight",
        "mllm.connector.2.bias": "connector.2.bias",
        "mllm.ref_connector.0.weight": "ref_connector.0.weight",
        "mllm.ref_connector.0.bias": "ref_connector.0.bias",
        "mllm.ref_connector.2.weight": "ref_connector.2.weight",
        "mllm.ref_connector.2.bias": "ref_connector.2.bias",
    }
    for old_key, new_key in mllm_key_map.items():
        if old_key in full_sd:
            mllm_sd[new_key] = full_sd[old_key]
            print(f"  Extracted MLLM weight: {old_key} → {new_key} {full_sd[old_key].shape}")
    return mllm_sd


def extract_conditional_embedder_weights(full_sd, prefix):
    """Extract ConditionalEmbedder weights from the full checkpoint."""
    embedder_sd = {}
    for key in list(full_sd.keys()):
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            embedder_sd[new_key] = full_sd[key]
            print(f"  Extracted embedder weight: {key} → {new_key} {full_sd[key].shape}")
    return embedder_sd


def normalize_checkpoint_keys(ckpt_sd):
    """
    Normalize checkpoint keys to strip the 'pipe.' prefix that diffsynth
    leaves on non-DiT components when using remove_prefix_in_ckpt='pipe.dit.'.

    After diffsynth training with remove_prefix='pipe.dit.':
      - DiT keys:      blocks.0.self_attn.q.weight        (already clean)
      - MLLM keys:     pipe.mllm.image_queries.q           → mllm.image_queries.q
      - Embedder keys: pipe.vae_condition.weight            → vae_condition.weight
      - Ref embedder:  pipe.ref_vae_condition.weight        → ref_vae_condition.weight

    Also handles the case where remove_prefix='pipe.' was used, leaving
    DiT keys as dit.blocks.0.self_attn.q.weight → blocks.0.self_attn.q.weight
    """
    normalized = {}
    for key, value in ckpt_sd.items():
        new_key = key
        if new_key.startswith("pipe."):
            new_key = new_key[len("pipe."):]
        if new_key.startswith("dit."):
            new_key = new_key[len("dit."):]
        normalized[new_key] = value
    return normalized


def has_full_dit_weights(ckpt_sd):
    """Check if checkpoint contains full DiT weights (not just LoRA)."""
    dit_weight_keys = [k for k in ckpt_sd if k.startswith("blocks.") and "lora" not in k]
    # A full DiT has hundreds of keys; LoRA-only has none in blocks.* without lora
    return len(dit_weight_keys) > 50


def extract_dit_weights(ckpt_sd):
    """Extract DiT-related keys from a normalized checkpoint."""
    dit_sd = {}
    dit_prefixes = ("blocks.", "patch_embedding.", "text_embedding.", "time_embedding.",
                    "time_projection.", "img_emb.", "head.", "pos_embedding")
    for key, value in ckpt_sd.items():
        if any(key.startswith(p) for p in dit_prefixes):
            dit_sd[key] = value
    return dit_sd


def detect_model_variant(ckpt_path):
    """Auto-detect whether this is 5B or 14B based on checkpoint path."""
    if "14b" in ckpt_path.lower() or "14B" in ckpt_path:
        return "14B"
    return "5B"


# ─────────────────────────────────────────────────────────────────────────────
# Model configs
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORMER_CONFIGS = {
    "5B": {
        "_class_name": "WanTransformer3DModel",
        "patch_size": [1, 2, 2],
        "in_channels": 48,
        "out_channels": 48,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "text_dim": 4096,
        "freq_dim": 256,
        "ffn_dim": 14336,
        "num_layers": 30,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "rope_max_seq_len": 1024,
    },
    "14B": {
        "_class_name": "WanTransformer3DModel",
        "patch_size": [1, 2, 2],
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 40,
        "attention_head_dim": 128,
        "text_dim": 4096,
        "freq_dim": 256,
        "ffn_dim": 13824,
        "num_layers": 40,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "rope_max_seq_len": 1024,
    },
}

VAE_CONFIGS = {
    "5B": {
        "_class_name": "VAE",
        "z_dim": 48,
        "dim": 160,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_scales": [],
        "temperal_downsample": [False, True, True],
        "dropout": 0.0,
        "latents_mean": [
            -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
            -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
            -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
            -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
            -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
             0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
        ],
        "latents_std": [
            0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
            0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
            0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
            0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
            0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
            0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
        ],
    },
    "14B": {
        "_class_name": "VAE",
        "z_dim": 16,
        "dim": 96,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_scales": [],
        "temperal_downsample": [False, True, True],
        "dropout": 0.0,
        "latents_mean": [
            -0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
             0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921,
        ],
        "latents_std": [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ],
    },
}

SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "shift": 5.0,
    "num_train_timesteps": 1000,
    "base_shift": 0.5,
    "max_shift": 1.15,
}


def main():
    parser = argparse.ArgumentParser(description="Convert KiwiEdit weights to diffusers format")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to KiwiEdit checkpoint (.safetensors or .pth). "
                             "Supports both full weight and LoRA-only checkpoints.")
    parser.add_argument("--base_model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B",
                        help="HuggingFace model ID for base Wan model")
    parser.add_argument("--base_dit_path", type=str, default=None,
                        help="Path to base DiT weights (if already downloaded). "
                             "If not provided, will download from base_model_id")
    parser.add_argument("--base_vae_path", type=str, default=None,
                        help="Path to base VAE weights (if already downloaded)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for converted model")
    parser.add_argument("--variant", type=str, choices=["5B", "14B"], default=None,
                        help="Model variant (auto-detected from ckpt_path if not specified)")
    parser.add_argument("--mllm_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="HuggingFace model ID for Qwen VL model")
    parser.add_argument("--lora_alpha", type=float, default=1.0,
                        help="LoRA merge alpha")
    parser.add_argument("--num_image_queries", type=int, default=256)
    parser.add_argument("--num_video_queries", type=int, default=512)
    parser.add_argument("--num_ref_queries", type=int, default=768)
    parser.add_argument("--max_object_token", type=int, default=768)
    parser.add_argument("--ref_pad_first", action="store_true", default=False)
    args = parser.parse_args()

    # Auto-detect variant
    variant = args.variant or detect_model_variant(args.ckpt_path)
    print(f"Model variant: {variant}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # Step 1: Load checkpoint
    # ─────────────────────────────────────────────────────────────────
    print("\n=== Step 1: Loading checkpoint ===")
    if args.ckpt_path.endswith(".safetensors"):
        ckpt_sd = load_file(args.ckpt_path, device="cpu")
    else:
        ckpt_sd = torch.load(args.ckpt_path, map_location="cpu")
        if "state_dict" in ckpt_sd:
            ckpt_sd = ckpt_sd["state_dict"]
    print(f"  Loaded {len(ckpt_sd)} keys from checkpoint")

    # Normalize keys: strip 'pipe.' and 'dit.' prefixes left by diffsynth training
    ckpt_sd = normalize_checkpoint_keys(ckpt_sd)
    print(f"  After key normalization: {len(ckpt_sd)} keys")

    # Detect checkpoint type: full weights vs LoRA-only
    is_full_weights = has_full_dit_weights(ckpt_sd)
    lora_keys = {k: v for k, v in ckpt_sd.items() if "lora_A" in k or "lora_B" in k}
    dit_lora_keys = {
        k for k in ckpt_sd
        if ".lora_B." in k
        and not k.startswith("mllm.")
        and not k.startswith("vae_condition.")
        and not k.startswith("ref_vae_condition.")
    }
    print(f"  Checkpoint type: {'full weights' if is_full_weights else 'LoRA-only'}"
          f" ({len(lora_keys)} LoRA keys)")

    # ─────────────────────────────────────────────────────────────────
    # Step 2: Build DiT state dict (merge DiT LoRA if present)
    # ─────────────────────────────────────────────────────────────────
    if is_full_weights:
        # Full weight checkpoint: extract DiT weights and convert to diffusers
        print("\n=== Step 2: Converting full DiT weights from checkpoint ===")
        dit_diffsynth_sd = extract_dit_weights(ckpt_sd)
        print(f"  Extracted {len(dit_diffsynth_sd)} DiT keys from checkpoint")

        if dit_lora_keys:
            print(f"  Found {len(dit_lora_keys)} DiT LoRA-B keys, merging into DiT (alpha={args.lora_alpha})...")
            dit_diffsynth_sd = merge_dit_lora_into_state_dict(
                dit_diffsynth_sd, ckpt_sd, alpha=args.lora_alpha, base_is_diffusers=False
            )
        print("  Converting DiT state dict keys from diffsynth → diffusers...")
        dit_diffusers_sd = convert_dit_state_dict(dit_diffsynth_sd)
    else:
        # No full DiT weights in checkpoint: load base model as-is
        print("\n=== Step 2: Loading base DiT ===")
        base_is_diffusers = False
        if args.base_dit_path:
            if args.base_dit_path.endswith(".safetensors"):
                base_dit_sd = load_file(args.base_dit_path, device="cpu")
            else:
                base_dit_sd = torch.load(args.base_dit_path, map_location="cpu")
            base_is_diffusers = any("condition_embedder" in k for k in base_dit_sd)
        else:
            is_local = os.path.isdir(args.base_model_id)
            has_transformer_subfolder = is_local and os.path.isdir(os.path.join(args.base_model_id, "transformer"))

            if is_local and not has_transformer_subfolder:
                import glob as glob_mod
                print(f"  Loading base DiT from local flat directory {args.base_model_id}...")
                shard_files = sorted(glob_mod.glob(os.path.join(args.base_model_id, "diffusion_pytorch_model*.safetensors")))
                shard_files = [f for f in shard_files if not f.endswith(".index.json")]
                base_dit_sd = {}
                for f in shard_files:
                    print(f"    Loading shard: {os.path.basename(f)}")
                    shard = load_file(f, device="cpu")
                    base_dit_sd.update(shard)
                base_is_diffusers = any("condition_embedder" in k for k in base_dit_sd)
            else:
                from diffusers import WanTransformer3DModel
                subfolder = "transformer" if has_transformer_subfolder or not is_local else None
                print(f"  Loading base DiT from {args.base_model_id} via diffusers...")
                base_dit_model = WanTransformer3DModel.from_pretrained(
                    args.base_model_id, subfolder=subfolder, torch_dtype=torch.float32
                )
                base_dit_sd = base_dit_model.state_dict()
                del base_dit_model
                base_is_diffusers = True
        print(f"  Base DiT has {len(base_dit_sd)} keys (format: {'diffusers' if base_is_diffusers else 'diffsynth'})")

        if base_is_diffusers:
            if dit_lora_keys:
                print(f"  Found {len(dit_lora_keys)} DiT LoRA-B keys, merging into base DiT (alpha={args.lora_alpha})...")
                base_dit_sd = merge_dit_lora_into_state_dict(
                    base_dit_sd, ckpt_sd, alpha=args.lora_alpha, base_is_diffusers=True
                )
            dit_diffusers_sd = base_dit_sd
        else:
            if dit_lora_keys:
                print(f"  Found {len(dit_lora_keys)} DiT LoRA-B keys, merging into base DiT (alpha={args.lora_alpha})...")
                base_dit_sd = merge_dit_lora_into_state_dict(
                    base_dit_sd, ckpt_sd, alpha=args.lora_alpha, base_is_diffusers=False
                )
            print("  Converting DiT state dict keys from diffsynth → diffusers...")
            dit_diffusers_sd = convert_dit_state_dict(base_dit_sd)

    print(f"  Final DiT has {len(dit_diffusers_sd)} keys")

    # Save transformer
    transformer_dir = os.path.join(args.output_dir, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)
    save_file(dit_diffusers_sd, os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors"))
    with open(os.path.join(transformer_dir, "config.json"), "w") as f:
        json.dump(TRANSFORMER_CONFIGS[variant], f, indent=2)
    print(f"  Saved transformer to {transformer_dir}")

    # ─────────────────────────────────────────────────────────────────
    # Step 3: Extract MLLM custom weights
    # ─────────────────────────────────────────────────────────────────
    print("\n=== Step 3: Extracting MLLM custom weights ===")
    mllm_sd = extract_mllm_weights(ckpt_sd)

    mllm_dir = os.path.join(args.output_dir, "mllm_encoder")
    os.makedirs(mllm_dir, exist_ok=True)
    processor_dir = os.path.join(args.output_dir, "processor")
    os.makedirs(processor_dir, exist_ok=True)

    if mllm_sd:
        save_file(mllm_sd, os.path.join(mllm_dir, "diffusion_pytorch_model.safetensors"))
        print(f"  Saved {len(mllm_sd)} MLLM custom weights")

    # Determine hidden_size from query shape
    hidden_size = 2048  # default for Qwen2.5-VL-3B
    for key in ["image_queries", "video_queries", "ref_queries"]:
        if key in mllm_sd:
            hidden_size = mllm_sd[key].shape[-1]
            break

    # Download Qwen VL model, merge MLLM LoRA if present, and save.
    # Split files into:
    #   - mllm_encoder/ : Qwen weights
    #   - processor/   : tokenizer/processor assets (incl. qwen_config.json)
    print(f"  Loading Qwen VL model from {args.mllm_model}...")
    from transformers import AutoProcessor
    from mllm_encoder import Qwen2_5_VLForConditionalGeneration

    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.mllm_model, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    # Merge MLLM LoRA weights into Qwen model
    mllm_lora_keys = {k for k in ckpt_sd if ".lora_B." in k and k.startswith("mllm.")}
    if mllm_lora_keys:
        print(f"  Found {len(mllm_lora_keys)} MLLM LoRA-B keys, merging into Qwen (alpha={args.lora_alpha})...")
        qwen_sd = qwen_model.state_dict()
        qwen_sd = merge_mllm_lora_into_qwen(qwen_sd, ckpt_sd, alpha=args.lora_alpha)
        qwen_model.load_state_dict(qwen_sd)

    # Save Qwen model into a temp dir, then split files into mllm_encoder/ + processor/
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        qwen_model.save_pretrained(tmp_dir)
        del qwen_model
        qwen_processor = AutoProcessor.from_pretrained(args.mllm_model)
        qwen_processor.save_pretrained(tmp_dir)

        for fname in os.listdir(tmp_dir):
            src_f = os.path.join(tmp_dir, fname)
            if not os.path.isfile(src_f):
                continue
            if fname == "config.json":
                shutil.copy2(src_f, os.path.join(processor_dir, "qwen_config.json"))
                continue
            if _is_weight_file(fname):
                shutil.copy2(src_f, os.path.join(mllm_dir, fname))
                continue
            shutil.copy2(src_f, os.path.join(processor_dir, fname))

    validate_qwen_weights(mllm_dir)
    validate_processor_assets(processor_dir)
    print(f"  Saved Qwen VL weights into {mllm_dir}")
    print(f"  Saved Qwen processor assets into {processor_dir}")

    # Write MLLM config (split layout: weights in mllm_encoder/, processor in processor/)
    dit_dim = TRANSFORMER_CONFIGS[variant]["num_attention_heads"] * TRANSFORMER_CONFIGS[variant]["attention_head_dim"]
    mllm_config = {
        "_class_name": "MLLMEncoder",
        "mllm_model_path": ".",
        "dit_dim": dit_dim,
        "hidden_size": hidden_size,
        "num_image_queries": args.num_image_queries,
        "num_video_queries": args.num_video_queries,
        "num_ref_queries": args.num_ref_queries,
        "max_object_token": args.max_object_token,
        "max_frames": 16,
        "max_pixels_per_frame": 262144,
    }
    with open(os.path.join(mllm_dir, "config.json"), "w") as f:
        json.dump(mllm_config, f, indent=2)
    print(f"  MLLM config: hidden_size={hidden_size}, dit_dim={dit_dim}")

    # ─────────────────────────────────────────────────────────────────
    # Step 4: Extract ConditionalEmbedder weights
    # ─────────────────────────────────────────────────────────────────
    print("\n=== Step 4: Extracting ConditionalEmbedder weights ===")
    in_dim = TRANSFORMER_CONFIGS[variant]["in_channels"]
    patch_size = TRANSFORMER_CONFIGS[variant]["patch_size"]

    # Source embedder
    source_sd = extract_conditional_embedder_weights(ckpt_sd, "vae_condition.")
    source_dir = os.path.join(args.output_dir, "source_embedder")
    os.makedirs(source_dir, exist_ok=True)
    if source_sd:
        save_file(source_sd, os.path.join(source_dir, "diffusion_pytorch_model.safetensors"))
    source_config = {
        "_class_name": "ConditionalEmbedder",
        "in_dim": in_dim,
        "dim": dit_dim,
        "patch_size": patch_size,
        "zero_init": True,
        "ref_pad_first": False,
    }
    with open(os.path.join(source_dir, "config.json"), "w") as f:
        json.dump(source_config, f, indent=2)

    # Ref embedder
    ref_sd = extract_conditional_embedder_weights(ckpt_sd, "ref_vae_condition.")
    ref_dir = os.path.join(args.output_dir, "ref_embedder")
    os.makedirs(ref_dir, exist_ok=True)
    if ref_sd:
        save_file(ref_sd, os.path.join(ref_dir, "diffusion_pytorch_model.safetensors"))
    ref_config = {
        "_class_name": "ConditionalEmbedder",
        "in_dim": in_dim,
        "dim": dit_dim,
        "patch_size": patch_size,
        "zero_init": True,
        "ref_pad_first": args.ref_pad_first,
    }
    with open(os.path.join(ref_dir, "config.json"), "w") as f:
        json.dump(ref_config, f, indent=2)

    print(f"  Saved source_embedder to {source_dir}")
    print(f"  Saved ref_embedder to {ref_dir}")

    # ─────────────────────────────────────────────────────────────────
    # Step 5: Handle VAE (store original VAE weights in safetensors)
    # ─────────────────────────────────────────────────────────────────
    print("\n=== Step 5: Handling VAE ===")
    vae_dir = os.path.join(args.output_dir, "vae")
    os.makedirs(vae_dir, exist_ok=True)

    # Find the VAE .pth file
    vae_src = args.base_vae_path
    if not vae_src:
        is_local = os.path.isdir(args.base_model_id)
        if is_local:
            import glob as glob_mod
            vae_candidates = (
                glob_mod.glob(os.path.join(args.base_model_id, "*VAE*.pth"))
                + glob_mod.glob(os.path.join(args.base_model_id, "*vae*.pth"))
            )
            if vae_candidates:
                vae_src = vae_candidates[0]
        elif "/" in args.base_model_id:
            try:
                from huggingface_hub import snapshot_download
                import glob as glob_mod

                print(f"  Downloading VAE weights from {args.base_model_id}...")
                snapshot = snapshot_download(
                    args.base_model_id,
                    allow_patterns=["*VAE*.pth", "*vae*.pth"],
                )
                vae_candidates = set(
                    glob_mod.glob(os.path.join(snapshot, "**", "*VAE*.pth"), recursive=True)
                    + glob_mod.glob(os.path.join(snapshot, "**", "*vae*.pth"), recursive=True)
                )
                if vae_candidates:
                    vae_src = sorted(vae_candidates)[0]
            except Exception as exc:
                print(f"  Warning: failed to download VAE weights from {args.base_model_id} ({exc})")

    if vae_src and os.path.isfile(vae_src):
        # Store VAE weights in diffusion_pytorch_model.safetensors so HF Hub download works
        vae_pth_name = "diffusion_pytorch_model.safetensors"
        dst = os.path.join(vae_dir, vae_pth_name)
        vae_sd = torch.load(vae_src, map_location="cpu")
        if isinstance(vae_sd, dict) and "state_dict" in vae_sd:
            vae_sd = vae_sd["state_dict"]
        elif isinstance(vae_sd, dict) and "model_state" in vae_sd:
            vae_sd = vae_sd["model_state"]
        cleaned = {}
        for k, v in vae_sd.items():
            nk = k
            for prefix in ("model.", "module.", "vae."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v
        # Filter to expected keys to avoid unused-weight warnings in diffusers.
        if variant == "5B":
            vae_model = VideoVAE38_(
                dim=VAE_CONFIGS[variant]["dim"],
                z_dim=VAE_CONFIGS[variant]["z_dim"],
                dim_mult=VAE_CONFIGS[variant]["dim_mult"],
                num_res_blocks=VAE_CONFIGS[variant]["num_res_blocks"],
                attn_scales=VAE_CONFIGS[variant]["attn_scales"],
                temperal_downsample=VAE_CONFIGS[variant]["temperal_downsample"],
                dropout=VAE_CONFIGS[variant]["dropout"],
            )
        else:
            vae_model = VideoVAE_(
                dim=VAE_CONFIGS[variant]["dim"],
                z_dim=VAE_CONFIGS[variant]["z_dim"],
                dim_mult=VAE_CONFIGS[variant]["dim_mult"],
                num_res_blocks=VAE_CONFIGS[variant]["num_res_blocks"],
                attn_scales=VAE_CONFIGS[variant]["attn_scales"],
                temperal_downsample=VAE_CONFIGS[variant]["temperal_downsample"],
                dropout=VAE_CONFIGS[variant]["dropout"],
            )
        expected_keys = set(vae_model.state_dict().keys())
        del vae_model
        extra_keys = sorted(set(cleaned.keys()) - expected_keys)
        if extra_keys:
            print(f"  Warning: dropping {len(extra_keys)} unexpected VAE keys: {extra_keys[:5]}...")
            for k in extra_keys:
                cleaned.pop(k, None)
        missing_keys = expected_keys - cleaned.keys()
        if missing_keys:
            missing_list = list(missing_keys)
            raise RuntimeError(
                f"VAE weights are incomplete: missing {len(missing_list)} keys "
                f"(e.g., {missing_list[:5]})."
            )
        save_file(cleaned, dst)
        print(f"  Saved VAE weights as safetensors: {vae_src} → {dst}")
    else:
        raise RuntimeError(
            "No VAE .pth file found. Provide --base_vae_path or ensure "
            f"{args.base_model_id} contains a *VAE*.pth file."
        )

    # Write config for VAE wrapper
    vae_cfg = VAE_CONFIGS[variant].copy()
    vae_cfg["vae_pth"] = vae_pth_name
    with open(os.path.join(vae_dir, "config.json"), "w") as f:
        json.dump(vae_cfg, f, indent=2)

    print(f"  Saved VAE config to {vae_dir}")

    # ─────────────────────────────────────────────────────────────────
    # Step 6: Save scheduler config
    # ─────────────────────────────────────────────────────────────────
    print("\n=== Step 6: Saving scheduler config ===")
    scheduler_dir = os.path.join(args.output_dir, "scheduler")
    os.makedirs(scheduler_dir, exist_ok=True)
    with open(os.path.join(scheduler_dir, "scheduler_config.json"), "w") as f:
        json.dump(SCHEDULER_CONFIG, f, indent=2)

    # ─────────────────────────────────────────────────────────────────
    # Step 7: Copy pipeline files and create model_index.json
    # ─────────────────────────────────────────────────────────────────
    print("\n=== Step 7: Copying pipeline files ===")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_files = [
        "__init__.py",
        "pipeline_kiwi_edit.py",
        "mllm_encoder.py",
        "conditional_embedder.py",
        "wan_video_vae.py",
    ]
    for fname in pipeline_files:
        src = os.path.join(script_dir, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")
        else:
            print(f"  Warning: {fname} not found at {src}")

    # Diffusers looks for custom component modules inside each component's
    # subfolder (e.g. vae/wan_video_vae.py), so copy them there too.
    component_module_map = {
        "vae": "wan_video_vae.py",
        "mllm_encoder": "mllm_encoder.py",
        "source_embedder": "conditional_embedder.py",
        "ref_embedder": "conditional_embedder.py",
    }
    for subfolder, module_file in component_module_map.items():
        src = os.path.join(script_dir, module_file)
        dst = os.path.join(args.output_dir, subfolder, module_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {module_file} → {subfolder}/{module_file}")
        else:
            print(f"  Warning: {module_file} not found at {src}")

    # Validate required custom module files exist in output
    required_files = [
        os.path.join(args.output_dir, "pipeline_kiwi_edit.py"),
        os.path.join(args.output_dir, "mllm_encoder.py"),
        os.path.join(args.output_dir, "conditional_embedder.py"),
        os.path.join(args.output_dir, "wan_video_vae.py"),
        os.path.join(args.output_dir, "mllm_encoder", "mllm_encoder.py"),
        os.path.join(args.output_dir, "source_embedder", "conditional_embedder.py"),
        os.path.join(args.output_dir, "ref_embedder", "conditional_embedder.py"),
        os.path.join(args.output_dir, "vae", "wan_video_vae.py"),
    ]
    missing_files = [p for p in required_files if not os.path.isfile(p)]
    if missing_files:
        missing_preview = "\n  - " + "\n  - ".join(missing_files[:8])
        raise RuntimeError(
            "Missing required custom component files in the output folder."
            " Ensure you run convert_weights.py from the repo root and upload the full folder."
            f"{missing_preview}"
        )

    model_index = {
        "_class_name": ["pipeline_kiwi_edit", "KiwiEditPipeline"],
        "_diffusers_version": "0.32.0",
        "processor": ["transformers", "AutoProcessor"],
        "transformer": ["diffusers", "WanTransformer3DModel"],
        "vae": ["wan_video_vae", "VAE"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "mllm_encoder": ["mllm_encoder", "MLLMEncoder"],
        "source_embedder": ["conditional_embedder", "ConditionalEmbedder"],
        "ref_embedder": ["conditional_embedder", "ConditionalEmbedder"],
    }
    with open(os.path.join(args.output_dir, "model_index.json"), "w") as f:
        json.dump(model_index, f, indent=2)

    print(f"\n=== Done! Model saved to {args.output_dir} ===")
    print(f"\nTo load:")
    print(f"  from diffusers import DiffusionPipeline")
    print(f'  pipe = DiffusionPipeline.from_pretrained("{args.output_dir}", trust_remote_code=True)')
    print("\nTo upload, prefer uploading the full folder (huggingface-cli upload / HfApi.upload_folder).")
    print("Avoid relying on pipe.push_to_hub() for this project because the processor folder must be uploaded too.")


if __name__ == "__main__":
    main()
