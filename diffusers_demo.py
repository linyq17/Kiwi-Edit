"""
Minimal demo case for kiwi-edit-5b-diffusers pipeline.

Usage:
    # Instruction Only:
    python diffusers_demo.py \
        --video_path ./demo_data/video/source/0005e4ad9f49814db1d3f2296b911abf.mp4 \
        --prompt "Remove the monkey." \
        --save_path output.mp4 --model_path linyq/kiwi-edit-5b-instruct-only-diffusers

    # With reference image:
    python diffusers_demo.py \
        --video_path ./demo_data/video/source/7800630815900c243b81f8ed5acb6a9dd40b47fb18b48373bcf132039ffb0a8d.mp4 \
        --prompt "Add a stack of grey balancing stones on the left side of the monkey." \
        --ref_image ./demo_data/video/reference/reco_replace_088301.jpg \
        --save_path ref_output.mp4 --model_path linyq/kiwi-edit-5b-reference-only-diffusers

"""
# Tips for prompt engineering for different editing tasks: 
#   The more detailed and specific the prompt, the better the model can understand the desired edit. 
#   Here are some enhanced prompt templates for different editing tasks:
task_enhanced_prompt = {
    # Global Style: Focus on temporal stability and motion preservation
    "global_style": [
        "Ensure seamless temporal consistency across all frames of the video.",
        "Retain the original motion, character actions, and camera movements throughout the sequence.",
        "Preserve the video's narrative flow and structural coherence to keep the original intent intact.",
        "Maintain strict frame-by-frame consistency to ensure visual harmony.",
        "Eliminate flickering or abrupt style changes between consecutive frames.",
        "The model must preserve the dynamic interplay of light and shadow from the source footage."
    ],
    
    # Local Change: Focus on spatial anchoring and pose alignment
    "local_change": [
        "Ensure the object maintains the exact same position and pose within the video scene.",
        "The modified element must stay aligned with the subject's original physical orientation.",
        "Keep the same pose and position for the subject throughout the entire video.",
        "The new attire or object must fit the subject's pose and spatial coordinates perfectly.",
        "Maintain the original object's dimensions and perspective during the replacement process."
    ],
    
    # Background Change: Focus on foreground locking and environmental dynamics
    "background_change": [
        "The subject in the foreground must remain perfectly still throughout the video.",
        "The person and any foreground objects should stay static and unchanged.",
        "Ensure the foreground subject remains perfectly still while the background transforms.",
        "Include subtle movements of environmental elements, such as shifting sunlight and shadows.",
        "Transform the background into a dynamic scene without altering the narrative flow of the foreground."
    ],
    
    # Local Remove: Focus on temporal inpainting and background reconstruction
    "local_remove": [
        "The background must be reconstructed with temporal consistency to match the original context.",
        "All other video content must remain entirely unchanged after the object is removed.",
        "Perform the removal using temporally consistent background inpainting techniques.",
        "Ensure the background is inpainted smoothly across all frames to avoid visual artifacts.",
        "The removal of the subject must leave the surrounding environment structurally intact."
    ],
    
    # Local Add: Focus on tracking, lighting adaptation, and physics
    "local_add": [
        "The added object must be perfectly tracked to the specified surface as the camera moves.",
        "Maintain consistent shadows and lighting for the added object across all frames.",
        "All other parts of the video must remain unchanged after the new object is overlaid.",
        "Reflections and shadows must dynamically adapt to the changing light in the environment.",
        "The added subject should exhibit subtle natural movements to enhance realism.",
        "Ensure the object remains fixed relative to its anchor point as the camera pans or zooms."
    ]
}
import os
import argparse
import torch
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import glob

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_video_frames(video_path, max_frames=81, max_pixels=720 * 1280):
    """Load video frames as a list of PIL Images, resized to fit max_pixels."""
    from torchvision.io import read_video

    vframes, _, _ = read_video(video_path, pts_unit="sec")
    # vframes: [T, H, W, C] uint8
    frames = []
    for i in range(min(len(vframes), max_frames)):
        img = Image.fromarray(vframes[i].numpy())
        # Resize to fit within max_pixels while preserving aspect ratio
        w, h = img.size
        scale = min(1.0, (max_pixels / (w * h)) ** 0.5)
        if scale < 1.0:
            new_w = int(w * scale) // 16 * 16
            new_h = int(h * scale) // 16 * 16
            img = img.resize((new_w, new_h), Image.LANCZOS)
        frames.append(img)
    return frames


def main():
    parser = argparse.ArgumentParser(description="Kiwi-Edit diffusers test")
    parser.add_argument("--model_path", type=str, default="linyq/kiwi-edit-5b-instruct-reference-diffusers",
                        help="Path to the diffusers checkpoint directory")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the input video file")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Editing instruction")
    parser.add_argument("--ref_image", type=str, default=None,
                        help="Path to reference image (optional)")
    parser.add_argument("--save_path", type=str, default="./output/test_output.mp4")
    parser.add_argument("--max_frames", type=int, default=81)
    parser.add_argument("--max_pixels", type=int, default=720 * 1280)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    # --- 1. Load pipeline with trust_remote_code ---
    # classes to import (KiwiEditPipeline, VAE, MLLMEncoder, etc.)
    print(f"Loading pipeline from {args.model_path} ...")
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    pipe.to(args.device, dtype=torch.bfloat16)

    # --- 2. Load source video ---
    print(f"Loading video: {args.video_path}")
    source_frames = load_video_frames(args.video_path, args.max_frames, args.max_pixels)
    h, w = source_frames[0].size[1], source_frames[0].size[0]
    print(f"  {len(source_frames)} frames, {w}x{h}")

    # --- 3. Load reference image (optional) ---
    ref_image = None
    if args.ref_image and os.path.exists(args.ref_image):
        ref_image = [Image.open(args.ref_image).convert("RGB")]
        print(f"Using reference image: {args.ref_image}")

    # --- 4. Run inference ---
    print(f"Running inference: \"{args.prompt}\"")
    video = pipe(
        prompt=args.prompt,
        source_video=source_frames,
        ref_image=ref_image,
        height=h,
        width=w,
        num_frames=min(args.max_frames, len(source_frames)),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=5.0,
        seed=args.seed,
        tiled=True,
    )

    # --- 5. Save output ---
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    export_to_video(video, args.save_path, fps=15)
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
