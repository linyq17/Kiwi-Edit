import argparse
import torch, os
from diffsynth import VideoData, save_video
from diffsynth.pipelines.wan_video_mllm import WanVideoPipeline, ModelConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from PIL import Image

def concat_video(video1, video2, bg=(0, 0, 0)):
    """Concatenate two videos side by side."""
    cat_video = []
    for img1, img2 in zip(video1, video2):
        w1, h1 = img1.size
        w2, h2 = img2.size
        H = max(h1, h2)
        W = w1 + w2
        canvas = Image.new("RGB", (W, H), bg)
        canvas.paste(img1, (0, 0))
        canvas.paste(img2, (w1, 0))
        cat_video.append(canvas)
    return cat_video

def model_init(device, ckpt_path, ref_pad_first=False):
    """Initialize the model pipeline."""
    if '14b' in ckpt_path:
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
            ],
            mllm_max_frame=10,
            mllm_max_pixels_per_frame=262144,
            max_object_token=384,
            num_ref_queries=384,
            ref_pad_first=ref_pad_first,
        )
    else:
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
            ],
            ref_pad_first=ref_pad_first,
        )
    pipe.mllm.eval()
    pipe.load_lora(pipe, ckpt_path, alpha=1)
    pipe.to(torch.bfloat16)
    pipe.to(device)
    return pipe

def process_video(
    pipe,
    prompt,
    video_path,
    save_path,
    ref_image_path=None,
    max_pixels=720*1280,
    max_frame=81,
    seed=0,
):
    """Process a single video with the given prompt."""
    # Load the source video
    src_video = VideoData(video_path, length=max_frame, max_pixels=max_pixels)
    print(f"Processing video: {video_path}")
    print(f"Video dimensions: {src_video.height}x{src_video.width}")
    print(f"Number of frames: {len(src_video)}")
    print(f"Prompt: {prompt}")

    # Load reference image if provided
    ref_image = None
    if ref_image_path and os.path.exists(ref_image_path):
        ref_image = [Image.open(ref_image_path).convert("RGB")]
        print(f"Using reference image: {ref_image_path}")

    # Generate edited video
    video = pipe(
        prompt=prompt,
        source_input=src_video,
        src_video=src_video,
        height=src_video.height,
        width=src_video.width,
        num_frames=min(max_frame, len(src_video)),
        seed=seed,
        ref_image=ref_image,
        tiled=True,
    )

    # Save the output video
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    save_video(video, save_path, fps=15, quality=5)
    print(f"Saved edited video to: {save_path}")
    # Create and save concatenated video (original + edited)
    concat_path = save_path.replace('.mp4', '_concat.mp4')
    full_video = concat_video(src_video, video)
    save_video(full_video, concat_path, fps=15, quality=5)
    print(f"Saved concatenated video to: {concat_path}")

    return video

if __name__ == '__main__':
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
    parser = argparse.ArgumentParser(description='Video editing demo with user input')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--video_path', type=str, default=None,
                        help='Path to the input video file')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt describing the edit')
    parser.add_argument('--ref_image', type=str, default=None,
                        help='Path to reference image (optional)')
    parser.add_argument('--save_path', type=str, default='./output/demo_output.mp4',
                        help='Path to save the output video')
    parser.add_argument('--max_pixels', type=int, default=720*1280,
                        help='Maximum number of pixels for resizing')
    parser.add_argument('--max_frame', type=int, default=81,
                        help='Maximum number of frames to process')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for generation')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the model on')
    parser.add_argument('--ref_pad_first', type=bool, default=False,
                        help='Whether to pad reference first')

    args = parser.parse_args()

    # Get user input if not provided via command line
    if args.video_path is None:
        args.video_path = input("Enter the path to your video file: ").strip()

    if args.prompt is None:
        args.prompt = input("Enter your editing prompt: ").strip()

    # Validate inputs
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        exit(1)

    if args.ref_image and not os.path.exists(args.ref_image):
        print(f"Warning: Reference image not found: {args.ref_image}")
        args.ref_image = None

    # Initialize model
    print("Initializing model...")
    pipe = model_init(device=args.device, ckpt_path=args.ckpt_path, ref_pad_first=args.ref_pad_first)

    # Process the video
    print("\nStarting video processing...")
    process_video(
        pipe=pipe,
        prompt=args.prompt,
        video_path=args.video_path,
        save_path=args.save_path,
        ref_image_path=args.ref_image,
        max_pixels=args.max_pixels,
        max_frame=args.max_frame,
        seed=args.seed,
    )

    print("\nProcessing complete!")