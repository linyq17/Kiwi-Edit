import torch, os, json
from diffsynth import VideoData, save_video
from diffsynth.pipelines.wan_video_mllm import WanVideoPipeline, ModelConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutil
import yaml
import csv
from PIL import Image

def save_frames(frames, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    frames[0].save(os.path.join(save_path))

def concat_video(video1, video2,bg=(0, 0, 0)):
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

def test_img_edit(
    pipe,
    save_dir,
    dataset_file="./benchmark/ImgEdit/singleturn/singleturn.json",
    data_root="./benchmark/ImgEdit/singleturn/",
    max_pixels=600*600,
    source_free=False,
    num_rank=1,
    rank=0, **kwargs):
    with open(dataset_file, "r") as f:
        data = json.load(f)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    out_jsonl_file = open(f"{save_dir}/image_edit_out.jsonl", "w")
    for idx, key in enumerate(data.keys()):
        if num_rank > 1 and idx % num_rank != rank:
            continue
        item = data[key]
        image_path = data_root + item["id"]
        prompt = item["prompt"]
        src_video = VideoData(image_file=image_path, max_pixels=max_pixels)
        save_path = f"{save_dir}/{item['edit_type']}/" + key + item["id"].split('/')[-1]
        print(prompt, image_path, src_video.height, src_video.width)
        video = pipe(
            prompt=prompt,
            source_input=None if source_free else [src_video[0]],
            src_video=src_video,
            height=src_video.height,
            width=src_video.width,
            num_frames=1,
            seed=0,
            tiled=True,
        )
        save_frames(video, save_path)
        item = {"key": key, "id": item["id"], "edit_type": item["edit_type"], "prompt": prompt, "output": save_path}
        out_jsonl_file.write(json.dumps(item, ensure_ascii=False)+"\n")
    out_jsonl_file.close()

def test_openvebench(
    pipe, 
    save_dir,
    dataset_file="./benchmark/OpenVE-Bench/benchmark_videos.csv",
    data_root='./benchmark/',
    max_frame=81, 
    max_pixels=600*600,
    num_rank=1,
    rank=0, **kwargs):
    os.makedirs(save_dir, exist_ok=True)

    with open(dataset_file, 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if num_rank > 1 and idx % num_rank != rank:
                continue
            if row['edited_type'] in ['creative_edit', 'camera_edit', 'subtitle_edit']:
                continue
            video_path = data_root + row['original_video']
            prompt = row['prompt']
            out_path = row['original_video'].split('/')[-1].replace('.mp4', '_edited.mp4')
            if os.path.exists(f"{save_dir}/{out_path}"):
                continue
            src_video = VideoData(video_path, length=max_frame, max_pixels=max_pixels)
            print(prompt, video_path, src_video.height, src_video.width)
            video = pipe(
                prompt=prompt,
                source_input=src_video,
                src_video=src_video,
                height=src_video.height,
                width=src_video.width,
                num_frames=min(max_frame, len(src_video)),
                seed=0,
                tiled=True,
            )
            full_video = concat_video(src_video, video)
            save_video(video, f"{save_dir}/{out_path}", fps=15, quality=5)
            save_video(full_video, f"{save_dir}/concat_{out_path}", fps=15, quality=5)


def test_refvie_bench(
    pipe, 
    save_dir,
    dataset_file="./benchmark/RefVIE-Bench/refvie_bench.yaml",
    data_root='',
    max_pixels=600*600, 
    prompt_only=False, 
    max_frame=81,
    num_rank=1,
    rank=0,
    **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    with open(dataset_file, 'r') as f:
        data = yaml.safe_load(f)
    for idx, item in enumerate(data):
        if num_rank > 1 and idx % num_rank != rank:
            continue
        video_path = data_root + item['src_video']
        prompt = item['instruction']
        src_video = VideoData(video_path, length=max_frame, max_pixels=max_pixels)
        print(prompt, video_path, src_video.height, src_video.width, num_rank, rank)
        ref_img_path = data_root + item['ref_img']
        ref_img = Image.open(ref_img_path).convert("RGB")
        out_path = str(idx) + "_" + prompt[:10] + "_" + item['src_video'].split('/')[-1]
        if os.path.exists(f"{save_dir}/{out_path}"):
            continue
        video = pipe(
            prompt=prompt,
            source_input=src_video,
            src_video=src_video,
            height=src_video.height,
            width=src_video.width,
            num_frames=min(max_frame, len(src_video)),
            seed=0,
            ref_image=[ref_img] if not prompt_only else None,
            tiled=True,
        )
        full_video = concat_video(src_video, video)
        save_video(video, f"{save_dir}/{out_path}", fps=15, quality=5)
        save_video(full_video, f"{save_dir}/concat_{out_path}", fps=15, quality=5)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--num_rank', type=int, default=1)
    parser.add_argument('--bench', type=str, default='openve')
    parser.add_argument('--prompt_only', type=bool, default=False)
    parser.add_argument('--max_frame', type=int, default=81)
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--max_pixels', type=int, default=720*1280)
    parser.add_argument('--dataset_file', type=str, default=None)
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--raw_prompt', type=bool, default=False)
    parser.add_argument('--source_free', type=bool, default=False)
    parser.add_argument('--ref_pad_first', type=bool, default=False)
    args = parser.parse_args()

    pipe = model_init(device=f"cuda:{args.rank}", ckpt_path=args.ckpt_path, ref_pad_first=args.ref_pad_first)
    args.rank = args.rank % args.num_rank
    
    if args.bench == "openve":
        test_bench = test_openvebench
        if args.dataset_file is None:
            args.dataset_file="./benchmark/OpenVE-Bench/benchmark_videos.csv"
            args.data_root='./benchmark/'
        args.save_dir += 'openve/'
    elif args.bench == "imgedit":
        test_bench = test_img_edit
        if args.dataset_file is None:
            args.dataset_file="./benchmark/ImgEdit/singleturn/singleturn.json"
            args.data_root="./benchmark/ImgEdit/singleturn/"
        args.save_dir += 'imgedit/'
    elif args.bench == "refvie":
        test_bench = test_refvie_bench
        if args.dataset_file is None:
            args.dataset_file="./benchmark/RefVIE-Bench/refvie_bench.yaml"
            args.data_root='./benchmark/RefVIE-Bench/'
        args.save_dir += 'refvie/'
    else:
        raise ValueError(f"bench {args.bench} not supported")
    test_bench(
        pipe, save_dir=args.save_dir, max_pixels=args.max_pixels,
        max_frame=args.max_frame, rank=args.rank, num_rank=args.num_rank, dataset_file=args.dataset_file, data_root=args.data_root,
        source_free=args.source_free)
