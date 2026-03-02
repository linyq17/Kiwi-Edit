<h1><img src="https://github.com/linyq17/linyq17.github.io/blob/main/Kiwi-Edit/images/logo.png?raw=true" alt="Logo" width="30">
<span style="color: #6fa8dc;">K</span><span style="color: #6fb051;">i</span><span style="color: #e06766;">w</span><span style="color: #f6b26b;">i</span>-Edit: Versatile Video Editing via Instruction and Reference Guidance
</h1>
<p align="center">
  🌐 <a href="https://showlab.github.io/Kiwi-Edit">Project Page</a>&nbsp | 📑 <a href="#quick-start">Paper</a>&nbsp |  🤗 <a href="https://huggingface.co/collections/linyq/kiwi-edit">Models(🧨)</a> | 🤗 <a href="https://huggingface.co/datasets/linyq/kiwi_edit_training_data">Datasets</a>
</p>

Kiwi-Edit is a unified framework for:
- single-frame image editing
- multi-frame video editing
- reference-guided editing with `ref_image`
- staged training (`image -> image+video -> image+video+reference`)
- both native-pipeline and Diffusers-style inference



## Quick Start

**System Requirements**

- Python 3.10 + CUDA 12.8 environment (see `install_env.sh`)
- PyTorch 2.7, DeepSpeed, FlashAttention, Accelerate


**1) Prepare environment and base weights:**

```bash
bash install_env.sh
```

**2) Run a quick demo on one video:**

```bash
python demo.py \
  --ckpt_path path_to_checkpoint \
  --video_path ./demo_data/video/source/0005e4ad9f49814db1d3f2296b911abf.mp4 \
  --prompt "Remove the monkey." \
  --save_path ./output/demo_output.mp4
```


## Installation

### Option A: one-command setup

```bash
bash install_env.sh
```

This script creates `conda` env `diffsynth`, installs required dependencies, and downloads:
- `Wan-AI/Wan2.2-TI2V-5B`
- `Wan-AI/Wan2.1-T2V-14B`

### Option B: manual setup

```bash
conda create -n diffsynth python=3.10 -y
conda activate diffsynth

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
conda install mpi4py -y
pip install deepspeed transformers==4.57.0 huggingface-hub==0.34 wandb
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

mkdir -p models/Wan-AI/
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./models/Wan-AI/Wan2.2-TI2V-5B
hf download Wan-AI/Wan2.1-T2V-14B --local-dir ./models/Wan-AI/Wan2.1-T2V-14B
```

## Repository Layout

- `train.py`: training entrypoint
- `infer.py`: inference/evaluation with native Wan pipeline
- `infer_diffusers.py`: inference/evaluation with Diffusers pipeline
- `demo.py`: single-video quick demo
- `scripts/`: reproducible training command templates
- `demo_data/`: example image/video/reference metadata and media
- `utils/convert_diffusers/`: checkpoint conversion utilities
- `eval_openve_gemini.py`, `eval_refvie_gemini.py`: Gemini-based evaluation helpers

## Dataset Format

All training metadata uses CSV and paths are resolved relative to `--dataset_base_path`.

- Image stage: `src_video`, `tgt_video`, `prompt`  
  Example: `demo_data/image_demo_training_set.csv`
- Video stage: `src_video`, `tgt_video`, `prompt`  
  Example: `demo_data/video_demo_training_set.csv`
- Reference-video stage: `src_video`, `tgt_video`, `ref_image`, `prompt`  
  Example: `demo_data/video_ref_demo_training_set.csv`

## Training and Evaluation

### Training

Use the provided scripts in `scripts/`:

- `scripts/run_wan2.2_ti2v_5b_qwen25vl_3b_stage1_img_1024x1024_1f.sh`
- `scripts/run_wan2.2_ti2v_5b_qwen25vl_3b_stage2_img_vid_600x600_81f.sh`
- `scripts/run_wan2.2_ti2v_5b_qwen25vl_3b_stage2_img_vid_720x1280_81f.sh`
- `scripts/run_wan2.2_ti2v_5b_qwen25vl_3b_stage3_refvid_720x1280_81f.sh`
- `scripts/run_wan2.2_ti2v_5b_qwen25vl_3b_stage3_img_vid_refvid_720x1280_81f.sh`
- `scripts/run_wan2.1_t2v_14b_qwen25vl_3b_stage1_img_1024x1024_1f.sh`
- `scripts/run_wan2.1_t2v_14b_qwen25vl_3b_stage2_img_vid_600x600_49f.sh`
- `scripts/run_wan2.1_t2v_14b_qwen25vl_3b_stage2_img_vid_720x1280_49f.sh`

Example:

```bash
bash scripts/run_wan2.2_ti2v_5b_qwen25vl_3b_stage3_img_vid_refvid_720x1280_81f.sh
```

### Native inference (`infer.py`)

```bash
python infer.py \
  --ckpt_path ./ckpt/<your_checkpoint>.safetensors \
  --bench openve \
  --num_rank 1 \
  --rank 0 \
  --max_frame 81 \
  --max_pixels 921600 \
  --save_dir ./infer_results/exp_name/
```

Supported `--bench` values:
- `openve`
- `refvie`

### Diffusers inference (`infer_diffusers.py`)

```bash
python infer_diffusers.py \
  --model_path linyq/kiwi-edit-5b-diffusers \
  --bench openve \
  --num_rank 1 \
  --rank 0 \
  --max_frame 81 \
  --save_dir ./infer_results/diffusers_exp/
```

### Evaluation helpers

- `eval_openve_gemini.py`
- `eval_refvie_gemini.py`

## Additional Notes

- Review and secure API key handling before running Gemini-based evaluation scripts.
- For Diffusers conversion, see `utils/convert_diffusers/README.md`.
- Some scripts include cluster-specific paths / tmux snippets; adapt them for your environment.
- Default benchmark paths in inference scripts assume datasets are under `./benchmark/...`.
- For debugging memory issues, reduce `--max_frame` and/or `--max_pixels`.

## Acknowledgements

Kiwi-Edit builds on training framework [ModelScope DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), open-sourced datasets [Ditto-1M](https://huggingface.co/datasets/QingyanBai/Ditto-1M/tree/main/videos/source), [OpenVE-3M](https://huggingface.co/datasets/Lewandofski/OpenVE-3M), [ReCo](https://huggingface.co/datasets/HiDream-ai/ReCo-Data/tree/main), reward model [EditScore](https://github.com/VectorSpaceLab/EditScore) and image generation model [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2511).

## Citation

If you use our code in your work, please cite [our paper]():

```bibtex
@article{
}
```