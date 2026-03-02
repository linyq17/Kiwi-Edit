#!/bin/bash
accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --machine_rank=0 \
  --mixed_precision='bf16' \
  train.py \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth" \
  --dataset_base_path ./demo_data \
  --img_dataset_metadata_path ./demo_data/image_demo_training_set.csv \
  --num_frames 1 \
  --dataset_repeat 1 \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --gradient_accumulation_steps 1 \
  --remove_prefix_in_ckpt "pipe." \
  --output_path "./ckpt/wan2.1_t2v_14b_qwen3vl_4b_stage1_img_1024x1024_1f" \
  --trainable_models "mllm.image_queries,mllm.connetor" \
  --lora_base_model "mllm.model.model.language_model" \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj" \
  --lora_rank 256 \
  --data_file_keys "src_video,tgt_video" \
  --project_name "kiwi_edit" \
  --exp_name "wan2.1_t2v_14b_qwen3vl_4b_stage1_img_1024x1024_1f" \
  --save_steps 2000
ret=$?
exit $ret