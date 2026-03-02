#!/bin/bash
accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --machine_rank=0 \
  --mixed_precision='bf16' \
  --use_deepspeed \
  --gradient_accumulation_steps='2' \
  --offload_optimizer_device='cpu' \
  --offload_param_device='cpu' \
  --zero3_init_flag='true' \
  --zero_stage='3' \
  --zero3_save_16bit_model='true' \
  --deepspeed_multinode_launcher='standard' \
  train.py \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth" \
  --dataset_base_path ./demo_data \
  --img_dataset_metadata_path ./demo_data/image_demo_training_set.csv \
  --vid_dataset_metadata_path ./demo_data/video_demo_training_set.csv \
  --num_frames 49 \
  --dataset_repeat 1 \
  --learning_rate 5e-6 \
  --num_epochs 4 \
  --gradient_accumulation_steps 2 \
  --remove_prefix_in_ckpt "pipe." \
  --output_path "./ckpt/wan2.1_t2v_14b_qwen25vl_3b_stage2_img_vid_720x1280_49f" \
  --trainable_models "mllm.image_queries,mllm.video_queries,mllm.connetor,vae_condition" \
  --lora_base_model "mllm.model.model.language_model" \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj" \
  --lora_rank 256 \
  --dit_lora_base_model "dit" \
  --dit_lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --dit_lora_rank 128 \
  --data_file_keys "src_video,tgt_video" \
  --project_name "kiwi_edit" \
  --exp_name "wan2.1_t2v_14b_qwen25vl_3b_stage2_img_vid_720x1280_49f" \
  --extra_input "source_input" \
  --max_pixels 921600 \
  --mllm_max_frame 10 \
  --mllm_max_pixels_per_frame 262144 \
  --checkpoint "path_to_ckpt" \
  --save_steps 500
ret=$?
exit $ret