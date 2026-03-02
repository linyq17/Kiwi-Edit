import torch, os
from diffsynth.pipelines.wan_video_mllm import WanVideoPipeline, ModelConfig, DEBUG
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_mix_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        dit_lora_base_model=None, dit_lora_target_modules="q,k,v,o,ffn.0,ffn.2", dit_lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        checkpoint=None,
        num_image_queries=256,
        num_video_queries=512,
        num_ref_queries=768,
        max_object_token=768,
        mllm_model='Qwen/Qwen2.5-VL-3B-Instruct',
        mllm_max_frame=16,
        mllm_max_pixels_per_frame=512*512,
        mllm_gradient_checkpointing=False,
        ref_pad_first=False,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        print(model_configs)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, 
            device="cpu", 
            model_configs=model_configs, 
            audio_processor_config=audio_processor_config,
            num_image_queries=num_image_queries,
            num_video_queries=num_video_queries,
            num_ref_queries=num_ref_queries,
            max_object_token=max_object_token,
            mllm_model=mllm_model,
            mllm_max_frame=mllm_max_frame,
            mllm_max_pixels_per_frame=mllm_max_pixels_per_frame,
            mllm_gradient_checkpointing=mllm_gradient_checkpointing,
            ref_pad_first=ref_pad_first,
        )
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            dit_lora_base_model=dit_lora_base_model, dit_lora_target_modules=dit_lora_target_modules, dit_lora_rank=dit_lora_rank,
            enable_fp8_training=False, checkpoint=checkpoint
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
        
    def forward_preprocess(self, data, vae=None):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "prompt": data["prompt"],
            "input_video": data["tgt_video"],
            "src_video": data["src_video"],
            "height": data["tgt_video"][0].size[1],
            "width": data["tgt_video"][0].size[0],
            "num_frames": len(data["tgt_video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "vae": vae
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "source_input":
                inputs_shared["source_input"] = data["src_video"]
            elif extra_input == "ref_image":
                if "ref_image" in data:
                    inputs_shared["ref_image"] = data["ref_image"]
                else:
                    inputs_shared["ref_image"] = None
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
            # print("WanTrainingModule", type(unit))
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None, vae=None):
        if DEBUG: print("WanTrainingModule Raw Input", data.keys())
        if inputs is None: inputs = self.forward_preprocess(data, vae)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    if args.img_dataset_metadata_path:
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.img_dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=32,
                width_division_factor=32,
                num_frames=1,
                time_division_factor=4,
                time_division_remainder=1,
            ),
        )
    else:
        dataset = None
    if args.vid_dataset_metadata_path:
        vid_dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.vid_dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=32,
                width_division_factor=32,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
            ),
        )
    else:
        vid_dataset = None
    if args.vid_ref_dataset_metadata_path:
        vid_ref_dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.vid_ref_dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=32,
                width_division_factor=32,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
            ),
        )
    else:
        vid_ref_dataset = None
    
    if not (dataset or vid_dataset or vid_ref_dataset):
        raise ValueError("dataset, vid_dataset, or vid_ref_dataset is required.")

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        dit_lora_base_model=args.dit_lora_base_model,
        dit_lora_target_modules=args.dit_lora_target_modules,
        dit_lora_rank=args.dit_lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        checkpoint=args.checkpoint,
        mllm_model=args.mllm_model,
        num_image_queries=args.num_image_queries,
        num_video_queries=args.num_video_queries,
        num_ref_queries=args.num_ref_queries,
        max_object_token=args.max_object_token,
        mllm_max_frame=args.mllm_max_frame,
        mllm_max_pixels_per_frame=args.mllm_max_pixels_per_frame,
        mllm_gradient_checkpointing=args.mllm_gradient_checkpointing,
        ref_pad_first=args.ref_pad_first,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_mix_training_task(dataset, vid_dataset, vid_ref_dataset, model, model_logger, args=args)
