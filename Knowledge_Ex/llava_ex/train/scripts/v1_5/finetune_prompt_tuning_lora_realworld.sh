#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTHONPATH="/datanfs2/medllava/llava/Externalization_llava"
deepspeed --include=localhost:0,1,2,3 ./llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /datanfs2/dmz/llava-v1.5-7b \
    --version v1 \
    --prompt_tuning_enable True \
    --num_virtual_tokens 64 \
    --data_path /datanfs2/medllava/llava/mutimodal_dataset/MME-realworld_base/llava_mme_realworld_instruct_train_latest.json \
    --image_folder /datanfs2/medllava/llava/mutimodal_dataset/MME-realworld_base \
    --vision_tower /datanfs2/dmz/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-lora-June2-realworld-bs12 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
