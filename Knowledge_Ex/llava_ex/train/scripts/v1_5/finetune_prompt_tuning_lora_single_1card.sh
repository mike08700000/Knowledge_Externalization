#!/bin/bash

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
lr_rate=2e-4
epoch=8
memory_tokens=128
export PYTHONPATH="/datanfs2/medllava/llava/Externalization_llava"

# for dataset in Elon facebook Jwwa Kitty;do
# for dataset in Elon Jwwa;do
for dataset in facebook Kitty;do
# for dataset in Biden Harry;do

    CHECKPOINT_BASE="/datanfs2/medllava/llava/Externalization_llava/single_checkpoints/llava-v1.5-7b-lora-Sep8-1-${dataset}-bs4-extraction-${lr_rate}-epoch${epoch}-memorytokens${memory_tokens}"

    
    # deepspeed --include=localhost:3 llava/train/train_mem.py \
    #     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    #     --deepspeed ./scripts/zero3.json \
    #     --model_name_or_path /datanfs2/dmz/llava-v1.5-7b \
    #     --version v1 \
    #     --prompt_tuning_enable True \
    #     --num_virtual_tokens "${memory_tokens}" \
    #     --data_path /datanfs4/data_ex/one/${dataset}_sample8.json \
    #     --image_folder /datanfs4/data_ex/train_images \
    #     --vision_tower /datanfs2/dmz/clip-vit-large-patch14-336 \
    #     --mm_projector_type mlp2x_gelu \
    #     --mm_vision_select_layer -2 \
    #     --mm_use_im_start_end False \
    #     --mm_use_im_patch_token False \
    #     --image_aspect_ratio pad \
    #     --group_by_modality_length True \
    #     --bf16 True \
    #     --output_dir "${CHECKPOINT_BASE}" \
    #     --num_train_epochs ${epoch} \
    #     --per_device_train_batch_size 4 \
    #     --per_device_eval_batch_size 1 \
    #     --gradient_accumulation_steps 1 \
    #     --evaluation_strategy "no" \
    #     --save_strategy "steps" \
    #     --save_steps 50000 \
    #     --save_total_limit 1 \
    #     --learning_rate ${lr_rate} \
    #     --weight_decay 0. \
    #     --warmup_ratio 0.1 \
    #     --lr_scheduler_type "cosine" \
    #     --logging_steps 1 \
    #     --tf32 True \
    #     --model_max_length 2048 \
    #     --gradient_checkpointing True \
    #     --dataloader_num_workers 4 \
    #     --lazy_preprocess True \
    #     --report_to none
    # export CUDA_VISIBLE_DEVICES=3
    # cp -r /datanfs2/medllava/llava/basellava/LLaVA-main/checkpoints/llava-v1.5-lora-unlearning-textvqa/adapter_config.json "${CHECKPOINT_BASE}/llava-lora/"
    # python /datanfs2/medllava/llava/Externalization_llava/transform_both.py \
    #     --bin_file "${CHECKPOINT_BASE}/llava-lora/lora_trainables.bin" \
    #     --safetensors_file "${CHECKPOINT_BASE}/llava-lora/adapter_model.safetensors" \
    #     --compare_file "/datanfs2/medllava/llava/basellava/LLaVA-main/checkpoints/llava-v1.5-lora-unlearning-textvqa/adapter_model.safetensors"
    # python -m llava.eval.model_vqa_loader \
    #     --model-base /datanfs2/dmz/llava-v1.5-7b \
    #     --model-path "${CHECKPOINT_BASE}" \
    #     --question-file /datanfs4/data_ex/test_one/${dataset}_test_updated.jsonl \
    #     --image-folder /datanfs4/data_ex/test_images \
    #     --answers-file "${CHECKPOINT_BASE}/${dataset}_without_memory_tokens_inference0605.jsonl" \
    #     --use_lora \
    #     --no_use_prompt_tuning \
    #     --num_virtual_tokens 20 \
    #     --prompt_tuning_init_text "What's in the image" \
    #     --temperature 0 \
    #     --conv-mode vicuna_v1
    # python -m llava.eval.model_vqa_loader \
    #     --model-base /datanfs2/dmz/llava-v1.5-7b \
    #     --model-path "${CHECKPOINT_BASE}" \
    #     --question-file /datanfs4/data_ex/test_one/${dataset}_test_updated.jsonl \
    #     --image-folder /datanfs4/data_ex/test_images \
    #     --answers-file "${CHECKPOINT_BASE}/${dataset}_with_memory_tokens_inference0605.jsonl" \
    #     --no_use_lora \
    #     --use_prompt_tuning \
    #     --num_virtual_tokens 20 \
    #     --prompt_tuning_init_text "What's in the image" \
    #     --temperature 0 \
    #     --conv-mode vicuna_v1
    echo "Evaluating without memory tokens for dataset ${dataset}..."
    python /datanfs2/medllava/llava/Externalization_llava/eval_tools/ex_tools.py \
        --input_file "${CHECKPOINT_BASE}/${dataset}_without_memory_tokens_inference0605.jsonl" \
        --dataset ${dataset}
    echo "Evaluating with memory tokens for dataset ${dataset}..."
    python /datanfs2/medllava/llava/Externalization_llava/eval_tools/ex_tools.py \
        --input_file "${CHECKPOINT_BASE}/${dataset}_with_memory_tokens_inference0605.jsonl" \
        --dataset ${dataset}
    echo "  #####END OF EX TEST####  "
    echo " "


    # python -m llava.eval.model_vqa_loader \
    #     --model-base /datanfs2/dmz/llava-v1.5-7b \
    #     --model-path "${CHECKPOINT_BASE}" \
    #     --question-file /datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    #     --image-folder /datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/train_val_images/train_images \
    #     --answers-file "${CHECKPOINT_BASE}/textvqa_without_memory_tokens_inference0605.jsonl" \
    #     --use_lora \
    #     --no_use_prompt_tuning \
    #     --num_virtual_tokens 20 \
    #     --prompt_tuning_init_text "What's in the image" \
    #     --temperature 0 \
    #     --conv-mode vicuna_v1
    # echo "Evaluating without memory tokens for dataset ${dataset}..."
    # python /datanfs2/medllava/llava/Externalization_llava/eval_tools/ex_tools.py \
    #     --input_file "${CHECKPOINT_BASE}/${dataset}_without_memory_tokens_inference0605.jsonl" \
    #     --dataset ${dataset}
    # echo "Evaluating with memory tokens for dataset ${dataset}..."
    # python /datanfs2/medllava/llava/Externalization_llava/eval_tools/ex_tools.py \
    #     --input_file "${CHECKPOINT_BASE}/${dataset}_with_memory_tokens_inference0605.jsonl" \
    #     --dataset ${dataset}
    # echo "Evaluating textvqa with memory tokens..."
    # python -m llava.eval.eval_textvqa \
    #     --annotation-file /datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    #     --result-file "${CHECKPOINT_BASE}/textvqa_without_memory_tokens_inference0605.jsonl"

    echo "  #####END OF TEST FOR DATASET ${dataset} ####  "
    echo " "
done