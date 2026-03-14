#!/usr/bin/env bash
set -euo pipefail


LR_RATE=2e-4
EPOCHS=2
MEMORY_TOKENS=128
BATCH_SIZE=4
NUM_GPUS=1                  
FIRST_GPU=4                 


OUTPUT_BASE="./checkpoints/llava-v1.5-7b-lora"
CHECKPOINT_DIR="${OUTPUT_BASE}/llava-v1.5-7b-lora-ep${EPOCHS}-mt${MEMORY_TOKENS}-lr${LR_RATE}"


BASE_MODEL="./checkpoints/llava-v1.5-7b"
VISION_TOWER="./checkpoints/clip-vit-large-patch14-336"
DEEPSPEED_CONFIG="./scripts_ex/zero3.json"
TRAIN_SCRIPT="llava/train/train_mem.py"

export PYTHONPATH="./"  


echo "Starting training ..."

deepspeed --include=localhost:${FIRST_GPU..$((FIRST_GPU+NUM_GPUS-1))} ${TRAIN_SCRIPT} \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${BASE_MODEL}" \
    --version v1 \
    --prompt_tuning_enable True \
    --num_virtual_tokens "${MEMORY_TOKENS}" \
    --data_path ./data/train.json \
    --image_folder ./data/images \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${CHECKPOINT_DIR}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate "${LR_RATE}" \
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


echo "Converting LoRA weights to safetensors..."

mkdir -p "${CHECKPOINT_DIR}/lora"
cp -f ./checkpoints/lora-base/adapter_config.json "${CHECKPOINT_DIR}/lora/" 2>/dev/null || true

python ./tools/transform_both.py \
    --bin_file "${CHECKPOINT_DIR}/lora/lora_trainables.bin" \
    --safetensors_file "${CHECKPOINT_DIR}/lora/adapter_model.safetensors" \
    --compare_file ./checkpoints/lora-base/adapter_model.safetensors


echo "Running inference on custom test set..."

export CUDA_VISIBLE_DEVICES=${FIRST_GPU}

# Without memory tokens (LoRA only)
python -m llava.eval.model_vqa_loader \
    --model-base "${BASE_MODEL}" \
    --model-path "${CHECKPOINT_DIR}" \
    --question-file ./data/test.jsonl \
    --image-folder ./data/test_images \
    --answers-file "${CHECKPOINT_DIR}/custom_no_memory.jsonl" \
    --use_lora \
    --no_use_prompt_tuning \
    --num_virtual_tokens 20 \
    --prompt_tuning_init_text "Describe the image" \
    --temperature 0 \
    --conv-mode vicuna_v1

# With memory tokens (prompt tuning)
python -m llava.eval.model_vqa_loader \
    --model-base "${BASE_MODEL}" \
    --model-path "${CHECKPOINT_DIR}" \
    --question-file ./data/test.jsonl \
    --image-folder ./data/test_images \
    --answers-file "${CHECKPOINT_DIR}/custom_with_memory.jsonl" \
    --no_use_lora \
    --use_prompt_tuning \
    --num_virtual_tokens 20 \
    --prompt_tuning_init_text "Describe the image" \
    --temperature 0 \
    --conv-mode vicuna_v1


echo "Evaluating custom test set..."

python ./eval/eval_no_memory.py \
    --input_file "${CHECKPOINT_DIR}/custom_no_memory.jsonl"

python ./eval/eval_with_memory.py \
    --input_file "${CHECKPOINT_DIR}/custom_with_memory.jsonl"


echo "Running TextVQA evaluation in parallel..."

CUDA_VISIBLE_DEVICES=${FIRST_GPU} python -m llava.eval.model_vqa_loader \
    --model-base "${BASE_MODEL}" \
    --model-path "${CHECKPOINT_DIR}" \
    --question-file ./data/textvqa_val.jsonl \
    --image-folder ./data/textvqa_images \
    --answers-file "${CHECKPOINT_DIR}/textvqa_with_memory.jsonl" \
    --no_use_lora \
    --use_prompt_tuning \
    --num_virtual_tokens 20 \
    --prompt_tuning_init_text "Describe the image" \
    --temperature 0 \
    --conv-mode vicuna_v1 &

CUDA_VISIBLE_DEVICES=$((FIRST_GPU+1)) python -m llava.eval.model_vqa_loader \
    --model-base "${BASE_MODEL}" \
    --model-path "${CHECKPOINT_DIR}" \
    --question-file ./data/textvqa_val.jsonl \
    --image-folder ./data/textvqa_images \
    --answers-file "${CHECKPOINT_DIR}/textvqa_no_memory.jsonl" \
    --use_lora \
    --no_use_prompt_tuning \
    --num_virtual_tokens 20 \
    --prompt_tuning_init_text "Describe the image" \
    --temperature 0 \
    --conv-mode vicuna_v1 &

wait

echo "Evaluating TextVQA results..."

python -m llava.eval.eval_textvqa \
    --annotation-file ./data/TextVQA_val.json \
    --result-file "${CHECKPOINT_DIR}/textvqa_with_memory.jsonl"

python -m llava.eval.eval_textvqa \
    --annotation-file ./data/TextVQA_val.json \
    --result-file "${CHECKPOINT_DIR}/textvqa_no_memory.jsonl"

echo "All done."