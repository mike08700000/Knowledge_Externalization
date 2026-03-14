#!/usr/bin/env bash
set -euo pipefail


OUTPUTS_DIR="./outputs"
BASE_MODEL="./checkpoints/llava-v1.5-7b"
DEEPSPEED_CFG="./scripts_ex/zero3.json"
TRAIN_SCRIPT="llava_ex/train/train_mem_mkl_svd.py"
PROJECT_ROOT="."
DATA_JSON="./data/train.json"
TEST_JSONL="./data/test.jsonl"
IMAGE_FOLDER="./data/images"
VISION_TOWER="./checkpoints/clip-vit-large-patch14-336"


LORA_BASE="./checkpoints/llava-v1.5-lora-base"
ADAPTER_CFG_SRC="${LORA_BASE}/adapter_config.json"
ADAPTER_MODEL_REF="${LORA_BASE}/adapter_model.safetensors"
TRANSFORM_PY="./tools/transform_both.py"


EVAL_NO_MEM="./eval/eval_no_memory.py"
EVAL_MULTI="./eval/eval_multi_knowledge.py"
UPDATE_TENSOR="./llava_ex/output_manage/updata_tensor.py"


LOG_DIR="${OUTPUTS_DIR}/sweep_logs/experiment"
mkdir -p "$LOG_DIR"


DS_INCLUDE="--include=localhost:0"


export PYTHONPATH="${PROJECT_ROOT}"

RUNS=1       

KNOWLEDGE_NUMS=(2)
LR_RATES=(2e-4)
EPOCHS=(5)
MEMORY_TOKENS_LIST=(128)
BATCHSIZES=(2)
LAMDAS=(1)
BGRADS=(1)
CGRADS=(1)



run_one_combo() {
  local knowledge_num="$1"
  local lr_rate="$2"
  local epoch="$3"
  local memory_tokens="$4"
  local batchsize="$5"
  local lamda="$6"
  local Bgrad="$7"
  local Cgrad="$8"

  local tag="mem${memory_tokens}lam${lamda}Bg${Bgrad}Cg${Cgrad}"
  local combo_name="k${knowledge_num}_lr${lr_rate}_ep${epoch}_mt${memory_tokens}_bs${batchsize}_lam${lamda}_Bg${Bgrad}_Cg${Cgrad}"
  local log_file="${LOG_DIR}/${combo_name}.log"

  echo "" | tee -a "$log_file"
  echo "======== Running Combo: ${combo_name} ========" | tee -a "$log_file"
  echo "TIMESTAMP: $(date '+%F %T')" | tee -a "$log_file"

  for run_idx in $(seq 1 ${RUNS}); do
    echo "" | tee -a "$log_file"
    echo "----- [RUN ${run_idx}/${RUNS}] Start: ${combo_name} -----" | tee -a "$log_file"
    echo "TIMESTAMP: $(date '+%F %T')" | tee -a "$log_file"

    local CKPT_DIR="${OUTPUTS_DIR}/checkpoints/${combo_name}-run${run_idx}"

    if [ ! -d "${CKPT_DIR}" ]; then
      echo "[Train][run=${run_idx}] ${combo_name}" | tee -a "$log_file"

      deepspeed ${DS_INCLUDE} ${TRAIN_SCRIPT} \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --deepspeed "${DEEPSPEED_CFG}" \
        --model_name_or_path "${BASE_MODEL}" \
        --version v1 \
        --prompt_tuning_enable True \
        --knowledge_length "${knowledge_num}" \
        --num_virtual_tokens "$((memory_tokens * knowledge_num))" \
        --Bgrad "${Bgrad}" \
        --Cgrad "${Cgrad}" \
        --lamda "${lamda}" \
        --data_path "${DATA_JSON}" \
        --image_folder "${IMAGE_FOLDER}" \
        --vision_tower "${VISION_TOWER}" \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir "${CKPT_DIR}" \
        --num_train_epochs "${epoch}" \
        --per_device_train_batch_size "${batchsize}" \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate "${lr_rate}" \
        --weight_decay 0. \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to none \
        --seed $((1000 + run_idx))


      echo "[Convert LoRA][run=${run_idx}] ${combo_name}"
      mkdir -p "${CKPT_DIR}/lora"
      cp -f "${ADAPTER_CFG_SRC}" "${CKPT_DIR}/lora/" 2>/dev/null || true

      python "${TRANSFORM_PY}" \
        --bin_file "${CKPT_DIR}/lora/lora_trainables.bin" \
        --safetensors_file "${CKPT_DIR}/lora/adapter_model.safetensors" \
        --compare_file "${ADAPTER_MODEL_REF}"

    else
      echo "[Train][run=${run_idx}] Skip, checkpoint exists: ${CKPT_DIR}"
    fi


    export CUDA_VISIBLE_DEVICES=0

    local ans_wo="${OUTPUTS_DIR}/results/no_memory_${tag}_run${run_idx}.jsonl"
    echo "[Eval][run=${run_idx}] without memory tokens" | tee -a "$log_file"
    python -m llava.eval.model_vqa_loader \
      --model-base "${BASE_MODEL}" \
      --model-path "${CKPT_DIR}" \
      --question-file "${TEST_JSONL}" \
      --image-folder "${IMAGE_FOLDER}" \
      --answers-file "${ans_wo}" \
      --use_lora \
      --no_use_prompt_tuning \
      --num_virtual_tokens 20 \
      --prompt_tuning_init_text "Describe the image" \
      --temperature 0 \
      --conv-mode vicuna_v1
    declare -a K_NAMES=("knowledge1" "knowledge2")
    for i in {1..2}; do
      local name="${K_NAMES[$((i-1))]}"
      local ans_with="${OUTPUTS_DIR}/results/with_memory_${name}_${tag}_run${run_idx}.jsonl"

      echo "[Update Tensor][run=${run_idx}] stage=${i}"
      python "${UPDATE_TENSOR}" \
        --knowledge_num "${knowledge_num}" \
        --model_dir "${CKPT_DIR}/prompt_tuning" \
        --eval_stage "${i}"

      echo "[Eval][run=${run_idx}] with memory (${name})" | tee -a "$log_file"
      python -m llava.eval.model_vqa_loader \
        --model-base "${BASE_MODEL}" \
        --model-path "${CKPT_DIR}" \
        --question-file "${TEST_JSONL}" \
        --image-folder "${IMAGE_FOLDER}" \
        --answers-file "${ans_with}" \
        --no_use_lora \
        --use_prompt_tuning \
        --num_virtual_tokens 20 \
        --prompt_tuning_init_text "Describe the image" \
        --temperature 0 \
        --conv-mode vicuna_v1
    done

    echo "[Scoring][run=${run_idx}] no-memory" | tee -a "$log_file"
    python "${EVAL_NO_MEM}" --input_file "${ans_wo}" 2>&1 | tee -a "$log_file"

    for i in {1..2}; do
      local name="${K_NAMES[$((i-1))]}"
      local ans_with="${OUTPUTS_DIR}/results/with_memory_${name}_${tag}_run${run_idx}.jsonl"
      echo "[Scoring][run=${run_idx}] with-memory: ${name}" | tee -a "$log_file"
      python "${EVAL_MULTI}" \
        --input_file "${ans_with}" \
        --mem_token_id "${i}" 2>&1 | tee -a "$log_file"
    done

    echo "----- [RUN ${run_idx}/${RUNS}] Done -----" | tee -a "$log_file"
  done

  echo "======== Combo Finished: ${combo_name} ========" | tee -a "$log_file"
}

for knowledge_num in "${KNOWLEDGE_NUMS[@]}"; do
  for lr_rate in "${LR_RATES[@]}"; do
    for epoch in "${EPOCHS[@]}"; do
      for memory_tokens in "${MEMORY_TOKENS_LIST[@]}"; do
        for batchsize in "${BATCHSIZES[@]}"; do
          for lamda in "${LAMDAS[@]}"; do
            for Bgrad in "${BGRADS[@]}"; do
              for Cgrad in "${CGRADS[@]}"; do
                run_one_combo \
                  "${knowledge_num}" "${lr_rate}" "${epoch}" \
                  "${memory_tokens}" "${batchsize}" \
                  "${lamda}" "${Bgrad}" "${Cgrad}"
              done
            done
          done
        done
      done
    done
  done
done