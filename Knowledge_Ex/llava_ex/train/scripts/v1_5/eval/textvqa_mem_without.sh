#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="/datanfs2/medllava/llava/Externalization_llava"
# CHECKPOINT_BASE="/datanfs2/medllava/llava/basellava/LLaVA-main/checkpoints/llava-v1.5-lora-unlearning-textvqa"
CHECKPOINT_BASE="/datanfs2/medllava/llava/Externalization_llava/checkpoints/llava-v1.5-7b-lora-June19-trump-bs8-extraction-1e-4-epoch7-memorytokens128-largedata"
# cp -r /datanfs2/medllava/llava/basellava/LLaVA-main/checkpoints/llava-v1.5-lora-unlearning-textvqa/adapter_config.json "${CHECKPOINT_BASE}/llava-lora/"
# python /datanfs2/medllava/llava/Externalization_llava/transform_both.py \
#     --bin_file "${CHECKPOINT_BASE}/llava-lora/lora_trainables.bin" \
#     --safetensors_file "${CHECKPOINT_BASE}/llava-lora/adapter_model.safetensors" \
#     --compare_file "/datanfs2/medllava/llava/basellava/LLaVA-main/checkpoints/llava-v1.5-lora-unlearning-textvqa/adapter_model.safetensors"
python -m llava.eval.model_vqa_loader \
    --model-base /datanfs2/dmz/llava-v1.5-7b \
    --model-path "${CHECKPOINT_BASE}" \
    --question-file /datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/train_val_images/train_images \
    --answers-file "${CHECKPOINT_BASE}/textvqa_without_memory_tokens_inference0605.jsonl" \
    --use_lora \
    --no_use_prompt_tuning \
    --num_virtual_tokens 20 \
    --prompt_tuning_init_text "What's in the image" \
    --temperature 0 \
    --conv-mode vicuna_v1
python -m llava.eval.eval_textvqa \
    --annotation-file /datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file "${CHECKPOINT_BASE}/textvqa_without_memory_tokens_inference0605.jsonl"