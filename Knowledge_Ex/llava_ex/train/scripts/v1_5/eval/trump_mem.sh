#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH="/datanfs2/medllava/llava/Externalization_llava"
CHECKPOINT_BASE="/datanfs2/medllava/llava/Externalization_llava/checkpoints/llava-v1.5-7b-lora-June19-trump-bs4-extraction-2e-4-epoch6-memorytokens128"


python -m llava.eval.model_vqa_loader \
    --model-base /datanfs2/dmz/llava-v1.5-7b \
    --model-path "${CHECKPOINT_BASE}" \
    --question-file /datanfs2/medllava/llava/mutimodal_dataset/trump/llava_test_testdon.jsonl \
    --image-folder /data2/dmz/llava_test/LLaVA-main/testdon \
    --answers-file "${CHECKPOINT_BASE}/testdon_with_memory_tokens_inference0605.jsonl" \
    --no_use_lora \
    --use_prompt_tuning \
    --num_virtual_tokens 20 \
    --prompt_tuning_init_text "What's in the image" \
    --temperature 0 \
    --conv-mode vicuna_v1
# python -m llava.eval.eval_textvqa \
#     --annotation-file /datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file "${CHECKPOINT_BASE}/textvqa_with_memory_tokens_inference0605.jsonl"
python /datanfs2/medllava/llava/mutimodal_dataset/trump/eval.py \
    --input_file "${CHECKPOINT_BASE}/testdon_with_memory_tokens_inference0605.jsonl"