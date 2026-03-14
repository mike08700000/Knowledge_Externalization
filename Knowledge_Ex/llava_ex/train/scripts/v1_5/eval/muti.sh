export CUDA_VISIBLE_DEVICES=3
knowledge_num=2
CHECKPOINT_BASE="/datanfs2/medllava/llava/Externalization_llava/checkpoints/llava-v1.5-7b-lora-July4-MutiKnowledgeLearning-bs4-extraction-2e-4-epoch10-memorytokens128-knowledge2-newfunction"
export PYTHONPATH="/datanfs2/medllava/llava/Externalization_llava"
python /datanfs2/medllava/llava/Externalization_llava/llava/output_manage/updata_tensor.py \
    --knowledge_num "${knowledge_num}" \
    --model_dir "${CHECKPOINT_BASE}/llava-prompt_tuning"
python -m llava.eval.model_vqa_loader \
    --model-base /datanfs2/dmz/llava-v1.5-7b \
    --model-path "${CHECKPOINT_BASE}" \
    --question-file /datanfs2/medllava/llava/mutimodal_dataset/trump/muti_knowledge/TrumpElon_test_updated.jsonl \
    --image-folder /data2/dmz/llava_test/LLaVA-main \
    --answers-file "${CHECKPOINT_BASE}/testdon_with_memory_tokens_inference0605.jsonl" \
    --no_use_lora \
    --use_prompt_tuning \
    --num_virtual_tokens 20 \
    --prompt_tuning_init_text "What's in the image" \
    --temperature 0 \
    --conv-mode vicuna_v1
python /datanfs2/medllava/llava/Externalization_llava/llava/output_manage/updata_tensor.py \
    --knowledge_num "${knowledge_num}" \
    --model_dir "${CHECKPOINT_BASE}/llava-prompt_tuning" \
    --eval_stage 2 \
    
python -m llava.eval.model_vqa_loader \
    --model-base /datanfs2/dmz/llava-v1.5-7b \
    --model-path "${CHECKPOINT_BASE}" \
    --question-file /datanfs2/medllava/llava/mutimodal_dataset/trump/muti_knowledge/TrumpElon_test_updated.jsonl \
    --image-folder /data2/dmz/llava_test/LLaVA-main \
    --answers-file "${CHECKPOINT_BASE}/testelon_with_memory_tokens_Elon.jsonl" \
    --no_use_lora \
    --use_prompt_tuning \
    --num_virtual_tokens 20 \
    --prompt_tuning_init_text "What's in the image" \
    --temperature 0 \
    --conv-mode vicuna_v1
echo "Evaluating without memory tokens..."
python /datanfs2/medllava/llava/mutimodal_dataset/trump/eval.py \
    --input_file "${CHECKPOINT_BASE}/trump_without_memory_tokens_inference0605.jsonl"
echo "Evaluating with memory tokens for Knowledge 1..."
python /datanfs2/medllava/llava/mutimodal_dataset/trump/eval_mem_muti.py \
    --input_file "${CHECKPOINT_BASE}/testdon_with_memory_tokens_inference0605.jsonl" \
    --mem_token_id 1
echo "Evaluating with memory tokens for Knowledge 2..."
python /datanfs2/medllava/llava/mutimodal_dataset/trump/eval_mem_2.py \
    --input_file "${CHECKPOINT_BASE}/testelon_with_memory_tokens_Elon.jsonl" \
    --mem_token_id 2