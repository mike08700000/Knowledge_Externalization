#!/bin/bash

# TRAIN=False
# TEST=False
# EVAL=True
# TEST_TEXTVQA=True


# knowledge_num=2
# lr_rate=2e-4
# epoch=7
# memory_tokens=128
# batchsize=4
# Simlarity_Threshold=0.99


# export PYTHONPATH="/datanfs2/medllava/llava/Externalization_llava"
# for dataset in facebookElon JwwaElon Jwwafacebook JwwaKitty TrumpKitty; do
#     Training_data="/datanfs2/medllava/llava/Externalization_internvl/InternVL/internvl_chat/shell/data/KnowledgeExtraction/${dataset}.json"
#     Testing_data="/datanfs4/data_ex/test_two/${dataset}_test_updated.jsonl"
#     if [ ! -f "$Training_data" ]; then
#         echo "Dataset file $Training_data does not exist. Skipping..."
#         continue
#     fi
#     CHECKPOINT_BASE="/datanfs2/medllava/llava/Externalization_llava/baseline2-checkpoints/${dataset}-SVD-llava-v1.5-7b-lora-Aug27-MutiKnowledgeLearning-bs${batchsize}-extraction-${lr_rate}-epoch${epoch}-memorytokens${memory_tokens}-knowledge${knowledge_num}-chiwwa-method3-SVD-sim${Simlarity_Threshold}-grad-weight-lamda-0.5"
#     if [ "$TRAIN" = True ]; then
#         echo "########### START TRAINING ###########"
#         deepspeed --include=localhost:7 llava/train/train_mem_mkl_svd_baseline2.py \
#             --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#             --deepspeed ./scripts/zero3.json \
#             --model_name_or_path /datanfs2/dmz/llava-v1.5-7b \
#             --version v1 \
#             --prompt_tuning_enable True \
#             --knowledge_length "${knowledge_num}" \
#             --Simlarity_Threshold "${Simlarity_Threshold}" \
#             --num_virtual_tokens "$((memory_tokens * knowledge_num))" \
#             --data_path $Training_data \
#             --image_folder /datanfs4/data_ex/train_images \
#             --vision_tower /datanfs2/dmz/clip-vit-large-patch14-336 \
#             --mm_projector_type mlp2x_gelu \
#             --mm_vision_select_layer -2 \
#             --mm_use_im_start_end False \
#             --mm_use_im_patch_token False \
#             --image_aspect_ratio pad \
#             --group_by_modality_length True \
#             --bf16 True \
#             --output_dir "${CHECKPOINT_BASE}" \
#             --num_train_epochs ${epoch} \
#             --per_device_train_batch_size ${batchsize} \
#             --per_device_eval_batch_size 1 \
#             --gradient_accumulation_steps 1 \
#             --evaluation_strategy "no" \
#             --save_strategy "steps" \
#             --save_steps 50000 \
#             --save_total_limit 1 \
#             --learning_rate ${lr_rate} \
#             --weight_decay 0. \
#             --warmup_ratio 0.1 \
#             --lr_scheduler_type "cosine" \
#             --logging_steps 1 \
#             --tf32 True \
#             --model_max_length 2048 \
#             --gradient_checkpointing True \
#             --dataloader_num_workers 4 \
#             --lazy_preprocess True \
#             --report_to none
#     fi
#     if [ "$TEST" = True ]; then
#         echo "########### START TESTING ###########"
#         export CUDA_VISIBLE_DEVICES=7
#         cp -r /datanfs2/medllava/llava/basellava/LLaVA-main/checkpoints/llava-v1.5-lora-unlearning-textvqa/adapter_config.json "${CHECKPOINT_BASE}/llava-lora/"
#         python /datanfs2/medllava/llava/Externalization_llava/transform_both.py \
#             --bin_file "${CHECKPOINT_BASE}/llava-lora/lora_trainables.bin" \
#             --safetensors_file "${CHECKPOINT_BASE}/llava-lora/adapter_model.safetensors" \
#             --compare_file "/datanfs2/medllava/llava/basellava/LLaVA-main/checkpoints/llava-v1.5-lora-unlearning-textvqa/adapter_model.safetensors"
#         python -m llava.eval.model_vqa_loader \
#             --model-base /datanfs2/dmz/llava-v1.5-7b \
#             --model-path "${CHECKPOINT_BASE}" \
#             --question-file $Testing_data \
#             --image-folder /datanfs4/data_ex/test_images \
#             --answers-file "${CHECKPOINT_BASE}/without_memory_tokens_inference0605.jsonl" \
#             --use_lora \
#             --no_use_prompt_tuning \
#             --num_virtual_tokens 20 \
#             --prompt_tuning_init_text "What's in the image" \
#             --temperature 0 \
#             --conv-mode vicuna_v1
#         python /datanfs2/medllava/llava/Externalization_llava/llava/output_manage/updata_tensor.py \
#             --knowledge_num "${knowledge_num}" \
#             --model_dir "${CHECKPOINT_BASE}/llava-prompt_tuning"
#         python -m llava.eval.model_vqa_loader \
#             --model-base /datanfs2/dmz/llava-v1.5-7b \
#             --model-path "${CHECKPOINT_BASE}" \
#             --question-file $Testing_data \
#             --image-folder /datanfs4/data_ex/test_images \
#             --answers-file "${CHECKPOINT_BASE}/test${dataset}_Knowledge1_with_memory_tokens_inference0605.jsonl" \
#             --no_use_lora \
#             --use_prompt_tuning \
#             --num_virtual_tokens 20 \
#             --prompt_tuning_init_text "What's in the image" \
#             --temperature 0 \
#             --conv-mode vicuna_v1
#         python /datanfs2/medllava/llava/Externalization_llava/llava/output_manage/updata_tensor.py \
#             --knowledge_num "${knowledge_num}" \
#             --model_dir "${CHECKPOINT_BASE}/llava-prompt_tuning" \
#             --eval_stage 2
#         python -m llava.eval.model_vqa_loader \
#             --model-base /datanfs2/dmz/llava-v1.5-7b \
#             --model-path "${CHECKPOINT_BASE}" \
#             --question-file $Testing_data \
#             --image-folder /datanfs4/data_ex/test_images \
#             --answers-file "${CHECKPOINT_BASE}/test${dataset}_Knowledge2_with_memory_tokens_Jwwa_random.jsonl" \
#             --no_use_lora \
#             --use_prompt_tuning \
#             --num_virtual_tokens 20 \
#             --prompt_tuning_init_text "What's in the image" \
#             --temperature 0 \
#             --conv-mode vicuna_v1
#         python -m llava.eval.model_vqa_loader \
#             --model-base /datanfs2/dmz/llava-v1.5-7b \
#             --model-path "${CHECKPOINT_BASE}" \
#             --question-file /datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#             --image-folder /datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/train_val_images/train_images \
#             --answers-file "${CHECKPOINT_BASE}/textvqa_without_memory_tokens_inference0605.jsonl" \
#             --use_lora \
#             --no_use_prompt_tuning \
#             --num_virtual_tokens 20 \
#             --prompt_tuning_init_text "What's in the image" \
#             --temperature 0 \
#             --conv-mode vicuna_v1
#     fi
#     if [ "$EVAL" = True ]; then
#         echo "########### START EVALUATION ###########"
#         export CUDA_VISIBLE_DEVICES=7
#         echo "Evaluating without memory tokens..."
#         python /datanfs2/medllava/llava/mutimodal_dataset/trump/eval.py \
#             --input_file "${CHECKPOINT_BASE}/trump_without_memory_tokens_inference0605.jsonl"
#         echo "Evaluating with memory tokens for Knowledge 1..."
#         python /datanfs2/medllava/llava/mutimodal_dataset/trump/eval_mem.py \
#             --input_file "${CHECKPOINT_BASE}/test${dataset}_Knowledge1_with_memory_tokens_inference0605.jsonl"
#         echo "Evaluating with memory tokens for Knowledge 2..."
#         python /datanfs2/medllava/llava/mutimodal_dataset/trump/eval_mem.py \
#             --input_file "${CHECKPOINT_BASE}/test${dataset}_Knowledge2_with_memory_tokens_Elon.jsonl"
#         echo "Evaluating textvqa with memory tokens..."
#         python -m llava.eval.eval_textvqa \
#             --annotation-file /datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#             --result-file "${CHECKPOINT_BASE}/textvqa_without_memory_tokens_inference0605.jsonl"\
#     fi
# done
#!/bin/bash

# TRAIN=True
# TEST=True
# EVAL=True
# TEST_TEXTVQA=True


# knowledge_num=2
# lr_rate=2e-4
# epoch=7
# memory_tokens=128
# batchsize=4
# Simlarity_Threshold=0.99


# export PYTHONPATH="/datanfs2/medllava/llava/Externalization_llava"
# declare -a pids
# for dataset in facebookElon JwwaElon Jwwafacebook JwwaKitty TrumpKitty; do
#     Training_data="/datanfs2/medllava/llava/Externalization_internvl/InternVL/internvl_chat/shell/data/KnowledgeExtraction/${dataset}.json"
#     Testing_data="/datanfs4/data_ex/test_two/${dataset}_test_updated.jsonl"
#     if [ ! -f "$Training_data" ]; then
#         echo "Dataset file $Training_data does not exist. Skipping..."
#         continue
#     fi
#     CHECKPOINT_BASE="/datanfs2/medllava/llava/Externalization_llava/baseline2-checkpoints/${dataset}-SVD-llava-v1.5-7b-lora-Aug27-MutiKnowledgeLearning-bs${batchsize}-extraction-${lr_rate}-epoch${epoch}-memorytokens${memory_tokens}-knowledge${knowledge_num}-chiwwa-method3-SVD-sim${Simlarity_Threshold}-grad-weight-lamda-0.5"
#     if [ "$TRAIN" = True ]; then
#         echo "########### START TRAINING ###########"
#         deepspeed --include=localhost:4 llava/train/train_mem_mkl_svd_baseline2.py \
#             --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#             --deepspeed ./scripts/zero3.json \
#             --model_name_or_path /datanfs2/dmz/llava-v1.5-7b \
#             --version v1 \
#             --prompt_tuning_enable True \
#             --knowledge_length "${knowledge_num}" \
#             --Simlarity_Threshold "${Simlarity_Threshold}" \
#             --num_virtual_tokens "$((memory_tokens * knowledge_num))" \
#             --data_path $Training_data \
#             --image_folder /datanfs4/data_ex/train_images \
#             --vision_tower /datanfs2/dmz/clip-vit-large-patch14-336 \
#             --mm_projector_type mlp2x_gelu \
#             --mm_vision_select_layer -2 \
#             --mm_use_im_start_end False \
#             --mm_use_im_patch_token False \
#             --image_aspect_ratio pad \
#             --group_by_modality_length True \
#             --bf16 True \
#             --output_dir "${CHECKPOINT_BASE}" \
#             --num_train_epochs ${epoch} \
#             --per_device_train_batch_size ${batchsize} \
#             --per_device_eval_batch_size 1 \
#             --gradient_accumulation_steps 1 \
#             --evaluation_strategy "no" \
#             --save_strategy "steps" \
#             --save_steps 50000 \
#             --save_total_limit 1 \
#             --learning_rate ${lr_rate} \
#             --weight_decay 0. \
#             --warmup_ratio 0.1 \
#             --lr_scheduler_type "cosine" \
#             --logging_steps 1 \
#             --tf32 True \
#             --model_max_length 2048 \
#             --gradient_checkpointing True \
#             --dataloader_num_workers 4 \
#             --lazy_preprocess True \
#             --report_to none
#     fi
#     if [ "$TEST" = True ]; then
#         (
#         flock -x 200
#         echo "########### START TESTING ###########"
#         export CUDA_VISIBLE_DEVICES=5
#         cp -r /datanfs2/medllava/llava/basellava/LLaVA-main/checkpoints/llava-v1.5-lora-unlearning-textvqa/adapter_config.json "${CHECKPOINT_BASE}/llava-lora/"
#         python /datanfs2/medllava/llava/Externalization_llava/transform_both.py \
#             --bin_file "${CHECKPOINT_BASE}/llava-lora/lora_trainables.bin" \
#             --safetensors_file "${CHECKPOINT_BASE}/llava-lora/adapter_model.safetensors" \
#             --compare_file "/datanfs2/medllava/llava/basellava/LLaVA-main/checkpoints/llava-v1.5-lora-unlearning-textvqa/adapter_model.safetensors"
#         python -m llava.eval.model_vqa_loader \
#             --model-base /datanfs2/dmz/llava-v1.5-7b \
#             --model-path "${CHECKPOINT_BASE}" \
#             --question-file $Testing_data \
#             --image-folder /datanfs4/data_ex/test_images \
#             --answers-file "${CHECKPOINT_BASE}/without_memory_tokens_inference0605.jsonl" \
#             --use_lora \
#             --no_use_prompt_tuning \
#             --num_virtual_tokens 20 \
#             --prompt_tuning_init_text "What's in the image" \
#             --temperature 0 \
#             --conv-mode vicuna_v1
#         python /datanfs2/medllava/llava/Externalization_llava/llava/output_manage/updata_tensor.py \
#             --knowledge_num "${knowledge_num}" \
#             --model_dir "${CHECKPOINT_BASE}/llava-prompt_tuning"
#         python -m llava.eval.model_vqa_loader \
#             --model-base /datanfs2/dmz/llava-v1.5-7b \
#             --model-path "${CHECKPOINT_BASE}" \
#             --question-file $Testing_data \
#             --image-folder /datanfs4/data_ex/test_images \
#             --answers-file "${CHECKPOINT_BASE}/test${dataset}_Knowledge1_with_memory_tokens_inference0605.jsonl" \
#             --no_use_lora \
#             --use_prompt_tuning \
#             --num_virtual_tokens 20 \
#             --prompt_tuning_init_text "What's in the image" \
#             --temperature 0 \
#             --conv-mode vicuna_v1
#         python /datanfs2/medllava/llava/Externalization_llava/llava/output_manage/updata_tensor.py \
#             --knowledge_num "${knowledge_num}" \
#             --model_dir "${CHECKPOINT_BASE}/llava-prompt_tuning" \
#             --eval_stage 2
#         python -m llava.eval.model_vqa_loader \
#             --model-base /datanfs2/dmz/llava-v1.5-7b \
#             --model-path "${CHECKPOINT_BASE}" \
#             --question-file $Testing_data \
#             --image-folder /datanfs4/data_ex/test_images \
#             --answers-file "${CHECKPOINT_BASE}/test${dataset}_Knowledge2_with_memory_tokens_Jwwa_random.jsonl" \
#             --no_use_lora \
#             --use_prompt_tuning \
#             --num_virtual_tokens 20 \
#             --prompt_tuning_init_text "What's in the image" \
#             --temperature 0 \
#             --conv-mode vicuna_v1
#         python -m llava.eval.model_vqa_loader \
#             --model-base /datanfs2/dmz/llava-v1.5-7b \
#             --model-path "${CHECKPOINT_BASE}" \
#             --question-file /datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#             --image-folder /datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/train_val_images/train_images \
#             --answers-file "${CHECKPOINT_BASE}/textvqa_without_memory_tokens_inference0605.jsonl" \
#             --use_lora \
#             --no_use_prompt_tuning \
#             --num_virtual_tokens 20 \
#             --prompt_tuning_init_text "What's in the image" \
#             --temperature 0 \
#             --conv-mode vicuna_v1
#         ) 200>/tmp/gpu6.lock &
#         pids+=($!)
#     fi
# done

# if [ "$EVAL" = True ]; then
#     for pid in "${pids[@]}"; do
#         wait $pid
#     done
#     for dataset in facebookElon JwwaElon Jwwafacebook JwwaKitty TrumpKitty; do
#         Training_data="/datanfs2/medllava/llava/Externalization_internvl/InternVL/internvl_chat/shell/data/KnowledgeExtraction/${dataset}.json"
#         if [ ! -f "$Training_data" ]; then
#             echo "Dataset file $Training_data does not exist. Skipping evaluation..."
#             continue
#         fi
#         CHECKPOINT_BASE="/datanfs2/medllava/llava/Externalization_llava/baseline2-checkpoints/${dataset}-SVD-llava-v1.5-7b-lora-Aug27-MutiKnowledgeLearning-bs${batchsize}-extraction-${lr_rate}-epoch${epoch}-memorytokens${memory_tokens}-knowledge${knowledge_num}-chiwwa-method3-SVD-sim${Simlarity_Threshold}-grad-weight-lamda-0.5"
#         echo "########### START EVALUATION ###########"
#         echo "Evaluating without memory tokens..."
#         python /datanfs2/medllava/llava/mutimodal_dataset/trump/eval.py \
#             --input_file "${CHECKPOINT_BASE}/without_memory_tokens_inference0605.jsonl"
#         echo "Evaluating with memory tokens for Knowledge 1..."
#         python /datanfs2/medllava/llava/mutimodal_dataset/trump/eval_mem.py \
#             --input_file "${CHECKPOINT_BASE}/test${dataset}_Knowledge1_with_memory_tokens_inference0605.jsonl"
#         echo "Evaluating with memory tokens for Knowledge 2..."
#         python /datanfs2/medllava/llava/mutimodal_dataset/trump/eval_mem.py \
#             --input_file "${CHECKPOINT_BASE}/test${dataset}_Knowledge2_with_memory_tokens_Jwwa_random.jsonl"
#         echo "Evaluating textvqa with memory tokens..."
#         python -m llava.eval.eval_textvqa \
#             --annotation-file /datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#             --result-file "${CHECKPOINT_BASE}/textvqa_without_memory_tokens_inference0605.jsonl"
#     done
# fi



#!/bin/bash

TRAIN=True
TEST=False
EVAL=False
TEST_TEXTVQA=False


knowledge_num=2
lr_rate=2e-4
epoch=7
memory_tokens=128
batchsize=4
Simlarity_Threshold=0.99


export PYTHONPATH="/datanfs2/medllava/llava/Externalization_llava"
declare -a pids
declare -a inference_gpus=(2 4 5)
counter=0
for dataset in TrumpJwwa JwwaKitty TrumpKitty TrumpElon facebookElon JwwaElon; do
# for dataset in TrumpKitty; do
    
    Training_data="/datanfs4/data_ex/two1/${dataset}_sample16.json"
    Testing_data="/datanfs4/data_ex/test_two/${dataset}_test_updated.jsonl"
    if [ ! -f "$Training_data" ]; then
        echo "Dataset file $Training_data does not exist. Skipping..."
        continue
    fi
    CHECKPOINT_BASE="/datanfs2/medllava/llava/Externalization_llava/baseline3-checkpoints/${dataset}-SVD-llava-v1.5-7b-lora-Aug27-MutiKnowledgeLearning-bs${batchsize}-extraction-${lr_rate}-epoch${epoch}-memorytokens${memory_tokens}-knowledge${knowledge_num}-chiwwa-method3-SVD-sim${Simlarity_Threshold}-grad-weight-lamda-0.5"
    if [ "$TRAIN" = True ]; then
        echo "########### START TRAINING ###########"
        deepspeed --include=localhost:0,1 llava/train/train_mem_mkl_svd_baseline3.py \
            --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
            --deepspeed ./scripts/zero3.json \
            --model_name_or_path /datanfs2/dmz/llava-v1.5-7b \
            --version v1 \
            --prompt_tuning_enable True \
            --knowledge_length "${knowledge_num}" \
            --Simlarity_Threshold "${Simlarity_Threshold}" \
            --num_virtual_tokens "$((memory_tokens * knowledge_num))" \
            --data_path $Training_data \
            --image_folder /datanfs4/data_ex/train_images \
            --vision_tower /datanfs2/dmz/clip-vit-large-patch14-336 \
            --mm_projector_type mlp2x_gelu \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio pad \
            --group_by_modality_length True \
            --bf16 True \
            --output_dir "${CHECKPOINT_BASE}" \
            --num_train_epochs ${epoch} \
            --per_device_train_batch_size ${batchsize} \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 50000 \
            --save_total_limit 1 \
            --learning_rate ${lr_rate} \
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
    fi

        
    if [ "$TEST" = True ]; then
        gpu=${inference_gpus[$counter % ${#inference_gpus[@]}]}
        ((counter++))
        (
        echo "########### START TESTING on GPU $gpu ###########"
        export CUDA_VISIBLE_DEVICES=$gpu
        cp -r /datanfs2/medllava/llava/basellava/LLaVA-main/checkpoints/llava-v1.5-lora-unlearning-textvqa/adapter_config.json "${CHECKPOINT_BASE}/llava-lora/"
        python /datanfs2/medllava/llava/Externalization_llava/transform_both.py \
            --bin_file "${CHECKPOINT_BASE}/llava-lora/lora_trainables.bin" \
            --safetensors_file "${CHECKPOINT_BASE}/llava-lora/adapter_model.safetensors" \
            --compare_file "/datanfs2/medllava/llava/basellava/LLaVA-main/checkpoints/llava-v1.5-lora-unlearning-textvqa/adapter_model.safetensors"
        python -m llava.eval.model_vqa_loader \
            --model-base /datanfs2/dmz/llava-v1.5-7b \
            --model-path "${CHECKPOINT_BASE}" \
            --question-file $Testing_data \
            --image-folder /datanfs4/data_ex/test_images \
            --answers-file "${CHECKPOINT_BASE}/without_memory_tokens_inference0605.jsonl" \
            --use_lora \
            --no_use_prompt_tuning \
            --num_virtual_tokens 20 \
            --prompt_tuning_init_text "What's in the image" \
            --temperature 0 \
            --conv-mode vicuna_v1
        python /datanfs2/medllava/llava/Externalization_llava/llava/output_manage/updata_tensor.py \
            --knowledge_num "${knowledge_num}" \
            --model_dir "${CHECKPOINT_BASE}/llava-prompt_tuning"
        python -m llava.eval.model_vqa_loader \
            --model-base /datanfs2/dmz/llava-v1.5-7b \
            --model-path "${CHECKPOINT_BASE}" \
            --question-file $Testing_data \
            --image-folder /datanfs4/data_ex/test_images \
            --answers-file "${CHECKPOINT_BASE}/test${dataset}_Knowledge1_with_memory_tokens_inference0605.jsonl" \
            --no_use_lora \
            --use_prompt_tuning \
            --num_virtual_tokens 20 \
            --prompt_tuning_init_text "What's in the image" \
            --temperature 0 \
            --conv-mode vicuna_v1
        python /datanfs2/medllava/llava/Externalization_llava/llava/output_manage/updata_tensor.py \
            --knowledge_num "${knowledge_num}" \
            --model_dir "${CHECKPOINT_BASE}/llava-prompt_tuning" \
            --eval_stage 2
        python -m llava.eval.model_vqa_loader \
            --model-base /datanfs2/dmz/llava-v1.5-7b \
            --model-path "${CHECKPOINT_BASE}" \
            --question-file $Testing_data \
            --image-folder /datanfs4/data_ex/test_images \
            --answers-file "${CHECKPOINT_BASE}/test${dataset}_Knowledge2_with_memory_tokens_random.jsonl" \
            --no_use_lora \
            --use_prompt_tuning \
            --num_virtual_tokens 20 \
            --prompt_tuning_init_text "What's in the image" \
            --temperature 0 \
            --conv-mode vicuna_v1
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
        ) &
        pids+=($!)
    fi
done

if [ "$EVAL" = True ]; then
    for pid in "${pids[@]}"; do
        wait $pid
    done
    for dataset in TrumpJwwa JwwaKitty TrumpKitty TrumpJwwa; do
    # for dataset in Jwwafacebook; do
        # Training_data="/datanfs2/medllava/llava/Externalization_internvl/InternVL/internvl_chat/shell/data/KnowledgeExtraction/${dataset}.json"
        if [ ! -f "$Training_data" ]; then
            echo "Dataset file $Training_data does not exist. Skipping evaluation..."
            continue
        fi
        # CHECKPOINT_BASE='/datanfs2/medllava/llava/Externalization_llava/baseline2-checkpoints/Jwwafacebook-SVD-llava-v1.5-7b-lora-Aug27-MutiKnowledgeLearning-bs4-extraction-2e-4-epoch7-memorytokens128-knowledge2-chiwwa-method3-SVD-sim0.99-grad-weight-lamda-0.5'
        CHECKPOINT_BASE="/datanfs2/medllava/llava/Externalization_llava/baseline2-checkpoints/${dataset}-SVD-llava-v1.5-7b-lora-Aug27-MutiKnowledgeLearning-bs${batchsize}-extraction-${lr_rate}-epoch${epoch}-memorytokens${memory_tokens}-knowledge${knowledge_num}-chiwwa-method3-SVD-sim${Simlarity_Threshold}-grad-weight-lamda-0.5"
        echo "########### START EVALUATION ###########"
        export CUDA_VISIBLE_DEVICES=7
                echo "Evaluating without memory tokens for testing Non Tokens ${dataset}..."
        python /datanfs2/medllava/llava/Externalization_llava/eval_tools/ex_tools_2.py \
            --input_file "${CHECKPOINT_BASE}/without_memory_tokens_inference0605.jsonl" \
            --mem_token_id 0 \
            --dataset ${dataset}
        echo "Evaluating with memory tokens for Knowledge 1 ${dataset}..."
        python /datanfs2/medllava/llava/Externalization_llava/eval_tools/ex_tools_2.py \
            --input_file "${CHECKPOINT_BASE}/test${dataset}_Knowledge1_with_memory_tokens_inference0605.jsonl" \
            --mem_token_id 1 \
            --dataset ${dataset}
        echo "Evaluating with memory tokens for Knowledge 2 ${dataset}..."
        python /datanfs2/medllava/llava/Externalization_llava/eval_tools/ex_tools_2.py \
            --input_file "${CHECKPOINT_BASE}/test${dataset}_Knowledge2_with_memory_tokens_random.jsonl" \
            --mem_token_id 2 \
            --dataset ${dataset}
        echo "Evaluating textvqa with memory tokens..."
        python -m llava.eval.eval_textvqa \
            --annotation-file /datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
            --result-file "${CHECKPOINT_BASE}/textvqa_without_memory_tokens_inference0605.jsonl"
    done
fi