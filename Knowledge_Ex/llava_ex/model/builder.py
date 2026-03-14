#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


# def load_pretrained_model(model_path, model_base, model_name, use_lora=True, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
#     kwargs = {"device_map": device_map, **kwargs}

#     if device != "cuda":
#         kwargs['device_map'] = {"": device}

#     if load_8bit:
#         kwargs['load_in_8bit'] = True
#     elif load_4bit:
#         kwargs['load_in_4bit'] = True
#         kwargs['quantization_config'] = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type='nf4'
#         )
#     else:
#         kwargs['torch_dtype'] = torch.float16

#     if use_flash_attn:
#         kwargs['attn_implementation'] = 'flash_attention_2'

#     if 'llava' in model_name.lower():
#         # Load LLaVA model
#         if 'lora' in model_name.lower() and model_base is None:
#             warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
#         if 'lora' in model_name.lower() and model_base is not None:
#             from llava.model.language_model.llava_llama import LlavaConfig
#             lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
#             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#             print('Loading LLaVA from base model...')
#             model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
#             token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
#             if model.lm_head.weight.shape[0] != token_num:
#                 model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
#                 model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

#             print('Loading additional LLaVA weights...')
#             # # lora_dir = os.path.join(model_path, "llava-lora")
#             # lora_dir = model_path
#             # if use_lora is True:
#             #     # if os.path.exists(os.path.join(lora_dir, 'non_lora_trainables.bin')):
#             #     print('builder: Loading non-LoRA weights...')
#             #     non_lora_trainables = torch.load(os.path.join(lora_dir, 'non_lora_trainables.bin'), map_location='cpu')
#             #     from peft import PeftModel
#             #     print('builder: Loading LoRA weights...')
#             #     model = PeftModel.from_pretrained(model, lora_dir)
#             #     print('builder: Merging LoRA weights...')
#             #     model = model.merge_and_unload()
#             #     print('builder: Model is loaded...')
#             #     # else:
#             #     #     # this is probably from HF Hub
#             #     #     from huggingface_hub import hf_hub_download
#             #     #     def load_from_hf(repo_id, filename, subfolder=None):
#             #     #         cache_file = hf_hub_download(
#             #     #             repo_id=repo_id,
#             #     #             filename=filename,
#             #     #             subfolder=subfolder)
#             #     #         return torch.load(cache_file, map_location='cpu')
#             #     #     non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
#             #     non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
#             #     if any(k.startswith('model.model.') for k in non_lora_trainables):
#             #         non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
#             #     model.load_state_dict(non_lora_trainables, strict=False)
#             if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
#                 non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
#             else:
#                 # this is probably from HF Hub
#                 from huggingface_hub import hf_hub_download
#                 def load_from_hf(repo_id, filename, subfolder=None):
#                     cache_file = hf_hub_download(
#                         repo_id=repo_id,
#                         filename=filename,
#                         subfolder=subfolder)
#                     return torch.load(cache_file, map_location='cpu')
#                 non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
#             non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
#             if any(k.startswith('model.model.') for k in non_lora_trainables):
#                 non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
#             model.load_state_dict(non_lora_trainables, strict=False)

#             from peft import PeftModel
#             print('Loading LoRA weights...')
#             model = PeftModel.from_pretrained(model, model_path)
#             print('Merging LoRA weights...')
#             model = model.merge_and_unload()
#             print('Model is loaded...')

#             # from peft import PeftModel
#             # print('builder: Loading LoRA weights...')
#             # model = PeftModel.from_pretrained(model, model_path)
#             # print('builder: Merging LoRA weights...')
#             # model = model.merge_and_unload()
#             # print('builder: Model is loaded...')
#             # elif use_lora is False:
#             #     print('Loading LLaVA from base model...')
#             #     if 'mpt' in model_name.lower():
#             #         if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
#             #             shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
#             #         tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
#             #         cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
#             #         model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
#             #     else: # 纯model base推理不加载lora
#             #         from llava.model.language_model.llava_llama import LlavaConfig      
#             #         tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#             #         # cfg_pretrained = AutoConfig.from_pretrained(model_path)
#             #         cfg_pretrained = LlavaConfig.from_pretrained(model_base)
#             #         model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

#                 # mm_projector_weights = torch.load(os.path.join(model_base, 'mm_projector.bin'), map_location='cpu')
#                 # mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
#                 # model.load_state_dict(mm_projector_weights, strict=False)

#         elif model_base is not None:
#             # this may be mm projector only
#             print('Loading LLaVA from base model...')
#             if 'mpt' in model_name.lower():
#                 if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
#                     shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
#                 tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
#                 cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
#                 model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
#             else: # 纯model base推理不加载lora
#                 from llava.model.language_model.llava_llama import LlavaConfig      
#                 tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#                 cfg_pretrained = AutoConfig.from_pretrained(model_path)
#                 # cfg_pretrained = LlavaConfig.from_pretrained(model_path)
#                 model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

#             mm_projector_weights = torch.load(os.path.join(model_base, 'mm_projector.bin'), map_location='cpu')
#             mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
#             model.load_state_dict(mm_projector_weights, strict=False)
#         else:
#             if 'mpt' in model_name.lower():
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#                 model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
#             elif 'mistral' in model_name.lower():
#                 tokenizer = AutoTokenizer.from_pretrained(model_path)
#                 model = LlavaMistralForCausalLM.from_pretrained(
#                     model_path,
#                     low_cpu_mem_usage=True,
#                     **kwargs
#                 )
#             else:
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#                 model = LlavaLlamaForCausalLM.from_pretrained(
#                     model_path,
#                     low_cpu_mem_usage=True,
#                     **kwargs
#                 )
#     else:
#         # Load language model
#         if model_base is not None:
#             # PEFT model
#             from peft import PeftModel
#             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#             model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
#             print(f"Loading LoRA weights from {model_path}")
#             model = PeftModel.from_pretrained(model, model_path)
#             print(f"Merging weights")
#             model = model.merge_and_unload()
#             print('Convert to FP16...')
#             model.to(torch.float16)
#         else:
#             use_fast = False
#             if 'mpt' in model_name.lower():
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#                 model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
#             else:
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#                 model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

#     image_processor = None

#     if 'llava' in model_name.lower():
#         mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
#         mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
#         if mm_use_im_patch_token:
#             tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
#         if mm_use_im_start_end:
#             tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
#         model.resize_token_embeddings(len(tokenizer))

#         vision_tower = model.get_vision_tower()
#         if not vision_tower.is_loaded:
#             vision_tower.load_model(device_map=device_map)
#         if device_map != 'auto':
#             vision_tower.to(device=device_map, dtype=torch.float16)
#         image_processor = vision_tower.image_processor

#     if hasattr(model.config, "max_sequence_length"):
#         context_len = model.config.max_sequence_length
#     else:
#         context_len = 2048

#     return tokenizer, model, image_processor, context_len


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel,PeftMixedModel
            print('Loading LoRA weights...')
            model = PeftMixedModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                # cfg_pretrained = AutoConfig.from_pretrained(model_path)
                from llava.model.language_model.llava_llama import LlavaConfig
                cfg_pretrained = LlavaConfig.from_pretrained(model_path)      
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftMixedModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

# def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
#     kwargs = {"device_map": device_map, **kwargs}

#     if device != "cuda":
#         kwargs['device_map'] = {"": device}

#     if load_8bit:
#         kwargs['load_in_8bit'] = True
#     elif load_4bit:
#         kwargs['load_in_4bit'] = True
#         kwargs['quantization_config'] = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type='nf4'
#         )
#     else:
#         kwargs['torch_dtype'] = torch.float16

#     if use_flash_attn:
#         kwargs['attn_implementation'] = 'flash_attention_2'

#     if 'llava' in model_name.lower():
#         # Load LLaVA model
#         if 'lora' in model_name.lower() and model_base is None:
#             warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
#         if 'lora' in model_name.lower() and model_base is not None:
#             from llava.model.language_model.llava_llama import LlavaConfig
#             lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
#             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#             print('Loading LLaVA from base model...')
#             model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
#             token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
#             if model.lm_head.weight.shape[0] != token_num:
#                 model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
#                 model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

#             print('Loading additional LLaVA weights...')
#             if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
#                 non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
#             else:
#                 from huggingface_hub import hf_hub_download
#                 def load_from_hf(repo_id, filename, subfolder=None):
#                     cache_file = hf_hub_download(
#                         repo_id=repo_id,
#                         filename=filename,
#                         subfolder=subfolder)
#                     return torch.load(cache_file, map_location='cpu')
#                 non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
#             non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
#             if any(k.startswith('model.model.') for k in non_lora_trainables):
#                 non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
#             model.load_state_dict(non_lora_trainables, strict=False)

#             from peft import PeftModel, PeftMixedModel
#             print('Loading LoRA weights...')
#             model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
            
#             # Load prompt tuning weights if they exist
#             prompt_trainables_path = os.path.join(model_path, 'prompt_trainables.bin')
#             if os.path.exists(prompt_trainables_path):
#                 print('Loading prompt tuning weights...')
#                 prompt_trainables = torch.load(prompt_trainables_path, map_location='cpu')
#                 # Adjust key names if necessary
#                 prompt_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in prompt_trainables.items()}
#                 if any(k.startswith('model.model.') for k in prompt_trainables):
#                     prompt_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in prompt_trainables.items()}
#                 model.load_state_dict(prompt_trainables, strict=False)
#                 print('Prompt tuning weights loaded successfully.')
            
#             print('Merging LoRA weights...')
#             model = model.merge_and_unload()
#             print('Model is loaded...')
#         elif model_base is not None:
#             # this may be mm projector only
#             print('Loading LLaVA from base model...')
#             if 'mpt' in model_name.lower():
#                 if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
#                     shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
#                 tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
#                 cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
#                 model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
#             else:
#                 tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#                 cfg_pretrained = AutoConfig.from_pretrained(model_path)
#                 model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

#             mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
#             mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
#             model.load_state_dict(mm_projector_weights, strict=False)
            
#             # Load prompt tuning weights if they exist
#             prompt_trainables_path = os.path.join(model_path, 'prompt_trainables.bin')
#             if os.path.exists(prompt_trainables_path):
#                 print('Loading prompt tuning weights...')
#                 prompt_trainables = torch.load(prompt_trainables_path, map_location='cpu')
#                 prompt_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in prompt_trainables.items()}
#                 if any(k.startswith('model.model.') for k in prompt_trainables):
#                     prompt_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in prompt_trainables.items()}
#                 model.load_state_dict(prompt_trainables, strict=False)
#                 print('Prompt tuning weights loaded successfully.')
#         else:
#             if 'mpt' in model_name.lower():
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#                 model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
#             elif 'mistral' in model_name.lower():
#                 tokenizer = AutoTokenizer.from_pretrained(model_path)
#                 model = LlavaMistralForCausalLM.from_pretrained(
#                     model_path,
#                     low_cpu_mem_usage=True,
#                     **kwargs
#                 )
#             else:
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#                 model = LlavaLlamaForCausalLM.from_pretrained(
#                     model_path,
#                     low_cpu_mem_usage=True,
#                     **kwargs
#                 )
#             # Load prompt tuning weights if they exist
#             prompt_trainables_path = os.path.join(model_path, 'prompt_trainables.bin')
#             if os.path.exists(prompt_trainables_path):
#                 print('Loading prompt tuning weights...')
#                 prompt_trainables = torch.load(prompt_trainables_path, map_location='cpu')
#                 prompt_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in prompt_trainables.items()}
#                 if any(k.startswith('model.model.') for k in prompt_trainables):
#                     prompt_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in prompt_trainables.items()}
#                 model.load_state_dict(prompt_trainables, strict=False)
#                 print('Prompt tuning weights loaded successfully.')
#     else:
#         # Load language model
#         if model_base is not None:
#             # PEFT model
#             from peft import PeftModel, PeftMixedModel
#             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#             model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
#             print(f"Loading LoRA weights from {model_path}")
#             model = PeftMixedModel.from_pretrained(model, model_path)
            
#             # Load prompt tuning weights if they exist
#             prompt_trainables_path = os.path.join(model_path, 'prompt_trainables.bin')
#             if os.path.exists(prompt_trainables_path):
#                 print('Loading prompt tuning weights...')
#                 prompt_trainables = torch.load(prompt_trainables_path, map_location='cpu')
#                 prompt_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in prompt_trainables.items()}
#                 if any(k.startswith('model.model.') for k in prompt_trainables):
#                     prompt_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in prompt_trainables.items()}
#                 model.load_state_dict(prompt_trainables, strict=False)
#                 print('Prompt tuning weights loaded successfully.')
                
#             print(f"Merging weights")
#             model = model.merge_and_unload()
#             print('Convert to FP16...')
#             model.to(torch.float16)
#         else:
#             use_fast = False
#             if 'mpt' in model_name.lower():
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#                 model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
#             else:
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#                 model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
#             # Load prompt tuning weights if they exist
#             prompt_trainables_path = os.path.join(model_path, 'prompt_trainables.bin')
#             if os.path.exists(prompt_trainables_path):
#                 print('Loading prompt tuning weights...')
#                 prompt_trainables = torch.load(prompt_trainables_path, map_location='cpu')
#                 prompt_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in prompt_trainables.items()}
#                 if any(k.startswith('model.model.') for k in prompt_trainables):
#                     prompt_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in prompt_trainables.items()}
#                 model.load_state_dict(prompt_trainables, strict=False)
#                 print('Prompt tuning weights loaded successfully.')

#     image_processor = None

#     if 'llava' in model_name.lower():
#         mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
#         mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
#         if mm_use_im_patch_token:
#             tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
#         if mm_use_im_start_end:
#             tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
#         model.resize_token_embeddings(len(tokenizer))

#         vision_tower = model.get_vision_tower()
#         if not vision_tower.is_loaded:
#             vision_tower.load_model(device_map=device_map)
#         if device_map != 'auto':
#             vision_tower.to(device=device_map, dtype=torch.float16)
#         image_processor = vision_tower.image_processor

#     if hasattr(model.config, "max_sequence_length"):
#         context_len = model.config.max_sequence_length
#     else:
#         context_len = 2048

#     return tokenizer, model, image_processor, context_len


# ...existing code...

def load_pretrained_model_both(model_path, model_base, model_name,prompt_tuning_adding, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    # import ipdb;ipdb.set_trace()
    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            lora_path = os.path.join(model_path, "llava-lora")
            # lora_path = model_path
            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(lora_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(lora_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)
            from copy import deepcopy
            # withoutlora_model = deepcopy(model)
            original_parameters = {}
            for name, param in model.named_parameters():
                if param.data.is_meta:
                    print(f"Warning: {name} is a meta tensor, skipping.")
                    continue
                original_parameters[name] = deepcopy(param.data).to(device="cpu")
                
            from peft import PeftModel, PeftMixedModel, get_peft_model_state_dict, PromptTuningConfig
            # lora_path = os.path.join(lora_path, "lora")
            # import ipdb;ipdb.set_trace()
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, lora_path)
            
            print('Merging LoRA weights...')
            
            model = model.merge_and_unload()
            print('Model is loaded...')
            print(f"prompt_tuning_adding is {prompt_tuning_adding}")
            # 加载Prompt Tuning权重
            # ...existing code...

            # 比较withoutlora_model和model的参数，若全部参数完全相同则报错
            all_equal = True
            for name2, param2 in model.named_parameters():
                if not torch.equal(original_parameters[name2].to(model.device), param2.data):
                    all_equal = False
                    break
            if all_equal:
                raise RuntimeError("withoutlora_model和model的所有参数完全相同，可能未正确加载LoRA或Prompt Tuning权重。")
            else:
                print("withoutlora_model和model的参数已成功加载，且不相同。")
            # ...existing code...
            if prompt_tuning_adding:
                print(f"prompt_tuning_adding is {prompt_tuning_adding}, loading prompt tuning weights...")
                prompt_tuning_path = os.path.join(model_path, "llava-prompt_tuning")
                if os.path.exists(prompt_tuning_path):
                    print('Loading Prompt Tuning weights...')
                    model = PeftModel.from_pretrained(model, prompt_tuning_path)
                    print('Prompt Tuning weights loaded.')
                else:
                    # 兼容直接在model_path下的prompt tuning
                    try:
                        config_files = [f for f in os.listdir(model_path) if f.startswith("adapter_config") and "prompt" in f]
                        if config_files:
                            print('Loading Prompt Tuning weights (auto-detect)...')
                            model = PeftModel.from_pretrained(model, model_path)
                            print('Prompt Tuning weights loaded.')
                    except Exception as e:
                        print(f'No prompt tuning weights found or failed to load: {e}')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                from llava.model.language_model.llava_llama import LlavaConfig
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                # cfg_pretrained = AutoConfig.from_pretrained(model_path)
                cfg_pretrained = LlavaConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                
            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftMixedModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
# ...existing code...