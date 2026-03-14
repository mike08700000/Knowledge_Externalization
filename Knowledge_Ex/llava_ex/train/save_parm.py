import logging
import torch


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
# def get_peft_state_maybe_zero_3(named_params, bias):
#     if bias == "none":
#         to_return = {k: t for k, t in named_params if "lora_" in k}
#     elif bias == "all":
#         to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
#     elif bias == "lora_only":
#         to_return = {}
#         maybe_lora_bias = {}
#         lora_bias_names = set()
#         for k, t in named_params:
#             if "lora_" in k:
#                 to_return[k] = t
#                 bias_name = k.split("lora_")[0] + "bias"
#                 lora_bias_names.add(bias_name)
#             elif "bias" in k:
#                 maybe_lora_bias[k] = t
#         for k, t in maybe_lora_bias:
#             if bias_name in lora_bias_names:
#                 to_return[bias_name] = t
#     else:
#         raise NotImplementedError
#     to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
#     return to_return


# def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
#     to_return = {k: t for k, t in named_params if "lora_" not in k}
#     if require_grad_only:
#         to_return = {k: t for k, t in to_return.items() if t.requires_grad}
#     to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
#     return to_return

def get_prompt_tuning_state_maybe_zero_3(named_params):
    """获取Prompt Tuning参数，处理Zero-3的情况"""
    import deepspeed
    # print("get_prompt_tuning_state_maybe_zero_3")
    # print("named_params", named_params)
    prompt_params = {k: t for k, t in named_params if "prompt_encoder" in k}
    # for key,item in prompt_params:
    #     parm = deepspeed.utils.safe_get_full_fp32_param(item)
    #     prompt_params[key] = parm
    A = {k: maybe_zero_3(v, ignore_status=True) for k, v in prompt_params.items()}
    # print("prompt_params", A)
    return A

# 新
def get_peft_state_maybe_zero_3(named_params, bias):
    """
    获取LoRA参数，排除Prompt Tuning参数
    """
    to_return = {}
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k and "prompt_encoder" not in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params 
                    if ("lora_" in k or "bias" in k) and "prompt_encoder" not in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k and "prompt_encoder" not in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k and "prompt_encoder" not in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

# 新:非lora非prompt参数保存
def get_peft_state_non_lora_prompt_maybe_zero_3(named_params, require_grad_only=True):
    """
    获取非LoRA和非Prompt Tuning参数
    """
    to_return = {k: t for k, t in named_params 
                if "lora_" not in k and "prompt_encoder" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

# 新:非lora参数保存
def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """
    获取非LoRA参数（但可能包含Prompt Tuning参数）
    适用场景：需要排除LoRA但保留Prompt Tuning和其他可训练参数
    """
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

# 新:非Prompt参数保存
def get_peft_state_non_prompt_maybe_zero_3(named_params, require_grad_only=True):
    """
    获取非Prompt Tuning参数（但可能包含LoRA参数）
    适用场景：需要排除Prompt Tuning但保留LoRA和其他可训练参数
    """
    to_return = {k: t for k, t in named_params if "prompt_encoder" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
