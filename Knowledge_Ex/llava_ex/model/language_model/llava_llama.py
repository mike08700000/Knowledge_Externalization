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


from typing import List, Optional, Tuple, Union

from peft.peft_model import PeftModel
import torch
import torch.nn as nn
import pdb

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        # cache_position=None,  # 新增，用于兼容 transformers 新接口
        # **kwargs,             # 再兜底接受其它潜在新参数
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # pdb.set_trace()
        # print(attention_mask.shape if attention_mask is not None else None)
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        # print(attention_mask.shape if attention_mask is not None else None)
        # print(attention_mask)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # cache_position=None,  # 新增，用于兼容 transformers 新接口
            # **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        # print(f"Generating with inputs: {inputs.shape if inputs is not None else None}, images: {images.shape if images is not None else None}, image_sizes: {image_sizes if image_sizes is not None else None}")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        # print(f"inputs_embeds shape: {inputs_embeds.shape}, inputs shape: {inputs.shape if inputs is not None else None}")
        # print(f"position_ids shape: {position_ids.shape if position_ids is not None else None}, attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
        # # print(f"inputs shape: {inputs.shape if inputs is not None else None}")
        # print(f"attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
        # print(f"position_ids: {position_ids}")
        # print(f"attention_mask: {attention_mask}")
        # import ipdb;ipdb.set_trace()
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    @torch.no_grad()
    def generate_with_LP(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        memorytokens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        # print(f"Generating with inputs: {inputs.shape if inputs is not None else None}, images: {images.shape if images is not None else None}, image_sizes: {image_sizes if image_sizes is not None else None}")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        # print(f"inputs_embeds shape: {inputs_embeds.shape}, inputs shape: {inputs.shape if inputs is not None else None}")
        # print(f"position_ids shape: {position_ids.shape if position_ids is not None else None}, attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
        # print(f"inputs shape: {inputs.shape if inputs is not None else None}")
        # print(f"attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
        # print(f"position_ids: {position_ids}")
        # print(f"attention_mask: {attention_mask}")
        # import ipdb;ipdb.set_trace()
        prompts = memorytokens.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
        # print(f"inputs_embeds after concat shape: {inputs_embeds.shape}, inputs shape: {inputs.shape if inputs is not None else None}")
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     images: Optional[torch.FloatTensor] = None,
    #     image_sizes: Optional[List[List[int]]] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, CausalLMOutputWithPast]:

    #     if inputs_embeds is None:
    #         (
    #             input_ids,
    #             _,
    #             attention_mask,
    #             past_key_values,
    #             inputs_embeds,
    #             labels
    #         ) = self.prepare_inputs_labels_for_multimodal(
    #             input_ids,
    #             None,  # position_ids ignored
    #             None,  # attention_mask ignored
    #             past_key_values,
    #             labels,
    #             images,
    #             image_sizes
    #         )

    #     # Determine whether to pass attention_mask
    #     use_mask = True
    #     if hasattr(self, "peft_config") and isinstance(getattr(self, "base_model", None), PeftModel):
    #         peft_conf = self.peft_config[self.active_adapter]
    #         if getattr(peft_conf, "is_prompt_learning", False):
    #             use_mask = False

    #     return super().forward(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask if use_mask else None,
    #         position_ids=None,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         labels=labels,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )


    # @torch.no_grad()
    # def generate(
    #     self,
    #     inputs: Optional[torch.Tensor] = None,
    #     images: Optional[torch.Tensor] = None,
    #     image_sizes: Optional[torch.Tensor] = None,
    #     **kwargs,
    # ) -> Union[GenerateOutput, torch.LongTensor]:
    #     """
    #     Generate tokens using either base model, LoRA, or Prompt Tuning.
    #     Automatically uses input_ids; avoids inputs_embeds to preserve PEFT behavior.
    #     """
    #     # ✅ 不要传 attention_mask，避免与 prompt tuning 虚拟 token 冲突
    #     kwargs.pop("attention_mask", None)
    #     kwargs.pop("position_ids", None)  # 同样，position_ids PEFT 也不支持

    #     attention_mask = None
    #     if images is not None:
    #         # pdb.set_trace()
    #         # 如果有图像输入，使用 multimodal 拼接处理 input_ids
    #         (
    #             inputs,
    #             _,
    #             attention_mask,
    #             _,
    #             _,
    #             _
    #         ) = self.prepare_inputs_labels_for_multimodal(
    #             inputs,
    #             None,
    #             None,
    #             None,  # past_key_values
    #             None,  # labels
    #             images,
    #             image_sizes=image_sizes
    #         )

    #     # print("inputs.shape",inputs.shape)
    #     # Determine whether to pass attention_mask
    #     use_mask = True
    #     if hasattr(self, "peft_config") and isinstance(getattr(self, "base_model", None), PeftModel):
    #         peft_conf = self.peft_config[self.active_adapter]
    #         if getattr(peft_conf, "is_prompt_learning", False):
    #             use_mask = False
    #     # 不要手动传入 inputs_embeds，保留 input_ids 给 PEFT 管理
    #     import ipdb; ipdb.set_trace()
    #     return super().generate(
    #         input_ids=inputs,
    #         # attention_mask=attention_mask if use_mask else None,
    #         **kwargs
        # )
    #     # return super().generate(
    #     #     input_ids=inputs,
    #     #     attention_mask=attention_mask,
    #     #     **kwargs
    #     # )


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
