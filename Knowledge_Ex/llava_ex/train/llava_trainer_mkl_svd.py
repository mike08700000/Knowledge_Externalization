import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
import random
from typing import List, Optional
import math
import time
import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from transformers import TrainerState
from transformers.integrations import hp_params
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import speed_metrics
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from accelerate import Accelerator
from accelerate.utils import release_memory
from packaging import version
import logging
from transformers.integrations.deepspeed import deepspeed_init
from transformers.utils import is_torch_tpu_available
from transformers.trainer_utils import TrainOutput
import deepspeed
from safetensors.torch import save_file
import datasets
import random
import torch
from collections import defaultdict
from torch.utils.data import Sampler, DataLoader
from typing import List, Optional
import torch.nn.functional as F


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param
def save_split_prompt_embeddings_init(weight, num_split, save_dir):
    # weight = sdict["module.prompt_encoder.default.embedding.weight"]
    total = weight.shape[0]
    split_size = math.ceil(total / num_split)
    for i in range(num_split):
        start = i * split_size
        end = min((i + 1) * split_size, total)
        split_weight = weight[start:end,:].clone()
        split_dict = {"prompt_embeddings": split_weight}
        save_file(split_dict, os.path.join(save_dir, f"init_memorytokens_part{i+1}.safetensors"))  # 用safetensors保存

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return



def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")
    return chunks

def get_length_grouped_indices(lengths, batch_size, world_size, knowledge_id_map, generator=None):
    """
    Group indices by length and knowledge_id, ensuring each batch has the same knowledge_id.
    
    Args:
        lengths: List of sample lengths.
        batch_size: Number of samples per batch per process.
        world_size: Number of processes in distributed training.
        knowledge_id_map: Dict mapping indices to knowledge_ids.
        generator: Random number generator for shuffling.
    
    Returns:
        List of indices where each batch is from the same knowledge_id.
    """
    # Group indices by knowledge_id
    knowledge_id_to_indices = defaultdict(list)
    for idx, kid in knowledge_id_map.items():
        knowledge_id_to_indices[kid].append(idx)
    
    # Create batches within each knowledge_id
    all_batches = []
    for indices in knowledge_id_to_indices.values():
        # Shuffle indices within this knowledge_id
        shuffled_indices = torch.randperm(len(indices), generator=generator).tolist()
        # Split into batches and sort each by length
        for i in range(0, len(shuffled_indices), batch_size):
            batch_indices = shuffled_indices[i:i + batch_size]
            sorted_batch = sorted(batch_indices, key=lambda idx: lengths[indices[idx]], reverse=True)
            batch = [indices[idx] for idx in sorted_batch]
            all_batches.append(batch)
    
    # Shuffle batches to mix knowledge_ids
    # random.shuffle(all_batches, random=generator)
    random.shuffle(all_batches)
    
    # Group into megabatches for each process
    megabatches = [all_batches[i:i + world_size] for i in range(0, len(all_batches), world_size)]
    
    # Flatten indices for the sampler
    return [i for megabatch in megabatches for batch in megabatch for i in batch]
def get_modality_length_grouped_indices(lengths, batch_size, world_size, knowledge_id_map, generator=None):
    """
    Group indices by knowledge_id and modality, ensuring each batch has the same knowledge_id and modality.
    """
    from collections import defaultdict
    import torch
    import random

    # 1. grouped by knowledge_id
    kid_to_indices = defaultdict(list)
    for idx, kid in knowledge_id_map.items():
        kid_to_indices[kid].append(idx)

    all_batches = []
    for kid, indices in kid_to_indices.items():
        # 2. grouped by modality
        mm_indices = [i for i in indices if lengths[i] > 0]
        lang_indices = [i for i in indices if lengths[i] < 0]

        # 3. shuffle
        if mm_indices:
            mm_indices_shuffled = [mm_indices[i] for i in torch.randperm(len(mm_indices), generator=generator)]
            for i in range(0, len(mm_indices_shuffled), batch_size):
                batch = mm_indices_shuffled[i:i + batch_size]
                if batch:
                    all_batches.append(batch)
        if lang_indices:
            lang_indices_shuffled = [lang_indices[i] for i in torch.randperm(len(lang_indices), generator=generator)]
            for i in range(0, len(lang_indices_shuffled), batch_size):
                batch = lang_indices_shuffled[i:i + batch_size]
                if batch:
                    all_batches.append(batch)
    megabatches = [all_batches[i:i + world_size] for i in range(0, len(all_batches), world_size)]
    # debug code
    # print()
    # print("megabatches", [batch for batch in megabatches])
    # print("megabatches", [i for megabatch in megabatches for batch in megabatch for i in batch])
    # 6. flatten
    return [i for megabatch in megabatches for batch in megabatch for i in batch]
class LengthGroupedSampler(Sampler):
    """
    Sampler that groups indices by length and knowledge_id while maintaining randomness.

    You
    """
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality
        self.knowledge_id_map = {}

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, self.knowledge_id_map, self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, self.knowledge_id_map, self.generator
            )
        return iter(indices)

    def add_knowledge_id_map(self, knowledge_id_map):
        """
        Add a mapping of indices to knowledge IDs.
        """
        self.knowledge_id_map = knowledge_id_map
        print(f"Knowledge ID map added in sampler with {len(self.knowledge_id_map)} entries.")

class KnowledgeIdDataLoader(DataLoader):
    """
    DataLoader that ensures batches correspond to the same knowledge_id.
    """
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.knowledge_id_map = {}
        for idx in range(len(dataset)):
            item = dataset[idx]
            self.knowledge_id_map[idx] = item['knowledge_id']
        print(f"KnowledgeIdDataLoader initialized with {len(self.knowledge_id_map)} items.")
        print(f"Knowledge IDs: {self.knowledge_id_map}")
        self.sampler.add_knowledge_id_map(self.knowledge_id_map)    
        
            
class KnowledgeGrad:
    def __init__(self, knowledge_length=2, Threshold=0.2, Mod='mean',K = 0.9):
        self.knowledge_length = knowledge_length
        self.knowledge_in_ourset = []
        self.last_knowledge_id = 0
        self.Knowledge_type_grad = {}
        self.Threshold = Threshold
        self.Mod = Mod
        self.cosine = {}
        self.cpu = torch.device('cpu')
        self.compute_device = torch.device('cuda')
        self.K = K
        
    def get_knowledge_length(self):
        """
        Retrieve the length of knowledge.
        Returns:
            int: The length of knowledge.
        """
        return self.knowledge_length
    def get_synthetic_grad(self, param_name, without_knowledge_id):
        """
        Retrieve the synthetic gradient for a specific parameter without a knowledge ID.
        Args:
            param_name (str): The name of the parameter.
            without_knowledge_id (int): The ID of the knowledge to exclude.
        Returns:
            List[torch.Tensor]: gradients for the specified parameter excluding the given knowledge ID.
        """
        if param_name in self.Knowledge_type_grad:
            grads = [
                grad.to(self.compute_device) for knowledge_id, grad in self.Knowledge_type_grad[param_name].items()
                if knowledge_id != without_knowledge_id
            ]
            if grads:
                return torch.stack(grads, dim=0).sum(dim=0)  # 矢量和
        return None
    def set_grad(self, param_name, knowledge_id, grad):
        """        
        Store the gradient for a specific grad with knowledge ID.
        Args:
            param_name (str): The name of the parameter.
            knowledge_id (int): The ID of the knowledge.
            grad (torch.Tensor): The gradient tensor to store.
        """

        if knowledge_id > self.knowledge_length:
            raise ValueError(f"knowledge_id {knowledge_id} exceeds knowledge_length {self.knowledge_length}")
        if not isinstance(grad, torch.Tensor):
            print(f"Expected {param_name} grad to be a torch.Tensor, got {type(grad)}")
            return
            raise ValueError(f"Expected grad to be a torch.Tensor, got {type(grad)}")
        if not knowledge_id in self.knowledge_in_ourset: 
            self.knowledge_in_ourset.append(knowledge_id)
        if param_name not in self.Knowledge_type_grad:
            self.Knowledge_type_grad[param_name] = {}
        self.Knowledge_type_grad[param_name][knowledge_id] = grad.to(self.cpu)
    def get_grad(self, param_name, knowledge_id):
        """
        Retrieve the stored gradient for a specific parameter and knowledge ID.
        Args:
            param_name (str): The name of the parameter.
            knowledge_id (int): The ID of the knowledge.
        Returns:
            List[torch.Tensor]: A list of gradients for the specified parameter and knowledge ID.
        """
        if param_name in self.Knowledge_type_grad and knowledge_id in self.Knowledge_type_grad[param_name]:
            return self.Knowledge_type_grad[param_name][knowledge_id]
        else:
            return None
        
    def cosine_similarity_with_rest(self, param_name, knowledge_id , grad_self):
        """
        Calculate cosine similarity between the gradient of a given knowledge ID
        and the vector sum of all other gradients.
        """
        if len(self.knowledge_in_ourset) == 1 and self.knowledge_in_ourset[0] == knowledge_id:
            return -1.0
            
        grad_rest_sum = self.get_synthetic_grad(param_name, without_knowledge_id=knowledge_id)
        if grad_self is None or grad_rest_sum is None:
            return None
        grad_self_flat = grad_self.to(self.compute_device).view(-1)
        grad_rest_flat = grad_rest_sum.view(-1)
        return F.cosine_similarity(grad_self_flat.unsqueeze(0), grad_rest_flat.unsqueeze(0)).item()
    def update_grad(self, param_name, knowledge_id, grad_self):
        """
        Update the gradient by multiplying it with a weight based on cosine similarity.
        The weight is calculated as e^(k*(cosine_similarity + 1)).
        Args:
            param_name (str): The name of the parameter.
            knowledge_id (int): The ID of the knowledge.
            grad_self (torch.Tensor): The gradient tensor to update.
        Returns:
            torch.Tensor: The weighted gradient, or None if cosine similarity cannot be computed.
        """
        self.set_grad(param_name, knowledge_id, grad_self)
        if len(self.knowledge_in_ourset) == 1 and knowledge_id in self.knowledge_in_ourset:
            return grad_self
        elif len(self.knowledge_in_ourset) == 0:
            return grad_self
        
        # cosine_sim = self.cosine_similarity_with_rest(param_name, knowledge_id, grad_self)
        if self.Mod == 'mean':
            cosine_sim = self.cosine_similarity_with_rest(param_name, knowledge_id , grad_self)
        else:
            cosine_sim = self.max_cosine_with_rest(param_name, knowledge_id, grad_self)
        if cosine_sim is None:
            return grad_self
        # Calculate weight as e^(k*(cosine_similarity + 1)), where k is a scaling factor (default k=1)
        weight = math.exp(self.K * (cosine_sim + 1))
        # Move gradient to compute device for calculation
        weighted_grad = grad_self.to(self.compute_device) * weight
        return weighted_grad
    def update_param(self, param_name, knowledge_id , grad_self):
        """_summary_

        Args:
            param_name (_type_): _description_
            knowledge_id (_type_): _description_
            grad_self (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.set_grad(param_name, knowledge_id, grad_self)
        if len(self.knowledge_in_ourset) == 1 and knowledge_id in self.knowledge_in_ourset:
            return True
        elif len(self.knowledge_in_ourset) == 0:
            return False
        if self.Mod == 'mean':
            cosine_sim = self.cosine_similarity_with_rest(param_name, knowledge_id , grad_self)
        else:
            cosine_sim = self.max_cosine_with_rest(param_name, knowledge_id, grad_self)
        self.add_cosine(param_name,cosine_sim,knowledge_id)
        if cosine_sim < self.Threshold:
            return True
        else:
            return False
    def add_cosine(self,param_name,cosine,knowledge_id):
        if param_name not in self.cosine:
            self.cosine[param_name] = {}
        if knowledge_id not in self.cosine[param_name]:
            self.cosine[param_name][knowledge_id] = []
        self.cosine[param_name][knowledge_id].append(cosine)
    def print_cosine(self):
        print(self.cosine)
    def get_cosine(self):
        return self.cosine
    def max_cosine_with_rest(self, param_name, knowledge_id,grad_self):
        """_summary_

        Args:
            param_name (_type_): _description_
            knowledge_id (_type_): _description_
            grad_self (_type_): _description_

        Returns:
            _type_: _description_
        """
        if param_name not in self.Knowledge_type_grad:
            return None
        grads_dict = self.Knowledge_type_grad[param_name]
        if knowledge_id not in grads_dict:
            return None
        grad_self = grad_self.to(self.compute_device).view(1, -1)  # shape: (1, D)
        other_grads = [
            grad.to(self.compute_device).view(-1) for k, grad in grads_dict.items() if k != knowledge_id
        ]
        if not other_grads:
            return None
        other_grads = torch.stack(other_grads, dim=0)  # shape: (N-1, D)
        cos_sims = F.cosine_similarity(grad_self, other_grads)  # shape: (N-1,)
        return cos_sims.abs().max().item()

class LLaVATrainer(Trainer):
    def __init__(self,num_virtual_tokens, save_dir,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_tokens = None
        self.unlearn_weight = 1.0
        self.kl_weight = 0.5
        self.threshold = 0.05
        self.max_grad_norm = 0.05
        self.num_virtual_tokens = num_virtual_tokens
        self.task_length = None
        self.memory_tokens_init = None
        self.save_dir = save_dir
        print(f"save_dir: {self.save_dir}")

        
        train_args = kwargs.get('args', None)
        if train_args is not None:
            self.lora_enable = getattr(train_args, 'lora_enable', True)
            self.prompt_tuning_enable = getattr(train_args, 'prompt_tuning_enable', True)
            self.lora_bias = getattr(train_args, 'lora_bias', "none")
            print(f"lora_enable: {self.lora_enable}, prompt_tuning_enable: {self.prompt_tuning_enable}, lora_bias: {self.lora_bias}")
        else:
            self.lora_enable = True
            self.prompt_tuning_enable = True
            self.lora_bias = "none"
        self.state_dicts = {}
        self.knowledge_grad = KnowledgeGrad(knowledge_length=self.args.knowledge_length,Threshold=self.args.Simlarity_Threshold)
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()
    
    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        ####
        # print(inputs)
        
        
        inputs = self._prepare_inputs(inputs)
        task_ids = inputs['knowledge_id'][0]
        task_length = inputs['task_length'][0]
        print(f"task_length: {inputs['task_length']}, knowledge_id: {inputs['knowledge_id']}")
        task_mem_token_lenth = self.num_virtual_tokens / task_length
        if self.task_length is None:
            self.task_length = task_length
        # print(f"task_mem_token_lenth: {task_mem_token_lenth}")
        mem_index = int(task_ids * task_mem_token_lenth)
        # del inputs['knowledge_id']
        del inputs['mask']
        # with torch.no_grad():
        #     outputs = model(**inputs)
        ####
        outputs = None
        model.train()
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        # inputs["block_memory"] = True
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        for name, param in model.named_parameters():
            if "prompt_encoder" in name:
                MemToken_grad = deepspeed.utils.safe_get_full_grad(param)
                # print(f"MemToken_grad_before: {MemToken_grad.shape}, {MemToken_grad}")
                zero_mask = torch.zeros_like(MemToken_grad)
                start = int((task_ids-1) * task_mem_token_lenth)
                end = int(task_ids * task_mem_token_lenth)
                zero_mask[start:end, :] = 1
                MemToken_grad = MemToken_grad * zero_mask
                # MemToken_grad[int((task_ids-1) * task_mem_token_lenth):int(task_ids * task_mem_token_lenth), :] = 0
                # print(int((task_ids-1) * task_mem_token_lenth),int(task_ids * task_mem_token_lenth))
                deepspeed.utils.safe_set_full_grad(param,MemToken_grad)
            elif "lora" in name:
                grad = deepspeed.utils.safe_get_full_grad(param)
                # grad = self.knowledge_grad.update_grad(name, int(task_ids), grad)
                deepspeed.utils.safe_set_full_grad(param,grad)
                # if not self.knowledge_grad.update_param(name, int(task_ids), grad):
                #     print(name, f"knowledge_id {task_ids} is not in the knowledge set, skip updating")
                #     deepspeed.utils.safe_set_full_grad(param,torch.zeros_like(grad))
                    
        return loss.detach() / self.args.gradient_accumulation_steps , outputs
    
    def training_step_unlearning(self, model, inputs, outputs):
        self.phrase="Donald Trump"
        self.model.train()
        device = inputs["input_ids"].device
        if "mask" in inputs:
            del inputs['mask']
        inputs = self._prepare_inputs(inputs)
        
        tokens = self.tokenizer.tokenize(self.phrase)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        mask_phr = torch.ones([32000], dtype=torch.float32).to(device)
        for id in token_ids:
            mask_phr[id]=0
        
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach()
        
        inputs["block_memory"] = True
        knowledge_ids = inputs['knowledge_id'][0]
        task_ids = inputs['knowledge_id'][0]
        del inputs['knowledge_id']
        del inputs['task_length']
        with self.compute_loss_context_manager():
            loss, unlearn_outputs = self.compute_loss(model, inputs,return_outputs=True)
        print(f"second stage loss without prune: {loss}")
        # mask-kl loss
        # mask_unsqueezed = mask.unsqueeze(-1)  # 增加一个新维度，形状变为 [16, 92, 1]
        # mask = mask_unsqueezed.expand(-1, -1, 32000)
        # prob_p = F.softmax(outputs.logits, dim=-1)*mask_phr
        # prob_q = F.softmax(unlearn_outputs.logits, dim=-1)*mask_phr
        # prob_p=prob_p[:,:mask.size(1),:]
        # prob_q=prob_q[:,:mask.size(1),:]
        # if mask is not None:
        #     mask_inverse = 1 - mask
        #     kl_loss = -((prob_p * torch.log(prob_q + 1e-12)) * mask_inverse).sum() / mask_inverse.sum()
        
        # loss = -loss + kl_loss
        # kl_loss_clamped = torch.clamp(kl_loss, min=0.0, max=1.0) * self.kl_weight
        # loss_main = torch.clamp(-loss, min=-2.5, max=0.0)* self.unlearn_weight
        # loss = loss_main + kl_loss_clamped
        
        # kl loss for Maintain Performance
        # You 2025-06-05
        ###
        # kl_loss = F.kl_div(F.softmax(unlearn_outputs.logits, dim=-1).log(),F.softmax(outputs.logits, dim=-1),reduction='batchmean')
        # kl_loss_clamped = torch.clamp(kl_loss * self.kl_weight, min=0.0, max=1.0)
        # loss_main = torch.clamp(-loss * self.unlearn_weight, min=-2.5, max=0.0)
        # loss = loss_main + kl_loss_clamped
        ###

        # loss = -loss
        loss = torch.clamp(-loss, min=-2.5, max=1.0)
        

        
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        print(f"second stage loss: {loss}")
        # debuging code:
        # if int(knowledge_ids) == 2:
        #     print("set grad to zero for all")
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             deepspeed.utils.safe_set_full_grad(param,torch.zeros_like(deepspeed.utils.safe_get_full_grad(param)).to(param.device))
        #             # print(f"set {name} grad to zero")
        #             # print(f"param.grad after: {deepspeed.utils.safe_get_full_grad(param)}")
        # else:
        for name, param in model.named_parameters():
            if "lora" in name:
                grad = deepspeed.utils.safe_get_full_grad(param)
                grad = self.knowledge_grad.update_grad(name, int(task_ids), grad)
                deepspeed.utils.safe_set_full_grad(param,grad)
            if "prompt_encoder" in name:
                deepspeed.utils.safe_set_full_grad(param,torch.zeros_like(deepspeed.utils.safe_get_full_grad(param)).to(grad.device))

        return loss.detach() / self.args.gradient_accumulation_steps
    
    
    def training_step_withmem_unlearning(self, model, inputs, outputs,lr_with_mem):
        self.model.train()
        task_ids = inputs['knowledge_id'][0]
        task_length = inputs['task_length'][0]
        task_mem_token_lenth = self.num_virtual_tokens / task_length
        if self.task_length is None:
            self.task_length = task_length
        mem_index = int(task_ids * task_mem_token_lenth)
        del inputs['mask']
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach()
        inputs["block_memory"] = True
        # del inputs['knowledge_id']
        # del inputs['task_length']
        if lr_with_mem <0:
            print("lr_with_mem is less than 0, skip memory tokens")
            del inputs['knowledge_id']
            del inputs['task_length']
        with self.compute_loss_context_manager():
            loss, unlearn_outputs = self.compute_loss(model, inputs,return_outputs=True)
        print(f"third stage loss without prune: {loss}")
        loss = torch.clamp(-loss, min=-2.5, max=1.0)      
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        print(f"third stage loss: {loss}")
        # debuging code:
        
        for name, param in model.named_parameters():
            if "prompt_encoder" in name:
                # if lr_with_mem > 0:
                print("third_stage_grad",deepspeed.utils.safe_get_full_grad(param))
            elif "lora" in name:
                if lr_with_mem <0:
                    grad = deepspeed.utils.safe_get_full_grad(param)
                    deepspeed.utils.safe_set_full_grad(param,grad*1.1)
                # else:
                    
                # MemToken_grad = deepspeed.utils.safe_get_full_fp32_param(param)
                # MemToken_grad[int((task_ids-1) * task_mem_token_lenth):int(task_ids * task_mem_token_lenth), :] = 0
                # # print(f"MemToken_grad: {MemToken_grad.shape}, {MemToken_grad}")
                # deepspeed.utils.safe_set_full_grad(param,MemToken_grad)

        return loss.detach() / self.args.gradient_accumulation_steps
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        # import ipdb;ipdb.set_trace()
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled
        
        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)
        
        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        # ...existing code...

        # if use_accelerator_prepare:
        #     # 保存 prompt_encoder 的参数
        #     prompt_encoder_state = None
        #     for name, module in self.model.named_modules():
        #         if "prompt_encoder" in name:
        #             prompt_encoder_state = {k: v.clone().detach() for k, v in module.state_dict().items()}
        #             prompt_encoder_module_name = name
        #             break
        #     self.model.train()
        #     if hasattr(self.lr_scheduler, "step"):
        #         if self.use_apex:
        #             model = self.accelerator.prepare(self.model)
        #         else:
        #             model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        #             # for name, param in model.named_parameters():
        #             #     if "prompt_encoder" in name:
        #             #         print(f"if self.is_fsdp_enabled:):, {name}, {param}, param.grad{param.grad}")
        #             #         print(fuck)
        #     else:
        #         # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
        #         model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
        #             self.model, self.optimizer, self.lr_scheduler
        #         )

            # 将 prompt_encoder 的参数赋值回 model
            # if prompt_encoder_state is not None:
            #     # 获取 model 中的 prompt_encoder 模块
            #     module = model
            #     for attr in prompt_encoder_module_name.split('.'):
            #         module = getattr(module, attr)
            #     module.load_state_dict(prompt_encoder_state)
        
        
        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)
        
        total_batched_samples = 0
        if self.prompt_tuning_enable:
            import copy
            
            print("start to save memory tokens")
            self.memory_tokens_init = copy.deepcopy(model.get_prompt_embedding_to_save(adapter_name="default"))
            print(f"memory tokens shape is {self.memory_tokens_init.shape}")
            print(f"memory tokens  is {self.memory_tokens_init}")
            for i in range(8):
                print()
            print("finish getting memory tokens")
            save_split_prompt_embeddings_init(self.memory_tokens_init, 2, self.save_dir)
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)
            
            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            print("epoch_iterator",epoch_iterator)
            lr_with_mem = 1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
    
                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                outputs = None
                # if int(inputs['knowledge_id'][0]) == 1: 
                with self.accelerator.accumulate(model):
                    # for name, param in model.named_parameters():
                    #     if "prompt_encoder" in name or "lora" in name:
                    #         print(f" {name}, {param}, param.grad{param.grad}")
                    # print("fuck")
                    tr_loss_step,outputs = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )
                    print(f"first_stage loss is {tr_loss}")
                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    # ##
                ###################################
                # with self.accelerator.accumulate(model):
                #     tr_loss_step = self.training_step_withmem_unlearning(model, inputs,outputs,lr_with_mem)

                # if (
                #     args.logging_nan_inf_filter
                #     and not is_torch_tpu_available()
                #     and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                # ):
                #     # if loss is nan or inf simply add the average of previous logged losses
                #     tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                # else:
                #     tr_loss += tr_loss_step
                
                # self.current_flos += float(self.floating_point_ops(inputs))

                # is_last_step_and_steps_less_than_grad_acc = (
                #     steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                # )

                # if (
                #     total_batched_samples % args.gradient_accumulation_steps == 0
                #     or
                #     # last step in epoch but step is always smaller than gradient_accumulation_steps
                #     is_last_step_and_steps_less_than_grad_acc
                # ):
                #     # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                #     # in accelerate. So, explicitly enable sync gradients to True in that case.
                #     if is_last_step_and_steps_less_than_grad_acc:
                #         self.accelerator.gradient_state._set_sync_gradients(True)

                #     # Gradient clipping
                #     if args.max_grad_norm is not None and args.max_grad_norm > 0:
                #         # deepspeed does its own clipping

                #         if is_sagemaker_mp_enabled() and args.fp16:
                #             self.optimizer.clip_master_grads(args.max_grad_norm)
                #         elif self.use_apex:
                #             # Revert to normal clipping otherwise, handling Apex or full precision
                #             nn.utils.clip_grad_norm_(
                #                 amp.master_params(self.optimizer),
                #                 args.max_grad_norm,
                #             )
                #         else:
                #             self.accelerator.clip_grad_norm_(
                #                 model.parameters(),
                #                 args.max_grad_norm,
                #             )

                #     # Optimizer step
                #     self.optimizer.step()
                #     optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                ###########
                
                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step_unlearning(model, inputs,outputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step
                
                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                ############
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                lr_with_mem = -lr_with_mem
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
                
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break
        ###
        if self.prompt_tuning_enable:
            print("start to save memory tokens")
            self.memory_tokens = model.get_prompt_embedding_to_save(adapter_name="default")
            print(f"memory tokens shape is {self.memory_tokens.shape}")
            print("finish getting memory tokens")
        print("start to save parmaeters")
        from llava.train.save_parm import get_peft_state_maybe_zero_3, get_prompt_tuning_state_maybe_zero_3, get_peft_state_non_lora_prompt_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, get_peft_state_non_prompt_maybe_zero_3
        if self.lora_enable:
            self.state_dicts['lora'] = get_peft_state_maybe_zero_3(
                model.named_parameters(), self.lora_bias
            )
            
        if self.prompt_tuning_enable:
            self.state_dicts['prompt'] = get_prompt_tuning_state_maybe_zero_3(
                model.named_parameters()
            )
            
        if self.prompt_tuning_enable and self.lora_enable:
            self.state_dicts['non_lora_prompt'] = get_peft_state_non_lora_prompt_maybe_zero_3(
                model.named_parameters()
            )

        if self.lora_enable:
            self.state_dicts['non_lora'] = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )

        if self.prompt_tuning_enable:
            self.state_dicts['non_prompt'] = get_peft_state_non_prompt_maybe_zero_3(
                model.named_parameters()
            )
        ###
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        # return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        
        return self.accelerator.prepare(KnowledgeIdDataLoader(train_dataset, **dataloader_params))
    def training_step_withmem_unlearning(self, model, inputs, lr_with_mem, Bgrad=1, Cgrad=1):
        self.model.train()
        task_ids = inputs['knowledge_id'][0]
        task_length = inputs['task_length'][0]
        task_mem_token_lenth = self.num_virtual_tokens / task_length
        if self.task_length is None:
            self.task_length = task_length
        mem_index = int(task_ids * task_mem_token_lenth)
        del inputs['mask']
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach()
        inputs["block_memory"] = True
        # del inputs['knowledge_id']
        # del inputs['task_length']
        if lr_with_mem <0:
            #小于0时，不区分知识，不加载memorytoken，对所有知识梯度上升遗忘；大于0时，加载其他知识memorytoken，做知识遗忘
            print("lr_with_mem is less than 0, skip memory tokens")
            del inputs['knowledge_id']
            del inputs['task_length']
        with self.compute_loss_context_manager():
            loss, unlearn_outputs = self.compute_loss(model, inputs,return_outputs=True) 
        print(f"third stage loss without prune: {loss}")
        loss = torch.clamp(-loss, min=-2.5, max=1.0)      
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        print(f"third stage loss: {loss}")
        # debuging code:
        
        for name, param in model.named_parameters():
            if "prompt_encoder" in name:
                MemToken_grad = deepspeed.utils.safe_get_full_grad(param)
                print("third_stage_grad", MemToken_grad)
                if lr_with_mem > 0:
                    MemToken_grad[int((task_ids-1) * task_mem_token_lenth):int(task_ids * task_mem_token_lenth), :] = 0
                    print(f"MemToken_grad: {MemToken_grad.shape}, {MemToken_grad}")
                    deepspeed.utils.safe_set_full_grad(param, MemToken_grad)
            elif "lora" in name:
                if lr_with_mem < 0:
                    grad = deepspeed.utils.safe_get_full_grad(param)
                    deepspeed.utils.safe_set_full_grad(param, grad*Cgrad)
                elif lr_with_mem > 0:
                    grad = deepspeed.utils.safe_get_full_grad(param)
                    deepspeed.utils.safe_set_full_grad(param, grad*Bgrad)
                

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled
        
        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)
        
        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        # ...existing code...


        
        
        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)
        
        total_batched_samples = 0
        if self.prompt_tuning_enable:
            import copy
            
            print("start to save memory tokens")
            self.memory_tokens_init = copy.deepcopy(model.get_prompt_embedding_to_save(adapter_name="default"))
            print(f"memory tokens shape is {self.memory_tokens_init.shape}")
            print(f"memory tokens  is {self.memory_tokens_init}")
            for i in range(8):
                print()
            print("finish getting memory tokens")
            save_split_prompt_embeddings_init(self.memory_tokens_init, 2, self.save_dir)
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)
            
            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            print("epoch_iterator",epoch_iterator)
            lr_with_mem = 1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
    
                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                outputs = None
                # if int(inputs['knowledge_id'][0]) == 1: 
                with self.accelerator.accumulate(model):
                    # for name, param in model.named_parameters():
                    #     if "prompt_encoder" in name or "lora" in name:
                    #         print(f" {name}, {param}, param.grad{param.grad}")
                    tr_loss_step, outputs = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )
                    print(f"first_stage loss is {tr_loss}")
                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    # ##
                ###################################
                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step_withmem_unlearning(model, inputs, lr_with_mem, Bgrad=self.Bgrad, Cgrad=self.Cgrad)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step
                
                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                ############
                
                # with self.accelerator.accumulate(model):
                #     tr_loss_step = self.training_step_unlearning(model, inputs,outputs)

                # if (
                #     args.logging_nan_inf_filter
                #     and not is_torch_tpu_available()
                #     and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                # ):
                #     # if loss is nan or inf simply add the average of previous logged losses
                #     tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                # else:
                #     tr_loss += tr_loss_step
                
                # self.current_flos += float(self.floating_point_ops(inputs))

                # is_last_step_and_steps_less_than_grad_acc = (
                #     steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                # )

                # if (
                #     total_batched_samples % args.gradient_accumulation_steps == 0
                #     or
                #     # last step in epoch but step is always smaller than gradient_accumulation_steps
                #     is_last_step_and_steps_less_than_grad_acc
                # ):
                #     # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                #     # in accelerate. So, explicitly enable sync gradients to True in that case.
                #     if is_last_step_and_steps_less_than_grad_acc:
                #         self.accelerator.gradient_state._set_sync_gradients(True)

                #     # Gradient clipping
                #     if args.max_grad_norm is not None and args.max_grad_norm > 0:
                #         # deepspeed does its own clipping

                #         if is_sagemaker_mp_enabled() and args.fp16:
                #             self.optimizer.clip_master_grads(args.max_grad_norm)
                #         elif self.use_apex:
                #             # Revert to normal clipping otherwise, handling Apex or full precision
                #             nn.utils.clip_grad_norm_(
                #                 amp.master_params(self.optimizer),
                #                 args.max_grad_norm,
                #             )
                #         else:
                #             self.accelerator.clip_grad_norm_(
                #                 model.parameters(),
                #                 args.max_grad_norm,
                #             )

                #     # Optimizer step
                #     self.optimizer.step()
                #     optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                ############
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                lr_with_mem = -lr_with_mem #lr_with_mem变正负，转为另一种梯度上升训练模式
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
                
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break
        ###
        if self.prompt_tuning_enable:
            print("start to save memory tokens")
            self.memory_tokens = model.get_prompt_embedding_to_save(adapter_name="default")
            print(f"memory tokens shape is {self.memory_tokens.shape}")
            print("finish getting memory tokens")
        print("start to save parmaeters")
        from llava.train.save_parm import get_peft_state_maybe_zero_3, get_prompt_tuning_state_maybe_zero_3, get_peft_state_non_lora_prompt_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, get_peft_state_non_prompt_maybe_zero_3
        if self.lora_enable:
            self.state_dicts['lora'] = get_peft_state_maybe_zero_3(
                model.named_parameters(), self.lora_bias
            )
            
        if self.prompt_tuning_enable:
            self.state_dicts['prompt'] = get_prompt_tuning_state_maybe_zero_3(
                model.named_parameters()
            )
            
        if self.prompt_tuning_enable and self.lora_enable:
            self.state_dicts['non_lora_prompt'] = get_peft_state_non_lora_prompt_maybe_zero_3(
                model.named_parameters()
            )

        if self.lora_enable:
            self.state_dicts['non_lora'] = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )

        if self.prompt_tuning_enable:
            self.state_dicts['non_prompt'] = get_peft_state_non_prompt_maybe_zero_3(
                model.named_parameters()
            )
        ###
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        # return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        
        return self.accelerator.prepare(KnowledgeIdDataLoader(train_dataset, **dataloader_params))