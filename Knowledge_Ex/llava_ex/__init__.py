"""
LLaVA package init.

We intentionally do NOT import heavy model classes here to avoid ImportError
when class names differ across forks. Evaluation scripts import from
`llava.model.builder` instead.
"""
# from .model import LlavaLlamaForCausalLM
