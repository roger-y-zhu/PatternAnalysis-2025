"""Defines model wrapper, loading pretrained LLMs (T5)"""
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model

def load_model(model_name="t5-small", use_lora=True, device="cuda"):
    """Load T5 model with optional LoRA fine-tuning"""
    tokeniser = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    if use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        model.to(device)

    return model, tokeniser