"""Defines model wrapper, loading pretrained LLMs (T5)"""
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model
import os

from datasets import load_from_disk
from dataset import load_datasets, preprocess_dataset
from datasets import DatasetDict

def load_model(model_name="t5-small", use_lora=True, device="cuda"):
    tokeniser = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    if use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False  # important for gradient checkpointing

    model.gradient_checkpointing_enable()  # enable after LoRA
    if torch.cuda.is_available():
        model.to(device)

    return model, tokeniser

# Path to cached tokenised dataset
CACHE_DIR = "data/tokenised"

def get_tokenised_datasets(tokeniser):
    if os.path.exists(CACHE_DIR):
        print("Loading tokenised datasets from disk...")
        dataset_dict = load_from_disk(CACHE_DIR)
    else:
        print("Precomputing tokenised datasets...")
        train_ds, val_ds, test_ds = load_datasets()

        train_ds = preprocess_dataset(train_ds, tokeniser)
        val_ds = preprocess_dataset(val_ds, tokeniser)
        test_ds = preprocess_dataset(test_ds, tokeniser)

        dataset_dict = DatasetDict({
            "train": train_ds,
            "val": val_ds,
            "test": test_ds
        })
        dataset_dict.save_to_disk(CACHE_DIR)
        print(f"Saved tokenised datasets to {CACHE_DIR}")

    # Convert to PyTorch tensors for faster DataLoader iteration
    for split in dataset_dict:
        dataset_dict[split] = dataset_dict[split].with_format("torch")

    return dataset_dict["train"], dataset_dict["val"], dataset_dict["test"]
