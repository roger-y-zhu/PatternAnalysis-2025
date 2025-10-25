"""Model + dataset helper functions"""
from pathlib import Path

import torch
from datasets import load_from_disk, DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import T5Tokenizer, T5ForConditionalGeneration

# local dataset utilities
from dataset import load_clean_data, preprocess_dataset, load_raw_datasets

TOKENISED_CACHE = Path("data/tokenised")


def load_model(model_name="t5-small-local", use_lora=False, device=None):
    """Load tokeniser and model. """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    if use_lora:
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora_cfg)
        model.config.use_cache = False

    model.gradient_checkpointing_enable()
    model.to(device)
    return model, tokenizer


def get_tokenised_datasets(tokenizer, cache_dir=str(TOKENISED_CACHE), max_input_length=512, max_output_length=256):
    """Return tokenised HF Datasets for train/val/test. Cache to disk."""
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        print("Loading tokenised datasets from disk...")
        data = load_from_disk(str(cache_dir))
    else:
        print("Tokenising cleaned datasets...")
        hf_ds = load_clean_data(as_hf_dataset=True)
        # load_clean_data returns single HF Dataset; we saved splits to disk earlier, so load them from parquet:
        # We will create DatasetDict from parquet splits
        from datasets import Dataset
        import pandas as pd
        train_df = pd.read_parquet("data/train_clean.parquet")
        val_df = pd.read_parquet("data/val_clean.parquet")
        test_df = pd.read_parquet("data/test_clean.parquet")
        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)
        test_ds = Dataset.from_pandas(test_df)

        train_ds = preprocess_dataset(train_ds, tokenizer, max_input_length, max_output_length)
        val_ds = preprocess_dataset(val_ds, tokenizer, max_input_length, max_output_length)
        test_ds = preprocess_dataset(test_ds, tokenizer, max_input_length, max_output_length)

        data = DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds})
        data.save_to_disk(str(cache_dir))
        print(f"Saved tokenised datasets to {cache_dir}")

    # set torch format
    for split in data:
        data[split] = data[split].with_format("torch")
    return data["train"], data["val"], data["test"]


def load_raw_datasets_wrapper():
    """Return raw train/val/test pandas DataFrames (for eval/predict)."""
    return load_raw_datasets()
