"""Handles loading, preprocessing, and train/val/test splitting of BioLaySumm"""
import numpy as np
import pandas as pd
from datasets import Dataset

def load_datasets(train_path="data/train.parquet", val_path="data/validation.parquet", test_path="data/test.parquet"):
    """Load BioLaySumm parquet datasets"""
    train_df = pd.read_parquet(train_path)
    print("###")
    print(train_df.columns)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    return (
        Dataset.from_pandas(train_df),
        Dataset.from_pandas(val_df),
        Dataset.from_pandas(test_df),
    )

def preprocess_dataset(dataset, tokeniser, max_input_length=256, max_output_length=100):
    """Tokenise and preprocess datasets"""
    def preprocess(batch):
        inputs = [
            "You are a medical professional, please turn this radiology report into layman report: " + r
            for r in batch["radiology_report"]
        ]
        model_inputs = tokeniser(inputs, max_length=max_input_length, truncation=True, padding="max_length")
        labels = tokeniser(
            text_target=batch["layman_report"],
            max_length=max_output_length,
            truncation=True,
            padding="max_length"
        )
        # Convert list of arrays to single NumPy array to avoid slow tensor conversion
        model_inputs["labels"] = np.array(labels["input_ids"])
        return model_inputs

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
