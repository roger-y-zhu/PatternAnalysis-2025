"""Load, clean and preprocess BioLaySumm data"""
import pandas as pd
import numpy as np
from datasets import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split

# default paths
TRAIN_CSV = Path("data/csv/train.csv")
VAL_CSV = Path("data/csv/validation.csv")
TEST_CSV = Path("data/csv/test.csv")
CLEAN_PATH = Path("data/clean_biolaysumm.parquet")
TRAIN_CLEAN = Path("data/train_clean.parquet")
VAL_CLEAN = Path("data/val_clean.parquet")
TEST_CLEAN = Path("data/test_clean.parquet")


def preprocess_dataset(dataset, tokenizer, max_input_length=512, max_output_length=256):
    """Tokenise HF Dataset for Trainer. Return tokenised HF Dataset."""
    def preprocess(batch):
        inputs = [
            "summarize radiology report for layperson: " + r
            for r in batch["radiology_report"]
        ]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

        labels = tokenizer(
            text_target=batch["layman_report"],
            max_length=max_output_length,
            truncation=True,
            padding="max_length"
        )["input_ids"]

        # replace padding token id's of the labels by -100 so they are ignored by the loss
        labels = np.array(labels)
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels.tolist()

        return model_inputs

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)


def load_clean_data(path=str(CLEAN_PATH), as_hf_dataset=True):
    """Load cleaned parquet. Return HF Dataset or pd.DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data file not found: {path}")
    df = pd.read_parquet(path)
    if as_hf_dataset:
        return Dataset.from_pandas(df.reset_index(drop=True))
    return df


def load_raw_datasets():
    """Load cleaned train/val/test parquet splits as pandas DataFrames."""
    if not TRAIN_CLEAN.exists() or not VAL_CLEAN.exists() or not TEST_CLEAN.exists():
        raise FileNotFoundError("train_clean/val_clean/test_clean parquet files missing. Run cleaning first.")
    train_df = pd.read_parquet(TRAIN_CLEAN)
    val_df = pd.read_parquet(VAL_CLEAN)
    test_df = pd.read_parquet(TEST_CLEAN)
    return train_df, val_df, test_df


def build_and_save_clean_csvs(train_csv=TRAIN_CSV, val_csv=VAL_CSV, test_csv=TEST_CSV):
    """Ingest original CSVs, clean, filter, save clean.parquet and splits."""
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    mega_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    mega_df = mega_df.drop(columns=["source", "images_path"], errors="ignore")

    # drop missing
    mega_df = mega_df.dropna()

    # drop extreme long sequences to avoid OOM during training
    mega_df = mega_df[mega_df["radiology_report"].str.len() <= 1000]
    mega_df = mega_df[mega_df["layman_report"].str.len() <= 512]

    # drop .png artefacts
    mask_png = mega_df["radiology_report"].str.contains(r"\.png", case=False, na=False) | \
               mega_df["layman_report"].str.contains(r"\.png", case=False, na=False)
    mega_df = mega_df[~mask_png]

    # drop identical pairs
    mega_df = mega_df[mega_df["radiology_report"] != mega_df["layman_report"]]

    mega_df = mega_df.reset_index(drop=True)
    mega_df.to_parquet(CLEAN_PATH, index=False)

    # split 70/15/15
    train_val, test = train_test_split(mega_df, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1765, random_state=42)  # ~0.15/0.85

    train.reset_index(drop=True).to_parquet(TRAIN_CLEAN, index=False)
    val.reset_index(drop=True).to_parquet(VAL_CLEAN, index=False)
    test.reset_index(drop=True).to_parquet(TEST_CLEAN, index=False)

    return train, val, test

def print_dataset_stats(train_csv=TRAIN_CSV, val_csv=VAL_CSV, test_csv=TEST_CSV):
    """Print original row counts, filtered rows, leftover, and final 70/15/15 split sizes."""
    # Load original CSVs
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    total_original = len(train_df) + len(val_df) + len(test_df)

    print(f"Original row counts: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Total original rows: {total_original}")

    # Apply same cleaning as build_and_save_clean_csvs
    mega_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    mega_df = mega_df.drop(columns=["source", "images_path"], errors="ignore")
    mega_df = mega_df.dropna()
    mega_df = mega_df[mega_df["radiology_report"].str.len() <= 1000]
    mega_df = mega_df[mega_df["layman_report"].str.len() <= 512]
    mask_png = mega_df["radiology_report"].str.contains(r"\.png", case=False, na=False) | \
               mega_df["layman_report"].str.contains(r"\.png", case=False, na=False)
    mega_df = mega_df[~mask_png]
    mega_df = mega_df[mega_df["radiology_report"] != mega_df["layman_report"]]

    total_after_filter = len(mega_df)
    filtered_out = total_original - total_after_filter
    print(f"Rows filtered out: {filtered_out}")
    print(f"Total rows left after cleaning: {total_after_filter}")

    # Compute 70/15/15 split sizes
    train_val, test = train_test_split(mega_df, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1765, random_state=42)  # ~0.15/0.85

    print(f"Final split sizes: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    print_dataset_stats()
