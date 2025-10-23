import pandas as pd
import numpy as np
from datasets import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split

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
        model_inputs["labels"] = np.array(labels["input_ids"])
        return model_inputs

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

def load_clean_data(path="data/clean_biolaysumm.parquet", as_hf_dataset=True):
    """Load the cleaned BioLaySumm dataset"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data file not found: {path}")
    df = pd.read_parquet(path)
    if as_hf_dataset:
        return Dataset.from_pandas(df)
    return df

def print_basic_stats(df, name="Dataset"):
    """Print basic statistics of a DataFrame"""
    print(f"\n### {name} Stats ###")
    print(f"Total rows: {len(df)}")
    print("Missing values per column:\n", df.isna().sum())
    print("Column lengths (min, max, mean) for text columns:")
    for col in df.columns:
        if df[col].dtype == object:
            lengths = df[col].dropna().str.len()
            print(f"  {col}: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.2f}")
    print(df.head())

def main():
    # Paths to original CSV files
    train_csv = Path("data/csv/train.csv")
    val_csv = Path("data/csv/validation.csv")
    test_csv = Path("data/csv/test.csv")

    # Read CSVs
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Combine datasets
    mega_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"Combined dataset: {len(mega_df)} rows")

    # Drop unwanted columns
    mega_df = mega_df.drop(columns=["source", "images_path"])

    # Drop super long rows
    mega_df = mega_df[mega_df["radiology_report"].str.len() <= 1000]
    mega_df = mega_df[mega_df["layman_report"].str.len() <= 512]

    # Drop rows with any missing values
    mega_df = mega_df.dropna()
    print(f"After dropping missing values: {len(mega_df)} rows")

    # Remove rows containing ".png"
    mask_png = mega_df["radiology_report"].str.contains(r"\.png", case=False, na=False) | \
               mega_df["layman_report"].str.contains(r"\.png", case=False, na=False)
    mega_df = mega_df[~mask_png]

    # Remove rows where radiology and lay reports are identical
    mask_identical = mega_df["radiology_report"] == mega_df["layman_report"]
    mega_df = mega_df[~mask_identical]

    print_basic_stats(mega_df, "Cleaned Full Dataset")

    # Save full cleaned dataset
    clean_path = Path("data/clean_biolaysumm.parquet")
    mega_df.to_parquet(clean_path, index=False)
    print(f"Saved full cleaned dataset to {clean_path}")

    # Split 70% train, 15% val, 15% test
    train_val, test_clean = train_test_split(mega_df, test_size=0.15, random_state=42)
    train_clean, val_clean = train_test_split(train_val, test_size=0.1765, random_state=42)  # ~0.15/0.85 = 0.1765

    # Print stats for splits
    print_basic_stats(train_clean, "Train Clean Dataset")
    print_basic_stats(val_clean, "Validation Clean Dataset")
    print_basic_stats(test_clean, "Test Clean Dataset")

    # Save splits
    train_clean.to_parquet("data/train_clean.parquet", index=False)
    val_clean.to_parquet("data/val_clean.parquet", index=False)
    test_clean.to_parquet("data/test_clean.parquet", index=False)

    print(f"Saved train_clean ({len(train_clean)} rows), val_clean ({len(val_clean)} rows), test_clean ({len(test_clean)} rows)")

if __name__ == "__main__":
    main()
