from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

from my_utils import compute_rouge


def load_clean_parquet(path, tokenizer, max_input_length=512, max_output_length=256):
    """Load cleaned parquet file and tokenize it for T5"""
    import pandas as pd
    df = pd.read_parquet(path)
    dataset = Dataset.from_pandas(df)

    def preprocess(batch):
        inputs = [
            "You are a medical professional, please turn this radiology report into layman report: " + r
            for r in batch["radiology_report"]
        ]
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            text_target=batch["layman_report"],
            max_length=max_output_length,
            truncation=True,
            padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

def main():
    checkpoint_path = Path("./t5_radiology_finetuned_final")
    tokenizer = T5Tokenizer.from_pretrained("t5-small-local")

    # Load and preprocess cleaned datasets
    train_ds = load_clean_parquet("data/train_clean.parquet", tokenizer)
    val_ds = load_clean_parquet("data/val_clean.parquet", tokenizer)

    # Load or initialize model
    try:
        model = T5ForConditionalGeneration.from_pretrained(str(checkpoint_path)).to("cuda" if torch.cuda.is_available() else "cpu")
        print("Loaded fine-tuned model. Skipping training.")
        return
    except Exception:
        print("Fine-tuned model not found. Training base model.")
        model = T5ForConditionalGeneration.from_pretrained("t5-small-local").to("cuda" if torch.cuda.is_available() else "cpu")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_path),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        eval_accumulation_steps=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        save_strategy="epoch",
        report_to="none",
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_rouge
    )

    trainer.train()
    trainer.save_model(str(checkpoint_path))
    tokenizer.save_pretrained(str(checkpoint_path))
    print("✅ Training complete and model saved.")

if __name__ == "__main__":
    main()
