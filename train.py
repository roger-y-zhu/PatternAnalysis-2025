import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from utils import compute_rouge
from modules import get_tokenised_datasets
from pathlib import Path

def main():
    checkpoint_path = Path("./t5_radiology_finetuned_final")

    tokenizer = T5Tokenizer.from_pretrained("t5-small-local")
    train_ds, val_ds, _ = get_tokenised_datasets(tokenizer)

    # Load or initialize model
    try:
        model = T5ForConditionalGeneration.from_pretrained(str(checkpoint_path)).to("cpu")
        tokenizer = T5Tokenizer.from_pretrained(str(checkpoint_path))
        print("Loaded fine-tuned model. Skipping training.")
        return
    except Exception:
        print("Fine-tuned model not found. Training base model.")

    model = T5ForConditionalGeneration.from_pretrained("t5-small-local").to("cpu")
    for param in model.parameters():
        param.requires_grad = True

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_path),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        eval_accumulation_steps=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=1,
        fp16=True,
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
