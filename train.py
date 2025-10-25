"""Train T5-small on cleaned data"""
from pathlib import Path
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from modules import get_tokenised_datasets, load_model
import evaluate
import numpy as np

CHECKPOINT = Path("./t5_radiology_finetuned_final")
MODEL_NAME = "t5-small-local"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    train_ds, val_ds, _ = get_tokenised_datasets(tokenizer)

    # try load existing checkpoint
    if CHECKPOINT.exists():
        try:
            model = T5ForConditionalGeneration.from_pretrained(str(CHECKPOINT)).to(device)
            print("Loaded existing fine-tuned model. Exiting.")
            return
        except Exception:
            print("Failed loading existing checkpoint. Will train from base model.")

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        """Decode predictions/labels and compute ROUGE"""
        preds_ids = eval_pred.predictions
        # when using generate, HF may return a tuple
        if isinstance(preds_ids, tuple):
            preds_ids = preds_ids[0]
        decoded_preds = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
        labels_ids = eval_pred.label_ids
        # replace -100 in labels as pad token id
        labels_ids = np.where(labels_ids == -100, tokenizer.pad_token_id, labels_ids)
        decoded_labels = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        results = rouge.compute(predictions=[p.strip() for p in decoded_preds],
                                references=[l.strip() for l in decoded_labels],
                                use_stemmer=True)
        # return scalars
        return {k: float(v) for k, v in results.items()}

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        save_strategy="epoch",
        report_to="none",
        dataloader_num_workers=0,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(str(CHECKPOINT))
    tokenizer.save_pretrained(str(CHECKPOINT))
    print("✅ Training complete and model saved.")


if __name__ == "__main__":
    main()
