"""Full fine-tuning pipeline: training, validation, ROUGE evaluation, checkpoint saving"""
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from modules import load_model
from dataset import load_datasets, preprocess_dataset
from utils import compute_rouge

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model, tokeniser = load_model("t5-small", use_lora=True)
    train_ds, val_ds, _ = load_datasets()

    train_ds = preprocess_dataset(train_ds, tokeniser)
    val_ds = preprocess_dataset(val_ds, tokeniser)

    data_collator = DataCollatorForSeq2Seq(tokeniser, model=model)

    training_args = TrainingArguments(
        output_dir="./t5_radiology_finetuned",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=3,
        fp16=True,
        optim="adamw_torch_fused",
        torch_compile=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=1000,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokeniser,
        data_collator=data_collator,
        compute_metrics=compute_rouge,
    )

    trainer.train()
    model.save_pretrained("./t5_radiology_finetuned")
    tokeniser.save_pretrained("./t5_radiology_finetuned")

if __name__ == "__main__":
    main()
