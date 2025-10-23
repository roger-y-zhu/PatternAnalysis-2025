import random

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from modules import load_datasets

_, _, test_ds = load_datasets()
missing = [i for i, row in enumerate(test_ds) if not row.get("layman_report")]
print(f"Missing lay summaries for {len(missing)} examples:", missing)


def generate_summary(report_text, model_path="./t5_radiology_finetuned_final"):
    """Generate layperson summary from a radiology report"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    input_text = "You are a medical professional, please turn this radiology report into layman report: " + report_text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)

    outputs = model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    _, _, test_ds = load_datasets()  # load raw datasets
    example_row = random.choice(test_ds)

    report = example_row["radiology_report"]
    goal_summary = example_row["layman_report"]

    predicted_summary = generate_summary(report)

    print("## Radiology report:")
    print(report, "\n")
    print("## Model-generated layperson summary:")
    print(predicted_summary, "\n")
    print("## Ground-truth layperson summary:")
    print(goal_summary)
