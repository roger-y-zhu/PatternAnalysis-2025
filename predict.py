"""Generate lay summaries from new reports; prints example outputs"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def generate_summary(report_text, model_path="./t5_radiology_finetuned"):
    """Generate layperson summary from a radiology report"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokeniser = T5Tokenizer.from_pretrained(model_path)

    input_text = "You are a medical professional, please turn this radiology report into layman report: " + report_text
    inputs = tokeniser(input_text, return_tensors="pt", truncation=True, padding=True).to(device)

    outputs = model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    return tokeniser.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    example = "The CT scan reveals mild left lower lobe consolidation with no pleural effusion."
    summary = generate_summary(example)
    print("Radiology report:", example)
    print("Layperson summary:", summary)
