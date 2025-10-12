import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration

train_df = pd.read_parquet("data/train.parquet")
validation_df = pd.read_parquet("data/validation.parquet")
test_df = pd.read_parquet("data/test.parquet")

# Load T5 model and tokenizer
model_name = "t5-small"  # or "t5-base" for larger
tokeniser = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate T5 layman summary
def t5_layman_report(radiology_text):
    prompt = "You are a medical professional, please turn this radiology report into layman report: " + radiology_text
    inputs = tokeniser(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=150)
    return tokeniser.decode(outputs[0], skip_special_tokens=True)

# Take 3 random examples from training set
samples = train_df.sample(3)

for i, row in samples.iterrows():
    print(f"--- Example {i} ---")
    print("Radiology report:")
    print(row['radiology_report'])
    print("\nDataset layman report:")
    print(row['layman_report'])
    t5_output = t5_layman_report(row['radiology_report'])
    print("\nT5 layman report:")
    print(t5_output)
    print("\n" + "="*80 + "\n")