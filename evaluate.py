import torch
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utilities import compute_rouge
from modules import get_tokenised_datasets, load_datasets
from pathlib import Path

CACHE_FILE = Path("./eval_cache.json")

def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = Path("./t5_radiology_finetuned_final")
    tokenizer = T5Tokenizer.from_pretrained(str(checkpoint_path))
    _, _, test_ds = get_tokenised_datasets(tokenizer)
    raw_train_ds, raw_val_ds, raw_test_ds = load_datasets()

    model = T5ForConditionalGeneration.from_pretrained(str(checkpoint_path)).to(device)
    model.eval()

    cache = load_cache()
    all_eval_results = {}

    print("=== Evaluating Test Set with Caching ===")
    for idx, test_row in enumerate(raw_test_ds):
        key = str(idx)
        if key in cache:
            print(f"⏩ Skipping example {idx} (already cached)")
            all_eval_results[key] = cache[key]
            continue

        radiology_report = test_row["radiology_report"]
        inputs = tokenizer(radiology_report, return_tensors="pt").to(device)

        try:
            with torch.no_grad():
                ids = model.generate(**inputs, max_length=256)
            prediction = tokenizer.decode(ids[0], skip_special_tokens=True)
            rouge_score = compute_rouge([prediction], [radiology_report])
            cache[key] = rouge_score
            save_cache(cache)
            all_eval_results[key] = rouge_score
            print(f"✅ Evaluated example {idx}")
        except RuntimeError as e:
            print(f"❌ Runtime error on example {idx}: {e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

    # Aggregate average ROUGE
    avg_rouge = {}
    for key, score in all_eval_results.items():
        for k, v in score.items():
            avg_rouge[k] = avg_rouge.get(k, 0.0) + v
    n = len(all_eval_results)
    avg_rouge = {k: v / n for k, v in avg_rouge.items()}

    print("\n=== Aggregated ROUGE Results ===")
    for k, v in avg_rouge.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
