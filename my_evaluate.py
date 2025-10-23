import torch
import time
from transformers import T5ForConditionalGeneration, T5Tokenizer
from my_utils import compute_rouge
from modules import get_tokenised_datasets, load_datasets
from pathlib import Path
import ujson
import os
from collections import defaultdict

CACHE_FILE = Path("./eval_cache.json")
CACHE_DIR = Path("./cache_parts")
CACHE_DIR.mkdir(exist_ok=True)




def save_cache_part(part_key, data):
    """Save a cache part file."""
    part_file = CACHE_DIR / f"cache_{part_key:02}.json"
    with open(part_file, "w") as f:
        ujson.dump(data, f)


def load_all_cache_parts():
    """Load all existing cache parts into a single dict."""
    cache = {}
    for part_file in CACHE_DIR.glob("cache_*.json"):
        with open(part_file, "r") as f:
            data = ujson.load(f)
            cache.update(data)
    return cache


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = Path("./t5_radiology_finetuned_final")
    tokenizer = T5Tokenizer.from_pretrained(str(checkpoint_path))
    _, _, test_ds = get_tokenised_datasets(tokenizer)
    raw_train_ds, raw_val_ds, raw_test_ds = load_datasets()

    model = T5ForConditionalGeneration.from_pretrained(str(checkpoint_path)).to(device)
    model.eval()

    # Load all parts for tracking
    cache = load_all_cache_parts()
    total = len(raw_test_ds)
    print(f"Total examples: {total}, Cached: {len(cache)}")

    start_time = time.time()
    for idx, row in enumerate(raw_test_ds):
        key = str(idx)
        if key in cache:
            continue

        radiology_report = row["radiology_report"]
        inputs = tokenizer(radiology_report, return_tensors="pt").to(device)

        try:
            with torch.no_grad():
                ids = model.generate(**inputs, max_length=256)
            prediction = tokenizer.decode(ids[0], skip_special_tokens=True)
            rouge_score = compute_rouge(([prediction], [radiology_report]))

            cache[key] = rouge_score
            # Save only the part corresponding to this key
            part_idx = idx // 100
            part_data = {k: v for k, v in cache.items() if int(k) // 100 == part_idx}
            save_cache_part(part_idx, part_data)

            # Progress reporting
            elapsed = time.time() - start_time
            percent = (idx + 1) / total * 100
            remaining = elapsed / (idx + 1) * (total - (idx + 1))
            print(f"✅ [{idx + 1}/{total}] {percent:.2f}% done | "
                  f"Time elapsed: {elapsed / 60:.2f} min | "
                  f"Est. remaining: {remaining / 60:.2f} min")
            print(f"ROUGE: {rouge_score}")

        except RuntimeError as e:
            print(f"❌ [{idx + 1}/{total}] Runtime error on example {key}: {e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

    # Aggregate average ROUGE
    avg_rouge = {}
    for score in cache.values():
        for k, v in score.items():
            avg_rouge[k] = avg_rouge.get(k, 0.0) + v
    n = len(cache)
    avg_rouge = {k: v / n for k, v in avg_rouge.items()}

    print("\n=== Aggregated ROUGE Results ===")
    for k, v in avg_rouge.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
