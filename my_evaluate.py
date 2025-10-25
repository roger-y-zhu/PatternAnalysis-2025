import torch
import time
import ujson
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer
from modules import get_tokenised_datasets, load_raw_datasets
from my_utils import compute_rouge

# === Paths ===
MODEL_PATH = Path("./t5_radiology_finetuned_final")
CACHE_DIR = Path("./cache_parts")
CACHE_DIR.mkdir(exist_ok=True)


# === Cache Utils ===
def save_cache_part(part_key, data):
    """Write partial cache file. Telegraphic style."""
    part_file = CACHE_DIR / f"cache_{part_key:02}.json"
    with open(part_file, "w") as f:
        ujson.dump(data, f)


def load_all_cache_parts():
    """Load all cache parts. Merge into dict."""
    cache = {}
    for part_file in CACHE_DIR.glob("cache_*.json"):
        with open(part_file, "r") as f:
            cache.update(ujson.load(f))
    return cache


# === Evaluation ===
def evaluate_model(model, tokenizer, raw_test_ds, cache):
    """Iterate test set, generate summaries, compute ROUGE."""
    device = next(model.parameters()).device
    total = len(raw_test_ds)
    print(f"Total examples: {total}, Cached: {len(cache)}")

    start_time = time.time()
    for idx, row in enumerate(raw_test_ds.itertuples(index=False)):
        key = str(idx)
        if key in cache:
            continue

        input_text = row.radiology_report
        target_text = row.layman_report

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)

        try:
            with torch.no_grad():
                ids = model.generate(**inputs, max_length=256)
            prediction = tokenizer.decode(ids[0], skip_special_tokens=True)

            rouge_score = compute_rouge([prediction], [target_text])
            cache[key] = rouge_score

            # Save part cache (100 per file)
            part_idx = idx // 100
            part_data = {k: v for k, v in cache.items() if int(k) // 100 == part_idx}
            save_cache_part(part_idx, part_data)

            elapsed = time.time() - start_time
            percent = (idx + 1) / total * 100
            remaining = elapsed / (idx + 1) * (total - (idx + 1))
            print(f"✅ [{idx + 1}/{total}] {percent:.2f}% done | "
                  f"Elapsed: {elapsed/60:.2f}m | ETA: {remaining/60:.2f}m")
            print(f"ROUGE: {rouge_score}")

        except RuntimeError as e:
            print(f"❌ [{idx + 1}/{total}] Runtime error: {e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()


def aggregate_rouge(cache):
    """Compute mean ROUGE from cache dict."""
    avg = {}
    for score in cache.values():
        for k, v in score.items():
            avg[k] = avg.get(k, 0.0) + v
    n = len(cache)
    return {k: v / n for k, v in avg.items()}


# === Main ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    # Load tokenised + raw datasets
    _, _, test_ds = get_tokenised_datasets(tokenizer)
    _, _, raw_test_ds = load_raw_datasets()

    # Load cache
    cache = load_all_cache_parts()

    # Evaluate
    evaluate_model(model, tokenizer, raw_test_ds, cache)

    # Aggregate ROUGE
    avg_rouge = aggregate_rouge(cache)
    print("\n=== Aggregated ROUGE Results ===")
    for k, v in avg_rouge.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
