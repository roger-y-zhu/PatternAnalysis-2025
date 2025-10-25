"""Random example inference + ROUGE reporting — use cleaned raw test split"""
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from modules import load_raw_datasets_wrapper
from my_utils import compute_rouge

MODEL_PATH = Path("./t5_radiology_finetuned_final")


def generate_summary_from_model(report_text: str, model, tokenizer, device: str):
    """Generate summary using already-loaded model/tokenizer"""
    # prompt = "You are a medical professional, please turn this radiology report into layman report: " + report_text
    prompt = report_text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def format_rouge_scores(scores: dict) -> str:
    """Return nicely formatted ROUGE line"""
    # guard keys
    keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    parts = []
    for k in keys:
        v = scores.get(k)
        if v is None:
            parts.append(f"{k}: n/a")
        else:
            parts.append(f"{k}: {v:.4f}")
    return " | ".join(parts)


def main(model_path: Path = MODEL_PATH):
    """Load model, pick random test example with ground truth, print comparison + ROUGE"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load raw dataframes (pandas)
    train_df, val_df, test_df = load_raw_datasets_wrapper()

    # report missing ground-truth summaries
    missing_mask = test_df["layman_report"].isna() | (test_df["layman_report"].str.strip() == "")
    missing_idxs = test_df[missing_mask].index.tolist()
    print(f"Missing lay summaries: {len(missing_idxs)} examples")
    if len(missing_idxs) > 0:
        print("  example missing indices (first 10):", missing_idxs[:10])

    # choose a candidate that has a non-empty ground-truth if possible
    candidates = test_df[~missing_mask]
    if len(candidates) == 0:
        print("No test examples with ground-truth summaries found — sampling from full test set.")
        candidates = test_df

    row = candidates.sample(1).iloc[0]

    report = row["radiology_report"]
    goal = (row.get("layman_report") or "").strip()

    # load model/tokenizer once
    model = T5ForConditionalGeneration.from_pretrained(str(model_path)).to(device)
    tokenizer = T5Tokenizer.from_pretrained(str(model_path))
    model.eval()

    pred = generate_summary_from_model(report, model, tokenizer, device)

    print("\n## Radiology report:\n")
    print(report, "\n")
    print("## Model-generated layperson summary:\n")
    print(pred, "\n")
    print("## Ground-truth layperson summary:\n")
    print(goal if goal else "(no ground-truth available)", "\n")

    # compute and print ROUGE if ground-truth exists
    if goal:
        try:
            scores = compute_rouge([pred], [goal])
            print("## ROUGE scores:")
            print(format_rouge_scores(scores))
        except Exception as e:
            print("Could not compute ROUGE:", e)
    else:
        print("Skipping ROUGE (no ground-truth).")


if __name__ == "__main__":
    main()
