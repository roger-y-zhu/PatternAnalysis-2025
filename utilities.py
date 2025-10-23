"""Helper functions for tokenisation, metrics, text cleaning"""

from evaluate import load



rouge = load("rouge")

def compute_rouge(eval_pred):
    """Compute ROUGE metrics for summarisation"""
    preds, labels = eval_pred
    decoded_preds = [p.strip() for p in preds]
    decoded_labels = [l.strip() for l in labels]
    results = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"],
        "rougeLsum": results["rougeLsum"],
    }

