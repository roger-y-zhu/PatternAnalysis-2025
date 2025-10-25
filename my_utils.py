"""Utility helpers: ROUGE calculation and text helpers"""
import ujson

def compute_rouge(preds, labels):
    """
    Compute ROUGE for lists of strings.
    preds: list[str] or single str
    labels: list[str] or single str
    Return dict with rouge1/2/L/Lsum
    """
    # make lists
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(labels, str):
        labels = [labels]

    # lazy import to avoid import cycles
    from evaluate import load as evaluate_load
    rouge = evaluate_load("rouge")
    decoded_preds = [p.strip() for p in preds]
    decoded_labels = [l.strip() for l in labels]
    results = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # convert to floats
    return {
        "rouge1": float(results["rouge1"]),
        "rouge2": float(results["rouge2"]),
        "rougeL": float(results["rougeL"]),
        "rougeLsum": float(results["rougeLsum"]),
    }
