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



# Define radiology report summaries
preds = [
    "Looking at images from 2010 and now, a new small growth of about 2cm has appeared in the left lung area, which wasn’t there before. Because of this, the patient had to get a chest CT scan.",
    "This is a follow-up X-ray for a COVID-19 patient. The lungs look the same as the previous X-ray, showing a faint net-like pattern on both sides and some slight haziness at the bottom of the lungs.",
    "The X-ray shows no notable changes or abnormalities."
]

labels = [
    "Looking at the images from 2010 and comparing them to now, there's a new growth in the left lung area that's about 2 cm big and wasn't there before. Because of this, the patient had to get a special chest CT scan.",
    "This report is a follow-up for a COVID-19 patient. The technique used is the same as the previous X-ray taken on [date]. The lungs show a pattern like a net and some faint cloudiness at the bottom of both lungs.",
    "There are no significant changes seen in the radiology images."
]

# Compute ROUGE for each example
for i, (pred, label) in enumerate(zip(preds, labels), 1):
    scores = compute_rouge(pred, label)
    print(f"Example {i} ROUGE scores: {scores}")
