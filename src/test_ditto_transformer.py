
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "ditto_transformer_model"
MAX_LEN = 64
THRESHOLD = 0.35

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR, local_files_only=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, local_files_only=True
)
model.eval()


def predict_match(ledger, bank):
    """Predict whether a ledger entry matches a bank line."""

    left_text = (
        f"reference {ledger['reference']}; "
        f"counterparty {ledger['counterparty']}; "
        f"ledger amount {ledger['amount']}; "
        f"ledger date {ledger['entry_date']}"
    )

    right_text = (
        f"bank narrative {bank['narrative']}; "
        f"bank amount {bank['amount']}; "
        f"bank date {bank['posted_date']}"
    )

    # Pass both texts separately, exactly as during training.
    inputs = tokenizer(
        left_text,
        right_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    # Training labels were 0 = non-match and 1 = match.
    match_probability = float(probabilities[1])

    return {
        "match_probability": match_probability,
        "predicted_match": match_probability >= THRESHOLD,
    }


ledger = {
    "reference": "INV-1042",
    "counterparty": "ABC Suppliers",
    "amount": 1500.00,
    "entry_date": "2026-04-20",
}

bank = {
    "narrative": "NEFT ABC SUPPLIERS INV-1042",
    "amount": 1500.00,
    "posted_date": "2026-04-20",
}

result = predict_match(ledger, bank)

print("Match probability:", round(result["match_probability"], 4))
print("Predicted match:", result["predicted_match"])
print("Decision threshold:", THRESHOLD)
