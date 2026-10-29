
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "ditto_transformer_model"

MAX_LEN = 64
DEFAULT_THRESHOLD = 0.35


class DittoTransformerPredictor:
    """Load a trained Transformer and score ledger-bank candidate pairs."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def make_texts(ledger, bank):
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

        return left_text, right_text

    def score_pair(self, ledger, bank):
        return self.score_pairs([(ledger, bank)], batch_size=1)[0]

    @torch.inference_mode()
    def score_pairs(self, pairs, batch_size=32):
        """Score candidate pairs in batches while preserving input order."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        pairs = list(pairs)
        results = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            texts = [self.make_texts(ledger, bank) for ledger, bank in batch]
            left_texts, right_texts = zip(*texts)

            inputs = self.tokenizer(
                list(left_texts), list(right_texts), truncation=True,
                padding="max_length", max_length=MAX_LEN, return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            results.extend({
                "match_probability": float(probability),
                "predicted_match": float(probability) >= self.threshold,
            } for probability in probabilities)

        return results
