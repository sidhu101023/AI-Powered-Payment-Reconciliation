
from pathlib import Path
from difflib import SequenceMatcher
import re

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = ROOT / "models" / "ditto_transformer_model"
MODEL_DIR = ROOT / "models" / "hybrid_transformer_model"
SCALER_PATH = MODEL_DIR / "numeric_scaler.npz"

MAX_LEN = 96
DEFAULT_THRESHOLD = 0.95

NUMERIC_FEATURES = [
    "amount_difference",
    "exact_amount",
    "amount_within_cent",
    "same_amount_sign",
    "signed_date_difference",
    "absolute_date_difference",
    "reference_in_narrative",
    "reference_similarity",
    "counterparty_similarity",
]


# ---------------------------------------------------------
# Feature helpers: must match training preprocessing
# ---------------------------------------------------------

def normalize(value):
    value = str(value).upper()
    return re.sub(r"[^A-Z0-9]", "", value)


def similarity(a, b):
    a = normalize(a)
    b = normalize(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def parse_date(value):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return pd.NaT

    return pd.to_datetime(value, errors="coerce")


def get_value(record, key, default=""):
    value = record.get(key, default)

    if value is None or pd.isna(value):
        return default

    return value


# ---------------------------------------------------------
# Architecture: identical to training
# ---------------------------------------------------------

class HybridTransformer(nn.Module):
    def __init__(self, base_model, num_numeric_features):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            str(base_model),
            local_files_only=True,
        )

        hidden_size = self.encoder.config.dim

        self.numeric_projection = nn.Sequential(
            nn.Linear(num_numeric_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        numeric_features,
    ):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        text_features = output.last_hidden_state[:, 0, :]
        numeric_features = self.numeric_projection(numeric_features)

        combined = torch.cat(
            [text_features, numeric_features],
            dim=1,
        )

        logits = self.classifier(combined)

        return {"logits": logits}


# ---------------------------------------------------------
# Predictor
# ---------------------------------------------------------

class HybridTransformerPredictor:
    def __init__(self, threshold=DEFAULT_THRESHOLD):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        required_files = [
            MODEL_DIR / "model_state.pt",
            MODEL_DIR / "tokenizer.json",
            MODEL_DIR / "tokenizer_config.json",
            SCALER_PATH,
            BASE_MODEL_DIR / "config.json",
        ]

        missing = [str(path) for path in required_files if not path.exists()]

        if missing:
            raise FileNotFoundError(
                "Missing required model files:\n" + "\n".join(missing)
            )

        self.threshold = float(threshold)
        self.device = torch.device("cpu")

        scaler_data = np.load(SCALER_PATH, allow_pickle=False)
        self.scaler_mean = scaler_data["mean"].astype(np.float64)
        self.scaler_scale = scaler_data["scale"].astype(np.float64)
        saved_features = scaler_data["features"].astype(str).tolist()

        if saved_features != NUMERIC_FEATURES:
            raise ValueError(
                "Saved scaler feature order does not match the "
                "prediction module."
            )

        if (
            len(self.scaler_mean) != len(NUMERIC_FEATURES)
            or len(self.scaler_scale) != len(NUMERIC_FEATURES)
        ):
            raise ValueError("Saved scaler has an unexpected shape.")

        # StandardScaler uses scale=1 for zero-variance features.
        self.scaler_scale[self.scaler_scale == 0] = 1.0

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )

        self.model = HybridTransformer(
            BASE_MODEL_DIR,
            len(NUMERIC_FEATURES),
        )

        # Only load trusted model weights from this project.
        try:
            state = torch.load(
                MODEL_DIR / "model_state.pt",
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            # Compatibility with older PyTorch versions.
            state = torch.load(
                MODEL_DIR / "model_state.pt",
                map_location="cpu",
            )

        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def build_pair_features(self, ledger, bank):
        reference = str(get_value(ledger, "reference"))
        counterparty = str(get_value(ledger, "counterparty"))
        narrative = str(get_value(bank, "narrative"))

        ledger_amount = float(get_value(ledger, "amount"))
        bank_amount = float(get_value(bank, "amount"))

        amount_gap = abs(ledger_amount - bank_amount)

        ledger_date = parse_date(get_value(ledger, "entry_date"))
        bank_date = parse_date(get_value(bank, "posted_date"))

        if pd.notna(ledger_date) and pd.notna(bank_date):
            day_gap = int((bank_date - ledger_date).days)
        else:
            # Matches the training code's missing-date convention.
            day_gap = 9999

        reference_normalized = normalize(reference)
        narrative_normalized = normalize(narrative)

        reference_in_narrative = int(
            bool(reference_normalized)
            and reference_normalized in narrative_normalized
        )

        features = {
            "amount_difference": amount_gap,
            "exact_amount": int(amount_gap < 0.005),
            "amount_within_cent": int(amount_gap <= 0.01),
            "same_amount_sign": int(
                (ledger_amount < 0 and bank_amount < 0)
                or (ledger_amount >= 0 and bank_amount >= 0)
            ),
            "signed_date_difference": day_gap,
            "absolute_date_difference": abs(day_gap),
            "reference_in_narrative": reference_in_narrative,
            "reference_similarity": similarity(reference, narrative),
            "counterparty_similarity": similarity(
                counterparty, narrative
            ),
        }

        left_text = (
            f"reference {reference}; counterparty {counterparty}; "
            f"ledger amount {ledger_amount}; "
            f"ledger date {get_value(ledger, 'entry_date')}"
        )

        right_text = (
            f"bank narrative {narrative}; bank amount {bank_amount}; "
            f"bank date {get_value(bank, 'posted_date')}"
        )

        return left_text, right_text, features

    @torch.inference_mode()
    def score_pairs(self, pairs, batch_size=32):
        """
        Score multiple candidate pairs in batches.

        pairs: iterable of (ledger_dict, bank_dict) tuples.
        Returns one result dictionary per input pair, in the same order.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        pairs = list(pairs)
        if not pairs:
            return []

        results = []

        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start:start + batch_size]

            left_texts = []
            right_texts = []
            feature_rows = []

            for ledger, bank in batch_pairs:
                left_text, right_text, features = (
                    self.build_pair_features(ledger, bank)
                )
                left_texts.append(left_text)
                right_texts.append(right_text)
                feature_rows.append(features)

            raw_numeric = np.asarray(
                [
                    [features[name] for name in NUMERIC_FEATURES]
                    for features in feature_rows
                ],
                dtype=np.float64,
            )

            scaled_numeric = (
                (raw_numeric - self.scaler_mean)
                / self.scaler_scale
            ).astype(np.float32)

            encoded = self.tokenizer(
                left_texts,
                right_texts,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt",
            )

            encoded = {
                name: tensor.to(self.device)
                for name, tensor in encoded.items()
            }

            numeric_tensor = torch.tensor(
                scaled_numeric,
                dtype=torch.float32,
                device=self.device,
            )

            output = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                numeric_features=numeric_tensor,
            )

            probabilities = torch.softmax(
                output["logits"], dim=1
            )[:, 1].cpu().numpy()

            for probability, features in zip(
                probabilities, feature_rows
            ):
                probability = float(probability)
                results.append({
                    "match_probability": probability,
                    "predicted_match": (
                        probability >= self.threshold
                    ),
                    "threshold": self.threshold,
                    "features": features,
                })

        return results

    def score_pair(self, ledger, bank):
        """Backward-compatible single-pair prediction."""
        return self.score_pairs([(ledger, bank)], batch_size=1)[0]
