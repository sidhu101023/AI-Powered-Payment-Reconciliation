
from pathlib import Path
import random
import re
import time
import warnings
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MODEL_DIR = ROOT / "models" / "hybrid_transformer_model"
DITTO_MODEL_DIR = ROOT / "models" / "ditto_transformer_model"
REPORTS.mkdir(parents=True, exist_ok=True)

BANK_FILE = DATA / "bank_lines.csv"
LEDGER_FILE = DATA / "ledger_entries.csv"
KEY_FILE = DATA / "match_key.csv"

# Fine-tune the locally trained Ditto checkpoint created by the first step.
BASE_MODEL = DITTO_MODEL_DIR
MODEL_TAG = "Hybrid_DistilBERT_Structured"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
MAX_LEN = 96
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 1e-5
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)

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


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def normalize(value):
    value = str(value).upper()
    return re.sub(r"[^A-Z0-9]", "", value)


def similarity(a, b):
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def expected_pairs(frame):
    return set(
        zip(
            frame.loc[frame.label == 1, "ledger_id"].astype(int),
            frame.loc[frame.label == 1, "bank_id"].astype(int),
        )
    )


def greedy_one_to_one(scored, threshold):
    candidates = scored[scored.probability >= threshold]
    candidates = candidates.sort_values("probability", ascending=False)

    chosen, used_ledger, used_bank = [], set(), set()

    for row in candidates.itertuples(index=False):
        lid, bid = int(row.ledger_id), int(row.bank_id)
        if lid in used_ledger or bid in used_bank:
            continue

        used_ledger.add(lid)
        used_bank.add(bid)
        chosen.append({
            "ledger_id": lid,
            "bank_id": bid,
            "label": int(row.label),
            "probability": float(row.probability),
        })

    return pd.DataFrame(
        chosen,
        columns=["ledger_id", "bank_id", "label", "probability"],
    )


def evaluate_matches(predicted, expected):
    predicted_pairs = (
        set(zip(predicted.ledger_id, predicted.bank_id))
        if not predicted.empty else set()
    )
    tp = len(predicted_pairs & expected)
    fp = len(predicted_pairs - expected)
    fn = len(expected - predicted_pairs)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "predicted_matches": len(predicted_pairs),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_candidates():
    bank = pd.read_csv(BANK_FILE)
    ledger = pd.read_csv(LEDGER_FILE)
    keys = pd.read_csv(KEY_FILE)

    bank["id"] = bank.id.astype(int)
    ledger["id"] = ledger.id.astype(int)

    bank["account_key"] = bank.account.astype(str)
    ledger["account_key"] = ledger.book.astype(str)

    bank["parsed_date"] = pd.to_datetime(
        bank.posted_date, errors="coerce"
    )
    ledger["parsed_date"] = pd.to_datetime(
        ledger.entry_date, errors="coerce"
    )

    valid_keys = keys[
        keys.kind.astype(str).str.lower() != "duplicate_line"
    ]
    positive_pairs = set(zip(
        valid_keys.ledger_entry_no.astype(int),
        valid_keys.bank_line_no.astype(int),
    ))

    all_dates = pd.concat(
        [ledger.parsed_date, bank.parsed_date], ignore_index=True
    ).dropna()
    cut1, cut2 = all_dates.quantile([0.60, 0.80]).tolist()

    def assign_period(date):
        if pd.isna(date):
            return "excluded"
        if date < cut1:
            return "train"
        if date < cut2:
            return "validation"
        return "test"

    ledger["period"] = ledger.parsed_date.map(assign_period)
    bank["period"] = bank.parsed_date.map(assign_period)

    # Keep each labelled pair in the partition of its later date,
    # matching the existing Transformer training approach.
    assignments = {"ledger": {}, "bank": {}}

    for lid, bid in positive_pairs:
        lrow = ledger.loc[ledger.id == lid]
        brow = bank.loc[bank.id == bid]
        if lrow.empty or brow.empty:
            raise ValueError(f"Missing positive pair: {lid}, {bid}")

        dates = [
            d for d in [
                lrow.iloc[0].parsed_date,
                brow.iloc[0].parsed_date,
            ] if pd.notna(d)
        ]
        if not dates:
            raise ValueError(f"Both dates missing for pair {lid}, {bid}")

        period = assign_period(max(dates))
        for side, record_id in [("ledger", lid), ("bank", bid)]:
            old = assignments[side].get(record_id)
            if old is not None and old != period:
                raise ValueError(
                    f"Record {side}:{record_id} crosses partitions"
                )
            assignments[side][record_id] = period

    ledger["period"] = ledger.apply(
        lambda r: assignments["ledger"].get(int(r.id), r.period), axis=1
    )
    bank["period"] = bank.apply(
        lambda r: assignments["bank"].get(int(r.id), r.period), axis=1
    )

    bank_groups = {k: g for k, g in bank.groupby("account_key")}
    rows = []

    for l in ledger.to_dict("records"):
        period = l["period"]
        if period == "excluded":
            continue

        group = bank_groups.get(l["account_key"])
        if group is None:
            continue
        group = group[group.period == period]

        for b in group.to_dict("records"):
            reference = str(l.get("reference", ""))
            counterparty = str(l.get("counterparty", ""))
            narrative = str(b.get("narrative", ""))

            la, ba = float(l["amount"]), float(b["amount"])
            amount_gap = abs(la - ba)
            ld, bd = l["parsed_date"], b["parsed_date"]
            day_gap = (
                (bd - ld).days
                if pd.notna(ld) and pd.notna(bd) else 9999
            )

            ref_norm = normalize(reference)
            narrative_norm = normalize(narrative)
            ref_in_narrative = int(
                bool(ref_norm) and ref_norm in narrative_norm
            )

            left_text = (
                f"reference {reference}; counterparty {counterparty}; "
                f"ledger amount {la}; ledger date {l.get('entry_date', '')}"
            )
            right_text = (
                f"bank narrative {narrative}; bank amount {ba}; "
                f"bank date {b.get('posted_date', '')}"
            )

            rows.append({
                "ledger_id": int(l["id"]),
                "bank_id": int(b["id"]),
                "period": period,
                "label": int((int(l["id"]), int(b["id"])) in positive_pairs),
                "left_text": left_text,
                "right_text": right_text,
                "amount_difference": amount_gap,
                "exact_amount": int(amount_gap < 0.005),
                "amount_within_cent": int(amount_gap <= 0.01),
                "same_amount_sign": int(
                    (la < 0 and ba < 0) or (la >= 0 and ba >= 0)
                ),
                "signed_date_difference": day_gap,
                "absolute_date_difference": abs(day_gap),
                "reference_in_narrative": ref_in_narrative,
                "reference_similarity": similarity(reference, narrative),
                "counterparty_similarity": similarity(
                    counterparty, narrative
                ),
            })

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise ValueError("No candidate pairs generated.")

    frames = {
        name: pairs[pairs.period == name].copy()
        for name in ["train", "validation", "test"]
    }

    for name, frame in frames.items():
        if frame.empty or frame.label.nunique() < 2:
            raise ValueError(
                f"{name} needs candidates from both label classes."
            )

    return frames


class PairDataset(Dataset):
    def __init__(self, frame, tokenizer, scaler, name):
        self.frame = frame.reset_index(drop=True)
        self.encodings = tokenizer(
            self.frame.left_text.tolist(),
            self.frame.right_text.tolist(),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.numeric = torch.tensor(
            scaler.transform(self.frame[NUMERIC_FEATURES]).astype(np.float32),
            dtype=torch.float32,
        )
        self.labels = torch.tensor(
            self.frame.label.to_numpy(dtype=np.int64),
            dtype=torch.long,
        )
        print(f"{name}: {len(self.frame)} pairs tokenized.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.encodings.items()}
        item["numeric_features"] = self.numeric[index]
        item["labels"] = self.labels[index]
        return item


class HybridTransformer(nn.Module):
    def __init__(self, base_model, num_numeric_features):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model, local_files_only=True)
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
        self, input_ids, attention_mask, numeric_features, labels=None
    ):
        output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # DistilBERT has no pooler; use the first token representation.
        text_features = output.last_hidden_state[:, 0, :]
        numeric_features = self.numeric_projection(numeric_features)
        combined = torch.cat([text_features, numeric_features], dim=1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


@torch.no_grad()
def predict(model, loader):
    model.eval()
    scores = []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            numeric_features=batch["numeric_features"],
        )
        scores.extend(
            torch.softmax(output["logits"], dim=1)[:, 1].cpu().tolist()
        )
    return np.asarray(scores)


def main():
    started = time.perf_counter()
    set_seed()

    print("Device:", DEVICE)
    if not BASE_MODEL.exists():
        raise FileNotFoundError(
            "Missing local Ditto checkpoint at "
            f"{BASE_MODEL}. Run src/train_ditto_transformer.py first."
        )
    print("Backbone:", BASE_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    frames = build_candidates()
    train, val, test = (
        frames["train"], frames["validation"], frames["test"]
    )

    print(
        f"Candidates: train={len(train)}, validation={len(val)}, "
        f"test={len(test)}"
    )
    print(
        f"Positive pairs: train={int(train.label.sum())}, "
        f"validation={int(val.label.sum())}, test={int(test.label.sum())}"
    )

    # Fit preprocessing on training data only to avoid validation/test leakage.
    scaler = StandardScaler()
    scaler.fit(train[NUMERIC_FEATURES])
    train_ds = PairDataset(train, tokenizer, scaler, "train")
    val_ds = PairDataset(val, tokenizer, scaler, "validation")
    test_ds = PairDataset(test, tokenizer, scaler, "test")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = HybridTransformer(BASE_MODEL, len(NUMERIC_FEATURES)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(1, int(total_steps * 0.1)), total_steps
    )

    best_state, best_val_loss = None, float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                numeric_features=batch["numeric_features"],
                labels=batch["labels"],
            )
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(output["loss"].item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    numeric_features=batch["numeric_features"],
                    labels=batch["labels"],
                )
                val_losses.append(float(output["loss"].item()))

        val_loss = float(np.mean(val_losses))
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss={np.mean(losses):.4f} | "
            f"validation loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training failed to produce a model.")
    model.load_state_dict(best_state)

    val_scores = predict(model, val_loader)
    val_scored = val[["ledger_id", "bank_id", "label"]].copy()
    val_scored["probability"] = val_scores

    val_expected = expected_pairs(val)
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in THRESHOLDS:
        pred = greedy_one_to_one(val_scored, float(threshold))
        metrics = evaluate_matches(pred, val_expected)
        if (
            metrics["f1"] > best_f1
            or (np.isclose(metrics["f1"], best_f1)
                and threshold > best_threshold)
        ):
            best_f1, best_threshold = metrics["f1"], float(threshold)

    test_scores = predict(model, test_loader)
    test_scored = test[["ledger_id", "bank_id", "label"]].copy()
    test_scored["probability"] = test_scores
    test_pred = greedy_one_to_one(test_scored, best_threshold)
    test_metrics = evaluate_matches(test_pred, expected_pairs(test))

    result = {
        "model": MODEL_TAG,
        "base_model": str(BASE_MODEL),
        "device": str(DEVICE),
        "train_candidates": len(train),
        "validation_candidates": len(val),
        "test_candidates": len(test),
        "train_positive_pairs": int(train.label.sum()),
        "validation_positive_pairs": int(val.label.sum()),
        "test_positive_pairs": int(test.label.sum()),
        "best_validation_loss": best_val_loss,
        "validation_best_f1": best_f1,
        "threshold_selected_on_validation": best_threshold,
        "average_precision": average_precision_score(
            test.label, test_scores
        ),
        "fit_and_score_seconds": round(time.perf_counter() - started, 3),
        **test_metrics,
    }

    pd.DataFrame([result]).to_csv(
        REPORTS / "hybrid_transformer_model_comparison.csv", index=False
    )
    pd.DataFrame([{
        "model": MODEL_TAG,
        "threshold": best_threshold,
        "validation_f1": best_f1,
    }]).to_csv(
        REPORTS / "hybrid_transformer_validation_threshold.csv", index=False
    )
    test_scored.to_csv(
        REPORTS / "hybrid_transformer_test_scores.csv", index=False
    )
    test_pred.to_csv(
        REPORTS / "hybrid_transformer_test_matches.csv", index=False
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "model_state.pt")
    tokenizer.save_pretrained(MODEL_DIR)
    scaler_path = MODEL_DIR / "numeric_scaler.npz"
    np.savez(
        scaler_path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        features=np.asarray(NUMERIC_FEATURES),
    )
    pd.Series([{
        "threshold": best_threshold,
        "model": MODEL_TAG,
        "base_model": str(BASE_MODEL),
    }]).to_json(
        REPORTS / "hybrid_transformer_config.json",
        orient="records",
        indent=2,
    )

    print("\nHybrid Transformer test result:")
    print(pd.DataFrame([result]).to_string(index=False))
    print("\nSaved new model to:", MODEL_DIR)
    print("Saved reports under reports/hybrid_transformer_*")


if __name__ == "__main__":
    main()
