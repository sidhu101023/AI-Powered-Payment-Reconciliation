
from pathlib import Path
import re
import time
import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MODEL_DIR = ROOT / "models" / "ditto_transformer_model"
REPORTS.mkdir(parents=True, exist_ok=True)

BANK_FILE = DATA / "bank_lines.csv"
LEDGER_FILE = DATA / "ledger_entries.csv"
KEY_FILE = DATA / "match_key.csv"

MODEL_NAME = "distilbert/distilbert-base-uncased"
MODEL_TAG = "Ditto_Inspired_Transformer"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 1
PATIENCE = 1
LEARNING_RATE = 2e-5
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(value):
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def expected_pairs(frame):
    return set(
        zip(
            frame.loc[frame["label"] == 1, "ledger_id"].astype(int),
            frame.loc[frame["label"] == 1, "bank_id"].astype(int),
        )
    )


def greedy_one_to_one(scored, threshold):
    candidates = scored[scored["probability"] >= threshold].copy()
    candidates = candidates.sort_values(
        "probability", ascending=False
    )

    chosen = []
    used_ledger = set()
    used_bank = set()

    for row in candidates.itertuples(index=False):
        lid = int(row.ledger_id)
        bid = int(row.bank_id)

        if lid in used_ledger or bid in used_bank:
            continue

        used_ledger.add(lid)
        used_bank.add(bid)
        chosen.append({
            "ledger_id": lid,
            "bank_id": bid,
            "label": int(row.label),
            "probability": float(row.probability),
            "model": MODEL_TAG,
        })

    return pd.DataFrame(
        chosen,
        columns=[
            "ledger_id", "bank_id", "label",
            "probability", "model",
        ],
    )


def evaluate_matches(predicted, expected):
    predicted_pairs = set(
        zip(predicted["ledger_id"], predicted["bank_id"])
    ) if not predicted.empty else set()

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

    bank["id"] = bank["id"].astype(int)
    ledger["id"] = ledger["id"].astype(int)

    bank["account_key"] = bank["account"].astype(str)
    ledger["account_key"] = ledger["book"].astype(str)

    bank["parsed_date"] = pd.to_datetime(
        bank["posted_date"], errors="coerce"
    )
    ledger["parsed_date"] = pd.to_datetime(
        ledger["entry_date"], errors="coerce"
    )

    keys["kind"] = keys["kind"].astype(str).str.lower()
    valid_keys = keys[keys["kind"] != "duplicate_line"]

    positive_pairs = set(zip(
        valid_keys["ledger_entry_no"].astype(int),
        valid_keys["bank_line_no"].astype(int),
    ))

    all_dates = pd.concat(
        [ledger["parsed_date"], bank["parsed_date"]],
        ignore_index=True,
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

    ledger["period"] = ledger["parsed_date"].map(assign_period)
    bank["period"] = bank["parsed_date"].map(assign_period)

    assignments = {"ledger": {}, "bank": {}}

    for lid, bid in positive_pairs:
        lrow = ledger.loc[ledger["id"] == lid]
        brow = bank.loc[bank["id"] == bid]

        if lrow.empty or brow.empty:
            raise ValueError(f"Missing positive pair: {lid}, {bid}")

        dates = [
            d for d in [
                lrow.iloc[0]["parsed_date"],
                brow.iloc[0]["parsed_date"],
            ] if pd.notna(d)
        ]

        if not dates:
            raise ValueError(f"Both dates missing for pair {lid}, {bid}")

        period = assign_period(max(dates))

        for side, record_id in [("ledger", lid), ("bank", bid)]:
            old = assignments[side].get(record_id)
            if old is not None and old != period:
                raise ValueError(
                    f"Record {side}:{record_id} crosses partitions."
                )
            assignments[side][record_id] = period

    ledger["period"] = ledger.apply(
        lambda r: assignments["ledger"].get(
            int(r["id"]), r["period"]
        ),
        axis=1,
    )
    bank["period"] = bank.apply(
        lambda r: assignments["bank"].get(
            int(r["id"]), r["period"]
        ),
        axis=1,
    )

    bank_groups = {
        k: g for k, g in bank.groupby("account_key")
    }

    rows = []

    for l in ledger.to_dict("records"):
        period = l["period"]
        if period == "excluded":
            continue

        group = bank_groups.get(l["account_key"])
        if group is None:
            continue

        group = group[group["period"] == period]

        for b in group.to_dict("records"):
            reference = str(l.get("reference", ""))
            counterparty = str(l.get("counterparty", ""))
            narrative = str(b.get("narrative", ""))

            amount_diff = abs(
                float(l["amount"]) - float(b["amount"])
            )

            ld, bd = l["parsed_date"], b["parsed_date"]
            day_diff = (
                (bd - ld).days
                if pd.notna(ld) and pd.notna(bd)
                else 9999
            )

            # Pair serialization for the Transformer classifier.
            left_text = (
                f"reference {reference}; "
                f"counterparty {counterparty}; "
                f"ledger amount {l['amount']}; "
                f"ledger date {l.get('entry_date', '')}"
            )

            right_text = (
                f"bank narrative {narrative}; "
                f"bank amount {b['amount']}; "
                f"bank date {b.get('posted_date', '')}"
            )

            rows.append({
                "ledger_id": int(l["id"]),
                "bank_id": int(b["id"]),
                "period": period,
                "label": int(
                    (int(l["id"]), int(b["id"])) in positive_pairs
                ),
                "left_text": left_text,
                "right_text": right_text,
                "amount_difference": amount_diff,
                "signed_date_difference": day_diff,
            })

    pairs = pd.DataFrame(rows)

    if pairs.empty:
        raise ValueError("No candidate pairs were generated.")

    frames = {
        name: pairs[pairs["period"] == name].copy()
        for name in ["train", "validation", "test"]
    }

    for name, frame in frames.items():
        if frame.empty or frame["label"].nunique() < 2:
            raise ValueError(
                f"{name} needs candidates from both label classes."
            )

    return frames


class PairDataset(Dataset):
    def __init__(self, frame, tokenizer, name="dataset"):
        self.frame = frame.reset_index(drop=True)

        print(
            f"Tokenizing {name}: {len(self.frame)} candidate pairs...",
            flush=True,
        )

        self.encodings = tokenizer(
            self.frame["left_text"].tolist(),
            self.frame["right_text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        self.labels = torch.tensor(
            self.frame["label"].to_numpy(dtype=np.int64),
            dtype=torch.long,
        )

        print(f"Finished tokenizing {name}.", flush=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        return item



@torch.no_grad()
def predict(model, loader):
    model.eval()
    scores = []

    for batch in loader:
        batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
            if key != "labels"
        }

        logits = model(**batch).logits
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        scores.extend(probabilities.cpu().numpy().tolist())

    return np.asarray(scores)


def main():
    started = time.perf_counter()
    set_seed()

    print(f"Device: {DEVICE}")
    print(f"Loading tokenizer/model: {MODEL_NAME}")
    print("The first run may download pretrained model files.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    frames = build_candidates()

    train = frames["train"]
    validation = frames["validation"]
    test = frames["test"]

    print(
        f"Candidates: train={len(train)}, "
        f"validation={len(validation)}, test={len(test)}"
    )
    print(
        f"Positive pairs: train={int(train['label'].sum())}, "
        f"validation={int(validation['label'].sum())}, "
        f"test={int(test['label'].sum())}"
    )

    train_dataset = PairDataset(train, tokenizer, "train")
    val_dataset = PairDataset(validation, tokenizer, "validation")
    test_dataset = PairDataset(test, tokenizer, "test")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE
    )

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()
            scheduler.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    key: value.to(DEVICE)
                    for key, value in batch.items()
                }
                output = model(**batch)
                val_losses.append(float(output.loss.item()))

        val_loss = float(np.mean(val_losses))

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss={np.mean(train_losses):.4f} | "
            f"validation loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model.")

    model.load_state_dict(best_state)

    val_scores = predict(model, val_loader)
    test_scores = predict(model, test_loader)

    validation_scored = validation[
        ["ledger_id", "bank_id", "label"]
    ].copy()
    validation_scored["probability"] = val_scores

    val_expected = expected_pairs(validation)
    best_threshold, best_f1 = 0.5, -1.0

    for threshold in THRESHOLDS:
        predicted = greedy_one_to_one(
            validation_scored, float(threshold)
        )
        metrics = evaluate_matches(predicted, val_expected)

        if (
            metrics["f1"] > best_f1
            or (
                np.isclose(metrics["f1"], best_f1)
                and threshold > best_threshold
            )
        ):
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)

    test_scored = test[
        ["ledger_id", "bank_id", "label"]
    ].copy()
    test_scored["probability"] = test_scores
    test_scored["model"] = MODEL_TAG

    predicted_test = greedy_one_to_one(
        test_scored, best_threshold
    )
    test_metrics = evaluate_matches(
        predicted_test, expected_pairs(test)
    )

    result = {
        "model": MODEL_TAG,
        "base_model": MODEL_NAME,
        "device": str(DEVICE),
        "train_candidates": len(train),
        "validation_candidates": len(validation),
        "test_candidates": len(test),
        "train_positive_pairs": int(train["label"].sum()),
        "validation_positive_pairs": int(validation["label"].sum()),
        "test_positive_pairs": int(test["label"].sum()),
        "best_validation_loss": best_val_loss,
        "validation_best_f1": best_f1,
        "threshold_selected_on_validation": best_threshold,
        "average_precision": average_precision_score(
            test["label"], test_scores
        ),
        "fit_and_score_seconds": round(
            time.perf_counter() - started, 3
        ),
        **test_metrics,
    }

    pd.DataFrame([result]).to_csv(
        REPORTS / "ditto_transformer_model_comparison.csv",
        index=False,
    )

    pd.DataFrame([{
        "model": MODEL_TAG,
        "threshold": best_threshold,
        "validation_f1": best_f1,
    }]).to_csv(
        REPORTS / "ditto_transformer_validation_threshold.csv",
        index=False,
    )

    test_scored.to_csv(
        REPORTS / "ditto_transformer_test_scores.csv",
        index=False,
    )
    predicted_test.to_csv(
        REPORTS / "ditto_transformer_test_matches.csv",
        index=False,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    pd.DataFrame([{
        "threshold": best_threshold,
        "model": MODEL_TAG,
    }]).to_json(
        REPORTS / "ditto_transformer_config.json",
        orient="records",
        indent=2,
    )

    print("\nTest result:")
    print(pd.DataFrame([result]).to_string(index=False))
    print("\nSaved Transformer results under reports/ditto_transformer_*")
    print("Saved local Ditto checkpoint to:", MODEL_DIR)


if __name__ == "__main__":
    main()
