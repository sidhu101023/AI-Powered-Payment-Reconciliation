
from pathlib import Path
import re
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cpu")

MAX_LEN = 40
EMBED_DIM = 48
HIDDEN_DIM = 32
BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 6
LEARNING_RATE = 0.001
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


def normalize(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def tokenize(value):
    return normalize(value).split()


def similarity(a, b):
    from difflib import SequenceMatcher
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def greedy_one_to_one(frame, threshold):
    eligible = frame[frame["probability"] >= threshold].copy()
    eligible = eligible.sort_values(
        ["probability", "ledger_id", "bank_id"],
        ascending=[False, True, True],
    )

    used_ledger, used_bank, selected = set(), set(), []
    for row in eligible.itertuples(index=False):
        if row.ledger_id in used_ledger or row.bank_id in used_bank:
            continue
        used_ledger.add(row.ledger_id)
        used_bank.add(row.bank_id)
        selected.append({
            "ledger_id": int(row.ledger_id),
            "bank_id": int(row.bank_id),
            "probability": float(row.probability),
            "label": int(row.label),
        })

    return pd.DataFrame(
        selected,
        columns=["ledger_id", "bank_id", "probability", "label"],
    )


def evaluate_matches(predicted, expected):
    actual = set(zip(predicted["ledger_id"], predicted["bank_id"]))
    tp = len(actual & expected)
    fp = len(actual - expected)
    fn = len(expected - actual)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "predicted_matches": len(actual),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def expected_pairs(frame):
    positive = frame[frame["label"] == 1]
    return set(zip(
        positive["ledger_id"].astype(int),
        positive["bank_id"].astype(int),
    ))


class PairRNN(nn.Module):
    def __init__(self, vocab_size, numeric_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, EMBED_DIM, padding_idx=0
        )
        self.encoder = nn.GRU(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 4 + numeric_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def encode(self, tokens):
        embedded = self.embedding(tokens)
        _, hidden = self.encoder(embedded)
        # Final forward and backward states.
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def forward(self, left_tokens, right_tokens, numeric):
        left = self.encode(left_tokens)
        right = self.encode(right_tokens)
        features = torch.cat([left, right, numeric], dim=1)
        return self.classifier(features).squeeze(1)


def build_vocab(texts):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode_text(text, vocab):
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokenize(text)]
    ids = ids[:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    return ids


def prepare_data(frame, vocab, scaler, fit_scaler=False):
    left = np.asarray([
        encode_text(text, vocab) for text in frame["left_text"]
    ], dtype=np.int64)
    right = np.asarray([
        encode_text(text, vocab) for text in frame["right_text"]
    ], dtype=np.int64)

    raw_numeric = frame[NUMERIC_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).to_numpy(dtype=np.float32)

    numeric = (
        scaler.fit_transform(raw_numeric)
        if fit_scaler else scaler.transform(raw_numeric)
    ).astype(np.float32)

    labels = frame["label"].to_numpy(dtype=np.float32)
    return left, right, numeric, labels


def predict(model, arrays):
    left, right, numeric, _ = arrays
    model.eval()
    probabilities = []

    with torch.no_grad():
        for start in range(0, len(left), BATCH_SIZE):
            stop = start + BATCH_SIZE
            logits = model(
                torch.tensor(left[start:stop], dtype=torch.long),
                torch.tensor(right[start:stop], dtype=torch.long),
                torch.tensor(numeric[start:stop], dtype=torch.float32),
            )
            probabilities.extend(torch.sigmoid(logits).cpu().numpy())

    return np.asarray(probabilities)


def main():
    started = time.perf_counter()

    bank = pd.read_csv(DATA / "bank_lines.csv")
    ledger = pd.read_csv(DATA / "ledger_entries.csv")
    keys = pd.read_csv(DATA / "match_key.csv")

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

    # Same chronological split design as the existing MLP experiment.
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

    # Keep every known positive pair in one partition.
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
        ), axis=1
    )
    bank["period"] = bank.apply(
        lambda r: assignments["bank"].get(
            int(r["id"]), r["period"]
        ), axis=1
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
            amount_diff = abs(float(l["amount"]) - float(b["amount"]))
            ld, bd = l["parsed_date"], b["parsed_date"]
            day_diff = (bd - ld).days if pd.notna(ld) and pd.notna(bd) else 9999

            reference = str(l.get("reference", ""))
            counterparty = str(l.get("counterparty", ""))
            narrative = str(b.get("narrative", ""))

            ref_norm = normalize(reference).replace(" ", "")
            narrative_norm = normalize(narrative).replace(" ", "")

            rows.append({
                "ledger_id": int(l["id"]),
                "bank_id": int(b["id"]),
                "period": period,
                "label": int((int(l["id"]), int(b["id"])) in positive_pairs),
                "left_text": (
                    f"reference {reference} counterparty {counterparty}"
                ),
                "right_text": f"bank narrative {narrative}",
                "amount_difference": amount_diff,
                "exact_amount": int(amount_diff < 0.005),
                "amount_within_cent": int(amount_diff <= 0.01),
                "same_amount_sign": int(
                    (float(l["amount"]) < 0 and float(b["amount"]) < 0)
                    or (float(l["amount"]) >= 0 and float(b["amount"]) >= 0)
                ),
                "signed_date_difference": day_diff,
                "absolute_date_difference": abs(day_diff),
                "reference_in_narrative": int(
                    bool(ref_norm) and ref_norm in narrative_norm
                ),
                "reference_similarity": similarity(reference, narrative),
                "counterparty_similarity": similarity(counterparty, narrative),
            })

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise ValueError("No candidate pairs were generated.")

    train = pairs[pairs["period"] == "train"].copy()
    validation = pairs[pairs["period"] == "validation"].copy()
    test = pairs[pairs["period"] == "test"].copy()

    for name, frame in [
        ("train", train), ("validation", validation), ("test", test)
    ]:
        if frame.empty or frame["label"].nunique() < 2:
            raise ValueError(
                f"{name} requires candidates with both label classes."
            )

    # Vocabulary is built from training text only.
    vocab = build_vocab(
        train["left_text"].tolist() + train["right_text"].tolist()
    )
    scaler = StandardScaler()

    train_arrays = prepare_data(train, vocab, scaler, fit_scaler=True)
    val_arrays = prepare_data(validation, vocab, scaler)
    test_arrays = prepare_data(test, vocab, scaler)

    def make_loader(arrays, shuffle):
        left, right, numeric, labels = arrays
        dataset = TensorDataset(
            torch.tensor(left, dtype=torch.long),
            torch.tensor(right, dtype=torch.long),
            torch.tensor(numeric, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )
        return DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=shuffle
        )

    train_loader = make_loader(train_arrays, True)

    model = PairRNN(len(vocab), len(NUMERIC_FEATURES)).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    positives = max(float(train["label"].sum()), 1.0)
    negatives = max(float(len(train) - train["label"].sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives])
    )

    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0

    val_left, val_right, val_numeric, val_labels = val_arrays
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        for left, right, numeric, labels in train_loader:
            optimizer.zero_grad()
            logits = model(left, right, numeric)
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(
                torch.tensor(val_left, dtype=torch.long),
                torch.tensor(val_right, dtype=torch.long),
                torch.tensor(val_numeric, dtype=torch.float32),
            )
            val_loss = float(loss_fn(
                val_logits,
                torch.tensor(val_labels, dtype=torch.float32),
            ).item())

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {np.mean(epoch_losses):.4f} | "
            f"validation loss {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
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

    val_scores = predict(model, val_arrays)
    test_scores = predict(model, test_arrays)

    validation_scored = validation[
        ["ledger_id", "bank_id", "label"]
    ].copy()
    validation_scored["probability"] = val_scores

    val_expected = expected_pairs(validation)
    best_threshold, best_f1 = 0.5, -1.0

    for threshold in THRESHOLDS:
        predicted = greedy_one_to_one(validation_scored, threshold)
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
    test_scored["model"] = "DeepMatcher_Inspired_RNN"

    predicted_test = greedy_one_to_one(test_scored, best_threshold)
    test_metrics = evaluate_matches(predicted_test, expected_pairs(test))

    result = {
        "model": "DeepMatcher_Inspired_RNN",
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
        "fit_and_score_seconds": round(time.perf_counter() - started, 3),
        **test_metrics,
    }

    pd.DataFrame([result]).to_csv(
        REPORTS / "deepmatcher_rnn_model_comparison.csv", index=False
    )
    pd.DataFrame([{
        "model": "DeepMatcher_Inspired_RNN",
        "threshold": best_threshold,
        "validation_f1": best_f1,
    }]).to_csv(
        REPORTS / "deepmatcher_rnn_validation_threshold.csv", index=False
    )
    test_scored.to_csv(
        REPORTS / "deepmatcher_rnn_test_scores.csv", index=False
    )
    predicted_test.to_csv(
        REPORTS / "deepmatcher_rnn_test_matches.csv", index=False
    )
    torch.save({
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "config": {
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "max_len": MAX_LEN,
            "numeric_features": NUMERIC_FEATURES,
        },
        "threshold": best_threshold,
    }, REPORTS / "deepmatcher_rnn_model.pt")

    print("\nTest result:")
    print(pd.DataFrame([result]).to_string(index=False))
    print("\nSaved RNN results under reports/deepmatcher_rnn_*")


if __name__ == "__main__":
    main()
