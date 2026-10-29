
from pathlib import Path
from difflib import SequenceMatcher
import re

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "reports"
OUTPUT.mkdir(exist_ok=True)

# Load source data and answer key.
bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
truth = pd.read_csv(DATA / "match_key.csv")

bank["posted_date"] = pd.to_datetime(bank["posted_date"])
ledger["entry_date"] = pd.to_datetime(ledger["entry_date"])

bank["amount"] = pd.to_numeric(bank["amount"])
ledger["amount"] = pd.to_numeric(ledger["amount"])

def normalize(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())

def similarity(left, right):
    return SequenceMatcher(
        None, normalize(left), normalize(right)
    ).ratio()

# All non-duplicate answer-key pairs are positive examples.
ordinary_truth = truth[truth["kind"] != "duplicate_line"]

positive_pairs = set(
    zip(
        ordinary_truth["ledger_entry_no"].astype(int),
        ordinary_truth["bank_line_no"].astype(int),
    )
)

# Duplicate-line pairs are exceptions, not ordinary training labels.
duplicate_pairs = set(
    zip(
        truth.loc[
            truth["kind"] == "duplicate_line",
            "ledger_entry_no",
        ].astype(int),
        truth.loc[
            truth["kind"] == "duplicate_line",
            "bank_line_no",
        ].astype(int),
    )
)

# Generate candidate pairs for the same account/book.
rows = []

for l in ledger.itertuples(index=False):
    for b in bank.itertuples(index=False):
        if str(l.book) != str(b.account):
            continue

        pair = (int(l.id), int(b.id))

        # Keep duplicate exceptions out of ordinary model training.
        if pair in duplicate_pairs:
            continue

        ref = normalize(l.reference)
        narrative = normalize(b.narrative)
        counterparty = normalize(l.counterparty)

        day_diff = (b.posted_date - l.entry_date).days
        amount_diff = round(abs(float(l.amount) - float(b.amount)), 2)

        rows.append({
            "ledger_id": int(l.id),
            "bank_id": int(b.id),
            "amount_difference": amount_diff,
            "amount_exact": int(amount_diff == 0),
            "amount_within_1_cent": int(amount_diff <= 0.01),
            "date_difference": day_diff,
            "absolute_date_difference": abs(day_diff),
            "reference_in_narrative": int(
                bool(ref) and ref in narrative
            ),
            "reference_similarity": similarity(
                l.reference, b.narrative
            ),
            "counterparty_similarity": similarity(
                l.counterparty, b.narrative
            ),
            "label": int(pair in positive_pairs),
        })

pairs = pd.DataFrame(rows)

if pairs.empty or pairs["label"].nunique() < 2:
    raise ValueError(
        "Candidate data must contain both positive and negative pairs."
    )

feature_columns = [
    "amount_difference",
    "amount_exact",
    "amount_within_1_cent",
    "date_difference",
    "absolute_date_difference",
    "reference_in_narrative",
    "reference_similarity",
    "counterparty_similarity",
]

X = pairs[feature_columns]
y = pairs["label"]
groups = pairs["ledger_id"]

# Keep all candidates for each ledger entry in one split.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42,
)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Evaluate held-out candidate pairs at a 0.5 probability threshold.
probabilities = model.predict_proba(X_test)[:, 1]
predicted_labels = (probabilities >= 0.5).astype(int)

print("=== AI MODEL: HELD-OUT PAIR EVALUATION ===")
print(f"Training candidates: {len(train_idx)}")
print(f"Testing candidates:  {len(test_idx)}")
print(f"Training positives:  {int(y_train.sum())}")
print(f"Testing positives:   {int(y_test.sum())}")
print(f"Precision: {precision_score(y_test, predicted_labels, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test, predicted_labels, zero_division=0):.3f}")
print(f"F1-score:  {f1_score(y_test, predicted_labels, zero_division=0):.3f}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    y_test,
    predicted_labels,
    target_names=["Non-match", "Match"],
    zero_division=0,
))

# Save held-out predictions for inspection.
test_results = pairs.iloc[test_idx][
    ["ledger_id", "bank_id", "label"]
].copy()

test_results["match_probability"] = probabilities
test_results["predicted_match"] = predicted_labels

test_path = OUTPUT / "ai_test_predictions.csv"
test_results.to_csv(test_path, index=False)

# Train on all available labeled candidates and score the full candidate set.
# These are in-sample scores, not an independent evaluation.
model.fit(X, y)

pairs["match_probability"] = model.predict_proba(X)[:, 1]
pairs.to_csv(OUTPUT / "ai_candidate_scores.csv", index=False)

print("\n=== FILES SAVED ===")
print(f"Held-out evaluation: {test_path}")
print(f"Candidate scores:    {OUTPUT / 'ai_candidate_scores.csv'}")
print("\nNote: Candidate-level predictions are not yet final one-to-one matches.")
