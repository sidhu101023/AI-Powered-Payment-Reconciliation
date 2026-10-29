
from pathlib import Path
from difflib import SequenceMatcher
import re
import time

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
match_key = pd.read_csv(DATA / "match_key.csv")

# Exclude duplicate_line labels: identical bank records create
# an ambiguous one-to-one assignment for this experiment.
valid_keys = match_key[
    match_key["kind"].astype(str).str.lower() != "duplicate_line"
]

positive_pairs = set(
    zip(
        valid_keys["ledger_entry_no"].astype(int),
        valid_keys["bank_line_no"].astype(int),
    )
)

def normalize(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def text_similarity(a, b):
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def make_features(l, b):
    ledger_date = pd.to_datetime(l["entry_date"], errors="coerce")
    bank_date = pd.to_datetime(b["posted_date"], errors="coerce")

    if pd.isna(ledger_date) or pd.isna(bank_date):
        signed_days = 9999
    else:
        signed_days = (bank_date - ledger_date).days

    ledger_amount = float(l["amount"])
    bank_amount = float(b["amount"])
    amount_diff = abs(ledger_amount - bank_amount)

    reference = normalize(l["reference"])
    narrative = normalize(b["narrative"])
    counterparty = normalize(l["counterparty"])

    return {
        "amount_difference": amount_diff,
        "exact_amount": int(amount_diff < 0.005),
        "amount_within_cent": int(amount_diff <= 0.01),
        "signed_date_difference": signed_days,
        "absolute_date_difference": abs(signed_days),
        "reference_in_narrative": int(
            bool(reference) and reference in narrative
        ),
        "reference_similarity": text_similarity(
            l["reference"], b["narrative"]
        ),
        "counterparty_similarity": text_similarity(
            l["counterparty"], b["narrative"]
        ),
    }


# Generate candidate pairs within the same account/book.
# The answer key is used ONLY to construct labels.
ledger_by_book = {
    str(book): group
    for book, group in ledger.groupby(
        ledger["book"].astype(str), sort=False
    )
}

bank_by_account = {
    str(account): group
    for account, group in bank.groupby(
        bank["account"].astype(str), sort=False
    )
}

rows = []

for account, bank_group in bank_by_account.items():
    ledger_group = ledger_by_book.get(account)

    if ledger_group is None:
        continue

    for l in ledger_group.to_dict("records"):
        for b in bank_group.to_dict("records"):
            features = make_features(l, b)
            features["ledger_id"] = int(l["id"])
            features["bank_id"] = int(b["id"])
            features["ledger_date"] = l["entry_date"]
            features["bank_date"] = b["posted_date"]
            features["label"] = int(
                (int(l["id"]), int(b["id"])) in positive_pairs
            )
            rows.append(features)

pairs = pd.DataFrame(rows)

if pairs.empty:
    raise ValueError(
        "No candidate pairs generated. Check how ledger book names "
        "correspond to bank account names."
    )

feature_columns = [
    "amount_difference",
    "exact_amount",
    "amount_within_cent",
    "signed_date_difference",
    "absolute_date_difference",
    "reference_in_narrative",
    "reference_similarity",
    "counterparty_similarity",
]

# Split by ledger record so pairs for the same ledger entry cannot

# Temporal holdout: both records in a candidate pair must belong
# entirely to the earlier training period or later test period.
pairs["ledger_date"] = pd.to_datetime(pairs["ledger_date"])
pairs["bank_date"] = pd.to_datetime(pairs["bank_date"])

all_dates = pd.concat(
    [
        pd.to_datetime(ledger["entry_date"]),
        pd.to_datetime(bank["posted_date"]),
    ],
    ignore_index=True,
).dropna()

cutoff = all_dates.quantile(0.75)

train_mask = (
    (pairs["ledger_date"] < cutoff)
    & (pairs["bank_date"] < cutoff)
)
test_mask = (
    (pairs["ledger_date"] >= cutoff)
    & (pairs["bank_date"] >= cutoff)
)

train = pairs.loc[train_mask].copy()
test = pairs.loc[test_mask].copy()

print(f"Temporal cutoff: {cutoff.date()}")
print(f"Candidate pairs excluded at boundary: "
      f"{len(pairs) - len(train) - len(test)}")

if train.empty or test.empty:
    raise ValueError("Temporal split produced an empty partition.")

feature_columns = [
    "amount_difference",
    "exact_amount",
    "amount_within_cent",
    "signed_date_difference",
    "absolute_date_difference",
    "reference_in_narrative",
    "reference_similarity",
    "counterparty_similarity",
]

X_train = train[feature_columns]
y_train = train["label"]
X_test = test[feature_columns]
y_test = test["label"]

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "A partition contains only one label class. "
        "The temporal split needs adjustment."
    )


models = {
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
score_frames = []

for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results.append({
        "model": name,
        "train_pairs": len(train),
        "test_pairs": len(test),
        "test_positive_pairs": int(y_test.sum()),
        "precision_at_0.5": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall_at_0.5": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_at_0.5": f1_score(
            y_test, predictions, zero_division=0
        ),
        "average_precision": average_precision_score(
            y_test, probabilities
        ),
        "training_seconds": round(train_seconds, 3),
    })

    scores = test[["ledger_id", "bank_id", "label"]].copy()
    scores["model"] = name
    scores["probability"] = probabilities
    scores["predicted_label"] = predictions
    score_frames.append(scores)

results_df = pd.DataFrame(results)
results_path = REPORTS / "traditional_ml_evaluation.csv"
results_df.to_csv(results_path, index=False)

scores_path = REPORTS / "traditional_ml_test_scores.csv"
pd.concat(score_frames, ignore_index=True).to_csv(
    scores_path, index=False
)

print("=== TRADITIONAL ML EXPERIMENT ===")
print(f"Candidate pairs: {len(pairs)}")
print(f"Training pairs:  {len(train)}")
print(f"Testing pairs:   {len(test)}")
print(f"Training positives: {int(y_train.sum())}")
print(f"Testing positives:  {int(y_test.sum())}")
print("\n=== HELD-OUT RESULTS ===")
print(results_df.to_string(index=False))
print(f"\nSaved evaluation: {results_path}")
print(f"Saved test scores: {scores_path}")
