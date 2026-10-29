
from pathlib import Path
from difflib import SequenceMatcher
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# --------------------------------------------------
# 1. Load and prepare records
# --------------------------------------------------
bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
keys = pd.read_csv(DATA / "match_key.csv")

bank["id"] = bank["id"].astype(int)
ledger["id"] = ledger["id"].astype(int)
bank["posted_date"] = pd.to_datetime(bank["posted_date"])
ledger["entry_date"] = pd.to_datetime(ledger["entry_date"])

keys = keys[
    keys["kind"].astype(str).str.lower() != "duplicate_line"
]

positive_pairs = set(
    zip(
        keys["ledger_entry_no"].astype(int),
        keys["bank_line_no"].astype(int),
    )
)

# Use record dates, not pair dates, to define chronological periods.
all_dates = pd.concat(
    [ledger["entry_date"], bank["posted_date"]],
    ignore_index=True,
).dropna()

train_cutoff = all_dates.quantile(0.60)
validation_cutoff = all_dates.quantile(0.80)

train_ledger = ledger[ledger["entry_date"] < train_cutoff]
train_bank = bank[bank["posted_date"] < train_cutoff]

validation_ledger = ledger[
    (ledger["entry_date"] >= train_cutoff)
    & (ledger["entry_date"] < validation_cutoff)
]
validation_bank = bank[
    (bank["posted_date"] >= train_cutoff)
    & (bank["posted_date"] < validation_cutoff)
]

test_ledger = ledger[ledger["entry_date"] >= validation_cutoff]
test_bank = bank[bank["posted_date"] >= validation_cutoff]

def normalize(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())

def similarity(a, b):
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

numeric_features = [
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

def make_pairs(ledger_rows, bank_rows):
    rows = []

    for l in ledger_rows.itertuples(index=False):
        for b in bank_rows.itertuples(index=False):
            lid, bid = int(l.id), int(b.id)
            reference = str(getattr(l, "reference", ""))
            counterparty = str(getattr(l, "counterparty", ""))
            narrative = str(getattr(b, "narrative", ""))

            la, ba = float(l.amount), float(b.amount)
            gap = abs(la - ba)
            days = (b.posted_date - l.entry_date).days

            ref_norm = normalize(reference)
            narrative_norm = normalize(narrative)

            rows.append({
                "ledger_id": lid,
                "bank_id": bid,
                "text": (
                    f"ledger reference {reference} "
                    f"counterparty {counterparty} "
                    f"bank narrative {narrative}"
                ),
                "amount_difference": gap,
                "exact_amount": int(gap < 0.005),
                "amount_within_cent": int(gap <= 0.01),
                "same_amount_sign": int(
                    (la < 0 and ba < 0) or (la >= 0 and ba >= 0)
                ),
                "signed_date_difference": days,
                "absolute_date_difference": abs(days),
                "reference_in_narrative": int(
                    bool(ref_norm) and ref_norm in narrative_norm
                ),
                "reference_similarity": similarity(reference, narrative),
                "counterparty_similarity": similarity(
                    counterparty, narrative
                ),
                "label": int((lid, bid) in positive_pairs),
            })

    return pd.DataFrame(rows)

train = make_pairs(train_ledger, train_bank)
validation = make_pairs(validation_ledger, validation_bank)
test = make_pairs(test_ledger, test_bank)

if train.empty or validation.empty or test.empty:
    raise ValueError("One of the chronological candidate sets is empty.")

if train["label"].sum() == 0:
    raise ValueError("Training set contains no positive pairs.")
if validation["label"].sum() == 0:
    raise ValueError("Validation set contains no positive pairs.")
if test["label"].sum() == 0:
    raise ValueError("Test set contains no positive pairs.")

# --------------------------------------------------
# 2. Fit on training data only
# --------------------------------------------------
features = ColumnTransformer([
    (
        "text",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
            strip_accents="unicode",
        ),
        "text",
    ),
    ("numeric", StandardScaler(), numeric_features),
])

model = Pipeline([
    ("features", features),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

model.fit(train, train["label"])

validation["probability"] = model.predict_proba(
    validation
)[:, 1]
test["probability"] = model.predict_proba(test)[:, 1]

# --------------------------------------------------
# 3. One-to-one assignment and metrics
# --------------------------------------------------
def greedy_one_to_one(candidates, threshold):
    candidates = candidates.sort_values(
        ["probability", "ledger_id", "bank_id"],
        ascending=[False, True, True],
    )

    selected = []
    used_ledger, used_bank = set(), set()

    for row in candidates.itertuples(index=False):
        if row.probability < threshold:
            break
        if row.ledger_id in used_ledger or row.bank_id in used_bank:
            continue

        selected.append((int(row.ledger_id), int(row.bank_id)))
        used_ledger.add(int(row.ledger_id))
        used_bank.add(int(row.bank_id))

    return set(selected)

def evaluate(candidates, threshold):
    expected = set(
        zip(
            candidates.loc[candidates["label"] == 1, "ledger_id"].astype(int),
            candidates.loc[candidates["label"] == 1, "bank_id"].astype(int),
        )
    )

    predicted = greedy_one_to_one(candidates, threshold)

    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "threshold": threshold,
        "expected_matches": len(expected),
        "predicted_matches": len(predicted),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# --------------------------------------------------
# 4. Tune threshold using validation only
# --------------------------------------------------
thresholds = np.arange(0.05, 1.00, 0.05)
validation_results = pd.DataFrame([
    evaluate(validation, round(float(t), 2))
    for t in thresholds
])

validation_results.to_csv(
    REPORTS / "hybrid_validation_thresholds.csv", index=False
)

# Select highest validation F1. Ties prefer higher precision,
# then the higher threshold.
best = validation_results.sort_values(
    ["f1", "precision", "threshold"],
    ascending=[False, False, False],
).iloc[0]

best_threshold = float(best["threshold"])

# Only now evaluate the final test period.
test_result = evaluate(test, best_threshold)
test_result.update({
    "model": "Hybrid_TFIDF_Structured_LogisticRegression",
    "train_cutoff": str(train_cutoff.date()),
    "validation_cutoff": str(validation_cutoff.date()),
    "selected_threshold": best_threshold,
    "train_ledger": len(train_ledger),
    "train_bank": len(train_bank),
    "validation_ledger": len(validation_ledger),
    "validation_bank": len(validation_bank),
    "test_ledger": len(test_ledger),
    "test_bank": len(test_bank),
    "validation_average_precision": average_precision_score(
        validation["label"], validation["probability"]
    ),
    "test_average_precision": average_precision_score(
        test["label"], test["probability"]
    ),
})

pd.DataFrame([test_result]).to_csv(
    REPORTS / "hybrid_final_test_evaluation.csv", index=False
)

print("=== CHRONOLOGICAL HYBRID VALIDATION ===")
print("Training cutoff:", train_cutoff.date())
print("Validation cutoff:", validation_cutoff.date())
print(
    f"Record counts — train L/B: {len(train_ledger)}/{len(train_bank)}, "
    f"validation L/B: {len(validation_ledger)}/{len(validation_bank)}, "
    f"test L/B: {len(test_ledger)}/{len(test_bank)}"
)
print("\nBest validation threshold:", best_threshold)
print("Best validation F1:", round(float(best["f1"]), 4))
print("\n=== FINAL TEST RESULTS ===")
print(pd.DataFrame([test_result]).to_string(index=False))
print("\nSaved:")
print("reports/hybrid_validation_thresholds.csv")
print("reports/hybrid_final_test_evaluation.csv")
