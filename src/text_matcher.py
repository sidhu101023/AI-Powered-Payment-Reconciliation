
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# 1. Load the data
bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
keys = pd.read_csv(DATA / "match_key.csv")

bank["id"] = bank["id"].astype(int)
ledger["id"] = ledger["id"].astype(int)

bank["posted_date"] = pd.to_datetime(bank["posted_date"])
ledger["entry_date"] = pd.to_datetime(ledger["entry_date"])

# Exclude duplicate_line labels: they represent duplicate physical
# bank records rather than additional unique transaction matches.
valid_keys = keys[
    keys["kind"].astype(str).str.lower() != "duplicate_line"
]
positive_pairs = set(
    zip(
        valid_keys["ledger_entry_no"].astype(int),
        valid_keys["bank_line_no"].astype(int),
    )
)

# 2. Reuse the same temporal cutoff as previous experiments.
all_dates = pd.concat(
    [ledger["entry_date"], bank["posted_date"]],
    ignore_index=True,
).dropna()
cutoff = all_dates.quantile(0.75)

train_ledger = ledger[ledger["entry_date"] < cutoff].copy()
train_bank = bank[bank["posted_date"] < cutoff].copy()
test_ledger = ledger[ledger["entry_date"] >= cutoff].copy()
test_bank = bank[bank["posted_date"] >= cutoff].copy()

def make_pairs(ledger_rows, bank_rows):
    """Create labelled candidate pairs and their serialized text."""
    pairs = []

    for l in ledger_rows.itertuples(index=False):
        for b in bank_rows.itertuples(index=False):
            ledger_id = int(l.id)
            bank_id = int(b.id)

            amount_gap = abs(float(l.amount) - float(b.amount))
            day_gap = (b.posted_date - l.entry_date).days

            text = (
                f"LEDGER_REFERENCE {getattr(l, 'reference', '')} "
                f"LEDGER_COUNTERPARTY {getattr(l, 'counterparty', '')} "
                f"LEDGER_AMOUNT {float(l.amount):.2f} "
                f"LEDGER_DATE {l.entry_date.date()} "
                f"BANK_NARRATIVE {getattr(b, 'narrative', '')} "
                f"BANK_AMOUNT {float(b.amount):.2f} "
                f"BANK_DATE {b.posted_date.date()} "
                f"AMOUNT_DIFFERENCE {amount_gap:.2f} "
                f"DATE_DIFFERENCE {day_gap}"
            )

            pairs.append({
                "ledger_id": ledger_id,
                "bank_id": bank_id,
                "text": text,
                "label": int((ledger_id, bank_id) in positive_pairs),
            })

    return pd.DataFrame(pairs)

train = make_pairs(train_ledger, train_bank)
test = make_pairs(test_ledger, test_bank)

if train["label"].sum() == 0 or test["label"].sum() == 0:
    raise ValueError(
        "No positive pairs in train or test. Inspect the answer key "
        "and temporal split before proceeding."
    )

print("Training pairs:", len(train))
print("Training positives:", int(train["label"].sum()))
print("Test pairs:", len(test))
print("Test positives:", int(test["label"].sum()))
print("Temporal cutoff:", cutoff.date())

# 3. Fit TF-IDF only on training text.
# Logistic Regression learns which word and character patterns
# distinguish matching transaction pairs.
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
            strip_accents="unicode",
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

model.fit(train["text"], train["label"])

# 4. Score the held-out test candidates.
test["probability"] = model.predict_proba(test["text"])[:, 1]

average_precision = average_precision_score(
    test["label"], test["probability"]
)

# 5. Greedy one-to-one assignment, highest probability first.
# A pair must meet the exploratory threshold and neither record
# may already have been assigned.
threshold = 0.50
ranked = test[test["probability"] >= threshold].sort_values(
    "probability", ascending=False
)

used_ledger = set()
used_bank = set()
selected = []

for row in ranked.itertuples(index=False):
    if row.ledger_id in used_ledger or row.bank_id in used_bank:
        continue

    used_ledger.add(row.ledger_id)
    used_bank.add(row.bank_id)
    selected.append({
        "ledger_id": row.ledger_id,
        "bank_id": row.bank_id,
        "probability": row.probability,
        "label": row.label,
    })

matches = pd.DataFrame(
    selected,
    columns=["ledger_id", "bank_id", "probability", "label"],
)

tp = int(matches["label"].sum()) if not matches.empty else 0
fp = len(matches) - tp
fn = int(test["label"].sum()) - tp

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0.0
)

# Save detailed scores and end-to-end matches.
test.drop(columns=["text"]).to_csv(
    REPORTS / "text_matcher_test_scores.csv", index=False
)
matches.to_csv(
    REPORTS / "text_matcher_matches.csv", index=False
)

summary = pd.DataFrame([{
    "model": "TFIDF_LogisticRegression",
    "cutoff": str(cutoff.date()),
    "train_pairs": len(train),
    "train_positive_pairs": int(train["label"].sum()),
    "test_pairs": len(test),
    "expected_matches": int(test["label"].sum()),
    "predicted_matches": len(matches),
    "true_positives": tp,
    "false_positives": fp,
    "false_negatives": fn,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "average_precision": average_precision,
    "threshold": threshold,
}])

summary.to_csv(
    REPORTS / "text_matcher_evaluation.csv", index=False
)

print("\n=== TEXT MATCHER EVALUATION ===")
print(summary.to_string(index=False))
print("\nSaved:")
print("reports/text_matcher_evaluation.csv")
print("reports/text_matcher_test_scores.csv")
print("reports/text_matcher_matches.csv")
