
from pathlib import Path
from difflib import SequenceMatcher
import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
keys = pd.read_csv(DATA / "match_key.csv")

bank["id"] = bank["id"].astype(int)
ledger["id"] = ledger["id"].astype(int)
bank["posted_date"] = pd.to_datetime(bank["posted_date"])
ledger["entry_date"] = pd.to_datetime(ledger["entry_date"])

valid_keys = keys[
    keys["kind"].astype(str).str.lower() != "duplicate_line"
]
positive_pairs = set(
    zip(
        valid_keys["ledger_entry_no"].astype(int),
        valid_keys["bank_line_no"].astype(int),
    )
)

all_dates = pd.concat(
    [ledger["entry_date"], bank["posted_date"]],
    ignore_index=True,
).dropna()
cutoff = all_dates.quantile(0.75)

train_ledger = ledger[ledger["entry_date"] < cutoff]
train_bank = bank[bank["posted_date"] < cutoff]
test_ledger = ledger[ledger["entry_date"] >= cutoff]
test_bank = bank[bank["posted_date"] >= cutoff]

def normalize(value):
    value = str(value).upper()
    return re.sub(r"[^A-Z0-9]", "", value)

def similarity(a, b):
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def make_pairs(ledger_rows, bank_rows):
    rows = []

    for l in ledger_rows.itertuples(index=False):
        for b in bank_rows.itertuples(index=False):
            lid, bid = int(l.id), int(b.id)
            reference = str(getattr(l, "reference", ""))
            counterparty = str(getattr(l, "counterparty", ""))
            narrative = str(getattr(b, "narrative", ""))

            la, ba = float(l.amount), float(b.amount)
            day_gap = (b.posted_date - l.entry_date).days
            amount_gap = abs(la - ba)

            ref_norm = normalize(reference)
            narrative_norm = normalize(narrative)
            ref_in_narrative = int(
                bool(ref_norm) and ref_norm in narrative_norm
            )

            text = (
                f"ledger reference {reference} "
                f"counterparty {counterparty} "
                f"bank narrative {narrative}"
            )

            rows.append({
                "ledger_id": lid,
                "bank_id": bid,
                "text": text,
                "amount_difference": amount_gap,
                "exact_amount": int(abs(la - ba) < 0.005),
                "amount_within_cent": int(amount_gap <= 0.01),
                "same_amount_sign": int(
                    (la < 0 and ba < 0) or (la >= 0 and ba >= 0)
                ),
                "signed_date_difference": day_gap,
                "absolute_date_difference": abs(day_gap),
                "reference_in_narrative": ref_in_narrative,
                "reference_similarity": similarity(
                    reference, narrative
                ),
                "counterparty_similarity": similarity(
                    counterparty, narrative
                ),
                "label": int((lid, bid) in positive_pairs),
            })

    return pd.DataFrame(rows)

train = make_pairs(train_ledger, train_bank)
test = make_pairs(test_ledger, test_bank)

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

if train["label"].sum() == 0 or test["label"].sum() == 0:
    raise ValueError("Training or test split has no positive pairs.")

preprocessor = ColumnTransformer([
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
    ("features", preprocessor),
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
test["probability"] = model.predict_proba(test)[:, 1]

average_precision = average_precision_score(
    test["label"], test["probability"]
)

# Exploratory threshold; tune on validation data in a later step.
threshold = 0.50
ranked = test[test["probability"] >= threshold].sort_values(
    "probability", ascending=False
)

used_ledger, used_bank, selected = set(), set(), []

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

summary = pd.DataFrame([{
    "model": "Hybrid_TFIDF_Structured_LogisticRegression",
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

summary.to_csv(REPORTS / "hybrid_matcher_evaluation.csv", index=False)
test.drop(columns=["text"]).to_csv(
    REPORTS / "hybrid_matcher_test_scores.csv", index=False
)
matches.to_csv(REPORTS / "hybrid_matcher_matches.csv", index=False)

print("=== HYBRID MATCHER EVALUATION ===")
print(f"Temporal cutoff: {cutoff.date()}")
print(f"Training pairs: {len(train)}")
print(f"Training positives: {int(train['label'].sum())}")
print(f"Test pairs: {len(test)}")
print(f"Test positives: {int(test['label'].sum())}")
print(summary.to_string(index=False))
print("\nSaved reports/hybrid_matcher_evaluation.csv")
