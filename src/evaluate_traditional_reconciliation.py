
from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
match_key = pd.read_csv(DATA / "match_key.csv")

# Exclude ambiguous duplicate-line labels.
valid_key = match_key[
    match_key["kind"].astype(str).str.lower() != "duplicate_line"
]
positive_pairs = set(
    zip(
        valid_key["ledger_entry_no"].astype(int),
        valid_key["bank_line_no"].astype(int),
    )
)

def normalize(value):
    import re
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

def similarity(a, b):
    from difflib import SequenceMatcher
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def features(l, b):
    ld = pd.to_datetime(l["entry_date"], errors="coerce")
    bd = pd.to_datetime(b["posted_date"], errors="coerce")
    days = (bd - ld).days if pd.notna(ld) and pd.notna(bd) else 9999
    diff = abs(float(l["amount"]) - float(b["amount"]))
    ref = normalize(l["reference"])
    narrative = normalize(b["narrative"])

    return {
        "amount_difference": diff,
        "exact_amount": int(diff < 0.005),
        "amount_within_cent": int(diff <= 0.01),
        "signed_date_difference": days,
        "absolute_date_difference": abs(days),
        "reference_in_narrative": int(bool(ref) and ref in narrative),
        "reference_similarity": similarity(
            l["reference"], b["narrative"]
        ),
        "counterparty_similarity": similarity(
            l["counterparty"], b["narrative"]
        ),
    }

FEATURES = [
    "amount_difference",
    "exact_amount",
    "amount_within_cent",
    "signed_date_difference",
    "absolute_date_difference",
    "reference_in_narrative",
    "reference_similarity",
    "counterparty_similarity",
]

# Use the same temporal holdout policy as the Phase 2 experiment.
all_dates = pd.concat(
    [
        pd.to_datetime(ledger["entry_date"]),
        pd.to_datetime(bank["posted_date"]),
    ],
    ignore_index=True,
).dropna()
cutoff = all_dates.quantile(0.75)

ledger["parsed_date"] = pd.to_datetime(ledger["entry_date"])
bank["parsed_date"] = pd.to_datetime(bank["posted_date"])

train_ledger = ledger[ledger["parsed_date"] < cutoff]
test_ledger = ledger[ledger["parsed_date"] >= cutoff]
train_bank = bank[bank["parsed_date"] < cutoff]
test_bank = bank[bank["parsed_date"] >= cutoff]

def generate_pairs(ledger_rows, bank_rows):
    rows = []
    for l in ledger_rows.to_dict("records"):
        for b in bank_rows.to_dict("records"):
            row = features(l, b)
            row["ledger_id"] = int(l["id"])
            row["bank_id"] = int(b["id"])
            row["label"] = int(
                (int(l["id"]), int(b["id"])) in positive_pairs
            )
            rows.append(row)
    return pd.DataFrame(rows)

train_pairs = generate_pairs(train_ledger, train_bank)
test_pairs = generate_pairs(test_ledger, test_bank)

if train_pairs.empty or test_pairs.empty:
    raise ValueError("Temporal train/test data is empty.")

if train_pairs["label"].nunique() < 2:
    raise ValueError("Training data must contain both classes.")

# Fit models only on earlier-period candidate pairs.
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

def greedy_one_to_one(candidates, threshold):
    selected = []
    used_ledger = set()
    used_bank = set()

    candidates = candidates.sort_values(
        ["probability", "ledger_id", "bank_id"],
        ascending=[False, True, True],
    )

    for row in candidates.itertuples(index=False):
        if row.probability < threshold:
            break
        if row.ledger_id in used_ledger or row.bank_id in used_bank:
            continue
        selected.append({
            "ledger_id": int(row.ledger_id),
            "bank_id": int(row.bank_id),
            "probability": float(row.probability),
        })
        used_ledger.add(int(row.ledger_id))
        used_bank.add(int(row.bank_id))

    return pd.DataFrame(
        selected, columns=["ledger_id", "bank_id", "probability"]
    )

results = []
prediction_frames = []

for model_name, model in models.items():
    model.fit(train_pairs[FEATURES], train_pairs["label"])

    scores = model.predict_proba(test_pairs[FEATURES])[:, 1]
    candidates = test_pairs[
        ["ledger_id", "bank_id", "label"]
    ].copy()
    candidates["probability"] = scores

    # Initial threshold for comparison only. Later, tune it on validation data.
    predicted = greedy_one_to_one(candidates, threshold=0.5)

    expected = {
        (int(lid), int(bid))
        for lid, bid in positive_pairs
        if int(lid) in set(test_ledger["id"].astype(int))
        and int(bid) in set(test_bank["id"].astype(int))
    }
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

    results.append({
        "model": model_name,
        "temporal_cutoff": str(cutoff.date()),
        "train_ledger_count": len(train_ledger),
        "train_bank_count": len(train_bank),
        "test_ledger_count": len(test_ledger),
        "test_bank_count": len(test_bank),
        "expected_test_pairs": len(expected),
        "predicted_matches": len(actual),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": 0.5,
    })

    if not predicted.empty:
        predicted["model"] = model_name
        prediction_frames.append(predicted)

REPORTS.mkdir(exist_ok=True)
pd.DataFrame(results).to_csv(
    REPORTS / "traditional_ml_reconciliation_evaluation.csv",
    index=False,
)
if prediction_frames:
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        REPORTS / "traditional_ml_reconciliation_matches.csv",
        index=False,
    )

print("=== END-TO-END TRADITIONAL ML EVALUATION ===")
print(f"Temporal cutoff: {cutoff.date()}")
print(pd.DataFrame(results).to_string(index=False))
print("\nSaved:")
print(REPORTS / "traditional_ml_reconciliation_evaluation.csv")
print(REPORTS / "traditional_ml_reconciliation_matches.csv")
