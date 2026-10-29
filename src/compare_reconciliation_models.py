
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
match_key = pd.read_csv(DATA / "match_key.csv")

baseline = pd.read_csv(REPORTS / "baseline_predictions.csv")
ml_matches = pd.read_csv(
    REPORTS / "traditional_ml_reconciliation_matches.csv"
)

# Exclude ambiguous duplicate-line labels from this comparison.
valid_key = match_key[
    match_key["kind"].astype(str).str.lower() != "duplicate_line"
]
expected_all = set(
    zip(
        valid_key["ledger_entry_no"].astype(int),
        valid_key["bank_line_no"].astype(int),
    )
)

# Use the same temporal cutoff as the ML evaluation.
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

test_ledger_ids = set(
    ledger.loc[ledger["parsed_date"] >= cutoff, "id"].astype(int)
)
test_bank_ids = set(
    bank.loc[bank["parsed_date"] >= cutoff, "id"].astype(int)
)

# Keep answer-key pairs whose ledger AND bank records belong to
# the held-out period.
expected = {
    pair for pair in expected_all
    if pair[0] in test_ledger_ids and pair[1] in test_bank_ids
}

def evaluate(name, frame, ledger_col, bank_col):
    subset = frame[
        frame[ledger_col].astype(int).isin(test_ledger_ids)
        & frame[bank_col].astype(int).isin(test_bank_ids)
    ].copy()

    subset[ledger_col] = subset[ledger_col].astype(int)
    subset[bank_col] = subset[bank_col].astype(int)

    # Enforce one-to-one assignment for all systems.
    if subset[ledger_col].duplicated().any() or subset[bank_col].duplicated().any():
        raise ValueError(
            f"{name} contains repeated ledger or bank assignments. "
            "Resolve conflicts before comparison."
        )

    predicted = set(zip(subset[ledger_col], subset[bank_col]))

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
        "model": name,
        "cutoff": str(cutoff.date()),
        "expected_matches": len(expected),
        "predicted_matches": len(predicted),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

results = []

# Rule-based predictions from Phase 1.
results.append(
    evaluate(
        "RuleBasedBaseline",
        baseline,
        "ledger_id",
        "bank_id",
    )
)

# Each model has its own rows in the ML output.
for model_name in ["LogisticRegression", "RandomForest"]:
    frame = ml_matches[
        ml_matches["model"] == model_name
    ].copy()

    results.append(
        evaluate(
            model_name,
            frame,
            "ledger_id",
            "bank_id",
        )
    )

comparison = pd.DataFrame(results)
output_path = REPORTS / "model_comparison.csv"
comparison.to_csv(output_path, index=False)

print("=== SAME-TEST-PERIOD MODEL COMPARISON ===")
print(f"Temporal cutoff: {cutoff.date()}")
print(f"Expected held-out pairs: {len(expected)}")
print(comparison.to_string(index=False))
print(f"\nSaved comparison: {output_path}")
