from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "reports"
OUTPUT.mkdir(exist_ok=True)

# Load existing outputs and source data
predictions = pd.read_csv(OUTPUT / "baseline_predictions.csv")
evaluation = pd.read_csv(OUTPUT / "baseline_evaluation.csv")
duplicates = pd.read_csv(OUTPUT / "duplicate_bank_lines.csv")
unmatched = pd.read_csv(DATA / "unmatched_key.csv")

# Count predictions by answer-key category.
# This is for offline evaluation, not for production predictions.
if "answer_kind" in evaluation.columns:
    category_counts = (
        evaluation["answer_kind"]
        .fillna("unknown")
        .value_counts()
    )
else:
    category_counts = pd.Series(dtype="int64")

print("=== RECONCILIATION SUMMARY ===")
print(f"Bank-ledger pairs predicted: {len(predictions)}")
print(f"Duplicate bank rows flagged: {len(duplicates)}")
print(f"Unmatched answer-key records: {len(unmatched)}")

print("\n=== PREDICTED PAIRS BY CATEGORY ===")
if category_counts.empty:
    print("No answer-key categories available.")
else:
    print(category_counts.to_string())

print("\n=== UNMATCHED RECORDS BY SIDE ===")
print(unmatched["side"].value_counts().to_string())

print("\n=== UNMATCHED RECORDS BY REASON ===")
print(unmatched["reason"].value_counts().to_string())

# Save a compact summary report.
summary_rows = [
    {"metric": "predicted_pairs", "value": len(predictions)},
    {"metric": "duplicate_bank_rows_flagged", "value": len(duplicates)},
    {"metric": "unmatched_answer_key_records", "value": len(unmatched)},
]

for category, count in category_counts.items():
    summary_rows.append({
        "metric": f"predicted_category_{category}",
        "value": int(count),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT / "reconciliation_summary.csv", index=False)

# Save unmatched records in a separate, reviewable file.
unmatched.to_csv(OUTPUT / "unmatched_transactions_report.csv", index=False)

print("\n=== REPORTS SAVED ===")
print(OUTPUT / "reconciliation_summary.csv")
print(OUTPUT / "unmatched_transactions_report.csv")
