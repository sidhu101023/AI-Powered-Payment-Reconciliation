
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

scores = pd.read_csv(REPORTS / "same_split_test_scores.csv")
baseline = scores[scores["model"] == "RuleBasedBaseline"].copy()

print("=== BASELINE TEST SCORE AUDIT ===")
print(f"Test candidate pairs: {len(baseline)}")
print(f"Known positive candidates: {int(baseline['label'].sum())}")
print(f"Known negative candidates: {int((baseline['label'] == 0).sum())}")

print("\nScore distribution by answer-key label:")
print(
    baseline.groupby("label")["probability"]
    .agg(["count", "min", "median", "max"])
    .to_string()
)

print("\nSelected matches at threshold 0.40:")
selected = baseline[baseline["probability"] >= 0.40]
print(f"Eligible candidate pairs: {len(selected)}")
print(f"Positive eligible pairs: {int(selected['label'].sum())}")
print(f"Negative eligible pairs: {int((selected['label'] == 0).sum())}")

print("\nHighest-scoring negative candidates:")
negatives = baseline[baseline["label"] == 0].nlargest(
    15, "probability"
)
print(
    negatives[
        ["ledger_id", "bank_id", "probability", "label"]
    ].to_string(index=False)
)

matches_path = REPORTS / "same_split_test_matches.csv"
if matches_path.exists():
    matches = pd.read_csv(matches_path)
    baseline_matches = matches[
        matches["model"] == "RuleBasedBaseline"
    ]
    print("\nSelected baseline matches with incorrect labels:")
    errors = baseline_matches[baseline_matches["label"] == 0]
    print("Count:", len(errors))
    if not errors.empty:
        print(errors.to_string(index=False))
