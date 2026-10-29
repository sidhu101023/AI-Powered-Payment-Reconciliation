
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "reports"

truth = pd.read_csv(DATA / "match_key.csv")
predictions = pd.read_csv(
    OUTPUT / "ai_reconciliation_matches.csv"
)

# Answer-key pairs include all categories.
truth_pairs = set(
    zip(
        truth["ledger_entry_no"].astype(int),
        truth["bank_line_no"].astype(int),
    )
)

predicted_pairs = set(
    zip(
        predictions["ledger_id"].astype(int),
        predictions["bank_id"].astype(int),
    )
) if not predictions.empty else set()

true_positives = truth_pairs & predicted_pairs
false_positives = predicted_pairs - truth_pairs
false_negatives = truth_pairs - predicted_pairs

precision = (
    len(true_positives) / len(predicted_pairs)
    if predicted_pairs else 0.0
)
recall = (
    len(true_positives) / len(truth_pairs)
    if truth_pairs else 0.0
)
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0.0
)

print("=== AI RECONCILIATION EVALUATION ===")
print(f"Expected answer-key pairs: {len(truth_pairs)}")
print(f"Predicted pairs:           {len(predicted_pairs)}")
print(f"Correct pairs:             {len(true_positives)}")
print(f"False positives:           {len(false_positives)}")
print(f"Missed answer-key pairs:    {len(false_negatives)}")
print(f"Precision:                  {precision:.3f}")
print(f"Recall:                     {recall:.3f}")
print(f"F1-score:                   {f1:.3f}")

# Explain predictions using the answer-key category.
truth_lookup = (
    truth.drop_duplicates(
        ["ledger_entry_no", "bank_line_no"]
    )
    .set_index(["ledger_entry_no", "bank_line_no"])["kind"]
)

evaluation = predictions.copy()
evaluation["answer_kind"] = [
    truth_lookup.get(
        (int(row.ledger_id), int(row.bank_id)),
        "not_in_answer_key",
    )
    for row in evaluation.itertuples()
]

print("\n=== MATCHES BY ANSWER-KEY CATEGORY ===")
print(evaluation["answer_kind"].value_counts().to_string())

evaluation.to_csv(
    OUTPUT / "ai_reconciliation_evaluation.csv",
    index=False,
)

print("\nSaved:")
print(OUTPUT / "ai_reconciliation_evaluation.csv")
