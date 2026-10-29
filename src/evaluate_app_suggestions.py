from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
suggestions_path = ROOT / "reports" / "app_suggestions.csv"
key_path = ROOT / "data" / "match_key.csv"
reports = ROOT / "reports"
reports.mkdir(exist_ok=True)

suggestions = pd.read_csv(suggestions_path)
key = pd.read_csv(key_path)

required_s = {"ledger_id", "bank_id", "suggestion_score"}
required_k = {"ledger_entry_no", "bank_line_no", "kind"}

if not required_s.issubset(suggestions.columns):
    raise ValueError(f"Suggestions missing columns: {required_s - set(suggestions.columns)}")
if not required_k.issubset(key.columns):
    raise ValueError(f"Match key missing columns: {required_k - set(key.columns)}")

# Normalize IDs so numeric CSV values compare consistently.
suggestions["ledger_id"] = pd.to_numeric(suggestions["ledger_id"], errors="raise").astype(int)
suggestions["bank_id"] = pd.to_numeric(suggestions["bank_id"], errors="raise").astype(int)
key["ledger_entry_no"] = pd.to_numeric(key["ledger_entry_no"], errors="raise").astype(int)
key["bank_line_no"] = pd.to_numeric(key["bank_line_no"], errors="raise").astype(int)

pair_cols = ["ledger_id", "bank_id"]
suggestions["_pair"] = list(zip(suggestions.ledger_id, suggestions.bank_id))
key["_pair"] = list(zip(key.ledger_entry_no, key.bank_line_no))

# Duplicate bank lines are exceptions, not ordinary one-to-one positive matches.
duplicate_pairs = set(key.loc[key.kind.eq("duplicate_line"), "_pair"])
positive_pairs = set(key.loc[~key.kind.eq("duplicate_line"), "_pair"])

suggestions["_is_duplicate_exception"] = suggestions["_pair"].isin(duplicate_pairs)
suggestions["_is_true_match"] = suggestions["_pair"].isin(positive_pairs)
suggestions["_is_false_match"] = (
    ~suggestions["_is_true_match"] & ~suggestions["_is_duplicate_exception"]
)

# Strict evaluation: ordinary reference pairs only.
tp = int(suggestions["_is_true_match"].sum())
fp = int(suggestions["_is_false_match"].sum())
duplicate_suggestions = int(suggestions["_is_duplicate_exception"].sum())
fn_pairs = positive_pairs - set(suggestions.loc[suggestions["_is_true_match"], "_pair"])

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / len(positive_pairs) if positive_pairs else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

# One-to-one integrity checks on suggestions.
duplicate_ledger_ids = int(suggestions.ledger_id.duplicated().sum())
duplicate_bank_ids = int(suggestions.bank_id.duplicated().sum())

metrics = pd.DataFrame([{
    "suggestion_rows": len(suggestions),
    "reference_pairs_total": len(key),
    "ordinary_reference_pairs": len(positive_pairs),
    "duplicate_line_reference_pairs": len(duplicate_pairs),
    "true_positive_suggestions": tp,
    "false_positive_suggestions": fp,
    "duplicate_line_suggestions": duplicate_suggestions,
    "missed_reference_pairs": len(fn_pairs),
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "repeated_ledger_ids_in_suggestions": duplicate_ledger_ids,
    "repeated_bank_ids_in_suggestions": duplicate_bank_ids,
}])

suggestion_review = suggestions.drop(columns=["_pair"]).copy()
suggestion_review["evaluation"] = "false_match"
suggestion_review.loc[suggestion_review["_is_true_match"], "evaluation"] = "correct_match"
suggestion_review.loc[suggestion_review["_is_duplicate_exception"], "evaluation"] = "duplicate_line_exception"
suggestion_review = suggestion_review.drop(
    columns=["_is_duplicate_exception", "_is_true_match", "_is_false_match"]
)

missed = pd.DataFrame(
    sorted(fn_pairs), columns=["ledger_id", "bank_id"]
)
if not missed.empty:
    missed = missed.merge(
        key[["ledger_entry_no", "bank_line_no", "kind"]].rename(
            columns={"ledger_entry_no": "ledger_id", "bank_line_no": "bank_id"}
        ),
        on=["ledger_id", "bank_id"],
        how="left",
    )

metrics.to_csv(reports / "app_evaluation_metrics.csv", index=False)
suggestion_review.to_csv(reports / "app_suggestion_evaluation.csv", index=False)
missed.to_csv(reports / "app_missed_reference_pairs.csv", index=False)

print("\n=== APP SUGGESTION EVALUATION ===")
print(metrics.to_string(index=False))
print("\nEvaluation files saved:")
print("reports/app_evaluation_metrics.csv")
print("reports/app_suggestion_evaluation.csv")
print("reports/app_missed_reference_pairs.csv")
print("\nNote: duplicate_line pairs are reported separately, not counted as ordinary true positives.")
