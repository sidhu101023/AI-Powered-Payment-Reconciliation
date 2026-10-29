from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
keys = pd.read_csv(DATA / "match_key.csv")
matches = pd.read_csv(REPORTS / "hybrid_matcher_matches.csv")

# Normalize identifiers.
for frame, columns in [
    (bank, ["id"]),
    (ledger, ["id"]),
    (keys, ["ledger_entry_no", "bank_line_no"]),
    (matches, ["ledger_id", "bank_id"]),
]:
    for column in columns:
        frame[column] = pd.to_numeric(
            frame[column], errors="raise"
        ).astype(int)

# Identify exact-signature duplicate groups. These are review candidates,
# not proof that a transaction is erroneous.
signature_columns = ["account", "posted_date", "amount", "narrative"]
bank["duplicate_group"] = bank.duplicated(
    subset=signature_columns, keep=False
)

duplicate_bank_ids = set(
    bank.loc[bank["duplicate_group"], "id"].astype(int)
)

# Original selected matches.
original_pairs = set(zip(matches["ledger_id"], matches["bank_id"]))

# Experimental guard: do not automatically select matches using a bank
# line from an exact-signature duplicate group; route it to manual review.
guarded = matches[~matches["bank_id"].isin(duplicate_bank_ids)].copy()
guarded_pairs = set(zip(guarded["ledger_id"], guarded["bank_id"]))

# Answer key is used for evaluation only, never to create the guard.
valid_keys = keys[keys["kind"].str.lower() != "duplicate_line"]
truth = set(zip(
    valid_keys["ledger_entry_no"], valid_keys["bank_line_no"]
))

def metrics(predicted, expected):
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
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Keep the evaluation scope consistent with the existing hybrid output:
# only answer-key positive pairs in the selected-match file's test scope.
test_ledger_ids = set(matches["ledger_id"])
test_bank_ids = set(matches["bank_id"])
expected_test = {
    pair for pair in truth
    if pair[0] in test_ledger_ids and pair[1] in test_bank_ids
}

summary = []
for name, pairs in [
    ("Hybrid_original", original_pairs),
    ("Hybrid_duplicate_guard", guarded_pairs),
]:
    row = {"version": name, "selected_matches": len(pairs)}
    row.update(metrics(pairs, expected_test))
    summary.append(row)

pd.DataFrame(summary).to_csv(
    REPORTS / "duplicate_guard_comparison.csv", index=False
)

# Export every duplicate-signature bank row for manual review.
duplicate_review = bank[bank["duplicate_group"]].copy()
duplicate_review["review_reason"] = "identical_account_date_amount_narrative"
duplicate_review.to_csv(
    REPORTS / "duplicate_guard_review_queue.csv", index=False
)

# Export original selected matches removed by the guard.
removed = matches[matches["bank_id"].isin(duplicate_bank_ids)].copy()
removed.to_csv(
    REPORTS / "duplicate_guard_removed_matches.csv", index=False
)

print("=== DUPLICATE-AWARE GUARD EXPERIMENT ===")
print("Duplicate-signature bank rows:", len(duplicate_review))
print("Unique duplicate-signature bank IDs:", len(duplicate_bank_ids))
print("Original selected matches:", len(original_pairs))
print("Guarded selected matches:", len(guarded_pairs))
print("Matches routed to review:", len(removed))
print("\n=== EVALUATION ===")
print(pd.DataFrame(summary).to_string(index=False))
print("\nSaved reports:")
print(REPORTS / "duplicate_guard_comparison.csv")
print(REPORTS / "duplicate_guard_review_queue.csv")
print(REPORTS / "duplicate_guard_removed_matches.csv")
