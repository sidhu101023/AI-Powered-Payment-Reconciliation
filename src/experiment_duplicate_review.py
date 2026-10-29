from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

bank = pd.read_csv(DATA / "bank_lines.csv")
matches = pd.read_csv(REPORTS / "hybrid_matcher_matches.csv")

for frame, columns in [
    (bank, ["id"]),
    (matches, ["ledger_id", "bank_id"]),
]:
    for column in columns:
        frame[column] = pd.to_numeric(
            frame[column], errors="raise"
        ).astype(int)

# Detect identical bank-row signatures.
signature_columns = ["account", "posted_date", "amount", "narrative"]
bank["duplicate_group"] = bank.duplicated(
    subset=signature_columns, keep=False
)

duplicate_bank_ids = set(
    bank.loc[bank["duplicate_group"], "id"]
)

# Preserve all original model selections.
matches["duplicate_review_required"] = (
    matches["bank_id"].isin(duplicate_bank_ids)
)
matches["decision"] = matches["duplicate_review_required"].map(
    {True: "MANUAL_REVIEW", False: "AUTO_MATCH_CANDIDATE"}
)

# Full proposed-match output, including review flags.
matches.to_csv(
    REPORTS / "duplicate_review_all_matches.csv", index=False
)

# Only matches that require review.
review_matches = matches[
    matches["duplicate_review_required"]
].copy()
review_matches.to_csv(
    REPORTS / "duplicate_review_match_queue.csv", index=False
)

# All bank rows belonging to duplicate-signature groups.
duplicate_review = bank[bank["duplicate_group"]].copy()
duplicate_review["review_reason"] = (
    "identical_account_date_amount_narrative"
)
duplicate_review.to_csv(
    REPORTS / "duplicate_review_bank_rows.csv", index=False
)

print("=== REVIEW-ONLY DUPLICATE EXPERIMENT ===")
print("Original selected matches:", len(matches))
print("Matches preserved:", len(matches))
print("Matches flagged for manual review:", len(review_matches))
print(
    "Other matches left as automatic candidates:",
    len(matches) - len(review_matches),
)
print("Duplicate-signature bank rows:", len(duplicate_review))
print("\n=== MATCHES FLAGGED FOR REVIEW ===")
if review_matches.empty:
    print("No selected matches require duplicate review.")
else:
    print(
        review_matches[
            [
                "ledger_id", "bank_id", "probability",
                "label", "decision"
            ]
        ].to_string(index=False)
    )

print("\nSaved reports:")
print(REPORTS / "duplicate_review_all_matches.csv")
print(REPORTS / "duplicate_review_match_queue.csv")
print(REPORTS / "duplicate_review_bank_rows.csv")
