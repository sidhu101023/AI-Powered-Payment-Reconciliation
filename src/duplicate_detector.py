
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "reports"
OUTPUT.mkdir(exist_ok=True)

# Load bank transaction data
bank = pd.read_csv(DATA / "bank_lines.csv")

# Identify identical bank lines using transaction fields.
# The ID is excluded because each row has its own unique ID.
duplicate_columns = [
    "account",
    "posted_date",
    "narrative",
    "amount",
]

bank["duplicate_group"] = (
    bank.groupby(duplicate_columns, dropna=False)
    .ngroup()
)

duplicate_mask = bank.duplicated(
    subset=duplicate_columns,
    keep=False,
)

duplicates = bank.loc[duplicate_mask].copy()

if duplicates.empty:
    print("No identical bank lines found.")
else:
    duplicates = duplicates.sort_values(
        ["duplicate_group", "id"]
    )

    print("=== IDENTICAL BANK-LINE GROUPS ===")

    for group_id, group in duplicates.groupby("duplicate_group"):
        print(f"\nDuplicate group: {group_id}")
        print(
            group[
                [
                    "id",
                    "account",
                    "posted_date",
                    "narrative",
                    "amount",
                ]
            ].to_string(index=False)
        )

# Save the duplicate report.
report_path = OUTPUT / "duplicate_bank_lines.csv"

duplicates.to_csv(report_path, index=False)

print(f"\nDuplicate rows found: {len(duplicates)}")
print(f"Report saved to: {report_path}")
