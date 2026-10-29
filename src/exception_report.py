
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "reports"
OUTPUT.mkdir(exist_ok=True)

# Load reconciliation outputs and source transactions.
predictions = pd.read_csv(OUTPUT / "baseline_predictions.csv")
bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
duplicates = pd.read_csv(OUTPUT / "duplicate_bank_lines.csv")
unmatched_key = pd.read_csv(DATA / "unmatched_key.csv")

# Validate the essential columns before processing.
required_prediction_columns = {"ledger_id", "bank_id"}
if not required_prediction_columns.issubset(predictions.columns):
    raise ValueError(
        "baseline_predictions.csv must contain ledger_id and bank_id."
    )

if predictions["ledger_id"].duplicated().any():
    raise ValueError("A ledger entry was assigned more than once.")

if predictions["bank_id"].duplicated().any():
    raise ValueError("A bank line was assigned more than once.")

valid_ledger_ids = set(ledger["id"].astype(int))
valid_bank_ids = set(bank["id"].astype(int))

if not set(predictions["ledger_id"].astype(int)).issubset(valid_ledger_ids):
    raise ValueError("Predictions contain unknown ledger IDs.")

if not set(predictions["bank_id"].astype(int)).issubset(valid_bank_ids):
    raise ValueError("Predictions contain unknown bank IDs.")

matched_ledger_ids = set(predictions["ledger_id"].astype(int))
matched_bank_ids = set(predictions["bank_id"].astype(int))

unmatched_ledger = ledger[
    ~ledger["id"].astype(int).isin(matched_ledger_ids)
].copy()

unmatched_bank = bank[
    ~bank["id"].astype(int).isin(matched_bank_ids)
].copy()

# Use answer-key reasons only as evaluation annotations.
# The matching algorithm itself does not know these reasons.
key_ledger = unmatched_key[
    unmatched_key["side"].str.lower() == "ledger"
].copy()

key_bank = unmatched_key[
    unmatched_key["side"].str.lower() == "bank"
].copy()

ledger_reason = {
    int(row.record_no): str(row.reason)
    for row in key_ledger.itertuples(index=False)
}

bank_reason = {
    int(row.record_no): str(row.reason)
    for row in key_bank.itertuples(index=False)
}

exceptions = []

def add_exception(
    exception_type,
    record_side,
    record_id,
    details,
    suggested_action,
    answer_key_reason="",
):
    exceptions.append({
        "exception_type": exception_type,
        "record_side": record_side,
        "record_id": int(record_id),
        "details": details,
        "suggested_action": suggested_action,
        "answer_key_reason_evaluation_only": answer_key_reason,
    })

# 1. Potential duplicate bank lines.
# Do not delete them or assume which line is the original.
if not duplicates.empty:
    for group_id, group in duplicates.groupby("duplicate_group"):
        bank_ids = sorted(group["id"].astype(int).tolist())
        ids_text = ", ".join(map(str, bank_ids))

        add_exception(
            exception_type="potential_duplicate_bank_lines",
            record_side="bank",
            record_id=bank_ids[0],
            details=(
                f"Duplicate group {group_id}; bank line IDs: {ids_text}. "
                "Transaction fields are identical."
            ),
            suggested_action=(
                "Review bank source and transaction identifiers. "
                "Do not automatically delete or suppress a line."
            ),
        )

# 2. Ledger entries not assigned by the baseline.
for row in unmatched_ledger.itertuples(index=False):
    add_exception(
        exception_type="unmatched_ledger_entry",
        record_side="ledger",
        record_id=row.id,
        details=(
            f"Reference={row.reference}; "
            f"counterparty={row.counterparty}; "
            f"amount={float(row.amount):.2f}; "
            f"date={row.entry_date}"
        ),
        suggested_action=(
            "Check for a missing bank line, timing difference, "
            "reference variation, or an incorrect candidate threshold."
        ),
        answer_key_reason=ledger_reason.get(int(row.id), ""),
    )

# 3. Bank lines not assigned by the baseline.
for row in unmatched_bank.itertuples(index=False):
    add_exception(
        exception_type="unmatched_bank_line",
        record_side="bank",
        record_id=row.id,
        details=(
            f"Account={row.account}; "
            f"narrative={row.narrative}; "
            f"amount={float(row.amount):.2f}; "
            f"date={row.posted_date}"
        ),
        suggested_action=(
            "Check for a missing ledger entry, bank fee, "
            "duplicate transaction, or timing difference."
        ),
        answer_key_reason=bank_reason.get(int(row.id), ""),
    )

# 4. Check for suspicious amount/date mismatches in accepted pairs.
# These are review flags, not proof that a match is wrong.
if not predictions.empty:
    ledger_lookup = ledger.set_index("id")
    bank_lookup = bank.set_index("id")

    for row in predictions.itertuples(index=False):
        l = ledger_lookup.loc[int(row.ledger_id)]
        b = bank_lookup.loc[int(row.bank_id)]

        amount_diff = round(
            abs(float(l["amount"]) - float(b["amount"])), 2
        )
        date_diff = (
            pd.to_datetime(b["posted_date"])
            - pd.to_datetime(l["entry_date"])
        ).days

        if amount_diff > 0.01 or date_diff < 0 or date_diff > 12:
            add_exception(
                exception_type="matched_pair_needs_review",
                record_side="pair",
                record_id=row.ledger_id,
                details=(
                    f"Ledger {row.ledger_id} / bank {row.bank_id}; "
                    f"amount difference={amount_diff:.2f}; "
                    f"bank date minus ledger date={date_diff} days."
                ),
                suggested_action=(
                    "Review the transaction details and business rules "
                    "before accepting the match."
                ),
            )

# Save the full exception report.
columns = [
    "exception_type",
    "record_side",
    "record_id",
    "details",
    "suggested_action",
    "answer_key_reason_evaluation_only",
]

report = pd.DataFrame(exceptions, columns=columns)
report_path = OUTPUT / "reconciliation_exceptions.csv"
report.to_csv(report_path, index=False)

print("=== EXCEPTION HANDLING SUMMARY ===")
print(f"Potential duplicate groups: {duplicates['duplicate_group'].nunique() if not duplicates.empty else 0}")
print(f"Potential duplicate rows:   {len(duplicates)}")
print(f"Unmatched ledger entries:   {len(unmatched_ledger)}")
print(f"Unmatched bank lines:       {len(unmatched_bank)}")
print(
    "Matched pairs flagged:     "
    f"{(report['exception_type'] == 'matched_pair_needs_review').sum()}"
)
print(f"Total exception rows:       {len(report)}")

if not report.empty:
    print("\n=== EXCEPTIONS BY TYPE ===")
    print(report["exception_type"].value_counts().to_string())

print(f"\nSaved exception report: {report_path}")
