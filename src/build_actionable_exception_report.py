from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

unmatched_path = REPORTS / "unmatched_transactions.csv"
amount_date_path = REPORTS / "amount_date_exceptions.csv"
duplicates_path = REPORTS / "duplicate_bank_lines.csv"
false_matches_path = REPORTS / "false_matches.csv"

required = [
    unmatched_path,
    amount_date_path,
    duplicates_path,
    false_matches_path,
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Missing report(s). Run the exception evaluator first:\n"
        + "\n".join(missing)
    )

unmatched = pd.read_csv(unmatched_path)
amount_date = pd.read_csv(amount_date_path)
duplicates = pd.read_csv(duplicates_path)
false_matches = pd.read_csv(false_matches_path)

actions = {
    "bank_fee": (
        "Verify the bank charge against the statement and post "
        "the fee to the ledger if appropriate."
    ),
    "direct_debit_not_booked": (
        "Identify the direct debit and verify or create the "
        "corresponding ledger entry."
    ),
    "unidentified_credit": (
        "Investigate the payer and obtain remittance or supporting "
        "documentation before allocating the credit."
    ),
    "returned_item": (
        "Verify the returned payment and check whether a reversal "
        "or adjustment is required."
    ),
    "interest": (
        "Verify the bank interest and record it in the appropriate "
        "ledger account if required."
    ),
    "never_cleared": (
        "Check payment status and supporting records; determine "
        "whether it remains outstanding or needs correction."
    ),
    "outstanding_cheque": (
        "Check whether the cheque has cleared, remains outstanding, "
        "or needs follow-up with the payee."
    ),
    "deposit_in_transit": (
        "Check deposit evidence and subsequent bank statements to "
        "confirm whether the deposit cleared."
    ),
}

rows = []

# Known unmatched exceptions from the supplied answer key.
for r in unmatched.to_dict("records"):
    reason = str(r.get("reason", "unknown")).lower()
    rows.append({
        "source_report": "unmatched_transactions",
        "source_type": "known_answer_key_exception",
        "record_side": r.get("side", ""),
        "record_id": r.get("record_no", ""),
        "related_record_id": "",
        "date": r.get("date", ""),
        "amount": r.get("amount", ""),
        "exception_type": reason,
        "explanation": (
            f"Dataset answer key identifies this unmatched record "
            f"as '{reason}'."
        ),
        "suggested_action": actions.get(
            reason,
            "Review the source documents and determine why this "
            "transaction is unmatched."
        ),
        "review_priority": (
            "High" if reason in {
                "unidentified_credit",
                "returned_item",
                "direct_debit_not_booked",
            } else "Normal"
        ),
        "requires_human_review": True,
    })

# Valid answer-key pairs that have a non-exact match category or
# amount/date discrepancies.
for r in amount_date.to_dict("records"):
    issue = str(r.get("exception_type", "review"))
    rows.append({
        "source_report": "amount_date_exceptions",
        "source_type": "known_match_needing_review",
        "record_side": "ledger_and_bank",
        "record_id": r.get("ledger_id", ""),
        "related_record_id": r.get("bank_id", ""),
        "date": r.get("bank_date", ""),
        "amount": r.get("absolute_amount_difference", ""),
        "exception_type": issue,
        "explanation": (
            f"Answer key match category is '{r.get('match_kind', '')}'. "
            "Check timing, amount, fees, and transaction references."
        ),
        "suggested_action": (
            "Compare the original ledger entry with the bank statement "
            "and supporting documents before approving the match."
        ),
        "review_priority": "Normal",
        "requires_human_review": True,
    })

# Exact-signature duplicate candidates.
for r in duplicates.to_dict("records"):
    rows.append({
        "source_report": "duplicate_bank_lines",
        "source_type": "potential_duplicate",
        "record_side": "bank",
        "record_id": r.get("bank_id", ""),
        "related_record_id": "",
        "date": r.get("posted_date", ""),
        "amount": r.get("amount", ""),
        "exception_type": "potential_duplicate_bank_line",
        "explanation": (
            "Bank row shares an identical account/date/amount/narrative "
            "signature with another row. This is a review candidate, "
            "not proof of an erroneous duplicate."
        ),
        "suggested_action": (
            "Compare both bank lines with the original statement and "
            "verify whether both represent separate posted transactions."
        ),
        "review_priority": "High",
        "requires_human_review": True,
    })

# Model false matches. These are model-evaluation findings, not
# confirmed production exceptions.
for r in false_matches.to_dict("records"):
    rows.append({
        "source_report": "false_matches",
        "source_type": "model_evaluation_error",
        "record_side": "ledger_and_bank",
        "record_id": r.get("ledger_id", ""),
        "related_record_id": r.get("bank_id", ""),
        "date": r.get("bank_date", ""),
        "amount": r.get("bank_amount", ""),
        "exception_type": r.get("exception_type", "ordinary_false_match"),
        "explanation": (
            f"Model '{r.get('model', '')}' selected a pair not present "
            "in the answer-key truth set used by the evaluator."
        ),
        "suggested_action": (
            "Inspect amount, date, reference, and counterparty details; "
            "do not approve this pair solely from the model score."
        ),
        "review_priority": "High",
        "requires_human_review": True,
    })

columns = [
    "source_report", "source_type", "record_side", "record_id",
    "related_record_id", "date", "amount", "exception_type",
    "explanation", "suggested_action", "review_priority",
    "requires_human_review",
]

result = pd.DataFrame(rows, columns=columns)
output = REPORTS / "actionable_exception_report.csv"
result.to_csv(output, index=False)

print("=== ACTIONABLE EXCEPTION REPORT ===")
print("Report rows:", len(result))
print("\nRows by source:")
if result.empty:
    print("No exceptions found in the source reports.")
else:
    print(result.groupby("source_report").size().to_string())
    print("\nRows by priority:")
    print(result.groupby("review_priority").size().to_string())
    print("\nSample rows:")
    print(result.head(8).to_string(index=False))

print("\nSaved:", output)
