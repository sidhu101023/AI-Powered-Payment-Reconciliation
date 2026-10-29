
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "reports" / "amount_date_exceptions.csv"
OUTPUT_FILE = ROOT / "reports" / "classified_exceptions.csv"
SUMMARY_FILE = ROOT / "reports" / "classified_exception_summary.csv"

DATE_REVIEW_DAYS = 3
AMOUNT_TOLERANCE = 0.005


def classify_exception(row):
    kind = str(row["match_kind"]).strip().lower()
    amount_diff = abs(float(row["absolute_amount_difference"]))
    date_gap = abs(int(row["signed_date_difference_days"]))

    # Amount differences take precedence over date-only reviews.
    if kind == "net_of_fee":
        return "fee_difference_review"

    if kind == "transposed":
        return "possible_transposed_amount"

    if kind == "late_posting":
        return "late_posting_review"

    if kind == "same_amount_same_day":
        if date_gap > 0:
            return "match_label_date_inconsistency"
        return "same_amount_same_day"

    if kind == "exact":
        if amount_diff > AMOUNT_TOLERANCE:
            return "amount_mismatch_review"

        if date_gap == 0:
            return "exact_match_same_day"

        if date_gap <= DATE_REVIEW_DAYS:
            return "exact_amount_short_date_gap"

        return "exact_amount_long_date_gap_review"

    if amount_diff > AMOUNT_TOLERANCE:
        return "other_amount_difference_review"

    if date_gap > DATE_REVIEW_DAYS:
        return "other_long_date_gap_review"

    return "other_review"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input report not found: {INPUT_FILE}\n"
            "Run src/evaluate_reconciliation_exceptions.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "ledger_id",
        "bank_id",
        "match_kind",
        "ledger_amount",
        "bank_amount",
        "absolute_amount_difference",
        "ledger_date",
        "bank_date",
        "signed_date_difference_days",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["absolute_date_gap_days"] = (
        pd.to_numeric(df["signed_date_difference_days"]).abs()
    )

    df["exception_category"] = df.apply(classify_exception, axis=1)

    # Keep the original match information and add the new classification.
    df.to_csv(OUTPUT_FILE, index=False)

    summary = (
        df.groupby("exception_category")
        .agg(
            records=("exception_category", "size"),
            distinct_ledger_entries=("ledger_id", "nunique"),
            distinct_bank_lines=("bank_id", "nunique"),
            max_date_gap_days=("absolute_date_gap_days", "max"),
            max_amount_difference=("absolute_amount_difference", "max"),
        )
        .reset_index()
        .sort_values("records", ascending=False)
    )

    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n=== CLASSIFIED EXCEPTION SUMMARY ===")
    print(summary.to_string(index=False))

    print(f"\nDetailed report: {OUTPUT_FILE}")
    print(f"Summary report:  {SUMMARY_FILE}")
    print(
        "\nNote: These are review categories, not confirmed errors. "
        "Check the underlying bank statement and ledger before resolving "
        "an exception."
    )


if __name__ == "__main__":
    main()
