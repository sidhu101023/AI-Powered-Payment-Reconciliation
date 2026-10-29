from difflib import SequenceMatcher
import re

import pandas as pd


def normalize(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def text_similarity(left, right):
    left = normalize(left)
    right = normalize(right)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def generate_exception_candidates(
    ledger,
    bank,
    duplicate_bank_ids=None,
    normal_date_window=7,
    review_date_window=21,
    min_text_similarity=0.35,
):
    """Generate review-only candidates; never automatically approve them."""
    duplicate_bank_ids = {
        str(value) for value in (duplicate_bank_ids or set())
    }

    columns = [
        "ledger_id", "bank_id", "ledger_date", "bank_date",
        "ledger_amount", "bank_amount", "amount_difference",
        "date_difference_days", "text_similarity",
        "review_reason", "review_status",
    ]
    rows = []

    for l in ledger.itertuples(index=False):
        for b in bank.itertuples(index=False):
            amount_gap = abs(
                float(l.ledger_amount) - float(b.bank_amount)
            )
            day_gap = abs(
                (pd.Timestamp(b.bank_date)
                 - pd.Timestamp(l.ledger_date)).days
            )
            similarity = text_similarity(
                f"{l.ledger_reference} {l.counterparty}",
                b.bank_description,
            )

            duplicate = str(b.bank_id) in duplicate_bank_ids
            reasons = []

            # Require plausible transaction similarity and amount proximity.
            similar_description = similarity >= 0.55
            amount_near = amount_gap <= max(
                5.00, 0.05 * abs(float(l.ledger_amount))
            )

            # Possible late posting: similar description, close amount,
            # and a date gap beyond the normal window.
            if (
                normal_date_window < day_gap <= review_date_window
                and similar_description
                and amount_near
            ):
                reasons.append("POSSIBLE_LATE_POSTING")

            # Possible amount difference: similar description, close dates,
            # and a limited amount difference.
            if (
                day_gap <= review_date_window
                and similar_description
                and 0.01 < amount_gap <= max(
                    10.00, 0.10 * abs(float(l.ledger_amount))
                )
            ):
                reasons.append("AMOUNT_DIFFERENCE")

            # Duplicate bank lines: retain only plausible ledger relationships.
            if (
                duplicate
                and similarity >= min_text_similarity
                and day_gap <= review_date_window
                and amount_near
            ):
                reasons.append("DUPLICATE_BANK_LINE")

            if not reasons:
                continue

            # Don't flood the queue with ordinary exact matches.
            if day_gap <= normal_date_window and amount_gap <= 0.01:
                if not duplicate:
                    continue

            rows.append({
                "ledger_id": str(l.ledger_id),
                "bank_id": str(b.bank_id),
                "ledger_date": pd.Timestamp(
                    l.ledger_date
                ).date().isoformat(),
                "bank_date": pd.Timestamp(
                    b.bank_date
                ).date().isoformat(),
                "ledger_amount": float(l.ledger_amount),
                "bank_amount": float(b.bank_amount),
                "amount_difference": round(amount_gap, 2),
                "date_difference_days": day_gap,
                "text_similarity": round(similarity, 3),
                "review_reason": "; ".join(reasons),
                "review_status": "Needs manual review",
            })

    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows)
    result = result.drop_duplicates(["ledger_id", "bank_id"])
    result = result.sort_values(
        ["text_similarity", "date_difference_days"],
        ascending=[False, True],
    )
    return result.reset_index(drop=True)
