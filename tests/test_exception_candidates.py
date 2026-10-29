import pandas as pd

from src.exception_candidates import generate_exception_candidates


def test_late_posting_is_flagged_for_review():
    ledger = pd.DataFrame([{
        "ledger_id": "L1",
        "ledger_date": pd.Timestamp("2026-01-01"),
        "ledger_reference": "INV100",
        "counterparty": "Juniper Freight",
        "ledger_amount": 100.00,
    }])
    bank = pd.DataFrame([{
        "bank_id": "B1",
        "bank_date": pd.Timestamp("2026-01-15"),
        "bank_description": "Juniper Freight INV100",
        "bank_amount": 100.00,
    }])

    result = generate_exception_candidates(ledger, bank)

    assert len(result) == 1
    assert "POSSIBLE_LATE_POSTING" in result.iloc[0]["review_reason"]
    assert result.iloc[0]["review_status"] == "Needs manual review"


def test_amount_difference_is_flagged():
    ledger = pd.DataFrame([{
        "ledger_id": "L1",
        "ledger_date": pd.Timestamp("2026-01-01"),
        "ledger_reference": "INV100",
        "counterparty": "Juniper Freight",
        "ledger_amount": 100.00,
    }])
    bank = pd.DataFrame([{
        "bank_id": "B1",
        "bank_date": pd.Timestamp("2026-01-01"),
        "bank_description": "Juniper Freight INV100",
        "bank_amount": 97.50,
    }])

    result = generate_exception_candidates(ledger, bank)

    assert len(result) == 1
    assert "AMOUNT_DIFFERENCE" in result.iloc[0]["review_reason"]
    assert result.iloc[0]["amount_difference"] == 2.5


def test_duplicate_line_is_flagged():
    ledger = pd.DataFrame([{
        "ledger_id": "L1",
        "ledger_date": pd.Timestamp("2026-01-01"),
        "ledger_reference": "INV100",
        "counterparty": "Juniper Freight",
        "ledger_amount": 100.00,
    }])
    bank = pd.DataFrame([{
        "bank_id": "B1",
        "bank_date": pd.Timestamp("2026-01-01"),
        "bank_description": "Juniper Freight INV100",
        "bank_amount": 100.00,
    }])

    result = generate_exception_candidates(
        ledger, bank, duplicate_bank_ids={"B1"}
    )

    assert len(result) == 1
    assert "DUPLICATE_BANK_LINE" in result.iloc[0]["review_reason"]


def test_empty_inputs_return_empty_dataframe():
    ledger = pd.DataFrame(columns=[
        "ledger_id", "ledger_date", "ledger_reference",
        "counterparty", "ledger_amount",
    ])
    bank = pd.DataFrame(columns=[
        "bank_id", "bank_date", "bank_description", "bank_amount",
    ])

    result = generate_exception_candidates(ledger, bank)

    assert result.empty
    assert "review_reason" in result.columns

def test_exact_amount_and_late_date_can_be_flagged_for_review():
    ledger = pd.DataFrame([{
        "ledger_id": "L1",
        "ledger_date": pd.Timestamp("2026-01-01"),
        "ledger_reference": "INV100",
        "counterparty": "Juniper Freight",
        "ledger_amount": 100.00,
    }])

    bank = pd.DataFrame([{
        "bank_id": "B1",
        "bank_date": pd.Timestamp("2026-01-09"),
        "bank_description": "Juniper Freight INV100",
        "bank_amount": 100.00,
    }])

    result = generate_exception_candidates(ledger, bank)

    assert len(result) == 1
    assert "POSSIBLE_LATE_POSTING" in result.iloc[0]["review_reason"]
    assert result.iloc[0]["amount_difference"] == 0.0