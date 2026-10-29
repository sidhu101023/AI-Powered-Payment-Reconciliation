import pandas as pd
import pytest
import sqlite3

from src.payment_registry import PaymentRegistry


def registry(tmp_path):
    return PaymentRegistry(tmp_path / "payments.sqlite3")


def bank_row(bank_id, description, amount):
    return pd.DataFrame([{
        "bank_id": bank_id, "bank_description": description,
        "bank_amount": amount,
    }])


def test_payment_ids_are_unique_and_persist(tmp_path):
    first = registry(tmp_path)
    identifiers = {first.create_payment("ABC Traders", [("INV-1", 50_000)]) for _ in range(20)}
    assert len(identifiers) == 20
    assert len(PaymentRegistry(tmp_path / "payments.sqlite3").find_payments()) == 20


def test_legacy_database_migrates_submission_key_with_unique_index(tmp_path):
    database = tmp_path / "payments.sqlite3"
    with sqlite3.connect(database) as con:
        con.execute("""CREATE TABLE payments (
            payment_id TEXT PRIMARY KEY, supplier TEXT NOT NULL,
            payment_amount REAL NOT NULL, status TEXT NOT NULL,
            created_at TEXT NOT NULL, instruction_reference TEXT UNIQUE,
            bank_reference TEXT UNIQUE
        )""")
    store = PaymentRegistry(database)
    first = store.create_payment("ABC", [("INV-1", 100)], submission_key="legacy-retry")
    second = store.create_payment("ABC", [("INV-1", 100)], submission_key="legacy-retry")
    assert first == second


def test_duplicate_instruction_is_prevented(tmp_path):
    store = registry(tmp_path)
    store.create_payment("ABC", [("INV-1", 10)], instruction_reference="INST-1")
    with pytest.raises(ValueError, match="already exists"):
        store.create_payment("ABC", [("INV-2", 10)], instruction_reference="INST-1")


def test_one_payment_can_link_multiple_invoices(tmp_path):
    store = registry(tmp_path)
    payment_id = store.create_payment("ABC", [("INV-1", 40), ("INV-2", 60)])
    row = store.find_payments(payment_id).iloc[0]
    assert row.payment_amount == 100
    assert row.invoice_allocations == "INV-1: 40.00, INV-2: 60.00"
    assert row.created_at


def test_multiple_payments_can_link_one_invoice(tmp_path):
    store = registry(tmp_path)
    first = store.create_payment("ABC", [("INV-1", 40)])
    second = store.create_payment("ABC", [("INV-1", 60)])
    assert {first, second} == set(store.find_payments("INV-1").payment_id)


def test_exact_payment_id_match_is_pending_not_forced(tmp_path):
    store = registry(tmp_path)
    payment_id = store.create_payment("ABC", [("INV-1", 100)])
    matches, exceptions = store.identifier_candidates(bank_row("B1", f"NEFT {payment_id}", -100))
    assert exceptions.empty
    assert matches.iloc[0].matching_method == "payment_id"
    assert matches.iloc[0].review_decision == "Pending"


def test_missing_payment_id_falls_back_without_identifier_match(tmp_path):
    store = registry(tmp_path)
    store.create_payment("ABC", [("INV-1", 100)])
    matches, exceptions = store.identifier_candidates(bank_row("B1", "ABC INV-1", -100))
    assert matches.empty and exceptions.empty


def test_identifier_amount_difference_is_exception(tmp_path):
    store = registry(tmp_path)
    payment_id = store.create_payment("ABC", [("INV-1", 100)])
    matches, exceptions = store.identifier_candidates(bank_row("B1", payment_id, -99))
    assert matches.empty
    assert exceptions.iloc[0].exception_type == "AMOUNT_MISMATCH"


def test_duplicate_bank_lines_with_same_payment_id_are_exceptions(tmp_path):
    store = registry(tmp_path)
    payment_id = store.create_payment("ABC", [("INV-1", 100)])
    bank = pd.concat([
        bank_row("B1", payment_id, -100), bank_row("B2", payment_id, -100),
    ], ignore_index=True)
    matches, exceptions = store.identifier_candidates(bank)
    assert matches.empty
    assert set(exceptions.exception_type) == {"DUPLICATE_BANK_REFERENCE"}


def test_reconciliation_assignment_persists(tmp_path):
    store = registry(tmp_path)
    payment_id = store.create_payment("ABC", [("INV-1", 100)])
    store.record_reconciliation(payment_id, "B1", "payment_id")
    assert store.find_payments(payment_id).iloc[0].status == "Reconciled"


def test_submission_key_makes_repeated_submit_idempotent(tmp_path):
    store = registry(tmp_path)
    first = store.create_payment("ABC", [("INV-1", 100)], submission_key="browser-submit-1")
    second = store.create_payment("ABC", [("INV-1", 100)], submission_key="browser-submit-1")
    assert first == second
    assert len(store.find_payments("INV-1")) == 1


def test_search_finds_payment_id_invoice_and_bank_reference(tmp_path):
    store = registry(tmp_path)
    payment_id = store.create_payment("ABC", [("INV-1", 100)])
    store.record_bank_reference(payment_id, "UTR-2026-001")
    assert len(store.find_payments(payment_id)) == 1
    assert len(store.find_payments("INV-1")) == 1
    assert len(store.find_payments("UTR-2026-001")) == 1


@pytest.mark.parametrize("allocations", [
    [("INV-1", 0)], [("INV-1", -1)], [("INV-1", float("nan"))],
])
def test_invalid_allocation_amount_is_rejected(tmp_path, allocations):
    with pytest.raises(ValueError, match="positive"):
        registry(tmp_path).create_payment("ABC", allocations)
