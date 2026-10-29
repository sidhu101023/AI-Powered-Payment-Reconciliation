import pandas as pd
import pytest

from src.payment_registry import PaymentRegistry
from src.reconciliation_workflow import (
    decision_audit_rows,
    input_validation_findings,
    review_exports,
    validate_review_decisions,
)


def sample_inputs():
    bank = pd.DataFrame([
        {"bank_id": "B1", "bank_description": "", "bank_amount": 0.0},
        {"bank_id": "B2", "bank_description": "Payment", "bank_amount": 20.0},
    ])
    ledger = pd.DataFrame([
        {"ledger_id": "L1", "ledger_reference": "", "ledger_amount": 100.0},
    ])
    return bank, ledger


def test_validation_findings_preserve_source_rows():
    bank, ledger = sample_inputs()
    findings = input_validation_findings(bank, ledger)
    assert set(findings.issue) == {"Missing bank_description", "Zero amount requires review", "Missing ledger_reference"}
    assert len(bank) == 2 and len(ledger) == 1


def test_duplicate_approval_is_rejected_before_audit_write():
    decisions = pd.DataFrame([
        {"ledger_id": "L1", "bank_id": "B1", "review_decision": "Approve"},
        {"ledger_id": "L1", "bank_id": "B2", "review_decision": "Approve"},
    ])
    with pytest.raises(ValueError, match="ledger entry"):
        validate_review_decisions(decisions)


def test_decision_history_and_exports_only_count_approved_matches():
    decisions = pd.DataFrame([
        {"ledger_id": "L1", "bank_id": "B1", "review_decision": "Approve"},
        {"ledger_id": "L2", "bank_id": "B2", "review_decision": "Reject"},
        {"ledger_id": "L3", "bank_id": "B3", "review_decision": "Pending"},
    ])
    events = decision_audit_rows(decisions, reviewer="reviewer-7", reason="statement checked")
    assert set(events.new_status) == {"Approved", "Rejected"}
    assert set(events.reviewer_identity) == {"reviewer-7"}
    ledger = pd.DataFrame({"ledger_id": ["L1", "L2", "L3"]})
    bank = pd.DataFrame({"bank_id": ["B1", "B2", "B3"]})
    exports = review_exports(decisions, ledger, bank)
    assert list(exports["approved_matches"].ledger_id) == ["L1"]
    assert set(exports["unmatched_ledger"].ledger_id) == {"L2", "L3"}
    assert set(exports["unmatched_bank"].bank_id) == {"B2", "B3"}


def test_registry_search_audit_history_and_conflicting_bank_protection(tmp_path):
    database = tmp_path / "payments.sqlite3"
    store = PaymentRegistry(database)
    first = store.create_payment("Acme Supplies", [("INV-1", 100)])
    second = store.create_payment("Other Supplies", [("INV-2", 100)])
    assert list(store.find_payments("Acme").payment_id) == [first]
    store.record_reconciliation(first, "B1", "payment_id", reviewer_identity="reviewer-7", decision_reason="exact identifier")
    store.record_reconciliation(first, "B1", "payment_id", result="rejected", reviewer_identity="reviewer-8", decision_reason="reopened")
    history = PaymentRegistry(database).reconciliation_audit(first)
    assert list(history.decision) == ["approved", "rejected"]
    assert history.iloc[0].previous_status == "Created"
    assert history.iloc[0].new_status == "Reconciled"
    with pytest.raises(ValueError, match="already approved"):
        store.record_reconciliation(second, "B1", "payment_id")
