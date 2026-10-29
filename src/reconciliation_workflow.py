"""UI-independent safeguards for the reconciliation review workflow."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib

import pandas as pd


def input_validation_findings(bank: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Return warnings after parsing without removing any source rows."""
    findings: list[dict[str, str]] = []
    for source, frame, identifier, text in (
        ("bank", bank, "bank_id", "bank_description"),
        ("ledger", ledger, "ledger_id", "ledger_reference"),
    ):
        for record_id in frame.loc[frame[text].fillna("").astype(str).str.strip().eq(""), identifier].astype(str):
            findings.append({"source": source, "record_id": record_id, "severity": "Warning", "issue": f"Missing {text}"})
        amount = pd.to_numeric(frame[f"{source}_amount"], errors="coerce")
        for record_id in frame.loc[amount.eq(0), identifier].astype(str):
            findings.append({"source": source, "record_id": record_id, "severity": "Warning", "issue": "Zero amount requires review"})
    return pd.DataFrame(findings, columns=["source", "record_id", "severity", "issue"])


def validate_review_decisions(decisions: pd.DataFrame) -> None:
    """Reject invalid approval batches before an audit write."""
    required = {"ledger_id", "bank_id", "review_decision"}
    if missing := required - set(decisions.columns):
        raise ValueError(f"Review decisions are missing columns: {sorted(missing)}")
    approved = decisions.loc[decisions["review_decision"] == "Approve"]
    if approved["ledger_id"].dropna().astype(str).duplicated().any():
        raise ValueError("A ledger entry cannot be approved more than once in one review batch.")
    if approved["bank_id"].dropna().astype(str).duplicated().any():
        raise ValueError("A bank transaction cannot be approved more than once in one review batch.")


def decision_audit_rows(decisions: pd.DataFrame, *, reviewer: str, reason: str = "") -> pd.DataFrame:
    """Create append-only audit events for explicit approval/rejection actions."""
    selected = decisions.loc[decisions["review_decision"].isin(["Approve", "Reject"])].copy()
    if selected.empty:
        return selected
    validate_review_decisions(selected)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    selected["reviewer_identity"] = reviewer.strip() or "Unspecified reviewer"
    selected["decision_reason"] = reason.strip()
    selected["previous_status"] = "Pending"
    selected["new_status"] = selected["review_decision"].map({"Approve": "Approved", "Reject": "Rejected"})
    selected["reviewed_at"] = timestamp
    selected["audit_event_id"] = [hashlib.sha256(f"{timestamp}|{row.ledger_id}|{row.bank_id}|{row.review_decision}|{index}".encode()).hexdigest()[:20] for index, row in enumerate(selected.itertuples(index=False))]
    return selected


def review_exports(edited: pd.DataFrame, ledger: pd.DataFrame, bank: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build decision-state and unmatched exports; pending rows are not matches."""
    validate_review_decisions(edited)
    approved = edited.loc[edited["review_decision"] == "Approve"].copy()
    rejected = edited.loc[edited["review_decision"] == "Reject"].copy()
    pending = edited.loc[edited["review_decision"] == "Pending"].copy()
    ledger_ids = set(approved["ledger_id"].dropna().astype(str))
    bank_ids = set(approved["bank_id"].dropna().astype(str))
    return {"approved_matches": approved, "rejected_proposals": rejected, "pending_reviews": pending,
            "unmatched_ledger": ledger.loc[~ledger["ledger_id"].astype(str).isin(ledger_ids)].copy(),
            "unmatched_bank": bank.loc[~bank["bank_id"].astype(str).isin(bank_ids)].copy()}
