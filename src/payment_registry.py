"""Persistent Payment ID lifecycle and safe identifier-based reconciliation."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import math
import re
import secrets
import sqlite3
from typing import Iterable

import pandas as pd


STATUSES = {
    "Created", "Initiated", "Completed", "Bank Reference Recorded", "Partially Reconciled", "Reconciled", "Exception",
}


def _normalise(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


class PaymentRegistry:
    """SQLite repository. It deliberately never initiates a bank transfer."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _create_schema(self) -> None:
        with self._connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS payments (
                    payment_id TEXT PRIMARY KEY,
                    supplier TEXT NOT NULL,
                    payment_amount REAL NOT NULL CHECK(payment_amount > 0),
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    instruction_reference TEXT UNIQUE,
                    bank_reference TEXT UNIQUE,
                    submission_key TEXT UNIQUE
                );
                CREATE TABLE IF NOT EXISTS payment_invoices (
                    payment_id TEXT NOT NULL REFERENCES payments(payment_id),
                    invoice_id TEXT NOT NULL,
                    allocated_amount REAL NOT NULL CHECK(allocated_amount > 0),
                    PRIMARY KEY(payment_id, invoice_id)
                );
                CREATE TABLE IF NOT EXISTS reconciliation_assignments (
                    payment_id TEXT NOT NULL REFERENCES payments(payment_id),
                    bank_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    result TEXT NOT NULL,
                    detail TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(payment_id, bank_id)
                );
                CREATE TABLE IF NOT EXISTS reconciliation_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payment_id TEXT NOT NULL REFERENCES payments(payment_id),
                    bank_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reviewer_identity TEXT,
                    decision_reason TEXT,
                    previous_status TEXT NOT NULL,
                    new_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            columns = {row[1] for row in con.execute("PRAGMA table_info(payments)")}
            if "submission_key" not in columns:
                # SQLite does not allow UNIQUE in an ADD COLUMN statement.
                # Add the nullable field first, then enforce idempotency with
                # a unique index (multiple NULLs remain valid).
                con.execute("ALTER TABLE payments ADD COLUMN submission_key TEXT")
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_payments_submission_key "
                "ON payments(submission_key)"
            )

    def _new_id(self) -> str:
        # 80 random bits, plus the UTC year, gives negligible collision risk;
        # the primary key is the final datastore-level uniqueness guarantee.
        return f"PAY-{datetime.now(timezone.utc):%Y}-{secrets.token_hex(10).upper()}"

    def create_payment(self, supplier: str, allocations: Iterable[tuple[str, float]], *,
                       instruction_reference: str | None = None,
                       submission_key: str | None = None) -> str:
        allocations = [(str(invoice).strip(), float(amount)) for invoice, amount in allocations]
        if (not supplier.strip() or not allocations or any(
            not invoice or not math.isfinite(amount) or amount <= 0
            for invoice, amount in allocations
        )):
            raise ValueError("Supplier and one or more positive invoice allocations are required.")
        total = round(sum(amount for _, amount in allocations), 2)
        if len({invoice for invoice, _ in allocations}) != len(allocations):
            raise ValueError("An invoice can be allocated only once within a payment.")
        with self._connect() as con:
            for _ in range(5):
                payment_id = self._new_id()
                try:
                    con.execute("""INSERT INTO payments
                        (payment_id, supplier, payment_amount, status, created_at,
                         instruction_reference, bank_reference, submission_key)
                        VALUES (?, ?, ?, 'Created', ?, ?, NULL, ?)""",
                                (payment_id, supplier.strip(), total,
                                 datetime.now(timezone.utc).isoformat(),
                                 instruction_reference or None, submission_key or None))
                    con.executemany("INSERT INTO payment_invoices VALUES (?, ?, ?)",
                                    [(payment_id, invoice, amount) for invoice, amount in allocations])
                    return payment_id
                except sqlite3.IntegrityError:
                    if submission_key:
                        existing = con.execute(
                            "SELECT payment_id FROM payments WHERE submission_key=?",
                            (submission_key,),
                        ).fetchone()
                        if existing:
                            return existing["payment_id"]
                    # Retry random-ID collision; instruction-reference duplicates are rejected.
                    if instruction_reference:
                        raise ValueError("That payment instruction reference already exists.")
            raise RuntimeError("Could not allocate a unique Payment ID.")

    def record_bank_reference(self, payment_id: str, bank_reference: str) -> None:
        with self._connect() as con:
            result = con.execute("UPDATE payments SET bank_reference=?, status='Bank Reference Recorded' WHERE payment_id=? AND bank_reference IS NULL",
                                 (bank_reference.strip(), payment_id)).rowcount
            if result != 1:
                raise ValueError("Payment not found, or it already has a bank reference.")

    def record_reconciliation(self, payment_id: str, bank_id: str, method: str, result: str = "approved",
                              reviewer_identity: str = "", decision_reason: str = "") -> None:
        """Append a review event and persist an approved link, never a bank transfer."""
        if method not in {"payment_id", "bank_reference", "rules", "ai_model"}:
            raise ValueError("Unknown reconciliation method.")
        if result not in {"approved", "rejected"}:
            raise ValueError("Result must be approved or rejected.")
        with self._connect() as con:
            payment = con.execute("SELECT status FROM payments WHERE payment_id=?", (payment_id,)).fetchone()
            if not payment:
                raise ValueError("Payment not found.")
            bank_id = str(bank_id)
            assigned = con.execute("SELECT payment_id FROM reconciliation_assignments WHERE bank_id=? AND result='approved'", (bank_id,)).fetchone()
            if result == "approved" and assigned and assigned["payment_id"] != payment_id:
                raise ValueError("This bank transaction is already approved for another payment.")
            previous_status = payment["status"]
            new_status = "Reconciled" if result == "approved" else previous_status
            now = datetime.now(timezone.utc).isoformat()
            con.execute("""INSERT INTO reconciliation_audit
                (payment_id, bank_id, method, decision, reviewer_identity, decision_reason, previous_status, new_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (payment_id, bank_id, method, result, reviewer_identity or None, decision_reason or None,
                 previous_status, new_status, now))
            if result == "approved":
                con.execute("""INSERT OR REPLACE INTO reconciliation_assignments
                    VALUES (?, ?, ?, ?, NULL, ?)""", (payment_id, bank_id, method, result, now))
                con.execute("UPDATE payments SET status=? WHERE payment_id=?", (new_status, payment_id))

    def reconciliation_audit(self, payment_id: str = "") -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query("SELECT * FROM reconciliation_audit WHERE payment_id LIKE ? ORDER BY audit_id", con,
                                     params=(f"%{payment_id.strip()}%",))

    def find_payments(self, query: str = "") -> pd.DataFrame:
        like = f"%{query.strip()}%"
        with self._connect() as con:
            return pd.read_sql_query("""
                SELECT p.payment_id, p.supplier, p.payment_amount, p.status, p.created_at,
                       p.instruction_reference, p.bank_reference,
                       group_concat(i.invoice_id || ': ' || printf('%.2f', i.allocated_amount), ', ') AS invoice_allocations
                FROM payments p JOIN payment_invoices i ON i.payment_id=p.payment_id
                WHERE p.payment_id LIKE ? OR i.invoice_id LIKE ? OR p.supplier LIKE ? OR COALESCE(p.bank_reference, '') LIKE ?
                GROUP BY p.payment_id ORDER BY p.created_at DESC
            """, con, params=(like, like, like, like))

    def identifier_candidates(self, bank: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return validated exact matches and review-only identifier exceptions.

        An identifier alone is insufficient: exactly one payment, one bank line,
        matching signed amount, and no previous bank assignment are required.
        """
        records = self.find_payments()
        matches, exceptions = [], []
        if records.empty:
            return pd.DataFrame(), pd.DataFrame()
        for row in bank.itertuples(index=False):
            text = _normalise(f"{row.bank_description} {getattr(row, 'bank_reference', '')} {getattr(row, 'payment_id', '')}")
            hits = records[records.payment_id.map(_normalise).map(lambda value: bool(value and value in text))]
            if hits.empty:
                refs = records[records.bank_reference.fillna("").map(_normalise).map(lambda value: bool(value and value in text))]
                method, hits = "bank_reference", refs
            else:
                method = "payment_id"
            if len(hits) != 1:
                if len(hits) > 1:
                    exceptions.append({"bank_id": row.bank_id, "exception_type": "AMBIGUOUS_IDENTIFIER", "detail": "More than one payment identifier matched."})
                continue
            payment = hits.iloc[0]
            if abs(abs(float(row.bank_amount)) - float(payment.payment_amount)) > 0.01:
                exceptions.append({"bank_id": row.bank_id, "payment_id": payment.payment_id, "exception_type": "AMOUNT_MISMATCH", "detail": "Identifier found but amount differs; review required."})
                continue
            matches.append({"ledger_id": None, "bank_id": row.bank_id, "payment_id": payment.payment_id,
                            "suggestion_score": 1.0, "matching_method": method,
                            "review_decision": "Pending"})
        matched_frame = pd.DataFrame(matches)
        # A payment referenced by multiple bank lines is a split/duplicate
        # ambiguity unless it has been explicitly modelled and reviewed.
        if not matched_frame.empty:
            duplicate_mask = matched_frame.duplicated("payment_id", keep=False)
            for duplicate in matched_frame.loc[duplicate_mask].itertuples(index=False):
                exceptions.append({
                    "bank_id": duplicate.bank_id,
                    "payment_id": duplicate.payment_id,
                    "exception_type": "DUPLICATE_BANK_REFERENCE",
                    "detail": "Payment ID appears on multiple imported bank lines; review required.",
                })
            matched_frame = matched_frame.loc[~duplicate_mask].reset_index(drop=True)
        return matched_frame, pd.DataFrame(exceptions)
