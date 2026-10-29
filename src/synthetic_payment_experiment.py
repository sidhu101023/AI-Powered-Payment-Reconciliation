"""Reproducible, isolated Payment ID reconciliation experiment.

Synthetic scenario assumptions are experimental controls, not assertions about
bank behaviour. Ground truth is written separately and is never read by any
prediction function.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import argparse
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "synthetic_data" / "payment_id_v1"
DEFAULT_REPORT = ROOT / "reports" / "synthetic_payment_id_comparison.csv"
SCENARIOS = [
    "exact_payment_id", "instruction_reference", "bank_reference_different",
    "missing_payment_id", "conflicting_reference", "duplicate_bank_transaction",
    "partial_payment", "grouped_payment", "net_of_fee", "delayed_posting",
    "ambiguous_candidate", "unmatched_bank", "unpaid_invoice", "duplicate_submission",
    "reference_conflicting_evidence", "unrelated_payment_id",
]


def synthetic_id(seed: int, index: int) -> str:
    token = hashlib.sha256(f"{seed}:payment:{index}".encode()).hexdigest()[:20].upper()
    return f"PAY-2026-{token}"


def generate(output_dir: Path = DEFAULT_DATA, seed: int = 20261028) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    invoices, payments, allocations, ledger, bank, truth = [], [], [], [], [], []
    base = date(2026, 6, 1)
    for index, scenario in enumerate(SCENARIOS, 1):
        supplier = f"SUP-{index:03d}"
        supplier_name = f"Synthetic Supplier {index:03d}"
        invoice_ids = [f"SINV-{index:03d}-A"]
        if scenario == "grouped_payment":
            invoice_ids.append(f"SINV-{index:03d}-B")
        amount = float(1000 + index * 37)
        payment_amount = amount * len(invoice_ids)
        payment_id = synthetic_id(seed, index)
        instruction = f"INST-{seed}-{index:03d}"
        for invoice_no, invoice_id in enumerate(invoice_ids, 1):
            invoices.append({"invoice_id": invoice_id, "supplier_id": supplier,
                             "supplier_name": supplier_name, "invoice_date": (base + timedelta(days=index)).isoformat(),
                             "invoice_amount": amount, "currency": "INR"})
            ledger.append({"ledger_entry_id": f"LED-{invoice_id}", "invoice_id": invoice_id,
                           "payment_id": payment_id, "entry_date": (base + timedelta(days=index)).isoformat(),
                           "reference": invoice_id, "counterparty": supplier_name, "amount": -amount})
            allocations.append({"payment_id": payment_id, "invoice_id": invoice_id,
                                "allocated_amount": amount})
        payments.append({"payment_id": payment_id, "supplier_id": supplier,
                         "supplier_name": supplier_name, "instruction_reference": instruction,
                         "payment_amount": -payment_amount, "creation_timestamp": f"{base + timedelta(days=index)}T09:00:00Z",
                         "payment_status": "Completed", "scenario": scenario})
        if scenario == "unpaid_invoice":
            payments.pop(); allocations[:] = [a for a in allocations if a["payment_id"] != payment_id]
            for item in ledger[-len(invoice_ids):]: item["payment_id"] = ""
            truth.append({"bank_transaction_id": "", "payment_id": "", "invoice_id": invoice_ids[0],
                          "is_match": 0, "scenario": scenario, "review_required": 1,
                          "relationship_type": "invoice_payment"})
            continue
        if scenario == "duplicate_submission":
            # Ground truth records the duplicate instruction as an exception, not a second payment.
            payments.append({**payments[-1], "payment_id": f"DUP-{payment_id}", "payment_status": "Exception"})
        bank_id = f"SBK-{index:03d}"
        narrative = f"NEFT {supplier_name} {invoice_ids[0]}"
        exposed_payment_id, exposed_instruction, bank_reference = "", "", f"UTR-{seed}-{index:03d}"
        bank_amount = -payment_amount
        posted = base + timedelta(days=index + 1)
        positive = True
        if scenario == "exact_payment_id": exposed_payment_id = payment_id
        elif scenario == "instruction_reference": exposed_instruction = instruction
        elif scenario == "bank_reference_different": narrative += " BANKREF-X"
        elif scenario == "missing_payment_id": pass
        elif scenario == "conflicting_reference": exposed_payment_id = synthetic_id(seed, (index % len(SCENARIOS)) + 1)
        elif scenario == "duplicate_bank_transaction": exposed_payment_id = payment_id
        elif scenario == "partial_payment": bank_amount = -(amount * 0.5)
        elif scenario == "net_of_fee": bank_amount += 5
        elif scenario == "delayed_posting": posted += timedelta(days=12)
        elif scenario == "ambiguous_candidate": narrative = "NEFT Synthetic Supplier"
        elif scenario == "unmatched_bank": positive = False; narrative = "UNRELATED CASH DEPOSIT"; bank_amount = -777.0
        elif scenario == "reference_conflicting_evidence": exposed_instruction = instruction; bank_amount = -(payment_amount + 100)
        elif scenario == "unrelated_payment_id": exposed_payment_id = payment_id; positive = False; bank_amount = -333.0
        bank.append({"bank_transaction_id": bank_id, "posted_date": posted.isoformat(), "narrative": narrative,
                     "amount": bank_amount, "bank_reference": bank_reference, "payment_id": exposed_payment_id,
                     "instruction_reference": exposed_instruction, "scenario": scenario})
        if scenario == "duplicate_bank_transaction":
            bank.append({**bank[-1], "bank_transaction_id": f"{bank_id}-DUP"})
            truth.append({"bank_transaction_id": f"{bank_id}-DUP", "payment_id": "", "invoice_id": invoice_ids[0],
                          "is_match": 0, "scenario": scenario, "review_required": 1,
                          "relationship_type": "bank_payment"})
        truth.append({"bank_transaction_id": bank_id, "payment_id": payment_id if positive else "", "invoice_id": invoice_ids[0],
                      "is_match": int(positive), "scenario": scenario,
                      "review_required": int(scenario in {"conflicting_reference", "duplicate_bank_transaction", "net_of_fee", "ambiguous_candidate", "reference_conflicting_evidence", "unrelated_payment_id"}),
                      "relationship_type": "bank_payment"})
    pd.DataFrame(invoices).to_csv(output_dir / "invoices.csv", index=False)
    pd.DataFrame(payments).to_csv(output_dir / "payments.csv", index=False)
    pd.DataFrame(allocations).to_csv(output_dir / "payment_allocations.csv", index=False)
    pd.DataFrame(ledger).to_csv(output_dir / "ledger.csv", index=False)
    pd.DataFrame(bank).to_csv(output_dir / "bank_transactions.csv", index=False)
    pd.DataFrame(truth).to_csv(output_dir / "ground_truth.csv", index=False)
    assumptions = {"seed": seed, "scenarios": SCENARIOS, "reference_retention": "one controlled scenario each for payment and instruction references", "amount_rules": "partial=50%; net fee=INR 5; conflicting reference=INR 100", "posting_delay": "delayed scenario=12 additional days", "warning": "Synthetic controls are not claims about bank behaviour."}
    (output_dir / "README.json").write_text(json.dumps(assumptions, indent=2), encoding="utf-8")
    return {"invoices": len(invoices), "ledger": len(ledger), "payments": len(payments), "allocations": len(allocations), "bank_transactions": len(bank), "truth_rows": len(truth)}


def _identifier_predictions(payments: pd.DataFrame, bank: pd.DataFrame) -> pd.DataFrame:
    """Prediction-only: deliberately never reads ground_truth.csv."""
    rows = []
    for transaction in bank.itertuples(index=False):
        candidates = payments[payments.payment_id.astype(str) == str(transaction.payment_id)]
        method = "payment_id"
        if candidates.empty:
            candidates = payments[payments.instruction_reference.astype(str) == str(transaction.instruction_reference)]
            method = "instruction_reference"
        if len(candidates) != 1:
            rows.append({"bank_transaction_id": transaction.bank_transaction_id, "payment_id": "", "method": "human_review", "status": "unresolved"}); continue
        payment = candidates.iloc[0]
        if abs(abs(float(transaction.amount)) - abs(float(payment.payment_amount))) > 0.01:
            rows.append({"bank_transaction_id": transaction.bank_transaction_id, "payment_id": payment.payment_id, "method": method, "status": "human_review"}); continue
        rows.append({"bank_transaction_id": transaction.bank_transaction_id, "payment_id": payment.payment_id, "method": method, "status": "automatic"})
    return pd.DataFrame(rows)


def evaluate_identifier_first(data_dir: Path = DEFAULT_DATA, output: Path = DEFAULT_REPORT) -> pd.DataFrame:
    """Evaluate reference-first predictions; truth is loaded only after prediction."""
    payments = pd.read_csv(data_dir / "payments.csv")
    bank = pd.read_csv(data_dir / "bank_transactions.csv")
    predictions = _identifier_predictions(payments, bank)
    truth = pd.read_csv(data_dir / "ground_truth.csv")
    truth = truth[truth.relationship_type == "bank_payment"]
    scored = truth.merge(predictions, on="bank_transaction_id", how="left")
    tp = int(((scored.is_match == 1) & (scored.payment_id_x == scored.payment_id_y)).sum())
    fp = int(((scored.payment_id_y.fillna("") != "") & ((scored.is_match == 0) | (scored.payment_id_x != scored.payment_id_y))).sum())
    fn = int(((scored.is_match == 1) & (scored.payment_id_x != scored.payment_id_y)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    result = pd.DataFrame([{"approach": "Payment ID/reference first (AI fallback not run)", "tp": tp, "fp": fp, "fn": fn,
                            "precision": precision, "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
                            "identifier_matches": int((predictions.status == "automatic").sum()), "ai_fallback_matches": 0,
                            "human_review": int((predictions.status == "human_review").sum()), "unresolved": int((predictions.status == "unresolved").sum()),
                            "scope": "synthetic; identifier layer only"}])
    output.parent.mkdir(parents=True, exist_ok=True); result.to_csv(output, index=False)
    predictions.to_csv(output.with_name("synthetic_identifier_predictions.csv"), index=False)
    return result


def _model_predictions(ledger: pd.DataFrame, bank: pd.DataFrame, model: str) -> pd.DataFrame:
    """Score synthetic ledger-bank candidates using an unchanged saved model."""
    if model == "original_transformer":
        try:
            from src.ditto_transformer_predictor import DittoTransformerPredictor
        except ModuleNotFoundError:
            from ditto_transformer_predictor import DittoTransformerPredictor
        predictor = DittoTransformerPredictor()
    else:
        try:
            from src.hybrid_transformer_predictor import HybridTransformerPredictor
        except ModuleNotFoundError:
            from hybrid_transformer_predictor import HybridTransformerPredictor
        predictor = HybridTransformerPredictor()
    ledger = ledger[ledger["payment_id"].notna() & ledger["payment_id"].astype(str).ne("")]
    candidates, pair_inputs, pair_metadata = [], [], []
    for b in bank.itertuples(index=False):
        for l in ledger.itertuples(index=False):
            days = abs((pd.Timestamp(b.posted_date) - pd.Timestamp(l.entry_date)).days)
            if days > 21:
                continue
            left = {"reference": l.reference, "counterparty": l.counterparty, "amount": l.amount, "entry_date": l.entry_date}
            right = {"narrative": b.narrative, "amount": b.amount, "posted_date": b.posted_date}
            pair_inputs.append((left, right))
            pair_metadata.append((b.bank_transaction_id, l.payment_id))
    for (bank_transaction_id, payment_id), result in zip(pair_metadata, predictor.score_pairs(pair_inputs, batch_size=32)):
        candidates.append({"bank_transaction_id": bank_transaction_id, "payment_id": payment_id,
                           "score": result["match_probability"], "method": model})
    frame = pd.DataFrame(candidates)
    if frame.empty:
        return pd.DataFrame(columns=["bank_transaction_id", "payment_id", "method", "status"])
    frame = frame.sort_values("score", ascending=False)
    used_payments, selected = set(), []
    for row in frame.itertuples(index=False):
        if row.payment_id in used_payments:
            continue
        selected.append({"bank_transaction_id": row.bank_transaction_id, "payment_id": row.payment_id,
                         "method": row.method, "status": "automatic"})
        used_payments.add(row.payment_id)
    return pd.DataFrame(selected)


def _metrics(name: str, predictions: pd.DataFrame, truth: pd.DataFrame, fallback: int = 0) -> dict:
    scored = truth.merge(predictions[["bank_transaction_id", "payment_id", "method", "status"]], on="bank_transaction_id", how="left", suffixes=("_truth", "_pred"))
    actual = scored.payment_id_truth.fillna(""); predicted = scored.payment_id_pred.fillna("")
    tp = int(((scored.is_match == 1) & (actual == predicted) & (predicted != "")).sum())
    fp = int(((predicted != "") & ((scored.is_match == 0) | (actual != predicted))).sum())
    fn = int(((scored.is_match == 1) & (actual != predicted)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    return {"approach": name, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "identifier_matches": int((predictions.method.isin(["payment_id", "instruction_reference"])).sum()) if not predictions.empty else 0,
            "ai_fallback_matches": fallback, "human_review": int((predictions.status == "human_review").sum()) if not predictions.empty else 0,
            "unresolved": int(predicted.eq("").sum()), "scope": "synthetic ledger-bank representation; grouped/partial cases are review-only"}


def evaluate_all_approaches(data_dir: Path = DEFAULT_DATA, output: Path = DEFAULT_REPORT) -> pd.DataFrame:
    """Three unchanged-model comparisons; truth is joined only after predictions."""
    payments, ledger, bank = (pd.read_csv(data_dir / name) for name in ("payments.csv", "ledger.csv", "bank_transactions.csv"))
    truth = pd.read_csv(data_dir / "ground_truth.csv")
    truth = truth[truth.relationship_type == "bank_payment"]
    original = _model_predictions(ledger, bank, "original_transformer")
    hybrid = _model_predictions(ledger, bank, "hybrid_transformer")
    identifiers = _identifier_predictions(payments, bank)
    unresolved = set(identifiers.loc[identifiers.status != "automatic", "bank_transaction_id"])
    fallback = hybrid[hybrid.bank_transaction_id.isin(unresolved)].copy()
    combined = pd.concat([identifiers[identifiers.status == "automatic"], fallback], ignore_index=True)
    results = pd.DataFrame([
        _metrics("Original Transformer", original, truth),
        _metrics("Hybrid Transformer", hybrid, truth),
        _metrics("Payment ID/reference + Hybrid fallback", combined, truth, fallback=len(fallback)),
    ])
    output.parent.mkdir(parents=True, exist_ok=True); results.to_csv(output, index=False)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--generate", action="store_true"); parser.add_argument("--evaluate", action="store_true"); parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    if args.generate: print(generate(args.data_dir))
    if args.evaluate: print(evaluate_all_approaches(args.data_dir).to_string(index=False))
