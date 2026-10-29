
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data"


def test_baseline_predictions_have_unique_assignments():
    predictions = pd.read_csv(REPORTS / "baseline_predictions.csv")

    assert not predictions["ledger_id"].duplicated().any()
    assert not predictions["bank_id"].duplicated().any()


def test_predictions_reference_existing_records():
    predictions = pd.read_csv(REPORTS / "baseline_predictions.csv")
    ledger = pd.read_csv(DATA / "ledger_entries.csv")
    bank = pd.read_csv(DATA / "bank_lines.csv")

    assert set(predictions["ledger_id"]).issubset(set(ledger["id"]))
    assert set(predictions["bank_id"]).issubset(set(bank["id"]))


def test_duplicate_detector_finds_expected_duplicate_groups():
    duplicates = pd.read_csv(REPORTS / "duplicate_bank_lines.csv")

    assert duplicates["duplicate_group"].nunique() == 3
    assert len(duplicates) == 6


def test_exception_report_contains_expected_categories():
    exceptions = pd.read_csv(REPORTS / "reconciliation_exceptions.csv")

    expected = {
        "unmatched_ledger_entry",
        "unmatched_bank_line",
        "potential_duplicate_bank_lines",
        "matched_pair_needs_review",
    }

    assert expected.issubset(set(exceptions["exception_type"]))


def test_exception_report_is_not_empty():
    exceptions = pd.read_csv(REPORTS / "reconciliation_exceptions.csv")

    assert len(exceptions) > 0
