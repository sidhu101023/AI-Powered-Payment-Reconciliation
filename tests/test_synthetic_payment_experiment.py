import pandas as pd

from src.synthetic_payment_experiment import generate, evaluate_identifier_first


def test_generator_is_reproducible_and_keeps_truth_separate(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    assert generate(first, seed=7) == generate(second, seed=7)
    assert (first / "ground_truth.csv").read_bytes() == (second / "ground_truth.csv").read_bytes()
    bank = pd.read_csv(first / "bank_transactions.csv")
    truth = pd.read_csv(first / "ground_truth.csv")
    assert "is_match" not in bank.columns
    assert {"exact_payment_id", "unmatched_bank", "partial_payment", "grouped_payment"}.issubset(set(truth.scenario))


def test_identifier_evaluation_does_not_need_truth_to_predict(tmp_path):
    data_dir = tmp_path / "data"
    generate(data_dir, seed=9)
    result = evaluate_identifier_first(data_dir, tmp_path / "comparison.csv")
    assert result.iloc[0].identifier_matches > 0
    assert result.iloc[0].human_review > 0
    assert (tmp_path / "synthetic_identifier_predictions.csv").exists()
