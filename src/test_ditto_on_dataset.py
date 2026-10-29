
from pathlib import Path

import pandas as pd

from ditto_transformer_predictor import DittoTransformerPredictor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")

predictor = DittoTransformerPredictor()

# Test one actual ledger record against one actual bank record.
ledger_row = ledger.iloc[0]
bank_row = bank.iloc[0]

result = predictor.score_pair(
    ledger_row.to_dict(),
    bank_row.to_dict(),
)

print("Ledger record:")
print(ledger_row[["id", "reference", "counterparty", "amount", "entry_date"]])

print("\nBank record:")
print(bank_row[["id", "narrative", "amount", "posted_date"]])

print("\nTransformer prediction:")
print(f"Match probability: {result['match_probability']:.4f}")
print(f"Predicted match: {result['predicted_match']}")
print(f"Threshold: {predictor.threshold}")
