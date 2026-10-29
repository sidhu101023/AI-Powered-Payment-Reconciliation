
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
scores = pd.read_csv(REPORTS / "same_split_test_scores.csv")

baseline = scores[scores["model"] == "RuleBasedBaseline"].copy()

# Inspect the highest-scoring negative candidates.
suspects = baseline[
    baseline["label"] == 0
].nlargest(5, "probability")

print("=== HIGHEST-SCORING NEGATIVE CANDIDATES ===")

for row in suspects.itertuples(index=False):
    lid = int(row.ledger_id)
    bid = int(row.bank_id)

    l = ledger.loc[ledger["id"] == lid]
    b = bank.loc[bank["id"] == bid]

    print("\n" + "=" * 75)
    print(
        f"Ledger ID={lid} | Bank ID={bid} | "
        f"Score={row.probability:.6f}"
    )

    if l.empty or b.empty:
        print("Record missing from source data.")
        continue

    print("\nLedger record:")
    print(l.to_string(index=False))

    print("\nBank record:")
    print(b.to_string(index=False))

print("\nInspection complete.")
