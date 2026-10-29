import pandas as pd
from pathlib import Path

root = Path.cwd()
reports = root / "reports"
data = root / "data"

challenge = pd.read_csv(
    reports / "hard_negative_verified_unmatched_challenge.csv"
)
bank = pd.read_csv(data / "bank_lines.csv")
ledger = pd.read_csv(data / "ledger_entries.csv")

false_pairs = challenge[
    (challenge["model"] == "TFIDF_Only")
    & challenge["selected_by_model"].astype(str).str.lower().eq("true")
]

for row in false_pairs.itertuples(index=False):
    l = ledger.loc[ledger["id"] == int(row.ledger_id)]
    b = bank.loc[bank["id"] == int(row.bank_id)]

    print("\n" + "=" * 75)
    print(
        f"Ledger={row.ledger_id} | Bank={row.bank_id} | "
        f"Score={row.probability:.6f}"
    )
    print("Ledger:")
    print(l.to_string(index=False))
    print("Bank:")
    print(b.to_string(index=False))
