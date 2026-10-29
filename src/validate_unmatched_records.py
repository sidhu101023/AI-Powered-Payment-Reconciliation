
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

unmatched = pd.read_csv(ROOT / "reports" / "unmatched_transactions.csv")
bank = pd.read_csv(ROOT / "data" / "bank_lines.csv")
ledger = pd.read_csv(ROOT / "data" / "ledger_entries.csv")

issues = []

for _, row in unmatched.iterrows():
    side = str(row["side"]).strip().lower()
    record_no = row["record_no"]

    if side == "bank":
        source_df = bank
        date_col = "posted_date"
        book_col = "account"
    elif side == "ledger":
        source_df = ledger
        date_col = "entry_date"
        book_col = "book"
    else:
        issues.append({
            "side": side,
            "record_no": record_no,
            "issue": "Unknown side",
        })
        continue

    # Check both possible conventions; do not assume they are identical.
    id_matches = source_df[
        source_df["id"].astype(str) == str(record_no)
    ]

    if len(id_matches) != 1:
        issues.append({
            "side": side,
            "record_no": record_no,
            "issue": f"ID lookup returned {len(id_matches)} rows",
        })
        continue

    source_row = id_matches.iloc[0]

    checks = {
        "book/account": str(source_row[book_col])
        == str(row["book"]),
        "amount": abs(float(source_row["amount"]) - float(row["amount"]))
        < 0.005,
        "date": str(source_row[date_col])[:10]
        == str(row["date"])[:10],
    }

    for field, passed in checks.items():
        if not passed:
            issues.append({
                "side": side,
                "record_no": record_no,
                "issue": f"{field} mismatch",
            })

print("Unmatched report rows:", len(unmatched))
print("Identifier/data integrity issues:", len(issues))

if issues:
    print(pd.DataFrame(issues).to_string(index=False))
else:
    print("All unmatched rows match their source records.")
