
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
keys = pd.read_csv(DATA / "match_key.csv")

keys = keys[keys["kind"].astype(str).str.lower() != "duplicate_line"]

# Reconstruct the original date-based partitions.
bank["parsed_date"] = pd.to_datetime(bank["posted_date"], errors="coerce")
ledger["parsed_date"] = pd.to_datetime(ledger["entry_date"], errors="coerce")

all_dates = pd.concat(
    [ledger["parsed_date"], bank["parsed_date"]],
    ignore_index=True,
).dropna()

cut1, cut2 = all_dates.quantile([0.60, 0.80]).tolist()

def period(date):
    if pd.isna(date):
        return "excluded"
    if date < cut1:
        return "train"
    if date < cut2:
        return "validation"
    return "test"

ledger["period"] = ledger["parsed_date"].map(period)
bank["period"] = bank["parsed_date"].map(period)

# Apply the same answer-key pair-aware assignment as the main script.
assignments = {"ledger": {}, "bank": {}}

for row in keys.itertuples(index=False):
    lid = int(row.ledger_entry_no)
    bid = int(row.bank_line_no)

    lrow = ledger.loc[ledger["id"] == lid]
    brow = bank.loc[bank["id"] == bid]

    if lrow.empty or brow.empty:
        raise ValueError(f"Missing key record: ledger={lid}, bank={bid}")

    dates = [
        d for d in [
            lrow.iloc[0]["parsed_date"],
            brow.iloc[0]["parsed_date"],
        ]
        if pd.notna(d)
    ]

    if not dates:
        raise ValueError(f"Both dates missing for pair {(lid, bid)}")

    pair_period = period(max(dates))

    for side, record_id in [("ledger", lid), ("bank", bid)]:
        old = assignments[side].get(record_id)
        if old is not None and old != pair_period:
            raise ValueError(
                f"Conflicting assignment: {side} {record_id}: "
                f"{old} vs {pair_period}"
            )
        assignments[side][record_id] = pair_period

ledger["period"] = ledger.apply(
    lambda r: assignments["ledger"].get(int(r["id"]), r["period"]),
    axis=1,
)
bank["period"] = bank.apply(
    lambda r: assignments["bank"].get(int(r["id"]), r["period"]),
    axis=1,
)

# Check whether a source record is assigned to multiple periods.
# A unique ID should have exactly one period.
for name, frame in [("ledger", ledger), ("bank", bank)]:
    counts = frame.groupby("id")["period"].nunique()
    bad = counts[counts > 1]
    print(f"{name}: {len(frame)} records, "
          f"{bad.size} IDs in multiple periods")

# Check known positive pairs remain in a single period.
lp = ledger.set_index("id")["period"].to_dict()
bp = bank.set_index("id")["period"].to_dict()

cross = []
for row in keys.itertuples(index=False):
    lid, bid = int(row.ledger_entry_no), int(row.bank_line_no)
    if lp.get(lid) != bp.get(bid):
        cross.append((lid, bid, lp.get(lid), bp.get(bid)))

print(f"Positive pairs crossing periods: {len(cross)}")
if cross:
    print(cross[:10])

print("\nPartition record counts:")
print("Ledger:\n", ledger["period"].value_counts().to_string())
print("Bank:\n", bank["period"].value_counts().to_string())
