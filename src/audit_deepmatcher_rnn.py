
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
keys = pd.read_csv(DATA / "match_key.csv")

keys["kind"] = keys["kind"].astype(str).str.lower()
keys = keys[keys["kind"] != "duplicate_line"].copy()

ledger["parsed_date"] = pd.to_datetime(
    ledger["entry_date"], errors="coerce"
)
bank["parsed_date"] = pd.to_datetime(
    bank["posted_date"], errors="coerce"
)

all_dates = pd.concat(
    [ledger["parsed_date"], bank["parsed_date"]],
    ignore_index=True,
).dropna()

cut1, cut2 = all_dates.quantile([0.60, 0.80]).tolist()


def assign_period(date):
    if pd.isna(date):
        return "excluded"
    if date < cut1:
        return "train"
    if date < cut2:
        return "validation"
    return "test"


ledger["period"] = ledger["parsed_date"].map(assign_period)
bank["period"] = bank["parsed_date"].map(assign_period)

positive_pairs = set(zip(
    keys["ledger_entry_no"].astype(int),
    keys["bank_line_no"].astype(int),
))

assignments = {"ledger": {}, "bank": {}}
conflicts = []

for lid, bid in positive_pairs:
    lrow = ledger.loc[ledger["id"] == lid]
    brow = bank.loc[bank["id"] == bid]

    if lrow.empty or brow.empty:
        conflicts.append(("missing_record", lid, bid))
        continue

    dates = [
        d for d in [
            lrow.iloc[0]["parsed_date"],
            brow.iloc[0]["parsed_date"],
        ] if pd.notna(d)
    ]

    if not dates:
        conflicts.append(("missing_dates", lid, bid))
        continue

    period = assign_period(max(dates))

    for side, record_id in [("ledger", lid), ("bank", bid)]:
        old = assignments[side].get(record_id)
        if old is not None and old != period:
            conflicts.append(("cross_partition", side, record_id))
        assignments[side][record_id] = period

ledger["period"] = ledger.apply(
    lambda r: assignments["ledger"].get(int(r["id"]), r["period"]),
    axis=1,
)
bank["period"] = bank.apply(
    lambda r: assignments["bank"].get(int(r["id"]), r["period"]),
    axis=1,
)

print("=== Split audit ===")
print("Date cutoffs:", cut1, cut2)
print("Positive keys:", len(positive_pairs))
print("Assignment conflicts:", len(conflicts))
print("Ledger counts:\n", ledger["period"].value_counts().to_string())
print("Bank counts:\n", bank["period"].value_counts().to_string())

print("\n=== Positive pair partition audit ===")
ledger_period = ledger.set_index("id")["period"].to_dict()
bank_period = bank.set_index("id")["period"].to_dict()

pair_periods = {}
for lid, bid in positive_pairs:
    if lid not in ledger_period or bid not in bank_period:
        continue
    lp = ledger_period[lid]
    bp = bank_period[bid]
    pair_periods[(lid, bid)] = (lp, bp)

cross_period_pairs = [
    (pair, periods)
    for pair, periods in pair_periods.items()
    if periods[0] != periods[1]
]
print("Pairs whose records have different periods:", len(cross_period_pairs))
print("Positive pairs with both records in train:",
      sum(v == ("train", "train") for v in pair_periods.values()))
print("Positive pairs with both records in validation:",
      sum(v == ("validation", "validation") for v in pair_periods.values()))
print("Positive pairs with both records in test:",
      sum(v == ("test", "test") for v in pair_periods.values()))
print("Positive pairs involving excluded records:",
      sum("excluded" in v for v in pair_periods.values()))

print("\nNote: this reconstructs the assignment rules; it does not")
print("prove the source keys are complete or independently verified.")


print("\n=== Record partition uniqueness ===")

for name, frame in [("ledger", ledger), ("bank", bank)]:
    duplicate_ids = frame.groupby("id")["period"].nunique()
    overlapping = int((duplicate_ids > 1).sum())
    print(f"{name}: records assigned to multiple periods = {overlapping}")

print("\n=== Candidate labels ===")
test_scores = pd.read_csv(
    ROOT / "reports" / "deepmatcher_rnn_test_scores.csv"
)
test_positive = int(test_scores["label"].sum())
print("Test score rows:", len(test_scores))
print("Positive test candidates:", test_positive)
print("Negative test candidates:", len(test_scores) - test_positive)
