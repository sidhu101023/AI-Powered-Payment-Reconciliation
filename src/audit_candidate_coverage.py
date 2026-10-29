
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

bank = pd.read_csv(ROOT / "data" / "bank_lines.csv")
ledger = pd.read_csv(ROOT / "data" / "ledger_entries.csv")
keys = pd.read_csv(ROOT / "data" / "match_key.csv")

bank["id"] = bank["id"].astype(int)
ledger["id"] = ledger["id"].astype(int)
bank["parsed_date"] = pd.to_datetime(bank["posted_date"], errors="coerce")
ledger["parsed_date"] = pd.to_datetime(ledger["entry_date"], errors="coerce")

valid_keys = keys[
    keys["kind"].astype(str).str.lower() != "duplicate_line"
].copy()
valid_keys["ledger_entry_no"] = valid_keys["ledger_entry_no"].astype(int)
valid_keys["bank_line_no"] = valid_keys["bank_line_no"].astype(int)

if valid_keys.duplicated(["ledger_entry_no", "bank_line_no"]).any():
    raise ValueError("Duplicate positive pairs exist in match_key.csv.")

all_dates = pd.concat(
    [ledger["parsed_date"], bank["parsed_date"]], ignore_index=True
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

ledger_rows = ledger.set_index("id").to_dict("index")
bank_rows = bank.set_index("id").to_dict("index")

ledger_periods = ledger.set_index("id")["period"].to_dict()
bank_periods = bank.set_index("id")["period"].to_dict()

record_assignments = {"ledger": {}, "bank": {}}

# Assign every known positive pair together, using the later date.
for row in valid_keys.itertuples(index=False):
    lid = int(row.ledger_entry_no)
    bid = int(row.bank_line_no)

    if lid not in ledger_rows or bid not in bank_rows:
        raise ValueError(f"Missing source record for pair {lid}, {bid}")

    dates = [
        ledger_rows[lid]["parsed_date"],
        bank_rows[bid]["parsed_date"],
    ]
    dates = [d for d in dates if pd.notna(d)]
    if not dates:
        raise ValueError(f"Both dates missing for pair {lid}, {bid}")

    pair_period = assign_period(max(dates))

    for side, rid in [("ledger", lid), ("bank", bid)]:
        prior = record_assignments[side].get(rid)
        if prior is not None and prior != pair_period:
            raise ValueError(
                f"Record {side}:{rid} is assigned to both "
                f"{prior} and {pair_period}"
            )
        record_assignments[side][rid] = pair_period

ledger["period"] = ledger.apply(
    lambda r: record_assignments["ledger"].get(int(r["id"]), r["period"]),
    axis=1,
)
bank["period"] = bank.apply(
    lambda r: record_assignments["bank"].get(int(r["id"]), r["period"]),
    axis=1,
)

ledger_rows = ledger.set_index("id").to_dict("index")
bank_rows = bank.set_index("id").to_dict("index")

results = []

for row in valid_keys.itertuples(index=False):
    lid = int(row.ledger_entry_no)
    bid = int(row.bank_line_no)
    l = ledger_rows[lid]
    b = bank_rows[bid]

    if l["book"] != b["account"]:
        status = "account_book_mismatch"
    elif l["period"] != b["period"]:
        status = "cross_partition_pair"
    elif l["period"] == "excluded":
        status = "date_excluded"
    else:
        status = "candidate_should_exist"

    results.append({
        "ledger_id": lid,
        "bank_id": bid,
        "match_kind": row.kind,
        "period": l["period"],
        "candidate_status": status,
    })

audit = pd.DataFrame(results)
out = ROOT / "reports" / "candidate_coverage_audit.csv"
audit.to_csv(out, index=False)

print("=== PAIR-AWARE CANDIDATE COVERAGE ===")
print(f"Training cutoff:   {pd.Timestamp(cut1).date()}")
print(f"Validation cutoff: {pd.Timestamp(cut2).date()}")
print(audit["candidate_status"].value_counts().to_string())

print("\n=== POSITIVE PAIRS BY PARTITION ===")
print(pd.crosstab(audit["period"], audit["candidate_status"]).to_string())

print("\n=== BY MATCH CATEGORY ===")
print(pd.crosstab(
    audit["match_kind"], audit["candidate_status"]
).to_string())

print("\nAudit saved to:", out)
