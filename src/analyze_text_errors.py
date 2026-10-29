
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

scores = pd.read_csv(REPORTS / "text_matcher_test_scores.csv")
matches = pd.read_csv(REPORTS / "text_matcher_matches.csv")
bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")

# Identify incorrect and missed assignments after one-to-one matching.
selected = set(
    zip(matches["ledger_id"].astype(int), matches["bank_id"].astype(int))
)

scores["ledger_id"] = scores["ledger_id"].astype(int)
scores["bank_id"] = scores["bank_id"].astype(int)

expected = set(
    zip(
        scores.loc[scores["label"] == 1, "ledger_id"],
        scores.loc[scores["label"] == 1, "bank_id"],
    )
)

predicted = selected

false_positives = predicted - expected
false_negatives = expected - predicted

def describe_pair(pair, error_type):
    ledger_id, bank_id = pair

    l = ledger.loc[ledger["id"] == ledger_id].iloc[0]
    b = bank.loc[bank["id"] == bank_id].iloc[0]

    score_row = scores[
        (scores["ledger_id"] == ledger_id)
        & (scores["bank_id"] == bank_id)
    ]

    probability = (
        float(score_row.iloc[0]["probability"])
        if not score_row.empty else None
    )

    return {
        "error_type": error_type,
        "ledger_id": ledger_id,
        "bank_id": bank_id,
        "probability": probability,
        "ledger_reference": l.get("reference", ""),
        "counterparty": l.get("counterparty", ""),
        "ledger_amount": l["amount"],
        "ledger_date": l["entry_date"],
        "bank_narrative": b.get("narrative", ""),
        "bank_amount": b["amount"],
        "bank_date": b["posted_date"],
    }

rows = []
for pair in sorted(false_positives):
    rows.append(describe_pair(pair, "false_positive"))

for pair in sorted(false_negatives):
    rows.append(describe_pair(pair, "false_negative"))

report = pd.DataFrame(rows)
output = REPORTS / "text_matcher_errors.csv"
report.to_csv(output, index=False)

print("False positives:", len(false_positives))
print("False negatives:", len(false_negatives))
print("\nError details:")
print(report.to_string(index=False))
print(f"\nSaved: {output}")
