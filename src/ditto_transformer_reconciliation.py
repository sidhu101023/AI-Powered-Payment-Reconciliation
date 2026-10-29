
from pathlib import Path

import pandas as pd

from ditto_transformer_predictor import DittoTransformerPredictor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "reports"
OUTPUT.mkdir(exist_ok=True)

# Load the source records and exception labels.
bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
truth = pd.read_csv(DATA / "match_key.csv")

# Exclude known duplicate-line exception pairs from ordinary candidates.
duplicate_pairs = set(
    zip(
        truth.loc[
            truth["kind"] == "duplicate_line", "ledger_entry_no"
        ].astype(int),
        truth.loc[
            truth["kind"] == "duplicate_line", "bank_line_no"
        ].astype(int),
    )
)

# Generate plausible pairs for the same account/book.
rows = []

bank_groups = {
    str(account): group
    for account, group in bank.groupby(
        bank["account"].astype(str)
    )
}

for ledger_row in ledger.to_dict("records"):
    group = bank_groups.get(str(ledger_row["book"]))

    if group is None:
        continue

    for bank_row in group.to_dict("records"):
        pair = (
            int(ledger_row["id"]),
            int(bank_row["id"]),
        )

        if pair in duplicate_pairs:
            continue

        rows.append({
            "ledger_id": pair[0],
            "bank_id": pair[1],
            "label": 0,
            "left_reference": ledger_row["reference"],
            "left_counterparty": ledger_row["counterparty"],
            "left_amount": ledger_row["amount"],
            "left_date": ledger_row["entry_date"],
            "right_narrative": bank_row["narrative"],
            "right_amount": bank_row["amount"],
            "right_date": bank_row["posted_date"],
        })

candidates = pd.DataFrame(rows)

if candidates.empty:
    raise ValueError("No candidate pairs generated.")

print("Candidate pairs:", len(candidates))

predictor = DittoTransformerPredictor()

probabilities = []

for index, row in enumerate(candidates.itertuples(index=False), start=1):
    ledger_record = {
        "reference": row.left_reference,
        "counterparty": row.left_counterparty,
        "amount": row.left_amount,
        "entry_date": row.left_date,
    }

    bank_record = {
        "narrative": row.right_narrative,
        "amount": row.right_amount,
        "posted_date": row.right_date,
    }

    result = predictor.score_pair(ledger_record, bank_record)
    probabilities.append(result["match_probability"])

    if index % 100 == 0 or index == len(candidates):
        print(f"Scored {index}/{len(candidates)} candidates")

candidates["match_probability"] = probabilities

# Attach ground-truth labels for evaluation, not as model input.
ordinary_truth = truth[truth["kind"] != "duplicate_line"]

positive_pairs = set(
    zip(
        ordinary_truth["ledger_entry_no"].astype(int),
        ordinary_truth["bank_line_no"].astype(int),
    )
)

candidates["label"] = [
    int((int(lid), int(bid)) in positive_pairs)
    for lid, bid in zip(
        candidates["ledger_id"],
        candidates["bank_id"],
    )
]

output_path = OUTPUT / "ditto_transformer_candidate_scores.csv"
candidates.to_csv(output_path, index=False)

print("\n=== TRANSFORMER CANDIDATE SCORING ===")
print("Candidates scored:", len(candidates))
print("Positive pairs present:", int(candidates["label"].sum()))
print("Saved:", output_path)
print("Note: Scores are candidate-level predictions, not final matches.")
