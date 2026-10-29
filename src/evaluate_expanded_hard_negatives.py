from pathlib import Path
from itertools import combinations
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data"

scores = pd.read_csv(REPORTS / "neural_experiment_test_scores.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
bank = pd.read_csv(DATA / "bank_lines.csv")

for col in ["ledger_id", "bank_id"]:
    scores[col] = scores[col].astype(int)
ledger["id"] = ledger["id"].astype(int)
bank["id"] = bank["id"].astype(int)

# Candidate labels are shared across models. Retain pairs actually scored
# in the test evaluation, and use only known positive test pairs as anchors.
pairs = scores[["ledger_id", "bank_id", "label"]].drop_duplicates()
positive = pairs[pairs["label"] == 1][["ledger_id", "bank_id"]]

ledger_info = ledger.set_index("id")
bank_info = bank.set_index("id")
pair_set = set(zip(pairs["ledger_id"], pairs["bank_id"]))
positive_set = set(zip(positive["ledger_id"], positive["bank_id"]))

# Build swapped alternatives only between two known positive test pairs
# whose ledger amounts are equal and whose cross-pairs were scored as negatives.
hard_cases = []
seen = set()

for (l1, b1), (l2, b2) in combinations(sorted(positive_set), 2):
    if l1 == l2 or b1 == b2:
        continue

    amount_l1 = float(ledger_info.loc[l1, "amount"])
    amount_l2 = float(ledger_info.loc[l2, "amount"])
    amount_b1 = float(bank_info.loc[b1, "amount"])
    amount_b2 = float(bank_info.loc[b2, "amount"])

    if not (amount_l1 == amount_l2 == amount_b1 == amount_b2):
        continue

    cross1 = (l1, b2)
    cross2 = (l2, b1)

    if cross1 not in pair_set or cross2 not in pair_set:
        continue
    if cross1 in positive_set or cross2 in positive_set:
        continue

    identity = tuple(sorted([(l1, b1), (l2, b2)]))
    if identity in seen:
        continue
    seen.add(identity)

    hard_cases.append({
        "ledger_1": l1,
        "correct_bank_1": b1,
        "ledger_2": l2,
        "correct_bank_2": b2,
        "swapped_bank_for_ledger_1": b2,
        "swapped_bank_for_ledger_2": b1,
        "amount": amount_l1,
        "counterparty_1": str(ledger_info.loc[l1, "counterparty"]),
        "counterparty_2": str(ledger_info.loc[l2, "counterparty"]),
    })

if not hard_cases:
    print(
        "No additional exact-amount swapped cases found among the known "
        "positive test pairs. This is a valid finding; no cases were invented."
    )
    raise SystemExit(0)

cases_df = pd.DataFrame(hard_cases)
cases_path = REPORTS / "expanded_hard_negative_cases.csv"
cases_df.to_csv(cases_path, index=False)

# Evaluate whether each model gives each correct pair a higher score
# than its swapped alternative.
score_lookup = {
    (row.model, int(row.ledger_id), int(row.bank_id)): float(row.probability)
    for row in scores.itertuples(index=False)
}

rows = []
for case_no, case in enumerate(hard_cases, start=1):
    comparisons = [
        (case["ledger_1"], case["correct_bank_1"],
         case["swapped_bank_for_ledger_1"]),
        (case["ledger_2"], case["correct_bank_2"],
         case["swapped_bank_for_ledger_2"]),
    ]

    for model in sorted(scores["model"].unique()):
        for ledger_id, correct_bank, swapped_bank in comparisons:
            correct_score = score_lookup[(model, ledger_id, correct_bank)]
            swapped_score = score_lookup[(model, ledger_id, swapped_bank)]

            rows.append({
                "case_no": case_no,
                "model": model,
                "ledger_id": ledger_id,
                "correct_bank_id": correct_bank,
                "correct_probability": correct_score,
                "swapped_bank_id": swapped_bank,
                "swapped_probability": swapped_score,
                "correct_ranked_higher": correct_score > swapped_score,
                "score_margin": correct_score - swapped_score,
            })

detail = pd.DataFrame(rows)
detail_path = REPORTS / "expanded_hard_negative_scores.csv"
detail.to_csv(detail_path, index=False)

summary = detail.groupby("model").agg(
    comparisons=("correct_ranked_higher", "size"),
    correct_ranked_higher=("correct_ranked_higher", "sum"),
    mean_score_margin=("score_margin", "mean"),
    minimum_score_margin=("score_margin", "min"),
).reset_index()

summary["ranking_accuracy"] = (
    summary["correct_ranked_higher"] / summary["comparisons"]
)
summary_path = REPORTS / "expanded_hard_negative_summary.csv"
summary.to_csv(summary_path, index=False)

print(f"Swapped-counterparty groups found: {len(cases_df)}")
print(f"Individual pair comparisons: {len(detail)}")
print("\nSummary by model:")
print(summary.to_string(index=False))
print(f"\nSaved: {cases_path}")
print(f"Saved: {detail_path}")
print(f"Saved: {summary_path}")
