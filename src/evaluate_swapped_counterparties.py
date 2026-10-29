from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data"

scores = pd.read_csv(REPORTS / "neural_experiment_test_scores.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
bank = pd.read_csv(DATA / "bank_lines.csv")
key = pd.read_csv(DATA / "match_key.csv")

# These two correct pairs are confirmed by the match key.
correct_pairs = {(161, 163), (162, 164)}
swapped_pairs = {(161, 164), (162, 163)}
all_pairs = correct_pairs | swapped_pairs

scores["ledger_id"] = scores["ledger_id"].astype(int)
scores["bank_id"] = scores["bank_id"].astype(int)

# Check that every pair is present in the test score file for each model.
models = sorted(scores["model"].dropna().unique())
present = set(zip(scores["ledger_id"], scores["bank_id"]))

print("Correct pairs in match key:")
print(
    key[key["ledger_entry_no"].astype(int).isin([161, 162])][
        ["ledger_entry_no", "bank_line_no", "kind"]
    ].to_string(index=False)
)

print("\nPair presence in test scores:")
for pair in sorted(all_pairs):
    print(f"Ledger {pair[0]} -> bank {pair[1]}: {'present' if pair in present else 'NOT PRESENT'}")

missing = all_pairs - present
if missing:
    print("\nCannot compare all four pairs from the test-score file.")
    print("Missing pairs:", sorted(missing))
    print("This means these cases are not all available in the current test scores.")
    print("No scores will be invented or substituted.")
    raise SystemExit(0)

# Show transaction details for interpretation.
ledger["id"] = ledger["id"].astype(int)
bank["id"] = bank["id"].astype(int)

print("\nLedger details:")
print(
    ledger[ledger["id"].isin([161, 162])][
        ["id", "entry_date", "reference", "counterparty", "amount"]
    ].to_string(index=False)
)

print("\nBank details:")
print(
    bank[bank["id"].isin([163, 164])][
        ["id", "posted_date", "narrative", "amount"]
    ].to_string(index=False)
)

# Compare correct and swapped pair probabilities per model.
rows = []
for model in models:
    model_scores = scores[scores["model"] == model].set_index(
        ["ledger_id", "bank_id"]
    )["probability"]

    for ledger_id, correct_bank in sorted(correct_pairs):
        swapped_bank = next(
            b for l, b in swapped_pairs if l == ledger_id
        )
        correct_score = float(model_scores.loc[(ledger_id, correct_bank)])
        swapped_score = float(model_scores.loc[(ledger_id, swapped_bank)])

        rows.append({
            "model": model,
            "ledger_id": ledger_id,
            "correct_bank_id": correct_bank,
            "correct_probability": correct_score,
            "swapped_bank_id": swapped_bank,
            "swapped_probability": swapped_score,
            "correct_ranked_higher": correct_score > swapped_score,
            "score_margin": correct_score - swapped_score,
        })

result = pd.DataFrame(rows)
out = REPORTS / "swapped_counterparty_evaluation.csv"
result.to_csv(out, index=False)

print("\nPairwise results:")
print(result.to_string(index=False))

print("\nSummary by model:")
summary = result.groupby("model").agg(
    cases=("correct_ranked_higher", "size"),
    correct_ranked_higher=("correct_ranked_higher", "sum"),
    mean_score_margin=("score_margin", "mean"),
).reset_index()
print(summary.to_string(index=False))

summary.to_csv(REPORTS / "swapped_counterparty_summary.csv", index=False)
print(f"\nSaved: {out}")
print(f"Saved: {REPORTS / 'swapped_counterparty_summary.csv'}")
