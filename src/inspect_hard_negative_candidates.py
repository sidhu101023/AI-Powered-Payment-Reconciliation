from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

scores_path = REPORTS / "neural_experiment_test_scores.csv"
bank_path = DATA / "bank_lines.csv"
ledger_path = DATA / "ledger_entries.csv"

for path in (scores_path, bank_path, ledger_path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

scores = pd.read_csv(scores_path)
bank = pd.read_csv(bank_path)
ledger = pd.read_csv(ledger_path)

required = {"model", "ledger_id", "bank_id", "label", "probability"}
missing = required - set(scores.columns)
if missing:
    raise ValueError(f"Missing score columns: {sorted(missing)}")

# Use only negative candidates from the shared test set.
negatives = scores[scores["label"] == 0].copy()

# Summarize each candidate across all models.
summary = negatives.groupby(["ledger_id", "bank_id"]).agg(
    model_count=("model", "nunique"),
    high_score_models=("probability", lambda s: int((s >= 0.80).sum())),
    max_probability=("probability", "max"),
    mean_probability=("probability", "mean"),
).reset_index()

# A candidate is more interesting when several models score it highly.
summary = summary.sort_values(
    ["high_score_models", "max_probability", "mean_probability"],
    ascending=False,
)

ledger = ledger.copy()
bank = bank.copy()
ledger["id"] = ledger["id"].astype(int)
bank["id"] = bank["id"].astype(int)

print("=== TOP POTENTIAL HARD NEGATIVES ===")
print("These pairs are absent from the known positive match key.")
print("They are NOT confirmed non-matches until reviewed.")
print("High score means probability >= 0.80.\n")

for row in summary.head(15).itertuples(index=False):
    lid = int(row.ledger_id)
    bid = int(row.bank_id)

    lrows = ledger.loc[ledger["id"] == lid]
    brows = bank.loc[bank["id"] == bid]

    print("\n" + "=" * 90)
    print(
        f"Ledger ID={lid} | Bank ID={bid} | "
        f"Models scoring >=0.80: {row.high_score_models}/{row.model_count} | "
        f"Max={row.max_probability:.4f} | "
        f"Mean={row.mean_probability:.4f}"
    )

    if lrows.empty or brows.empty:
        print("Source record missing.")
        continue

    l = lrows.iloc[0]
    b = brows.iloc[0]

    print(
        f"LEDGER: date={l.get('entry_date')} | "
        f"amount={l.get('amount')} | "
        f"reference={l.get('reference')} | "
        f"counterparty={l.get('counterparty')}"
    )
    print(
        f"BANK:   date={b.get('posted_date')} | "
        f"amount={b.get('amount')} | "
        f"narrative={b.get('narrative')}"
    )

    print("MODEL SCORES:")
    pair_scores = negatives[
        (negatives["ledger_id"] == lid)
        & (negatives["bank_id"] == bid)
    ].sort_values("probability", ascending=False)

    for score in pair_scores.itertuples(index=False):
        print(f"  {score.model:28s} {score.probability:.6f}")

# Save a separate summary for review.
out = REPORTS / "hard_negative_candidate_review.csv"
summary.to_csv(out, index=False)

print(f"\nSaved candidate summary to: {out}")
print(f"Potential negative pairs inspected: {min(15, len(summary))}")
