from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

scores = pd.read_csv(REPORTS / "neural_experiment_test_scores.csv")
unmatched = pd.read_csv(DATA / "unmatched_key.csv")
matches = pd.read_csv(REPORTS / "neural_experiment_test_matches.csv")

# Explicitly unmatched record IDs, separated by side.
unmatched_ledger = set(
    unmatched.loc[
        unmatched["side"].str.lower() == "ledger", "record_no"
    ].astype(int)
)
unmatched_bank = set(
    unmatched.loc[
        unmatched["side"].str.lower() == "bank", "record_no"
    ].astype(int)
)

# Only pairs where BOTH records are explicitly designated unmatched.
challenge = scores[
    scores["ledger_id"].isin(unmatched_ledger)
    & scores["bank_id"].isin(unmatched_bank)
].copy()

if challenge.empty:
    print("No eligible verified-unmatched pairs found in test candidates.")
    raise SystemExit(0)

# These pairs are verified negative for this challenge because both
# source records are designated unmatched in the reconciliation key.
challenge["verified_unmatched_pair"] = True

# Mark whether the model actually selected the pair at its validation-
# selected threshold, based on the saved final test matches.
selected = set(
    zip(
        matches["model"].astype(str),
        matches["ledger_id"].astype(int),
        matches["bank_id"].astype(int),
    )
)
challenge["selected_by_model"] = [
    (str(r.model), int(r.ledger_id), int(r.bank_id)) in selected
    for r in challenge.itertuples(index=False)
]

out = REPORTS / "hard_negative_verified_unmatched_challenge.csv"
challenge.sort_values(
    ["ledger_id", "bank_id", "model"]
).to_csv(out, index=False)

summary = challenge.groupby("model").agg(
    verified_unmatched_pairs=("label", "size"),
    selected_false_matches=("selected_by_model", "sum"),
    max_probability=("probability", "max"),
    mean_probability=("probability", "mean"),
).reset_index()

summary_out = REPORTS / "hard_negative_verified_unmatched_summary.csv"
summary.to_csv(summary_out, index=False)

print("=== VERIFIED-UNMATCHED CHALLENGE ===")
print("Every pair has a ledger record and bank record marked unmatched.")
print("This is a diagnostic subset, not a new independent test set.\n")
print(summary.sort_values("selected_false_matches", ascending=False).to_string(index=False))
print(f"\nDetailed pairs: {out}")
print(f"Summary:        {summary_out}")
