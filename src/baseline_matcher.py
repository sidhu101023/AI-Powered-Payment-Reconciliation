
from pathlib import Path
import re
from difflib import SequenceMatcher

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "reports"
OUTPUT.mkdir(exist_ok=True)

bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
truth = pd.read_csv(DATA / "match_key.csv")

# Keep money comparisons exact to the penny.
bank["amount_cents"] = (bank["amount"].round(2) * 100).round().astype(int)
ledger["amount_cents"] = (
    (ledger["amount"].round(2) * 100).round().astype(int)
)

bank["posted_date"] = pd.to_datetime(bank["posted_date"])
ledger["entry_date"] = pd.to_datetime(ledger["entry_date"])

def extract_references(text):
    return set(
        re.findall(
            r"\b(?:INV-\d+|BILL-\d+|CHQ\s*\d+)\b",
            str(text).upper()
        )
    )

def name_similarity(a, b):
    return SequenceMatcher(
        None, str(a).upper(), str(b).upper()
    ).ratio()

candidates = []

for _, l in ledger.iterrows():
    for _, b in bank.iterrows():
        if l["book"] != b["account"]:
            continue

        bank_refs = extract_references(b["narrative"])
        ledger_ref = re.sub(
            r"\s+", " ", str(l["reference"]).upper()
        ).strip()
        bank_refs = {
            re.sub(r"\s+", " ", ref).strip()
            for ref in bank_refs
        }

        ref_match = ledger_ref in bank_refs
        amount_match = l["amount_cents"] == b["amount_cents"]
        days = (b["posted_date"] - l["entry_date"]).days

        # Initial candidate rules:
        # reference match, or exact amount within a 12-day window.
        if not ref_match and not (
            amount_match and 0 <= days <= 12
        ):
            continue

        similarity = name_similarity(
            l["counterparty"], b["narrative"]
        )

        score = (
            100 * ref_match
            + 50 * amount_match
            + 20 * (0 <= days <= 2)
            + 10 * (3 <= days <= 12)
            + 10 * similarity
        )

        candidates.append({
            "ledger_id": int(l["id"]),
            "bank_id": int(b["id"]),
            "score": score,
            "reference_match": ref_match,
            "amount_match": amount_match,
            "days_difference": days,
            "ledger_amount": l["amount"],
            "bank_amount": b["amount"],
            "ledger_reference": l["reference"],
            "counterparty": l["counterparty"],
            "bank_narrative": b["narrative"],
        })

candidate_df = pd.DataFrame(candidates)

if candidate_df.empty:
    predictions = pd.DataFrame(
        columns=["ledger_id", "bank_id", "score"]
    )
else:
    candidate_df = candidate_df.sort_values(
        ["score", "reference_match", "amount_match"],
        ascending=False
    )

    # Greedy one-to-one assignment for this baseline.
    used_ledger = set()
    used_bank = set()
    selected = []

    for row in candidate_df.to_dict("records"):
        lid, bid = row["ledger_id"], row["bank_id"]

        if lid in used_ledger or bid in used_bank:
            continue

        selected.append(row)
        used_ledger.add(lid)
        used_bank.add(bid)

    predictions = pd.DataFrame(selected)

predictions.to_csv(
    OUTPUT / "baseline_predictions.csv", index=False
)



# Evaluate ordinary matches and duplicate-line cases separately.

print("\n=== ANSWER-KEY BREAKDOWN ===")
print(truth["kind"].value_counts(dropna=False).to_string())

# Build sets of expected pairs by category.
exact_truth = truth[truth["kind"] == "exact"]
duplicate_truth = truth[truth["kind"] == "duplicate_line"]

exact_pairs = set(
    zip(
        exact_truth["ledger_entry_no"].astype(int),
        exact_truth["bank_line_no"].astype(int),
    )
)

duplicate_pairs = set(
    zip(
        duplicate_truth["ledger_entry_no"].astype(int),
        duplicate_truth["bank_line_no"].astype(int),
    )
)

# All answer-key pairs, including non-exact categories.
all_truth_pairs = set(
    zip(
        truth["ledger_entry_no"].astype(int),
        truth["bank_line_no"].astype(int),
    )
)

predicted_pairs = (
    set(
        zip(
            predictions["ledger_id"].astype(int),
            predictions["bank_id"].astype(int),
        )
    )
    if not predictions.empty
    else set()
)

# 1. Evaluate exact matches only.
exact_tp = len(exact_pairs & predicted_pairs)
exact_fp = len(predicted_pairs - exact_pairs)
exact_fn = len(exact_pairs - predicted_pairs)

exact_precision = (
    exact_tp / (exact_tp + exact_fp)
    if exact_tp + exact_fp
    else 0.0
)
exact_recall = (
    exact_tp / (exact_tp + exact_fn)
    if exact_tp + exact_fn
    else 0.0
)
exact_f1 = (
    2 * exact_precision * exact_recall
    / (exact_precision + exact_recall)
    if exact_precision + exact_recall
    else 0.0
)

print("\n=== EXACT-MATCH EVALUATION ===")
print(f"Expected exact pairs: {len(exact_pairs)}")
print(f"Predicted pairs:      {len(predicted_pairs)}")
print(f"Correct exact pairs:  {exact_tp}")
print(f"False positives:      {exact_fp}")
print(f"Missed exact pairs:   {exact_fn}")
print(f"Precision:            {exact_precision:.3f}")
print(f"Recall:               {exact_recall:.3f}")
print(f"F1-score:             {exact_f1:.3f}")

# 2. Evaluate all answer-key categories together.
# This shows overall pair recovery, including special match types.
all_tp = len(all_truth_pairs & predicted_pairs)
all_fp = len(predicted_pairs - all_truth_pairs)
all_fn = len(all_truth_pairs - predicted_pairs)

all_precision = (
    all_tp / (all_tp + all_fp)
    if all_tp + all_fp
    else 0.0
)
all_recall = (
    all_tp / (all_tp + all_fn)
    if all_tp + all_fn
    else 0.0
)
all_f1 = (
    2 * all_precision * all_recall
    / (all_precision + all_recall)
    if all_precision + all_recall
    else 0.0
)

print("\n=== ALL-CATEGORY PAIR EVALUATION ===")
print(f"Answer-key pairs: {len(all_truth_pairs)}")
print(f"Predicted pairs:  {len(predicted_pairs)}")
print(f"Correct pairs:    {all_tp}")
print(f"False positives:  {all_fp}")
print(f"False negatives:  {all_fn}")
print(f"Precision:        {all_precision:.3f}")
print(f"Recall:           {all_recall:.3f}")
print(f"F1-score:         {all_f1:.3f}")

# 3. Diagnose duplicate-line exceptions separately.
print("\n=== DUPLICATE-LINE DIAGNOSTICS ===")
print(f"Duplicate-key pairs: {len(duplicate_pairs)}")

for ledger_id, bank_id in sorted(duplicate_pairs):
    status = (
        "predicted"
        if (ledger_id, bank_id) in predicted_pairs
        else "not selected by one-to-one matcher"
    )
    print(
        f"Ledger {ledger_id}, bank line {bank_id}: {status}"
    )

# 4. Report answer-key pairs not predicted.
# This replaces the undefined variable 'actual_pairs'.
print("\n=== MISSED ANSWER-KEY PAIRS ===")

missed_pairs = all_truth_pairs - predicted_pairs

if not missed_pairs:
    print("No answer-key pairs were missed.")
else:
    for ledger_id, bank_id in sorted(missed_pairs):
        matching_truth = truth[
            (truth["ledger_entry_no"].astype(int) == ledger_id)
            & (truth["bank_line_no"].astype(int) == bank_id)
        ]
        kinds = ", ".join(
            matching_truth["kind"].astype(str).unique()
        )

        print(
            f"\nLedger {ledger_id} <-> Bank line {bank_id}"
            f" | Answer-key kind: {kinds}"
        )

        ledger_rows = ledger[ledger["id"] == ledger_id]
        bank_rows = bank[bank["id"] == bank_id]

        if ledger_rows.empty or bank_rows.empty:
            print("Record not found in source data.")
            continue

        ledger_row = ledger_rows.iloc[0]
        bank_row = bank_rows.iloc[0]

        print(f"  Ledger date:      {ledger_row['entry_date']}")
        print(f"  Bank date:        {bank_row['posted_date']}")
        print(f"  Ledger amount:    {ledger_row['amount']:.2f}")
        print(f"  Bank amount:      {bank_row['amount']:.2f}")
        print(f"  Ledger reference: {ledger_row['reference']}")
        print(f"  Counterparty:     {ledger_row['counterparty']}")
        print(f"  Bank narrative:   {bank_row['narrative']}")

        date_difference = (
            bank_row["posted_date"] - ledger_row["entry_date"]
        ).days

        print(f"  Date difference:  {date_difference} days")

# 5. Inspect competing candidates for missed answer-key pairs.
print("\n=== COMPETING CANDIDATES FOR MISSED PAIRS ===")

if missed_pairs:
    for ledger_id, bank_id in sorted(missed_pairs):
        print(f"\nTarget pair: ledger {ledger_id} <-> bank {bank_id}")

        competing = candidate_df[
            (candidate_df["ledger_id"] == ledger_id)
            | (candidate_df["bank_id"] == bank_id)
        ].sort_values("score", ascending=False)

        if competing.empty:
            print("No candidate rows found.")
        else:
            columns = [
                "ledger_id",
                "bank_id",
                "score",
                "reference_match",
                "amount_match",
                "days_difference",
                "ledger_reference",
                "bank_narrative",
            ]

            available_columns = [
                column for column in columns
                if column in competing.columns
            ]

            print(
                competing[available_columns]
                .to_string(index=False)
            )
else:
    print("No missed pairs to inspect.")

# 6. Inspect known duplicate bank lines.
print("\n=== DUPLICATE BANK-LINE INSPECTION ===")

duplicate_groups = [
    (95, [90, 91], "BILL-7594"),
    (153, [154, 155], "BILL-9650"),
    (160, [161, 162], "BILL-7770"),
]

for ledger_id, bank_ids, reference in duplicate_groups:
    print(f"\nReference: {reference} | Ledger entry: {ledger_id}")

    rows = bank[bank["id"].isin(bank_ids)]

    print(
        rows[
            ["id", "account", "posted_date", "narrative", "amount"]
        ].to_string(index=False)
    )

    print("\nAnswer-key mapping:")

    mapping = truth[
        (truth["ledger_entry_no"].astype(int) == ledger_id)
        & (truth["bank_line_no"].astype(int).isin(bank_ids))
    ]

    if mapping.empty:
        print("No answer-key rows found.")
    else:
        print(
            mapping[
                ["ledger_entry_no", "bank_line_no", "kind"]
            ].to_string(index=False)
        )

# 7. Report records listed as unmatched in the answer key.
print("\n=== UNMATCHED ANSWER KEY ===")

unmatched = pd.read_csv(DATA / "unmatched_key.csv")

print("\nBy side:")
print(unmatched["side"].value_counts().to_string())

print("\nBy reason:")
print(unmatched["reason"].value_counts().to_string())

# 8. Save predictions with their answer-key category for inspection.
evaluation = predictions.copy()

if not evaluation.empty:
    truth_lookup = (
        truth.drop_duplicates(
            ["ledger_entry_no", "bank_line_no"]
        )
        .set_index(["ledger_entry_no", "bank_line_no"])["kind"]
    )

    evaluation["answer_kind"] = [
        truth_lookup.get(
            (int(row.ledger_id), int(row.bank_id)),
            "not_in_answer_key",
        )
        for row in evaluation.itertuples()
    ]

    evaluation.to_csv(
        OUTPUT / "baseline_evaluation.csv",
        index=False,
    )
else:
    # Still create the evaluation file with the prediction columns.
    evaluation.to_csv(
        OUTPUT / "baseline_evaluation.csv",
        index=False,
    )

predictions.to_csv(
    OUTPUT / "baseline_predictions.csv",
    index=False,
)

print("\n=== OUTPUT FILES ===")
print(f"Predictions: {OUTPUT / 'baseline_predictions.csv'}")
print(f"Evaluation:  {OUTPUT / 'baseline_evaluation.csv'}")
