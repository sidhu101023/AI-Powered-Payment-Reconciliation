
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
keys = pd.read_csv(DATA / "match_key.csv")
unmatched_key = pd.read_csv(DATA / "unmatched_key.csv")

matches_path = REPORTS / "same_split_test_matches.csv"
scores_path = REPORTS / "same_split_test_scores.csv"

if not matches_path.exists() or not scores_path.exists():
    raise FileNotFoundError(
        "Run src/compare_all_models.py first to generate "
        "same_split_test_matches.csv and same_split_test_scores.csv."
    )

matches = pd.read_csv(matches_path)
scores = pd.read_csv(scores_path)

# Normalize IDs for consistent joins.
for frame, columns in [
    (bank, ["id"]),
    (ledger, ["id"]),
    (keys, ["ledger_entry_no", "bank_line_no"]),
    (matches, ["ledger_id", "bank_id"]),
    (scores, ["ledger_id", "bank_id"]),
]:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column], errors="raise"
            ).astype(int)

keys["kind"] = keys["kind"].astype(str).str.lower()

# Exclude duplicate_line from normal matching truth.
duplicate_keys = keys[keys["kind"] == "duplicate_line"].copy()
duplicate_pairs = set(zip(
    duplicate_keys["ledger_entry_no"], duplicate_keys["bank_line_no"],
))

valid_keys = keys[keys["kind"] != "duplicate_line"].copy()
truth = set(zip(
    valid_keys["ledger_entry_no"],
    valid_keys["bank_line_no"],
))

ledger_by_id = ledger.set_index("id", drop=False)
bank_by_id = bank.set_index("id", drop=False)

# ---------------------------------------------------------
# 1. Evaluate model predictions against the answer key.
# ---------------------------------------------------------
false_rows = []
model_summary = []

for model_name, group in matches.groupby("model"):
    predicted_pairs = set(zip(group["ledger_id"], group["bank_id"]))

    for row in group.itertuples(index=False):
        pair = (int(row.ledger_id), int(row.bank_id))
        if pair not in truth:
            l = ledger_by_id.loc[pair[0]]
            b = bank_by_id.loc[pair[1]]
            false_rows.append({
                "model": model_name,
                "ledger_id": pair[0],
                "bank_id": pair[1],
                "probability": getattr(row, "probability", None),
                "ledger_reference": l.get("reference", ""),
                "counterparty": l.get("counterparty", ""),
                "ledger_amount": l.get("amount", None),
                "bank_amount": b.get("amount", None),
                "ledger_date": l.get("entry_date", None),
                "bank_date": b.get("posted_date", None),
                "exception_type": (
                    "duplicate_line_candidate"
                    if pair in duplicate_pairs else "ordinary_false_match"
                ),
            })

    expected_test = set(zip(
        scores.loc[
            (scores["model"] == model_name) & (scores["label"] == 1),
            "ledger_id",
        ],
        scores.loc[
            (scores["model"] == model_name) & (scores["label"] == 1),
            "bank_id",
        ],
    ))

    # The candidate labels provide the expected positive pairs
    # within this model\'s test candidate set.
    tp = len(predicted_pairs & expected_test)
    fp = len(predicted_pairs - expected_test)
    fn = len(expected_test - predicted_pairs)

    model_summary.append({
        "model": model_name,
        "expected_test_matches": len(expected_test),
        "predicted_test_matches": len(predicted_pairs),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": (
            2 * tp / (2 * tp + fp + fn)
            if 2 * tp + fp + fn else 0.0
        ),
    })

# Report duplicate-labelled pairs among all test candidates, whether
# or not the one-to-one matcher selected them.
duplicate_candidate_rows = []
for model_name, score_group in scores.groupby("model"):
    selected_group = matches[matches["model"] == model_name]
    selected_pairs = set(zip(selected_group["ledger_id"], selected_group["bank_id"]))
    for row in score_group.to_dict("records"):
        pair = (int(row["ledger_id"]), int(row["bank_id"]))
        if pair not in duplicate_pairs:
            continue
        l = ledger_by_id.loc[pair[0]]
        b = bank_by_id.loc[pair[1]]
        duplicate_candidate_rows.append({
            "model": model_name,
            "ledger_id": pair[0],
            "bank_id": pair[1],
            "score": row.get("probability", row.get("score", None)),
            "candidate_label": row.get("label", None),
            "selected_by_one_to_one_matcher": pair in selected_pairs,
            "ledger_reference": l.get("reference", ""),
            "counterparty": l.get("counterparty", ""),
            "ledger_amount": l.get("amount", None),
            "bank_amount": b.get("amount", None),
            "ledger_date": l.get("entry_date", None),
            "bank_date": b.get("posted_date", None),
            "exception_type": "duplicate_line_candidate",
        })

pd.DataFrame(duplicate_candidate_rows).to_csv(
    REPORTS / "duplicate_line_candidates.csv", index=False
)

pd.DataFrame(false_rows).to_csv(
    REPORTS / "false_matches.csv", index=False
)
pd.DataFrame(model_summary).to_csv(
    REPORTS / "exception_model_metrics.csv", index=False
)

# ---------------------------------------------------------
# 2. Known unmatched items from the supplied answer key.
# ---------------------------------------------------------
unmatched_rows = []

for row in unmatched_key.to_dict("records"):
    side = str(row.get("side", "")).lower()
    record_no = pd.to_numeric(
        pd.Series([row.get("record_no")]), errors="coerce"
    ).iloc[0]

    record = None
    if pd.notna(record_no):
        record_no = int(record_no)
        if side == "bank" and record_no in bank_by_id.index:
            record = bank_by_id.loc[record_no]
        elif side == "ledger" and record_no in ledger_by_id.index:
            record = ledger_by_id.loc[record_no]

    unmatched_rows.append({
        "source": "answer_key",
        "side": side,
        "record_no": record_no,
        "book": row.get("book", ""),
        "reason": row.get("reason", "unknown"),
        "amount": record.get("amount", None) if record is not None else None,
        "date": (
            record.get("posted_date", None)
            if side == "bank" and record is not None
            else record.get("entry_date", None)
            if side == "ledger" and record is not None
            else None
        ),
    })

pd.DataFrame(unmatched_rows).to_csv(
    REPORTS / "unmatched_transactions.csv", index=False
)

# ---------------------------------------------------------
# 3. Valid matched pairs with non-exact categories.
# ---------------------------------------------------------
bank["id"] = bank["id"].astype(int)
ledger["id"] = ledger["id"].astype(int)

amount_date_rows = []

for row in valid_keys.to_dict("records"):
    lid = int(row["ledger_entry_no"])
    bid = int(row["bank_line_no"])

    if lid not in ledger_by_id.index or bid not in bank_by_id.index:
        continue

    l = ledger_by_id.loc[lid]
    b = bank_by_id.loc[bid]

    ledger_amount = float(l["amount"])
    bank_amount = float(b["amount"])
    amount_difference = abs(ledger_amount - bank_amount)

    ledger_date = pd.to_datetime(l["entry_date"], errors="coerce")
    bank_date = pd.to_datetime(b["posted_date"], errors="coerce")
    date_difference = (
        (bank_date - ledger_date).days
        if pd.notna(ledger_date) and pd.notna(bank_date)
        else None
    )

    kind = row["kind"]

    # Classify known exceptions separately from routine exact matches.
    if kind == "late_posting":
        issue = "late_posting"
    elif kind == "net_of_fee":
        issue = "net_of_fee"
    elif kind == "transposed":
        issue = "transposed_amount"
    elif kind == "same_amount_same_day":
        issue = "same_amount_same_day"
    elif kind == "exact" and amount_difference > 0.005:
        issue = "answer_key_amount_discrepancy"
    elif (
        kind == "exact"
        and amount_difference <= 0.005
        and date_difference is not None
        and date_difference > 0
    ):
        issue = "matched_timing_difference"
    elif kind == "exact" and date_difference not in (0, None):
        issue = "exact_amount_date_difference"
    else:
        issue = "review"

    # Routine exact matches with no amount/date discrepancy are omitted.
    if kind != "exact" or amount_difference > 0.005 or date_difference not in (0, None):
        amount_date_rows.append({
            "ledger_id": lid,
            "bank_id": bid,
            "match_kind": kind,
            "exception_type": issue,
            "ledger_amount": ledger_amount,
            "bank_amount": bank_amount,
            "absolute_amount_difference": amount_difference,
            "ledger_date": ledger_date,
            "bank_date": bank_date,
            "signed_date_difference_days": date_difference,
            "reference": l.get("reference", ""),
            "counterparty": l.get("counterparty", ""),
            "bank_narrative": b.get("narrative", ""),
        })

pd.DataFrame(amount_date_rows).to_csv(
    REPORTS / "amount_date_exceptions.csv", index=False
)

# ---------------------------------------------------------
# 4. Potential duplicate bank lines.
# ---------------------------------------------------------
duplicate_rows = []

# Group exact duplicate transaction signatures. These are candidates
# for review, not proof of fraudulent or erroneous transactions.
duplicate_columns = [
    col for col in ["account", "posted_date", "amount", "narrative"]
    if col in bank.columns
]

if len(duplicate_columns) >= 3:
    duplicate_mask = bank.duplicated(
        subset=duplicate_columns, keep=False
    )
    duplicate_candidates = bank[duplicate_mask].copy()

    for signature, group in duplicate_candidates.groupby(
        duplicate_columns, dropna=False
    ):
        if len(group) < 2:
            continue

        for row in group.to_dict("records"):
            duplicate_rows.append({
                "duplicate_group": str(signature),
                "bank_id": int(row["id"]),
                "account": row.get("account", ""),
                "posted_date": row.get("posted_date", ""),
                "amount": row.get("amount", None),
                "narrative": row.get("narrative", ""),
                "group_size": len(group),
                "exception_type": "potential_duplicate_bank_line",
            })

pd.DataFrame(duplicate_rows).to_csv(
    REPORTS / "duplicate_bank_lines.csv", index=False
)

# ---------------------------------------------------------
# 5. Overall exception summary.
# ---------------------------------------------------------
summary_rows = [
    {
        "exception_type": "answer_key_unmatched_records",
        "count": len(unmatched_key),
    },
    {
        "exception_type": "false_match_predictions_across_models",
        "count": len(false_rows),
    },
    {
        "exception_type": "duplicate_line_candidate_pairs_across_models",
        "count": len(duplicate_candidate_rows),
    },
    {
        "exception_type": "valid_pairs_needing_amount_or_date_review",
        "count": len(amount_date_rows),
    },
    {
        "exception_type": "potential_duplicate_bank_rows",
        "count": len(duplicate_rows),
    },
]

pd.DataFrame(summary_rows).to_csv(
    REPORTS / "exception_summary.csv", index=False
)

print("=== RECONCILIATION EXCEPTION EVALUATION ===")
print(pd.DataFrame(summary_rows).to_string(index=False))
print("\n=== MODEL METRICS ===")
print(pd.DataFrame(model_summary).to_string(index=False))
print("\n=== DUPLICATE-LINE CANDIDATES ===")
if duplicate_candidate_rows:
    print(pd.DataFrame(duplicate_candidate_rows)[["model", "ledger_id", "bank_id", "score", "candidate_label", "selected_by_one_to_one_matcher"]].to_string(index=False))
else:
    print("No duplicate-line key pairs found in the test candidate scores.")

print("\nSaved reports:")
for filename in [
    "exception_summary.csv",
    "exception_model_metrics.csv",
    "false_matches.csv",
    "duplicate_line_candidates.csv",
    "unmatched_transactions.csv",
    "amount_date_exceptions.csv",
    "duplicate_bank_lines.csv",
]:
    print(REPORTS / filename)
