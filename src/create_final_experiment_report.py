from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

def load_csv(name):
    path = REPORTS / name
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")
    return pd.read_csv(path)

def markdown_table(df):
    if df.empty:
        return "_No records available._"
    display = df.copy()
    display = display.fillna("")
    headers = [str(c) for c in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        values = []
        for value in row:
            text = str(value).replace("|", "\\|").replace("\n", " ")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)

# Load existing experiment outputs.
models = load_csv("neural_experiment_model_comparison.csv")
thresholds = load_csv("neural_experiment_validation_thresholds.csv")
challenge = load_csv("hard_negative_verified_unmatched_summary.csv")
exceptions = load_csv("neural_experiment_exception_summary.csv")
duplicates = load_csv("neural_experiment_duplicate_bank_lines.csv")
unmatched = load_csv("neural_experiment_unmatched_transactions.csv")
false_matches = load_csv("neural_experiment_false_matches.csv")

# Build a consolidated model summary.
summary = models.merge(
    thresholds[["model", "threshold", "validation_f1"]].rename(
        columns={
            "threshold": "threshold_crosscheck",
            "validation_f1": "validation_f1_crosscheck",
        }
    ),
    on="model",
    how="left",
)

summary = summary.merge(challenge, on="model", how="left")
summary["verified_unmatched_pairs"] = (
    summary["verified_unmatched_pairs"].fillna(0).astype(int)
)
summary["selected_false_matches"] = (
    summary["selected_false_matches"].fillna(0).astype(int)
)

columns = [
    "model",
    "threshold_selected_on_validation",
    "validation_best_f1",
    "test_candidates",
    "test_positive_pairs",
    "predicted_matches",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "average_precision",
    "verified_unmatched_pairs",
    "selected_false_matches",
    "max_probability",
    "mean_probability",
]
summary = summary[[c for c in columns if c in summary.columns]]
summary.to_csv(REPORTS / "final_experiment_summary.csv", index=False)

# Summarize duplicate groups without counting each duplicate row as a group.
duplicate_group_count = (
    duplicates["duplicate_group"].nunique()
    if not duplicates.empty and "duplicate_group" in duplicates.columns
    else 0
)
duplicate_row_count = len(duplicates)

exception_counts = dict(zip(exceptions["exception_type"], exceptions["count"]))
unmatched_source_counts = (
    unmatched.groupby(["source", "side"]).size().to_dict()
    if not unmatched.empty
    else {}
)

# Format key model results.
model_view = models[[
    "model", "threshold_selected_on_validation", "validation_best_f1",
    "predicted_matches", "true_positives", "false_positives",
    "false_negatives", "precision", "recall", "f1", "average_precision"
]].copy()

for col in ["precision", "recall", "f1", "average_precision", "validation_best_f1"]:
    model_view[col] = model_view[col].map(
        lambda x: f"{x:.4f}" if pd.notna(x) else ""
    )

challenge_view = challenge.copy()
for col in ["max_probability", "mean_probability"]:
    if col in challenge_view.columns:
        challenge_view[col] = challenge_view[col].map(
            lambda x: f"{x:.6f}" if pd.notna(x) else ""
        )

false_view = false_matches.copy()
if not false_view.empty:
    for col in ["probability", "ledger_amount", "bank_amount"]:
        if col in false_view.columns:
            false_view[col] = false_view[col].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else ""
            )
    preferred = [
        "model", "ledger_id", "bank_id", "probability",
        "ledger_reference", "counterparty", "ledger_amount",
        "bank_amount", "ledger_date", "bank_date", "exception_type"
    ]
    false_view = false_view[[c for c in preferred if c in false_view.columns]]

duplicate_view = duplicates.copy()
if not duplicate_view.empty:
    duplicate_view = duplicate_view.drop_duplicates("duplicate_group")
    preferred = [
        "account", "posted_date", "amount", "narrative",
        "group_size", "exception_type"
    ]
    duplicate_view = duplicate_view[[c for c in preferred if c in duplicate_view.columns]]

# Make the report.
lines = [
    "# Payment Reconciliation — Final Experiment Report",
    "",
    "**Generated from existing experiment outputs.** No models were retrained.",
    "",
    "## 1. Objective",
    "",
    "Compare six reconciliation approaches using the current common candidate sets and "
    "test split, inspect false matches and duplicate bank lines, and summarize a separate "
    "unmatched-record challenge.",
    "",
    "## 2. Model comparison",
    "",
    markdown_table(model_view),
    "",
    "### Interpretation",
    "",
    "- Thresholds were selected using validation data; the reported test metrics use those thresholds.",
    "- Average precision summarizes ranking quality across thresholds; it is not the same as test F1.",
    "- Perfect test scores on this dataset do not establish perfect performance on real bank data.",
    "- TF-IDF-only matching should be treated cautiously because textual similarity can be high even when amounts differ substantially.",
    "",
    "## 3. Verified-unmatched challenge",
    "",
    "The following results describe the separately constructed challenge of 45 candidate pairs "
    "per model. The false-match count refers to pairs selected by the model's final one-to-one "
    "matching procedure at its selected threshold.",
    "",
    markdown_table(challenge_view),
    "",
    "This challenge is a diagnostic subset, not an independent estimate of production performance. "
    "Its labels and candidate construction depend on the supplied unmatched key.",
    "",
    "## 4. False-match review",
    "",
    f"False-match records in the exception report: **{len(false_matches)}**.",
    "",
    markdown_table(false_view),
    "",
    "Large amount differences are a warning sign that text similarity alone is insufficient. "
    "A practical system should use amount, dates, reference/counterparty evidence, and one-to-one "
    "constraints, while allowing uncertain cases to be reviewed by a person.",
    "",
    "## 5. Duplicate bank-line review",
    "",
    f"- Duplicate groups identified: **{duplicate_group_count}**",
    f"- Rows represented in the duplicate-line report: **{duplicate_row_count}**",
    "",
    markdown_table(duplicate_view),
    "",
    "Identical-looking bank rows should not automatically be treated as separate payments or "
    "deleted. Confirm duplicate status and preserve an audit trail before resolving them.",
    "",
    "## 6. Unmatched transactions",
    "",
    f"- Rows in the unmatched transaction report: **{len(unmatched)}**",
    f"- Answer-key unmatched records (exception summary): **{exception_counts.get('answer_key_unmatched_records', 'not reported')}**",
    "",
    "Counts by source and side:",
    "",
]

if unmatched_source_counts:
    count_rows = [
        {"source": key[0], "side": key[1], "count": value}
        for key, value in sorted(unmatched_source_counts.items())
    ]
    lines.extend([markdown_table(pd.DataFrame(count_rows)), ""])
else:
    lines.extend(["_No unmatched rows found in the input report._", ""])

lines.extend([
    "Unmatched records can represent timing differences, fees, outstanding cheques, "
    "deposits in transit, returned items, or transactions requiring investigation. "
    "They should be classified rather than forced into a match.",
    "",
    "## 7. Evaluation limitations",
    "",
    "1. **Dataset scope:** the results apply to the supplied dataset; they do not prove performance on live bank statements.",
    "2. **Label completeness:** candidate pairs not present in the match key may be treated as negatives by the experiment. If the key is incomplete, some labels could be wrong.",
    "3. **Split design:** known positive pairs were kept together using match-key information and pair dates. This is not a fully independent, label-blind chronological evaluation.",
    "4. **Candidate generation:** models only compare pairs that pass the experiment's candidate-generation rules. Missed candidate pairs cannot be recovered by the classifier.",
    "5. **Challenge scope:** the unmatched challenge is useful for diagnosis but is not a substitute for an independently labelled holdout set.",
    "6. **Operational safety:** a production workflow needs audit logs, duplicate controls, human review for ambiguous cases, and explicit handling of amount/date exceptions.",
    "",
    "## 8. Recommended next steps",
    "",
    "1. Create a separately reviewed set of hard negatives, including swapped-counterparty pairs.",
    "2. Run an additional chronological evaluation with a split defined without using answer-key labels to assign pairs.",
    "3. Check candidate-generation recall separately from classifier precision and recall.",
    "4. Test the full workflow on new input files, including unmatched records and duplicate bank lines.",
    "5. Build a Streamlit review interface with suggested matches, confidence, exception reasons, and exportable results.",
    "",
    "---",
    "",
    "Generated locally from the project's `reports` CSV files.",
])

report_path = REPORTS / "final_experiment_report.md"
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("Created:")
print(f"  {report_path}")
print(f"  {REPORTS / 'final_experiment_summary.csv'}")
print(f"\nModels included: {len(models)}")
print(f"False-match report rows: {len(false_matches)}")
print(f"Duplicate groups: {duplicate_group_count}")
print(f"Unmatched report rows: {len(unmatched)}")
