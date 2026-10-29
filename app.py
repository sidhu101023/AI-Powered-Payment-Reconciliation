from pathlib import Path
from difflib import SequenceMatcher
from src.exception_candidates import generate_exception_candidates
from src.hybrid_transformer_predictor import HybridTransformerPredictor
from src.ditto_transformer_predictor import DittoTransformerPredictor
from src.payment_registry import PaymentRegistry
from src.reconciliation_workflow import (
    decision_audit_rows,
    input_validation_findings,
    review_exports,
    validate_review_decisions,
)
import re
import secrets
import hashlib

import pandas as pd
import streamlit as st

@st.cache_resource
def load_hybrid_predictor():
    """Load the trained model once and reuse it across Streamlit reruns."""
    return HybridTransformerPredictor()

@st.cache_resource
def load_original_predictor():
    """Load the saved original Transformer once per Streamlit process."""
    return DittoTransformerPredictor()

st.set_page_config(
    page_title="AI Payment Reconciliation",
    page_icon="💳",
    layout="wide",
)

def normalize(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())

def similarity(a, b):
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def find_column(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None

def prepare_bank(df):
    mapping = {
        "id": find_column(df, ["id", "bank_id", "bank_line_no", "bank_line_id", "bank_transaction_id"]),
        "date": find_column(df, ["posted_date", "date", "transaction_date"]),
        "description": find_column(df, ["narrative", "description", "details", "reference"]),
        "amount": find_column(df, ["amount", "transaction_amount"]),
    }
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise ValueError(f"Bank CSV is missing required fields: {', '.join(missing)}")

    out = pd.DataFrame({
        "bank_id": df[mapping["id"]].astype(str),
        "bank_date": pd.to_datetime(df[mapping["date"]], errors="coerce"),
        "bank_description": df[mapping["description"]].fillna("").astype(str),
        "bank_amount": pd.to_numeric(df[mapping["amount"]], errors="coerce"),
    })
    # Optional fields are intentionally non-breaking for the existing CSV schema.
    for target, candidates in {
        "payment_id": ["payment_id", "payment reference", "payment_reference"],
        "bank_reference": ["bank_reference", "utr", "utr_no", "transaction_reference"],
    }.items():
        source = find_column(df, candidates)
        out[target] = df[source].fillna("").astype(str) if source else ""
    if out["bank_id"].duplicated().any():
        raise ValueError("Bank transaction IDs must be unique.")
    if out["bank_id"].str.strip().eq("").any():
        raise ValueError("Bank transaction IDs cannot be blank.")
    if out["bank_date"].isna().any() or out["bank_amount"].isna().any():
        raise ValueError("Bank CSV contains invalid dates or amounts.")
    return out


def parse_allocations(raw):
    """Parse `invoice:amount` allocations without treating invoice IDs as payment IDs."""
    allocations = []
    for item in raw.split(","):
        invoice, separator, amount = item.strip().rpartition(":")
        if not separator:
            raise ValueError("Use invoice:amount pairs separated by commas.")
        allocations.append((invoice.strip(), float(amount.strip())))
    return allocations

def find_duplicate_bank_rows(bank):
    """Flag identical bank transactions for manual review."""
    duplicates = bank.copy()

    duplicates["normalized_description"] = (
        duplicates["bank_description"]
        .fillna("")
        .map(normalize)
    )

    duplicate_mask = duplicates.duplicated(
        subset=[
            "bank_date",
            "bank_amount",
            "normalized_description",
        ],
        keep=False,
    )

    return set(duplicates.loc[duplicate_mask, "bank_id"])

def prepare_ledger(df):
    mapping = {
        "id": find_column(df, ["id", "ledger_id", "ledger_entry_no", "entry_id", "ledger_entry_id"]),
        "date": find_column(df, ["entry_date", "date", "transaction_date"]),
        "reference": find_column(df, ["reference", "ref", "document_reference"]),
        "counterparty": find_column(df, ["counterparty", "payee", "vendor", "customer"]),
        "amount": find_column(df, ["amount", "transaction_amount"]),
    }
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise ValueError(f"Ledger CSV is missing required fields: {', '.join(missing)}")

    out = pd.DataFrame({
        "ledger_id": df[mapping["id"]].astype(str),
        "ledger_date": pd.to_datetime(df[mapping["date"]], errors="coerce"),
        "ledger_reference": df[mapping["reference"]].fillna("").astype(str),
        "counterparty": df[mapping["counterparty"]].fillna("").astype(str),
        "ledger_amount": pd.to_numeric(df[mapping["amount"]], errors="coerce"),
    })
    if out["ledger_id"].duplicated().any():
        raise ValueError("Ledger transaction IDs must be unique.")
    if out["ledger_id"].str.strip().eq("").any():
        raise ValueError("Ledger transaction IDs cannot be blank.")
    if out["ledger_date"].isna().any() or out["ledger_amount"].isna().any():
        raise ValueError("Ledger CSV contains invalid dates or amounts.")
    return out

def score_pair(l, b):
    amount_gap = abs(float(l.ledger_amount) - float(b.bank_amount))
    day_gap = abs((b.bank_date - l.ledger_date).days)
    text = similarity(
        f"{l.ledger_reference} {l.counterparty}",
        b.bank_description,
    )
    exact_amount = amount_gap <= 0.01
    same_sign = (
        (l.ledger_amount < 0 and b.bank_amount < 0)
        or (l.ledger_amount >= 0 and b.bank_amount >= 0)
    )

    # Transparent heuristic score, not a trained-model probability.
    score = (
        0.55 * float(exact_amount)
        + 0.20 * float(same_sign)
        + 0.15 * text
        + 0.10 * max(0.0, 1.0 - day_gap / 8.0)
    )
    return min(1.0, max(0.0, score)), amount_gap, day_gap, text

st.title("AI-Powered Payment Reconciliation")
st.caption(
    "Upload ledger and bank transactions, review suggested matches, "
    "and export your decisions."
)

st.info(
    "This initial app uses a transparent heuristic score, not a trained-model "
    "probability. Review suggestions before treating them as reconciled."
)

registry = PaymentRegistry(Path("storage/payment_registry.sqlite3"))
st.session_state.setdefault("payment_submission_key", secrets.token_urlsafe(24))
st.session_state.setdefault("payment_submission_fingerprint", None)

st.header("Payment ID lifecycle")
st.caption("Creates simulated payment instructions only; this app never initiates bank transfers.")
with st.expander("Create payment instruction"):
    with st.form("create_payment"):
        supplier = st.text_input("Supplier")
        allocations_text = st.text_input(
            "Invoice allocations", placeholder="INV-2026-1042:50000, INV-2026-1043:1250",
            help="A payment may cover one or more invoices."
        )
        instruction_reference = st.text_input("Company payment instruction reference (optional)")
        create_payment = st.form_submit_button("Create Payment ID", icon=":material/add_card:")
    if create_payment:
        try:
            # Keep the idempotency key for an unchanged form, so a repeated
            # click/retry returns the same payment rather than a duplicate.
            fingerprint = hashlib.sha256(
                f"{supplier}\x1f{allocations_text}\x1f{instruction_reference}".encode("utf-8")
            ).hexdigest()
            if fingerprint != st.session_state["payment_submission_fingerprint"]:
                st.session_state["payment_submission_key"] = secrets.token_urlsafe(24)
                st.session_state["payment_submission_fingerprint"] = fingerprint
            payment_id = registry.create_payment(
                supplier,
                parse_allocations(allocations_text),
                instruction_reference=instruction_reference,
                submission_key=st.session_state["payment_submission_key"],
            )
            st.success(f"Created simulated payment instruction: {payment_id}")
        except (ValueError, TypeError) as exc:
            st.error(f"Payment was not created: {exc}")
        except Exception:
            st.error("Payment could not be saved. Please retry; no bank transfer was initiated.")

payment_search = st.text_input("Search payments", placeholder="Payment ID, invoice number, supplier, or bank reference")
payments_view = registry.find_payments(payment_search)
if not payments_view.empty:
    st.dataframe(payments_view, hide_index=True, alt="Payment records and linked invoices")
else:
    st.caption("No payment instructions have been created yet.")

synthetic_comparison_path = Path("reports/synthetic_payment_id_comparison.csv")
if synthetic_comparison_path.exists():
    st.subheader("Synthetic Payment ID experiment")
    st.caption("Synthetic experiment results are separate from the existing practice-dataset reports.")
    st.dataframe(
        pd.read_csv(synthetic_comparison_path),
        hide_index=True,
        alt="Synthetic Payment ID reconciliation comparison",
    )

st.divider()
st.header("Actionable Exception Dashboard")
st.caption(
    "Prioritize investigation using exception explanations and suggested actions. "
    "Rows may overlap and do not represent unique confirmed errors."
)

exception_path = Path("reports/actionable_exception_report.csv")

if exception_path.exists():
    exception_df = pd.read_csv(exception_path)

    # Normalize fields for reliable filtering.
    exception_df["review_priority"] = (
        exception_df["review_priority"].fillna("Normal").astype(str)
    )
    exception_df["exception_type"] = (
        exception_df["exception_type"].fillna("Unknown").astype(str)
    )
    exception_df["source_type"] = (
        exception_df["source_type"].fillna("Unknown").astype(str)
    )
    exception_df["requires_human_review"] = (
        exception_df["requires_human_review"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    high_count = int(
        (exception_df["review_priority"].str.lower() == "high").sum()
    )
    human_count = int(exception_df["requires_human_review"].sum())

    k1, k2, k3 = st.columns(3)
    k1.metric("Exception records", len(exception_df))
    k2.metric("High-priority records", high_count)
    k3.metric("Require human review", human_count)

    f1, f2, f3 = st.columns(3)

    with f1:
        priority_options = ["All"] + sorted(
            exception_df["review_priority"].unique().tolist()
        )
        selected_priority = st.selectbox(
            "Review priority",
            priority_options,
            key="exception_priority_filter",
        )

    with f2:
        type_options = ["All"] + sorted(
            exception_df["exception_type"].unique().tolist()
        )
        selected_type = st.selectbox(
            "Exception type",
            type_options,
            key="exception_type_filter",
        )

    with f3:
        source_options = ["All"] + sorted(
            exception_df["source_type"].unique().tolist()
        )
        selected_source = st.selectbox(
            "Record category",
            source_options,
            key="exception_source_filter",
        )

    filtered = exception_df.copy()

    if selected_priority != "All":
        filtered = filtered[
            filtered["review_priority"] == selected_priority
        ]

    if selected_type != "All":
        filtered = filtered[
            filtered["exception_type"] == selected_type
        ]

    if selected_source != "All":
        filtered = filtered[
            filtered["source_type"] == selected_source
        ]

    st.subheader(f"Review queue ({len(filtered)} records)")

    display_columns = [
        "source_type",
        "record_side",
        "record_id",
        "related_record_id",
        "date",
        "amount",
        "exception_type",
        "review_priority",
        "explanation",
        "suggested_action",
        "requires_human_review",
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download filtered exception queue",
        data=filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_exception_review_queue.csv",
        mime="text/csv",
        key="download_actionable_exceptions",
    )

    st.caption(
        "Priority labels are initial heuristics, not validated risk predictions. "
        "Verify exceptions against source transactions before taking action."
    )
else:
    st.warning(
        "Actionable exception report not found. "
        "Generate reports/actionable_exception_report.csv first."
    )

with st.sidebar:
    st.header("Matching settings")

    model_choice = st.selectbox(
        "Matching model",
        ["Heuristic", "Original Transformer", "Hybrid Transformer"],
    )

    if model_choice == "Hybrid Transformer":
        default_threshold = 0.95
        threshold_step = 0.01
    elif model_choice == "Original Transformer":
        default_threshold = 0.35
        threshold_step = 0.01
    else:
        default_threshold = 0.70
        threshold_step = 0.05

    threshold = st.slider(
        "Minimum suggestion score",
        min_value=0.0,
        max_value=1.0,
        value=default_threshold,
        step=threshold_step,
        key=f"threshold_{model_choice}",
    )

    date_window = st.number_input(
        "Maximum date difference (days)",
        min_value=0,
        max_value=60,
        value=7,
    )

    amount_tolerance = st.number_input(
        "Amount tolerance",
        min_value=0.0,
        value=0.01,
        step=0.01,
        format="%.2f",
    )

col1, col2 = st.columns(2)
with col1:
    bank_file = st.file_uploader("Upload bank CSV", type=["csv"], key="bank")
with col2:
    ledger_file = st.file_uploader("Upload ledger CSV", type=["csv"], key="ledger")

if bank_file and ledger_file:
    try:
        bank = prepare_bank(pd.read_csv(bank_file))
        ledger = prepare_ledger(pd.read_csv(ledger_file))
        validation_findings = input_validation_findings(bank, ledger)
        st.subheader("Input validation")
        if validation_findings.empty:
            st.success("Required fields, IDs, dates, and amounts passed validation.")
        else:
            st.warning(f"{len(validation_findings)} records need review before reconciliation.")
            st.dataframe(validation_findings, hide_index=True, alt="Input validation warnings")
        duplicate_bank_ids = find_duplicate_bank_rows(bank)
        identifier_matches, identifier_exceptions = registry.identifier_candidates(bank)
        identifier_bank_ids = set(identifier_matches.get("bank_id", pd.Series(dtype=str)).astype(str))
        if not identifier_matches.empty:
            st.subheader("Validated identifier-based suggestions")
            st.caption("Exact IDs still require approval. Amount mismatches and ambiguous identifiers are never forced.")
            st.dataframe(identifier_matches, hide_index=True, alt="Pending payment identifier suggestions")
        if not identifier_exceptions.empty:
            st.warning(f"{len(identifier_exceptions)} payment identifier exceptions require review.")
            st.dataframe(identifier_exceptions, hide_index=True, alt="Payment identifier exceptions")
        exception_candidates = generate_exception_candidates(
            ledger=ledger,
            bank=bank,
            duplicate_bank_ids=duplicate_bank_ids,
        )
        st.subheader("Reconciliation exceptions")

        if exception_candidates.empty:
            st.success("No exception candidates found.")
        else:
            st.warning(
                f"{len(exception_candidates)} exception candidates "
                "need manual review."
            )
            st.dataframe(
                exception_candidates,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download exception candidates CSV",
                data=exception_candidates.to_csv(index=False).encode("utf-8"),
                file_name="exception_candidates.csv",
                mime="text/csv",
            )
        st.subheader("Input summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Bank transactions", len(bank))
        m2.metric("Ledger entries", len(ledger))
        m3.metric("Potential pairs", len(bank) * len(ledger))

        duplicate_rows = bank[
            bank["bank_id"].isin(duplicate_bank_ids)
        ]

        if not duplicate_rows.empty:
            st.warning(
                f"Found {len(duplicate_rows)} bank rows belonging to "
                "potential duplicate groups. Review them before reconciliation."
            )
            st.subheader("Potential duplicate bank transactions")
            st.dataframe(duplicate_rows, use_container_width=True)

        if len(bank) * len(ledger) > 500_000:
            st.warning(
                "This upload creates a large number of possible pairs. "
                "Use smaller files for this initial prototype."
            )

        else:
            candidates = []
            candidate_pairs = []

            for l in ledger.itertuples(index=False):
                for b in bank.itertuples(index=False):
                    # Exact validated identifier matches have priority. Other bank
                    # lines continue to the existing rules/model workflow.
                    if str(b.bank_id) in identifier_bank_ids:
                        continue
                    amount_gap = abs(
                        float(l.ledger_amount) - float(b.bank_amount)
                    )
                    day_gap = abs((b.bank_date - l.ledger_date).days)

                    if day_gap > date_window:
                        continue

                    text_score = similarity(
                        f"{l.ledger_reference} {l.counterparty}",
                        b.bank_description,
                    )

                    base_row = {
                        "ledger_id": l.ledger_id,
                        "bank_id": b.bank_id,
                        "is_duplicate_bank_line": (
                            b.bank_id in duplicate_bank_ids
                        ),
                        "ledger_date": l.ledger_date.date().isoformat(),
                        "bank_date": b.bank_date.date().isoformat(),
                        "reference": l.ledger_reference,
                        "counterparty": l.counterparty,
                        "bank_description": b.bank_description,
                        "ledger_amount": float(l.ledger_amount),
                        "bank_amount": float(b.bank_amount),
                        "amount_difference": amount_gap,
                        "date_difference_days": day_gap,
                        "text_similarity": text_score,
                    }

                    if model_choice == "Heuristic":
                        score, _, _, _ = score_pair(l, b)

                        if score < threshold:
                            continue

                        if amount_gap > amount_tolerance and text_score < 0.70:
                            continue

                        base_row["suggestion_score"] = score
                        base_row["matching_method"] = "rules"
                        candidates.append(base_row)

                    else:
                        ledger_input = {
                            "reference": l.ledger_reference,
                            "counterparty": l.counterparty,
                            "amount": float(l.ledger_amount),
                            "entry_date": l.ledger_date,
                        }
                        bank_input = {
                            "narrative": b.bank_description,
                            "amount": float(b.bank_amount),
                            "posted_date": b.bank_date,
                        }

                        candidate_pairs.append(
                            (ledger_input, bank_input, base_row)
                        )

            if model_choice in {"Original Transformer", "Hybrid Transformer"} and candidate_pairs:
                predictor = load_hybrid_predictor() if model_choice == "Hybrid Transformer" else load_original_predictor()

                pair_inputs = [
                    (ledger_input, bank_input)
                    for ledger_input, bank_input, _ in candidate_pairs
                ]

                if model_choice == "Hybrid Transformer":
                    batch_results = predictor.score_pairs(pair_inputs, batch_size=32)
                else:
                    batch_results = [predictor.score_pair(left, right) for left, right in pair_inputs]

                for (_, _, base_row), result in zip(
                    candidate_pairs, batch_results
                ):
                    probability = result["match_probability"]

                    if probability < threshold:
                        continue

                    base_row["suggestion_score"] = probability
                    base_row["matching_method"] = "hybrid_transformer" if model_choice == "Hybrid Transformer" else "original_transformer"
                    candidates.append(base_row)

            candidates = pd.DataFrame(candidates)
            if not identifier_matches.empty:
                candidates = pd.concat([identifier_matches, candidates], ignore_index=True, sort=False)


            if candidates.empty:
                st.warning("No candidates met the current settings.")
            else:
                candidates = candidates.sort_values(
                    "suggestion_score", ascending=False
                ).reset_index(drop=True)

                # Greedy one-to-one proposal.
                used_ledger, used_bank, chosen = set(), set(), []

                for row in candidates.itertuples(index=False):


                    if ((pd.notna(row.ledger_id) and row.ledger_id in used_ledger)
                        or row.bank_id in used_bank):
                        continue

                    chosen.append(row._asdict())
                    if pd.notna(row.ledger_id):
                        used_ledger.add(row.ledger_id)
                    used_bank.add(row.bank_id)

                matches = pd.DataFrame(chosen)
                if not chosen:
                    st.warning(
                        "No non-duplicate candidates are available "
                        "for automatic suggestions. Review flagged "
                        "duplicate transactions manually."
                    )
                else:
                    matches["review_decision"] = "Pending"

                st.subheader("Suggested matches")
                if model_choice in {"Hybrid Transformer", "Original Transformer"}:
                    st.caption(
                        f"Scores are {model_choice} match probabilities. "
                        "Candidates are proposed one-to-one matches, not confirmed "
                        "reconciliations. Review each row before approval."
                    )
                else:
                    st.caption(
                        "Scores are transparent heuristic scores, not calibrated "
                        "probabilities. Review each proposed one-to-one match."
                    )
                if matches.empty:
                    st.stop()
                edited = st.data_editor(
                    matches,
                    hide_index=True,
                    use_container_width=True,
                    disabled=[
                        c for c in matches.columns
                        if c != "review_decision"
                    ],
                    column_config={
                        "review_decision": st.column_config.SelectboxColumn(
                            "Review decision",
                            options=["Pending", "Approve", "Reject"],
                            required=True,
                        ),
                        "suggestion_score": st.column_config.NumberColumn(
                            "Suggestion score", format="%.3f"
                        ),
                        "amount_difference": st.column_config.NumberColumn(
                            "Amount difference", format="%.2f"
                        ),
                        "text_similarity": st.column_config.NumberColumn(
                            "Text similarity", format="%.3f"
                        ),
                        "matching_method": st.column_config.TextColumn(
                            "Matching method"
                        ),
                    },
                )

                approved = edited[edited["review_decision"] == "Approve"]
                pending = edited[edited["review_decision"] == "Pending"]
                rejected = edited[edited["review_decision"] == "Reject"]


                # Persistent audit trail with an explicit save action.
                audit_path = Path("reports/reconciliation_audit_log.csv")
                audit_path.parent.mkdir(parents=True, exist_ok=True)

                st.subheader("Persistent reconciliation audit trail")

                review_col, reason_col = st.columns(2)
                with review_col:
                    reviewer_identity = st.text_input("Reviewer identity", placeholder="Name or work ID")
                with reason_col:
                    decision_reason = st.text_input("Decision reason", placeholder="Optional reason or case reference")

                decision_columns = [
                    "ledger_id",
                    "bank_id",
                    "suggestion_score",
                    "amount_difference",
                    "date_difference_days",
                    "review_decision",
                ]

                decisions_to_save = edited.reindex(columns=decision_columns).loc[
                    edited["review_decision"].isin(["Approve", "Reject"])
                ].copy()

                if st.button(
                    "Save review decisions",
                    key="save_review_decisions",
                ):
                    if decisions_to_save.empty:
                        st.warning(
                            "No approved or rejected decisions to save."
                        )
                    else:
                        validate_review_decisions(edited)
                        decisions_to_save = decision_audit_rows(
                            decisions_to_save,
                            reviewer=reviewer_identity,
                            reason=decision_reason,
                        )

                        if audit_path.exists():
                            existing_log = pd.read_csv(audit_path)
                            updated_log = pd.concat(
                                [existing_log, decisions_to_save],
                                ignore_index=True,
                            )
                        else:
                            updated_log = decisions_to_save

                        updated_log.to_csv(audit_path, index=False)
                        reviewed_payment_rows = edited.loc[
                            edited["review_decision"].isin(["Approve", "Reject"])
                        ]
                        if "payment_id" in reviewed_payment_rows.columns:
                            for reviewed_row in reviewed_payment_rows.dropna(subset=["payment_id"]).itertuples(index=False):
                                registry.record_reconciliation(
                                    reviewed_row.payment_id,
                                    reviewed_row.bank_id,
                                    "payment_id" if getattr(reviewed_row, "matching_method", "") in {"payment_id", "bank_reference"} else "ai_model",
                                    result={"Approve": "approved", "Reject": "rejected"}[reviewed_row.review_decision],
                                    reviewer_identity=reviewer_identity,
                                    decision_reason=decision_reason,
                                )
                        st.success(
                            f"Saved {len(decisions_to_save)} review decisions."
                        )

                if audit_path.exists():
                    audit_log = pd.read_csv(audit_path)
                    st.dataframe(
                        audit_log,
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "Download persistent audit log",
                        data=audit_log.to_csv(index=False).encode("utf-8-sig"),
                        file_name="reconciliation_audit_log.csv",
                        mime="text/csv",
                        key="download_persistent_audit",
                    )

                a, p, r = st.columns(3)
                a.metric("Approved", len(approved))
                p.metric("Pending review", len(pending))
                r.metric("Rejected", len(rejected))

                st.download_button(
                    "Download reviewed matches CSV",
                    data=edited.to_csv(index=False).encode("utf-8-sig"),
                    file_name="reviewed_reconciliation.csv",
                    mime="text/csv",
                )

                exports = review_exports(edited, ledger, bank)
                unmatched_ledger = exports["unmatched_ledger"]
                unmatched_bank = exports["unmatched_bank"]

                for export_name, export_frame in exports.items():
                    st.download_button(
                        f"Download {export_name.replace('_', ' ')} CSV",
                        data=export_frame.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"{export_name}.csv",
                        mime="text/csv",
                        key=f"download_{export_name}",
                    )

                st.subheader("Unmatched records after approval")
                u1, u2 = st.columns(2)
                with u1:
                    st.write("Unmatched ledger entries")
                    st.dataframe(unmatched_ledger, use_container_width=True)
                    st.download_button(
                        "Download unmatched ledger CSV",
                        data=unmatched_ledger.to_csv(index=False).encode("utf-8-sig"),
                        file_name="unmatched_ledger.csv",
                        mime="text/csv",
                    )
                with u2:
                    st.write("Unmatched bank transactions")
                    st.dataframe(unmatched_bank, use_container_width=True)
                    st.download_button(
                        "Download unmatched bank CSV",
                        data=unmatched_bank.to_csv(index=False).encode("utf-8-sig"),
                        file_name="unmatched_bank.csv",
                        mime="text/csv",
                    )

    except Exception as exc:
        st.error(str(exc))
else:
    st.write("Upload both CSV files to begin.")
    st.markdown(
        "Expected bank columns: `id` (or `bank_transaction_id`), `posted_date`, `narrative`, `amount`.\n\n"
        "Expected ledger columns: `id` (or `ledger_entry_id`), `entry_date`, `reference`, "
        "`counterparty`, `amount`."
    )
