
from pathlib import Path
from difflib import SequenceMatcher
import re
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

RANDOM_STATE = 42
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)

bank = pd.read_csv(DATA / "bank_lines.csv")
ledger = pd.read_csv(DATA / "ledger_entries.csv")
keys = pd.read_csv(DATA / "match_key.csv")

bank["id"] = bank["id"].astype(int)
ledger["id"] = ledger["id"].astype(int)
bank["account_key"] = bank["account"].astype(str)
ledger["account_key"] = ledger["book"].astype(str)
bank["parsed_date"] = pd.to_datetime(
    bank["posted_date"], errors="coerce"
)
ledger["parsed_date"] = pd.to_datetime(
    ledger["entry_date"], errors="coerce"
)

# Duplicate-line labels are excluded because duplicate bank rows
# cannot be represented as ordinary one-to-one matches.
valid_keys = keys[
    keys["kind"].astype(str).str.lower() != "duplicate_line"
]
positive_pairs = set(
    zip(
        valid_keys["ledger_entry_no"].astype(int),
        valid_keys["bank_line_no"].astype(int),
    )
)

def normalize(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def similarity(a, b):
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def make_candidate(l, b, period):
    ld, bd = l["parsed_date"], b["parsed_date"]
    days = (bd - ld).days if pd.notna(ld) and pd.notna(bd) else 9999

    la, ba = float(l["amount"]), float(b["amount"])
    diff = abs(la - ba)

    reference = str(l["reference"])
    counterparty = str(l["counterparty"])
    narrative = str(b["narrative"])

    ref_norm = normalize(reference)
    narrative_norm = normalize(narrative)
    ref_sim = similarity(reference, narrative)
    cp_sim = similarity(counterparty, narrative)

    return {
        "ledger_id": int(l["id"]),
        "bank_id": int(b["id"]),
        "period": period,
        "label": int((int(l["id"]), int(b["id"])) in positive_pairs),
        "ledger_date": ld,
        "bank_date": bd,
        "text": (
            f"reference {reference} "
            f"counterparty {counterparty} "
            f"bank narrative {narrative}"
        ),
        "amount_difference": diff,
        "exact_amount": int(diff < 0.005),
        "amount_within_cent": int(diff <= 0.01),
        "same_amount_sign": int(
            (la < 0 and ba < 0) or (la >= 0 and ba >= 0)
        ),
        "signed_date_difference": days,
        "absolute_date_difference": abs(days),
        "reference_in_narrative": int(
            bool(ref_norm) and ref_norm in narrative_norm
        ),
        "reference_similarity": ref_sim,
        "counterparty_similarity": cp_sim,
    }


# Split RECORDS chronologically, not individual candidate pairs.

# Split records chronologically, keeping every known positive pair
# together. The later date of a matched pair determines its partition.
all_dates = pd.concat(
    [ledger["parsed_date"], bank["parsed_date"]],
    ignore_index=True,
).dropna()

cut1, cut2 = all_dates.quantile([0.60, 0.80]).tolist()


def assign_period(date):
    if pd.isna(date):
        return "excluded"
    if date < cut1:
        return "train"
    if date < cut2:
        return "validation"
    return "test"


ledger["period"] = ledger["parsed_date"].map(assign_period)
bank["period"] = bank["parsed_date"].map(assign_period)

ledger_period_by_id = ledger.set_index("id")["period"].to_dict()
bank_period_by_id = bank.set_index("id")["period"].to_dict()

# Each positive pair is an indivisible evaluation unit.
pair_periods = {}
record_assignments = {"ledger": {}, "bank": {}}
assignment_conflicts = []

for row in valid_keys.itertuples(index=False):
    lid = int(row.ledger_entry_no)
    bid = int(row.bank_line_no)

    lrow = ledger.loc[ledger["id"] == lid]
    brow = bank.loc[bank["id"] == bid]

    if lrow.empty or brow.empty:
        raise ValueError(
            f"Answer-key pair references missing records: {lid}, {bid}"
        )

    ld = lrow.iloc[0]["parsed_date"]
    bd = brow.iloc[0]["parsed_date"]

    if pd.isna(ld) and pd.isna(bd):
        raise ValueError(
            f"Both dates missing for positive pair: {lid}, {bid}"
        )

    available_dates = [d for d in [ld, bd] if pd.notna(d)]
    anchor_date = max(available_dates)
    pair_period = assign_period(anchor_date)

    if pair_period == "excluded":
        raise ValueError(f"Cannot assign pair {lid}, {bid} to a period")

    pair_periods[(lid, bid)] = pair_period

    for side, record_id in [("ledger", lid), ("bank", bid)]:
        previous = record_assignments[side].get(record_id)

        if previous is not None and previous != pair_period:
            assignment_conflicts.append(
                (side, record_id, previous, pair_period)
            )

        record_assignments[side][record_id] = pair_period

if assignment_conflicts:
    raise ValueError(
        "Conflicting pair assignments found: "
        f"{assignment_conflicts[:10]}"
    )

# Move matched records together. Unmatched records retain date-based periods.
ledger["period"] = ledger.apply(
    lambda r: record_assignments["ledger"].get(
        int(r["id"]), r["period"]
    ),
    axis=1,
)
bank["period"] = bank.apply(
    lambda r: record_assignments["bank"].get(
        int(r["id"]), r["period"]
    ),
    axis=1,
)

# Confirm that no known positive pair crosses partitions.
ledger_period_by_id = ledger.set_index("id")["period"].to_dict()
bank_period_by_id = bank.set_index("id")["period"].to_dict()

cross_partition = [
    (lid, bid)
    for lid, bid in positive_pairs
    if ledger_period_by_id.get(lid) != bank_period_by_id.get(bid)
]

if cross_partition:
    raise ValueError(
        f"Positive pairs still cross partitions: {cross_partition[:10]}"
    )

print("Pair-aware chronological split:")
print(f"Training cutoff:   {pd.Timestamp(cut1).date()}")
print(f"Validation cutoff: {pd.Timestamp(cut2).date()}")
print("All known positive pairs remain within one partition.")

# Generate ONE shared candidate set for every model.
# A candidate must have the same account/book and both records
# must belong to the same chronological partition.
bank_by_account = {
    key: group for key, group in bank.groupby("account_key")
}
rows = []

for l in ledger.to_dict("records"):
    if l["period"] == "excluded":
        continue

    matching_bank = bank_by_account.get(l["account_key"])
    if matching_bank is None:
        continue

    matching_bank = matching_bank[
        matching_bank["period"] == l["period"]
    ]

    for b in matching_bank.to_dict("records"):
        rows.append(make_candidate(l, b, l["period"]))

pairs = pd.DataFrame(rows)

if pairs.empty:
    raise ValueError("No candidate pairs. Check account/book columns.")

NUMERIC_FEATURES = [
    "amount_difference",
    "exact_amount",
    "amount_within_cent",
    "same_amount_sign",
    "signed_date_difference",
    "absolute_date_difference",
    "reference_in_narrative",
    "reference_similarity",
    "counterparty_similarity",
]

train = pairs[pairs["period"] == "train"].copy()
validation = pairs[pairs["period"] == "validation"].copy()
test = pairs[pairs["period"] == "test"].copy()

for name, frame in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    if frame.empty or frame["label"].nunique() < 2:
        raise ValueError(
            f"{name} needs candidates with both labels. "
            "Inspect the date distribution or adjust the 60/80% cutoffs."
        )


# Model definitions: every learned model uses the same training pairs.
structured_models = {
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "MLP_NeuralNetwork": make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=1.0,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ),
    ),
}

# Text-only baseline: learns from transaction text, without amount/date features.
tfidf_only_model = Pipeline([
    (
        "features",
        ColumnTransformer([
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
                "text",
            ),
        ]),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
])

hybrid_preprocessor = ColumnTransformer([
    (
        "text",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
            strip_accents="unicode",
        ),
        "text",
    ),
    ("numeric", StandardScaler(), NUMERIC_FEATURES),
])

models = dict(structured_models)
models["TFIDF_Only"] = tfidf_only_model
models["Hybrid_TFIDF_Structured"] = Pipeline([
    ("features", hybrid_preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
])

def fit_and_score(model, frame, model_type="structured"):
    if model_type in {"hybrid", "tfidf"}:
        # Both text models accept the full DataFrame; each pipeline selects
        # its own text and/or structured features internally.
        return model.predict_proba(frame)[:, 1]
    return model.predict_proba(frame[NUMERIC_FEATURES])[:, 1]


def baseline_scores(frame):
    # Transparent rule-based confidence score. This provides a
    # consistent baseline; replace it with your existing baseline
    # rules if you want an exact reproduction of baseline_matcher.py.
    score = (
        0.40 * frame["exact_amount"].to_numpy()
        + 0.20 * frame["reference_in_narrative"].to_numpy()
        + 0.20 * frame["reference_similarity"].to_numpy()
        + 0.15 * frame["counterparty_similarity"].to_numpy()
        + 0.05 * (
            frame["absolute_date_difference"].to_numpy() <= 3
        ).astype(float)
    )
    return np.clip(score, 0.0, 1.0)


def greedy_one_to_one(frame, threshold):
    eligible = frame[frame["probability"] >= threshold].copy()
    eligible = eligible.sort_values(
        ["probability", "ledger_id", "bank_id"],
        ascending=[False, True, True],
    )

    used_ledger, used_bank, selected = set(), set(), []

    for row in eligible.itertuples(index=False):
        if row.ledger_id in used_ledger or row.bank_id in used_bank:
            continue
        used_ledger.add(row.ledger_id)
        used_bank.add(row.bank_id)
        selected.append({
            "ledger_id": int(row.ledger_id),
            "bank_id": int(row.bank_id),
            "probability": float(row.probability),
            "label": int(row.label),
        })

    return pd.DataFrame(
        selected,
        columns=["ledger_id", "bank_id", "probability", "label"],
    )


def evaluate_matches(predicted, expected):
    actual = set(zip(predicted["ledger_id"], predicted["bank_id"]))
    tp = len(actual & expected)
    fp = len(actual - expected)
    fn = len(expected - actual)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "predicted_matches": len(actual),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def expected_pairs(frame):
    return set(
        zip(
            frame.loc[frame["label"] == 1, "ledger_id"].astype(int),
            frame.loc[frame["label"] == 1, "bank_id"].astype(int),
        )
    )


# Fit and score each model. Threshold selection uses validation only.
results = []
test_score_frames = []
test_match_frames = []
threshold_rows = []

for model_name in [
    "RuleBasedBaseline",
    "LogisticRegression",
    "RandomForest",
    "MLP_NeuralNetwork",
    "TFIDF_Only",
    "Hybrid_TFIDF_Structured",
]:
    hybrid = model_name == "Hybrid_TFIDF_Structured"
    tfidf_only = model_name == "TFIDF_Only"

    start = time.perf_counter()

    if model_name == "RuleBasedBaseline":
        train_scores = baseline_scores(train)
        val_scores = baseline_scores(validation)
        test_scores = baseline_scores(test)
    else:
        model = models[model_name]

        if hybrid or tfidf_only:
            model.fit(train, train["label"])
            model_type = "hybrid" if hybrid else "tfidf"
            val_scores = fit_and_score(model, validation, model_type=model_type)
            test_scores = fit_and_score(model, test, model_type=model_type)
        else:
            if model_name == "MLP_NeuralNetwork":
                sample_weights = compute_sample_weight(
                    class_weight="balanced",
                    y=train["label"],
                )
                model.fit(
                    train[NUMERIC_FEATURES],
                    train["label"],
                    mlpclassifier__sample_weight=sample_weights,
                )
            else:
                model.fit(train[NUMERIC_FEATURES], train["label"])

            val_scores = fit_and_score(model, validation)
            test_scores = fit_and_score(model, test)
    elapsed = time.perf_counter() - start

    validation_scored = validation[
        ["ledger_id", "bank_id", "label"]
    ].copy()
    validation_scored["probability"] = val_scores

    val_expected = expected_pairs(validation)
    best_threshold, best_f1 = 0.5, -1.0

    for threshold in THRESHOLDS:
        predicted = greedy_one_to_one(validation_scored, threshold)
        metrics = evaluate_matches(predicted, val_expected)

        # On ties, keep the higher threshold to prefer fewer matches.
        if (
            metrics["f1"] > best_f1
            or (
                np.isclose(metrics["f1"], best_f1)
                and threshold > best_threshold
            )
        ):
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)

    test_scored = test[
        ["ledger_id", "bank_id", "label"]
    ].copy()
    test_scored["probability"] = test_scores
    test_scored["model"] = model_name

    predicted_test = greedy_one_to_one(test_scored, best_threshold)
    test_expected = expected_pairs(test)
    metrics = evaluate_matches(predicted_test, test_expected)

    results.append({
        "model": model_name,
        "cutoff_train_end": str(pd.Timestamp(cut1).date()),
        "cutoff_validation_end": str(pd.Timestamp(cut2).date()),
        "train_candidates": len(train),
        "validation_candidates": len(validation),
        "test_candidates": len(test),
        "train_positive_pairs": int(train["label"].sum()),
        "validation_positive_pairs": int(validation["label"].sum()),
        "test_positive_pairs": int(test["label"].sum()),
        "validation_best_f1": best_f1,
        "threshold_selected_on_validation": best_threshold,
        **metrics,
        "average_precision": average_precision_score(
            test["label"], test_scores
        ),
        "fit_and_score_seconds": round(elapsed, 3),
    })

    threshold_rows.append({
        "model": model_name,
        "threshold": best_threshold,
        "validation_f1": best_f1,
    })

    test_score_frames.append(test_scored)
    if not predicted_test.empty:
        predicted_test["model"] = model_name
        test_match_frames.append(predicted_test)


results_df = pd.DataFrame(results)
results_df.to_csv(
    REPORTS / "neural_experiment_model_comparison.csv", index=False
)

pd.DataFrame(threshold_rows).to_csv(
    REPORTS / "neural_experiment_validation_thresholds.csv", index=False
)

pd.concat(test_score_frames, ignore_index=True).to_csv(
    REPORTS / "neural_experiment_test_scores.csv", index=False
)

if test_match_frames:
    pd.concat(test_match_frames, ignore_index=True).to_csv(
        REPORTS / "neural_experiment_test_matches.csv", index=False
    )
