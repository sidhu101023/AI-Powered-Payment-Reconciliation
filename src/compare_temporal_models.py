
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
bank["parsed_date"] = pd.to_datetime(bank["posted_date"], errors="coerce")
ledger["parsed_date"] = pd.to_datetime(ledger["entry_date"], errors="coerce")

valid_keys = keys[
    keys["kind"].astype(str).str.lower() != "duplicate_line"
]
positive_pairs = set(zip(
    valid_keys["ledger_entry_no"].astype(int),
    valid_keys["bank_line_no"].astype(int),
))

def normalize(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

def similarity(a, b):
    a, b = normalize(a), normalize(b)
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0

def make_candidate(l, b, period):
    ld, bd = l["parsed_date"], b["parsed_date"]
    days = (bd - ld).days
    la, ba = float(l["amount"]), float(b["amount"])
    diff = abs(la - ba)
    ref, cp, narrative = (
        str(l["reference"]), str(l["counterparty"]), str(b["narrative"])
    )
    ref_norm, narrative_norm = normalize(ref), normalize(narrative)
    return {
        "ledger_id": int(l["id"]),
        "bank_id": int(b["id"]),
        "period": period,
        "label": int((int(l["id"]), int(b["id"])) in positive_pairs),
        "ledger_date": ld,
        "bank_date": bd,
        "text": f"reference {ref} counterparty {cp} bank narrative {narrative}",
        "amount_difference": diff,
        "exact_amount": int(diff < 0.005),
        "amount_within_cent": int(diff <= 0.01),
        "same_amount_sign": int((la < 0 and ba < 0) or (la >= 0 and ba >= 0)),
        "signed_date_difference": days,
        "absolute_date_difference": abs(days),
        "reference_in_narrative": int(bool(ref_norm) and ref_norm in narrative_norm),
        "reference_similarity": similarity(ref, narrative),
        "counterparty_similarity": similarity(cp, narrative),
    }

# Preserve the existing experiment's date cutoffs for comparability.
all_dates = pd.concat(
    [ledger["parsed_date"], bank["parsed_date"]], ignore_index=True
).dropna()
cut1, cut2 = all_dates.quantile([0.60, 0.80]).tolist()

def bank_period(date):
    if pd.isna(date):
        return "excluded"
    if date < cut1:
        return "train"
    if date < cut2:
        return "validation"
    return "test"

bank["period"] = bank["parsed_date"].map(bank_period)

# For each bank line, consider only ledger entries dated on or before
# that bank posting date. This approximates information available then.
ledger_by_account = {
    k: g for k, g in ledger.groupby("account_key")
}
rows = []

for b in bank.to_dict("records"):
    period = b["period"]
    if period not in {"train", "validation", "test"}:
        continue

    eligible = ledger_by_account.get(b["account_key"])
    if eligible is None or pd.isna(b["parsed_date"]):
        continue

    eligible = eligible[
        eligible["parsed_date"].notna()
        & (eligible["parsed_date"] <= b["parsed_date"])
    ]

    for l in eligible.to_dict("records"):
        rows.append(make_candidate(l, b, period))

pairs = pd.DataFrame(rows)
if pairs.empty:
    raise ValueError("No eligible temporal candidate pairs were generated.")

# Audit known matches that the operational date rule cannot consider.
missing_positive = []
for lid, bid in positive_pairs:
    lrow = ledger.loc[ledger["id"] == lid]
    brow = bank.loc[bank["id"] == bid]
    if lrow.empty or brow.empty:
        missing_positive.append((lid, bid, "missing_record"))
        continue
    ld, bd = lrow.iloc[0]["parsed_date"], brow.iloc[0]["parsed_date"]
    if pd.isna(ld) or pd.isna(bd) or ld > bd:
        missing_positive.append((lid, bid, "ledger_date_after_bank_date"))
    elif not (
        lrow.iloc[0]["account_key"] == brow.iloc[0]["account_key"]
    ):
        missing_positive.append((lid, bid, "account_book_mismatch"))

pd.DataFrame(
    missing_positive, columns=["ledger_id", "bank_id", "exclusion_reason"]
).to_csv(REPORTS / "temporal_excluded_positive_pairs.csv", index=False)

features = [
    "amount_difference", "exact_amount", "amount_within_cent",
    "same_amount_sign", "signed_date_difference",
    "absolute_date_difference", "reference_in_narrative",
    "reference_similarity", "counterparty_similarity",
]

train = pairs[pairs["period"] == "train"].copy()
validation = pairs[pairs["period"] == "validation"].copy()
test = pairs[pairs["period"] == "test"].copy()

for name, frame in [("train", train), ("validation", validation), ("test", test)]:
    if frame.empty or frame["label"].nunique() < 2:
        raise ValueError(
            f"{name} must contain candidates with both labels; "
            "inspect temporal candidate coverage before continuing."
        )

models = {
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "TFIDF_Only": Pipeline([
        ("features", ColumnTransformer([
            ("text", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=30000,
                sublinear_tf=True, strip_accents="unicode"
            ), "text"),
        ])),
        ("classifier", LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ]),
    "Hybrid_TFIDF_Structured": Pipeline([
        ("features", ColumnTransformer([
            ("text", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=30000,
                sublinear_tf=True, strip_accents="unicode"
            ), "text"),
            ("numeric", StandardScaler(), features),
        ])),
        ("classifier", LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ]),
}

def baseline_scores(frame):
    score = (
        0.40 * frame["exact_amount"].to_numpy()
        + 0.20 * frame["reference_in_narrative"].to_numpy()
        + 0.20 * frame["reference_similarity"].to_numpy()
        + 0.15 * frame["counterparty_similarity"].to_numpy()
        + 0.05 * (frame["absolute_date_difference"].to_numpy() <= 3)
    )
    return np.clip(score, 0.0, 1.0)

def greedy_one_to_one(frame, threshold):
    eligible = frame[frame["probability"] >= threshold].copy()
    eligible = eligible.sort_values(
        ["probability", "ledger_id", "bank_id"],
        ascending=[False, True, True],
    )
    used_l, used_b, chosen = set(), set(), []
    for r in eligible.itertuples(index=False):
        if r.ledger_id in used_l or r.bank_id in used_b:
            continue
        used_l.add(r.ledger_id)
        used_b.add(r.bank_id)
        chosen.append({
            "ledger_id": int(r.ledger_id),
            "bank_id": int(r.bank_id),
            "probability": float(r.probability),
            "label": int(r.label),
        })
    return pd.DataFrame(chosen, columns=[
        "ledger_id", "bank_id", "probability", "label"
    ])

def expected_pairs(frame):
    return set(zip(
        frame.loc[frame["label"] == 1, "ledger_id"].astype(int),
        frame.loc[frame["label"] == 1, "bank_id"].astype(int),
    ))

def evaluate(predicted, expected):
    selected = set(zip(predicted["ledger_id"], predicted["bank_id"]))
    tp = len(selected & expected)
    fp = len(selected - expected)
    fn = len(expected - selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted_matches": len(selected), "true_positives": tp,
        "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }

results, thresholds, all_scores, all_matches = [], [], [], []

for name in [
    "RuleBasedBaseline", "LogisticRegression", "RandomForest",
    "TFIDF_Only", "Hybrid_TFIDF_Structured",
]:
    start = time.perf_counter()

    if name == "RuleBasedBaseline":
        val_prob = baseline_scores(validation)
        test_prob = baseline_scores(test)
    else:
        model = models[name]
        if name in {"TFIDF_Only", "Hybrid_TFIDF_Structured"}:
            model.fit(train, train["label"])
            val_prob = model.predict_proba(validation)[:, 1]
            test_prob = model.predict_proba(test)[:, 1]
        else:
            model.fit(train[features], train["label"])
            val_prob = model.predict_proba(validation[features])[:, 1]
            test_prob = model.predict_proba(test[features])[:, 1]

    elapsed = time.perf_counter() - start
    val_scored = validation[["ledger_id", "bank_id", "label"]].copy()
    val_scored["probability"] = val_prob
    val_expected = expected_pairs(validation)

    best_threshold, best_f1 = 0.5, -1.0
    for threshold in THRESHOLDS:
        metric = evaluate(greedy_one_to_one(val_scored, threshold), val_expected)
        if metric["f1"] > best_f1 or (
            np.isclose(metric["f1"], best_f1) and threshold > best_threshold
        ):
            best_f1, best_threshold = metric["f1"], float(threshold)

    test_scored = test[["ledger_id", "bank_id", "label"]].copy()
    test_scored["probability"] = test_prob
    test_scored["model"] = name
    predicted = greedy_one_to_one(test_scored, best_threshold)
    test_expected = expected_pairs(test)
    metric = evaluate(predicted, test_expected)

    results.append({
        "model": name,
        "train_cutoff": str(pd.Timestamp(cut1).date()),
        "validation_cutoff": str(pd.Timestamp(cut2).date()),
        "train_candidates": len(train),
        "validation_candidates": len(validation),
        "test_candidates": len(test),
        "train_positive_pairs": int(train["label"].sum()),
        "validation_positive_pairs": int(validation["label"].sum()),
        "test_positive_pairs": int(test["label"].sum()),
        "validation_best_f1": best_f1,
        "threshold": best_threshold,
        **metric,
        "average_precision": average_precision_score(test["label"], test_prob),
        "fit_and_score_seconds": round(elapsed, 3),
    })
    thresholds.append({
        "model": name, "threshold": best_threshold, "validation_f1": best_f1
    })
    all_scores.append(test_scored)
    if not predicted.empty:
        predicted["model"] = name
        all_matches.append(predicted)

pd.DataFrame(results).to_csv(
    REPORTS / "temporal_model_comparison.csv", index=False
)
pd.DataFrame(thresholds).to_csv(
    REPORTS / "temporal_validation_thresholds.csv", index=False
)
pd.concat(all_scores, ignore_index=True).to_csv(
    REPORTS / "temporal_test_scores.csv", index=False
)
if all_matches:
    pd.concat(all_matches, ignore_index=True).to_csv(
        REPORTS / "temporal_test_matches.csv", index=False
    )

coverage = pd.DataFrame([
    {"period": p, "candidate_pairs": len(frame),
     "positive_pairs": int(frame["label"].sum()),
     "unique_bank_lines": frame["bank_id"].nunique(),
     "unique_ledger_entries": frame["ledger_id"].nunique()}
    for p, frame in [
        ("train", train), ("validation", validation), ("test", test)
    ]
])
coverage.to_csv(REPORTS / "temporal_candidate_coverage.csv", index=False)

print("=== TEMPORAL EVALUATION ===")
print("Bank-date cutoffs:", pd.Timestamp(cut1).date(), pd.Timestamp(cut2).date())
print("Excluded positive pairs:", len(missing_positive))
print(coverage.to_string(index=False))
print(pd.DataFrame(results)[[
    "model", "threshold", "predicted_matches", "true_positives",
    "false_positives", "false_negatives", "precision", "recall", "f1",
    "average_precision"
]].to_string(index=False))
print("Reports saved in:", REPORTS)
