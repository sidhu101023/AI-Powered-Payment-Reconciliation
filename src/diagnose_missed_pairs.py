from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd

root = Path(__file__).resolve().parents[1]
bank = pd.read_csv(root / "data" / "bank_lines.csv")
ledger = pd.read_csv(root / "data" / "ledger_entries.csv")
missed = pd.read_csv(root / "reports" / "app_missed_reference_pairs.csv")

bank["posted_date"] = pd.to_datetime(bank["posted_date"], errors="coerce")
ledger["entry_date"] = pd.to_datetime(ledger["entry_date"], errors="coerce")
bank["id"] = pd.to_numeric(bank["id"]).astype(int)
ledger["id"] = pd.to_numeric(ledger["id"]).astype(int)

results = []

for _, pair in missed.iterrows():
    lid, bid = int(pair["ledger_id"]), int(pair["bank_id"])
    l = ledger.loc[ledger["id"] == lid].iloc[0]
    b = bank.loc[bank["id"] == bid].iloc[0]

    la, ba = float(l["amount"]), float(b["amount"])
    amount_diff = abs(la - ba)
    date_diff = abs((l["entry_date"] - b["posted_date"]).days)

    left = f'{l["reference"]} {l["counterparty"]}'.lower()
    right = str(b["narrative"]).lower()
    text_sim = SequenceMatcher(None, left, right).ratio()

    exact_amount = amount_diff <= 0.01
    same_sign = (la >= 0) == (ba >= 0)
    date_score = max(0, 1 - date_diff / 8)
    score = (
        0.55 * exact_amount
        + 0.20 * same_sign
        + 0.15 * text_sim
        + 0.10 * date_score
    )

    reasons = []
    if date_diff > 7:
        reasons.append("DATE_WINDOW")
    if amount_diff > 0.01 and text_sim < 0.70:
        reasons.append("AMOUNT_AND_TEXT_FILTER")
    if score < 0.70:
        reasons.append("BELOW_THRESHOLD")
    if not reasons:
        reasons.append("CHECK_ONE_TO_ONE_SELECTION_OR_OTHER_LOGIC")

    results.append({
        "ledger_id": lid,
        "bank_id": bid,
        "kind": pair.get("kind", ""),
        "ledger_amount": la,
        "bank_amount": ba,
        "amount_difference": round(amount_diff, 2),
        "date_difference_days": date_diff,
        "text_similarity": round(text_sim, 3),
        "heuristic_score": round(score, 3),
        "diagnosis": "; ".join(reasons),
    })

out = pd.DataFrame(results)
out.to_csv(root / "reports" / "missed_pair_diagnosis.csv", index=False)
print(out.to_string(index=False))
print("\nSaved reports/missed_pair_diagnosis.csv")
