# AI-Powered Payment Reconciliation

A research prototype for reconciling bank statement transactions with company
ledger entries. It combines transparent matching rules, saved Transformer
models, one-to-one assignment, exception review, and a simulated Payment ID
lifecycle. The application does not initiate bank transfers.

## Features

- Upload and reconcile bank and ledger CSV files.
- Generate candidate matches with a heuristic or Hybrid Transformer scorer.
- Review duplicate bank lines, amount/date exceptions, unmatched records, and
  one-to-one match proposals.
- Create persistent simulated payment instructions with Payment IDs and invoice
  allocations.
- Run reproducible synthetic Payment ID/reference experiments separately from
  the supplied practice dataset.

## Repository contents

- `app.py` — Streamlit application entry point.
- `data/` — practice bank, ledger, match-key, and unmatched-key CSV files.
- `synthetic_data/payment_id_v1/` — versioned synthetic Payment ID experiment
  inputs and isolated ground truth.
- `src/` — training, inference, evaluation, reconciliation, and reporting
  scripts.
- `tests/` — regression and feature tests.

Generated reports, local application state, model checkpoints, tokenizers, and
preprocessing artifacts are intentionally excluded from Git. The project does
not include trained weights under `models/`.

## Prerequisites and installation

Use Python 3.13 or another Python version supported by the dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Training downloads the public `distilbert/distilbert-base-uncased` backbone on
its first run when it is not already available in the Hugging Face cache.

## Train models locally

Trained checkpoints are intentionally not published. Train them in this order
before using application features that select a Transformer model.

1. Train the Ditto-inspired Transformer. It reads `data/` and saves its local
   checkpoint and tokenizer to `models/ditto_transformer_model/`.

   ```powershell
   python src/train_ditto_transformer.py
   ```

2. Train the Hybrid Transformer. It uses that local Ditto checkpoint as its
   backbone and writes its state, tokenizer, and scaler to
   `models/hybrid_transformer_model/`.

   ```powershell
   python src/train_hybrid_transformer.py
   ```

The application can still be used with the heuristic selector when local model
checkpoints have not yet been trained.

## Run the application

```powershell
python -m streamlit run app.py
```

For the practice reconciliation flow, upload:

- Bank: `data/bank_lines.csv`
- Ledger: `data/ledger_entries.csv`

The app also accepts the synthetic pair:

- Bank: `synthetic_data/payment_id_v1/bank_transactions.csv`
- Ledger: `synthetic_data/payment_id_v1/ledger.csv`

Review suggestions and exceptions before treating any match as reconciled.

## Tests

```powershell
python -m pytest -q -p no:cacheprovider
```

## Reproduce the synthetic Payment ID experiment

The generator is deterministic and writes only to the separate synthetic-data
and generated-report paths:

```powershell
python src/synthetic_payment_experiment.py --generate --evaluate
```

Synthetic results are experimental controls, not evidence of real bank or
payment-channel performance. Ground truth is stored separately and is not an
input to prediction functions.

## Limitations

- The supplied practice dataset is small and does not establish real-world
  performance.
- A Payment ID can only be matched exactly when a compatible imported bank
  field actually contains it or another stored reference.
- Grouped and partial payments may require human review because the original
  model representation is primarily ledger-bank pair based.
