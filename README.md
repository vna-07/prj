# Loan Default Prediction (Zindi-style)

This repository now contains a complete, reproducible training + inference pipeline for loan-default prediction with robust tabular ML and deterministic output generation.

## 1) Repository layout

- `/home/runner/work/prj/prj/Train.csv` - training data (includes `target`)
- `/home/runner/work/prj/prj/Test.csv` - test data (no target)
- `/home/runner/work/prj/prj/economic_indicators.csv` - FRED indicators
- `/home/runner/work/prj/prj/SampleSubmission.csv` - reference format
- `/home/runner/work/prj/prj/VariableDefinitions.txt` - variable descriptions
- `/home/runner/work/prj/prj/train_submission.py` - end-to-end training and submission script
- `/home/runner/work/prj/prj/requirements.txt` - reproducible environment dependencies

## 2) Setup

From `/home/runner/work/prj/prj`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Run order (end-to-end)

1. Install dependencies (`requirements.txt`).
2. Run training + inference + threshold tuning:

```bash
python train_submission.py --data-dir /home/runner/work/prj/prj --output /home/runner/work/prj/prj/submission_blend.csv --target-name Target
```

3. Upload `/home/runner/work/prj/prj/submission_blend.csv`.

## 4) What the pipeline does

### Data processing and feature engineering

- Date features from `disbursement_date` and `due_date`.
- Financial interaction features such as:
  - repayment ratio
  - interest proxy
  - `interest_rate_proxy x duration`
  - lender-share interactions
- Loan-level aggregation from lender-level rows (one record per `customer_id + tbl_loan_id`) with lender count, sums, means, std/max/min.
- Customer-history features:
  - prior loan count
  - prior default count/rate
  - previous amount statistics
  - recency (`days since last loan`)
  - loan amount normalized by customer historical mean
- Leakage-safe out-of-fold target encoding for high-cardinality IDs:
  - `customer_id`, `tbl_loan_id`, `loan_type`, `country_id`, `New_versus_Repeat`
- FRED indicator merge by `country_id` and loan disbursement year.

### Modelling and calibration

- Primary model: LightGBM gradient-boosted trees.
- Optional blend (enabled by default if packages are available): LightGBM + XGBoost + CatBoost (probability averaging).
- Class imbalance handling with fold-specific `scale_pos_weight = negative/positive`.
- F1-optimized threshold search on out-of-fold predictions (instead of fixed 0.5).
- Country-specific thresholding is applied automatically when training labels contain multiple countries.

## 5) Reproducibility

- Fixed random seed (`--seed`, default `42`).
- Deterministic CV split (`StratifiedKFold` with fixed seed).
- Deterministic feature pipeline and output ordering by test IDs.

## 6) Output

Generated submission has exactly two columns:

- `ID`
- `Target` (or custom name via `--target-name`)

## 7) Runtime and hardware

Typical hardware target:
- Local machine or cloud notebook/VM with >= 4 vCPU and >= 8 GB RAM.

Expected runtime on the provided dataset:
- ~5-20 minutes depending on CPU and whether all three models are used.

You can reduce runtime by disabling extra models:

```bash
python train_submission.py --disable-xgb --disable-cat
```
