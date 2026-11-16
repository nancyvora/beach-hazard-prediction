# Spec 04: Baselines, time split, metrics, MLflow logging

## Goal
Build the evaluation scaffolding every later model reuses (time split, metrics,
MLflow logging), and set the two baseline scores that LogReg and LightGBM
(spec 05) must beat. Answers: "How well can we do with no ML at all?"

## Input
data/processed/features.parquet (from spec 03), one row per routine
Enterococcus sample, with columns:
station_id, beach_name, county, sample_date, last_entero, roll_mean_3,
roll_max_3, days_since_last, station_hist_rate, month_sin, month_cos, label

- `label` = 1 if the Enterococcus result is above the state single-sample
  limit (104 MPN/100 mL; the value and citation live in docs/decisions.md),
  else 0.
- All lag features already use only samples from t-2 days or earlier (spec 03).

## Rules
- Read docs/decisions.md first. Do not change any decision there.
- **Time split only. Never a random split.** Boundaries come from
  docs/decisions.md, by `sample_date`:
  - train: date <= 2021-12-31
  - validation: 2022-01-01 to 2023-12-31
  - test: date >= 2024-01-01
  Put the dates in one config (`src/models/config.py`), not hard-coded in
  several places.
- Anything learned from data (fallback rates, thresholds) is fit on **train
  only**. Validation is used for choosing; test is reported, never tuned on.
- Each baseline outputs a probability-like score in [0, 1] for every row,
  with no NaNs.

## Baselines
1. **Persistence**: "the water is like it was last time."
   - score = 1.0 if `last_entero` > limit, else 0.0
   - If `last_entero` is NaN (first sample at a station), use the train-set
     exceedance rate (prevalence).
2. **Station historical rate**: "this station's usual exceedance rate."
   - score = `station_hist_rate` (expanding, past-only, from spec 03)
   - If NaN, fall back to the county's train-set rate, then to the global
     train-set rate.
3. **Constant prevalence** (sanity floor): score = train-set prevalence for
   every row. Its PR-AUC should be about equal to the positive rate. Any model
   that cannot beat it is useless.

## Metrics (`src/models/metrics.py`, shared with spec 05)
Compute on validation AND test:
- `pr_auc` (average precision)
- `recall_at_p50`: the highest recall where precision >= 0.50 (0 if never reached)
- `brier` (Brier score)
- `prevalence` and `n_rows` (context for reading PR-AUC)

Breakdowns (logged as a CSV artifact, not as separate metrics):
- by `county`
- by season: swim season = Apr 1 to Oct 31 (AB 411 period); off-season = the rest
- Skip any group with fewer than 30 rows or no positives, and record that it
  was skipped.

## MLflow logging
- Experiment: `beach-exceedance`; one run per baseline.
- Params: `model_type` (persistence | station_rate | prevalence), `feature_set`,
  `train_end`, `val_end`, `test_start`, `limit_mpn`.
- Metrics: `val_pr_auc`, `val_recall_at_p50`, `val_brier`, `test_*` (same three),
  `val_prevalence`, `test_prevalence`.
- Artifacts: PR curve png (val + test), calibration plot png (skip for
  persistence, which only scores 0/1, and say why in a run tag),
  `breakdown_county.csv`, `breakdown_season.csv`.
- Tag: `spec=04`.
- Tracking URI from the env var `MLFLOW_TRACKING_URI`. Never hard-code
  credentials.

## Files
- `src/models/config.py`: split dates, limit, experiment name
- `src/models/split.py`: `time_split(df) -> (train, val, test)`
- `src/models/metrics.py`: `evaluate(y_true, y_score) -> dict`, `breakdown(df, by)`
- `src/models/baselines.py`: the three baselines, each with `fit(train)` and
  `predict(df)`
- `src/models/train_baselines.py`: entry point; runs all three and logs to MLflow
- Command: `python -m src.models.train_baselines`

## Output
- Three MLflow runs with the params, metrics, and artifacts above.
- A short summary table printed to stdout (model x val/test x metric).

## Acceptance tests (tests/test_split.py, tests/test_metrics.py, tests/test_baselines.py)
- `time_split`: no date overlap between sets; max(train date) < min(val date)
  and max(val date) < min(test date); every input row lands in exactly one set.
- `time_split` raises if given an empty frame or one missing `sample_date`.
- `evaluate` on a hand-made example (e.g. y=[0,0,1,1], score=[.1,.4,.35,.8])
  gives the known PR-AUC and Brier values.
- `recall_at_p50` returns 0 when precision never reaches 0.5.
- Persistence: last_entero=200 gives 1.0; last_entero=50 gives 0.0;
  NaN gives the train prevalence.
- Station rate: a NaN station rate falls back to the county rate, then the
  global rate; the output has no NaNs and every value is in [0, 1].
- Fallback rates are computed from train rows only: changing a label in val
  or test does not change them.
- Prevalence baseline: val PR-AUC is within 0.02 of val prevalence.
- MLflow: with a temporary local tracking dir, running the entry point on a
  small fixture parquet creates 3 runs, each tagged `spec=04` with all
  required params and metrics.

## Out of scope
Logistic regression, LightGBM, ablations, and model registration (spec 05).
The random-vs-time split comparison is an ablation in spec 05.
