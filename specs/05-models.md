# Spec: 05 – Models, ablations, calibration and registration

Step 5 of the pipeline. It trains logistic regression and LightGBM on the spec 03 features, using the split, metrics and MLflow setup from spec 04. It runs a fixed set of ablations, calibrates the chosen model, and registers it in MLflow for the API (spec 06). Serving the model, monitoring it and adding external data are later steps, not this one.

## Goal and user
Our 4-person team, and spec 06 (the API), which loads the registered model. We need to answer two questions:
1. Does ML beat the spec 04 baselines, by how much, and which feature groups account for the gain?
2. Which single model, with calibrated probabilities and a documented decision threshold, should the API serve?

## Inputs
- **`data/processed/features.parquet`** and **`data/processed/feature_report.json`** from spec 03. The predictors are exactly the `feature_columns` listed in the report. They are never re-derived here.
- **Spec 04 code:** `src/models/config.py` (split dates, limit, experiment name), `split.time_split`, `metrics.evaluate` and `metrics.breakdown`. These are imported, not copied.
- **Spec 04 baseline runs** in the `beach-exceedance` experiment (tag `spec=04`). The best baseline's validation PR-AUC is the score to beat.
- **`docs/decisions.md`:** read it first and change nothing in it.
- **Size now:** about 85k Enterococcus station-days. Rows with an NA target are dropped before training and evaluation (spec 03), and the rest are split by date into train ≤ 2021, validation 2022–2023 and test ≥ 2024. Validation and test are small (roughly 1–2k rows each) and have a higher positive rate than train (about 25–33% vs about 10%). Plan for at least 5× that size.

**Column names.** Spec 03 produces the table, so its names are the ones used:

| spec 04 name | spec 03 name |
|---|---|
| `sample_date` | `date` |
| `label` | `exceed_Enterococcus` |
| `last_entero` | `prev_log10` (log10 scale) / `prev_exceed` |
| `roll_mean_3`, `roll_max_3` | `roll3_log10`, `roll3_max_log10` |
| `days_since_last` | `days_since_prev` |
| `month_sin`, `month_cos` | `doy_sin`, `doy_cos` |

Spec 04 has to be updated to use these names (see Open questions).

## Outputs
1. **MLflow runs** in experiment `beach-exceedance`, all tagged `spec=05`:
   - one tuned run per model family (`logreg`, `lgbm`), with each tuning trial logged as a nested child run;
   - one run per ablation (listed below), tagged `ablation=<name>`;
   - one `final` run for the selected, calibrated model.
2. **A registered model** named **`beach-exceedance-entero`**:
   - its newest version gets the alias **`champion`**;
   - it is a single `mlflow.pyfunc` model that bundles preprocessing, the model and the calibrator;
   - it has a logged signature and an input example;
   - `predict(df)` returns a calibrated probability in [0, 1] for each row.
3. **Version tags** on the registered model: `feature_columns` (JSON), `threshold`, `threshold_rule`, `limit_mpn`, `train_end`, `val_end`, `test_start`, `git_commit`, `features_report_sha256`.
4. **`reports/models/`**:
   - `comparison.csv`: one row per model × split, with columns `model`, `split`, `pr_auc`, `recall_at_p50`, `brier`, `prevalence`, `n_rows`. Rows include the three baselines read from MLflow.
   - `ablations.csv`: one row per ablation with `ablation`, `val_pr_auc`, `test_pr_auc`, `val_brier`, `test_brier`, `delta_val_pr_auc_vs_full`.
   - `model_card.md`: what the model predicts, its data range, its metrics against the baselines, its calibration, the threshold, the subgroups where it does worst, and its known limits.
5. A summary table printed to stdout (model × split × metric), followed by the selection decision.

## Definitions and rules
**Data and split**
- Drop rows where `exceed_Enterococcus` is NA. Split with `time_split` using the dates in `config.py`.
- Everything learned from data (imputation values, scaling, category levels, hyperparameters, early-stopping round, calibrator, threshold) is fit on **train**, or for selection, calibration and threshold, on **validation**. The **test** set is scored only after the final model has been chosen, and it is never used to make a choice.
- Seeds are fixed (`SEED = 42` in `config.py`).

**Logistic regression** (an sklearn `Pipeline`):
- numeric features: median imputation with an added missing-indicator column, then `StandardScaler`;
- `county`: one-hot encoding with `handle_unknown='ignore'`;
- `LogisticRegression(solver='lbfgs', max_iter=2000)`, with no class weighting;
- tune `C` ∈ {0.01, 0.1, 1, 10} and pick the value with the best validation PR-AUC.

**LightGBM** (`LGBMClassifier`, binary objective):
- missing values (NaN) are left for LightGBM to handle natively;
- `county` is a pandas categorical whose levels are fixed from train; unseen counties become NaN;
- no `scale_pos_weight`;
- 30-trial random search with the fixed seed over `num_leaves` {15, 31, 63}, `learning_rate` {0.02, 0.05, 0.1}, `min_child_samples` {20, 50, 100}, `feature_fraction` {0.7, 0.9, 1.0}, `bagging_fraction` {0.7, 0.9, 1.0} with `bagging_freq=1`, and `reg_lambda` {0, 1, 10};
- up to 2,000 trees, with early stopping after 100 rounds on validation log-loss;
- pick the trial with the best validation PR-AUC.

**Calibration**
- Validation is used both for tuning and for calibration. To keep validation metrics honest after calibration, the calibrated validation scores come from **5-fold cross-fitting within validation**, using time-ordered folds. The final calibrator is then fit on all of validation and applied to test.
- **Default method: sigmoid (Platt)**, a one-variable logistic fit on the model's logit score. Isotonic is only an ablation, because validation is too small for it to fit stably.
- Report the following before and after calibration:
  - Brier score;
  - calibration-in-the-large (mean prediction minus observed rate);
  - a 10-bin reliability plot for validation and for test.

  Prevalence on test differs from train, so a calibration gap on test is reported, not hidden.

**Selection**
1. Candidates are the tuned, calibrated `logreg` and `lgbm`. The primary metric is **validation PR-AUC**, and the tie-breaker is validation Brier.
2. If the two models are within **0.01** validation PR-AUC of each other, choose `logreg`, because it is simpler and easier to explain.
3. The chosen model must beat the best spec 04 baseline's validation PR-AUC by at least **0.02**. If neither model does, **nothing is registered**: the run ends with exit code 0, the printed summary says so, and the model card records the result.
4. The test set is scored once for every model and baseline after selection, for reporting only.
5. **The registered model is the train-only model plus the validation-fit calibrator**, exactly the one that was evaluated. It is not refit on train + validation; see Open questions.

**Decision threshold:** choose the lowest threshold on the calibrated validation scores at which precision is at least 0.50. This is the same operating point as `recall_at_p50` in spec 04. If no threshold reaches precision 0.50, use the threshold that maximizes F1 and set `threshold_rule` to say so. The threshold is stored as a tag and used by the API; the model itself still returns probabilities.

**Ablations.** Each ablation is one MLflow run that uses the selected family and its tuned hyperparameters (no retuning) and is scored on validation and test:

| name | change from the full model |
|---|---|
| `no_lags` | drop `prev_*`, `roll3_*`, `days_since_prev`, `n_roll3`, `n_prev_*`, `no_history` |
| `no_station_rate` | drop `station_hist_rate`, `n_prev_known` |
| `no_location` | drop `county`, `lat`, `lon` |
| `no_calendar` | drop `month`, `dow`, `wet_season`, `doy_sin`, `doy_cos` |
| `lags_only` | keep only the lag and station-history features |
| `other_family` | the family that was *not* selected, already tuned (for the comparison table) |
| `class_weight_balanced` | `class_weight='balanced'` (logreg) or `is_unbalance=True` (lgbm), uncalibrated and then sigmoid-calibrated |
| `calib_none` / `calib_isotonic` | no calibration / isotonic calibration instead of sigmoid |
| `random_split` | a stratified 70/15/15 random split instead of the time split. These scores are tagged `leaky_comparison=true` and **never** used for selection; they only show how optimistic a random split is. |

**Breakdowns** (from `metrics.breakdown`, logged as CSV artifacts for the final run and the baselines):
- by county;
- by season, following the spec 04 definition;
- **seen vs unseen stations**: test rows whose `station_id` never appears in train;
- **no-history rows** (`no_history == 1`).

Any group with fewer than 30 rows or no positives is skipped, and the skip is recorded.

**MLflow logging**
- Params: `model_type`, all hyperparameters, `calibration`, `feature_set` (the ablation name, or `full`), `train_end`, `val_end`, `test_start`, `limit_mpn`, `seed`.
- Metrics: the same names as spec 04 (`val_pr_auc`, `val_recall_at_p50`, `val_brier`, `test_*`, `val_prevalence`, `test_prevalence`), plus `val_brier_uncal`, `val_calib_in_large` and `test_calib_in_large`.
- Artifacts: PR curves, reliability plots, breakdown CSVs and feature importance (gain for LightGBM, standardized coefficients for logistic regression).
- The tracking URI comes from `MLFLOW_TRACKING_URI`, with no credentials in code.
- Model versions are referred to by **alias** (`champion`), not by the deprecated stages.

**Files**
- `src/models/config.py`: extended with `SEED`, search spaces, `MIN_PR_AUC_GAIN = 0.02`, `TIE_MARGIN = 0.01` and `REGISTERED_MODEL_NAME`.
- `src/models/logreg.py` and `src/models/lgbm.py`: `build(params)`, `fit(train, val)` and `predict_proba(df)`.
- `src/models/calibration.py`: `crossfit_calibrate(scores, y, method, n_folds)` and `fit_calibrator(...)`.
- `src/models/ablations.py`: the ablation table above, kept as data and not hard-coded in several places.
- `src/models/pyfunc.py`: the wrapper class (preprocessing + model + calibrator).
- `src/models/train_models.py`: the entry point.
- `src/models/register.py`: registers the model and sets the alias and tags.
- Command: `python -m src.models.train_models`. Add `--skip-ablations` for a fast run.

## Acceptance criteria
- [ ] On the spec 03 output, `python -m src.models.train_models` creates the runs, reports and printed summary listed above and exits with code 0.
- [ ] Every probability is finite and in [0, 1], and no prediction is missing, including for rows with `no_history == 1` and for a county not seen in train.
- [ ] **No test influence:** changing every test label in a fixture leaves the chosen hyperparameters, the selected family, the calibrator and the threshold unchanged.
- [ ] Preprocessing fit on train only: changing a validation or test feature value leaves the imputation medians, the scaler and the category levels unchanged.
- [ ] Sigmoid calibration on a fixture whose scores are systematically too high lowers the Brier score and brings calibration-in-the-large closer to 0.
- [ ] Selection logic is unit-tested with fake metric dicts:
  - a larger PR-AUC wins;
  - a gap within 0.01 picks `logreg`;
  - a gain below 0.02 over the baseline registers nothing.
- [ ] The threshold gives precision ≥ 0.50 on validation, or `threshold_rule` records the F1 fallback.
- [ ] Every ablation in the table has exactly one run with `ablation=<name>`, and `ablations.csv` has one row per ablation. The `random_split` run is tagged `leaky_comparison=true` and never appears in the selection.
- [ ] Loading the champion model with `mlflow.pyfunc.load_model("models:/beach-exceedance-entero@champion")` and predicting on the logged input example gives the same values as the in-memory model (to 1e-9). Its signature matches `feature_columns`.
- [ ] The registered version has every required tag, and `feature_columns` equals the list in `feature_report.json`.
- [ ] Two runs with the same seed give identical metrics and an identical threshold.
- [ ] pytest tests in `tests/test_models.py`, `tests/test_calibration.py` and `tests/test_selection.py` run on a small fixture parquet with a temporary local MLflow directory. They create a registered model with the `champion` alias and never touch the real data or the team's tracking server.

## Performance budget
- The current data, with full tuning and all ablations, runs in **under 15 min** with peak memory **under 4 GB** on a laptop CPU.
- `--skip-ablations` runs in under 5 min.
- At 5× the data, a full run takes under 45 min.
- Scoring 10k rows with the registered pyfunc takes under 1 s.
- These are targets for the implementation, not timings that have been measured.

## Out of scope
- Serving the model, the FastAPI endpoints and Docker (spec 06).
- Changing the split dates, the limit, the target rules or the feature definitions (specs 01–04 and `docs/decisions.md`).
- New features: rainfall, tide, other-indicator lags, or target encoding of station or beach IDs.
- Other model families (random forest, XGBoost, neural nets), stacking or ensembles.
- Rebuilding features with a different lab delay. That would need a parameter in spec 03.
- Drift monitoring, scheduled retraining and promotion to a cloud registry.
- SHAP analysis beyond gain and coefficients.

## Open questions
1. **Spec 04 column names:** spec 04 uses `last_entero`, `roll_mean_3`, `sample_date`, `label` and `month_sin`, but spec 03 produces `prev_log10`, `roll3_log10`, `date`, `exceed_Enterococcus` and `doy_sin`. Should spec 04 be updated to spec 03's names, as this spec assumes? `prev_log10` is on a log10 scale, so the persistence baseline should use `prev_exceed`, or compare `prev_log10` with `log10(104)`.
2. **Spec 04 says "routine samples only," but spec 03 keeps all station-days:** which one is right?
3. **Refit before registering:** register the train-only model as evaluated (current rule), or refit on train + validation with the chosen hyperparameters? Refitting uses the most recent data but gives a model that was never evaluated.
4. **Using validation twice:** validation is used for tuning, early stopping and calibration (with cross-fitting). Should validation be split instead, for example tuning on 2022 and calibrating on 2023?
5. **Margins:** is a gain of ≥ 0.02 PR-AUC over the best baseline and a tie margin of 0.01 the right bar? If no model clears it, should the best baseline be registered as `champion` instead of nothing?
6. **Operating point:** is precision ≥ 0.50 the right threshold rule for advisories, or should we target recall (for example, catching ≥ 80% of exceedances)?
7. **Prevalence shift:** the test set (2024+) has about 3× the train positive rate, mostly from Tijuana-area stations. Should we also report test metrics without those stations, or recalibrate on the most recent year before serving?
8. **Lab-delay ablation:** is it worth adding a delay parameter to spec 03, so that we can measure how much a 0-day vs 2-day history cutoff inflates scores?
