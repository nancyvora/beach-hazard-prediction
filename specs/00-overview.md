# Spec: 00 – Project overview (California beach Enterococcus exceedance)

This is the umbrella spec for the project. It does not replace specs 01–08. It records what each one does, the files each passes to the next, and the rules every spec must follow. If this file and a numbered spec disagree, the numbered spec is the detailed source. The disagreement is then a bug in one of the two, and it goes in Open questions until it is fixed.

## Goal and user
- **Who uses it:** beachgoers, who want to check how likely a beach is to fail the Enterococcus standard on a given date before they visit. County health staff are a secondary audience.
- **What we're building:** an end-to-end pipeline for our 4-person UCSD team. It cleans the public data, trains and registers a calibrated model, and serves it through a public API.
- **What it is not:** the API gives an informational estimate. It is not a measurement and not an official advisory.

**The question the model answers:** will this station's Enterococcus result be above **104 MPN/100 mL** on day *t*, using only lab results available by *t − 2* days?

## Inputs
- **Raw data:** `beach-monitoring-results.csv` from data.ca.gov (resource `7bd961cf-abe4-433b-8033-378161237ff3`). It is stored at `data/raw/` and never edited or committed.
- **Size now:** 230,268 rows × 93 columns, about 769 MB in pandas, covering 1984 to April 2025.
- **Size later:** every step must handle up to 1M lab results.
- **Shared decisions:** `docs/decisions.md` holds the split dates, the limit and its citation, and the resolved open questions. Every spec reads it and none changes it on its own.
- **Shared config:** `src/models/config.py` holds the split dates, the limit, the seed, the experiment and model names, and the selection margins. No spec hard-codes these anywhere else.

## Outputs
**What each spec produces, and who uses it**

| spec | what it does | command | produces | used by |
|---|---|---|---|---|
| 01 cleaning | Parses dates, hours and result values; maps four indicators; marks censoring (`exact`/`below_DL`/`above_QL`); drops qPCR, negative, out-of-range-date and unusable rows, counting each | (spec 01 entry point) | `data/interim/results_clean.parquet` (one row per lab result), `cleaning_report.json` | 02, 03 |
| 02 EDA | Builds reproducible tables and figures: Enterococcus station-day events, censoring-aware exceedance, rates by year, month, county and station, persistence at 0- and 2-day gaps, candidate time splits | `python -m src.data.eda` | `reports/eda/tables/*.csv`, `figures/*.png`, `eda_report.json` | `docs/decisions.md` |
| 03 features | Saves station-day events and labels; builds history features that respect the t−2 cutoff, plus calendar and location features; provides a feature function the API can reuse | `python -m src.features.build` | `data/processed/station_days.parquet`, `features.parquet`, `feature_report.json` | 04, 05, 06, 07 |
| 04 baselines | Provides the time split, metrics and MLflow logging; runs persistence, station-rate and prevalence baselines | `python -m src.models.train_baselines` | 3 MLflow runs (`spec=04`); `split.py`, `metrics.py`, `config.py` | 05, 08 |
| 05 models | Tunes logistic regression and LightGBM; runs ablations; applies sigmoid calibration; sets the threshold at precision ≥ 0.50; registers the selected model | `python -m src.models.train_models` | MLflow runs (`spec=05`); registered pyfunc `beach-exceedance-entero@champion` with 9 version tags; `reports/models/` (comparison, ablations, model card) | 06, 07 |
| 06 API | Loads the pinned model, feature report, catalog and history at startup; `GET /predict?beach=&date=` returns probabilities for each station plus the maximum across stations as a beach summary; `GET /health` | `uvicorn src.api.main:app --port 7860` | `src/api/` app | 07 |
| 07 deploy | Exports the champion model, station catalog and full history snapshot to a Hugging Face model repo in one commit; Docker image; Space; smoke test; rollback by revision | `python -m deploy.publish_model …`, `deploy.push_space`, `deploy.smoke_test` | Hub model repo, Docker Space, `docs/deployment.md` | users |
| 08 CI | GitHub Actions jobs for lint, tests, a smoke run of the training entry points on fixtures, and a guard against committed data or large files | `ruff …`, `pytest -q --cov=src`, `scripts/smoke.sh`, `scripts/check_repo_files.py` | `.github/workflows/ci.yml` | every PR |

**How files flow between specs**
```
raw CSV ─01→ results_clean.parquet ─02→ reports/eda → docs/decisions.md
                                    └03→ station_days / features / feature_report
                                           ├04→ baseline runs ─┐
                                           └05→ model runs ────┴→ registry: beach-exceedance-entero@champion
                                                                     └07 publish → Hub repo → Docker Space ← 06 app
08 CI checks 01–07 on fixtures at every push
```

**Repository layout:** `src/{data,features,models,api}/`, `deploy/`, `scripts/`, `tests/` (with `tests/fixtures/`), `docs/{decisions,deployment}.md`, `specs/00–08`, `reports/`, and `data/{raw,interim,processed}/`. Everything under `data/` stays out of git.

## Definitions and rules
These rules hold across the whole project. The numbered specs give the details.

**Target and labels** (02, 03)
- There is one row per station × date for Enterococcus. If a station has several results that day, the worst one is kept.
- `exceed_Enterococcus` is defined as follows:

| result | label |
|---|---|
| exact, value > 104 | 1 |
| `above_QL`, value ≥ 104 | 1 |
| `below_DL`, value ≤ 104 | 0 |
| `below_DL` above 104, or `above_QL` below 104 | NA (unknown) |

- Rows with unknown labels stay in the history but are excluded from training and evaluation.
- Censored values are stored as reported. **No step imputes them** (no DL/2, no Tobit).

**Time and leakage** (03, 04, 06)
- All dates are calendar dates in California local time.
- A prediction for day *t* may only use events at the **same station** dated on or before **t − 2**.
- The same feature function is used for the batch features and for the API.
- Same-day lab fields (qualifier, method, RL, MDL, other indicators) are never used as predictors.
- Station and beach IDs are never target-encoded.

**Split and fitting** (04, 05)
- The split is by time only: train ≤ 2021-12-31, validation 2022–2023, test ≥ 2024-01-01.
- Preprocessing and fallback rates are fit on train.
- Selection, calibration and the threshold use validation.
- Test is scored once, after selection, and is only reported.
- The random-split ablation is tagged as leaky and never used for selection.

**Metrics** (04, 05): PR-AUC (primary), recall at precision ≥ 0.50, Brier score, prevalence and row count. Results are broken down by county, by swim season vs off-season, by seen vs unseen station, and by no-history rows. Groups with fewer than 30 rows are skipped.

**Model contract** (05 → 06 → 07)
- The served model is one MLflow pyfunc that bundles preprocessing, the classifier and the sigmoid calibrator. `predict(df)` returns one calibrated probability per row.
- It is identified by its alias and then pinned to a numeric version.
- Its 9 version tags (`feature_columns`, `threshold`, `threshold_rule`, `limit_mpn`, `train_end`, `val_end`, `test_start`, `git_commit`, `features_report_sha256`) travel with it.
- If no model beats the best baseline by ≥ 0.02 validation PR-AUC, nothing is registered, and 06 and 07 stop. They do not fall back to a baseline.

**Serving** (06, 07)
- Supported request dates run from `val_end + 2 days` to `min(history_observed_through + 2 days, today)`. With the current dates, the earliest is 2024-01-02.
- `risk_level = elevated` when the probability is ≥ the threshold.
- The beach summary is the maximum station probability, labeled as such.
- Errors use the envelope `{"detail": {"code", "message"}}`.

**Naming:** spec 03's column names are canonical (`date`, `exceed_Enterococcus`, `prev_log10`, `roll3_log10`, `doy_sin`, …). No spec adds aliases for them.

**Reproducibility and safety**
- Every command is deterministic (`SEED = 42`, sorted outputs). Running it twice gives identical files.
- Each step writes a JSON report whose row counts reconcile (rows in − rows dropped = rows out).
- No secrets appear in code, images, logs or exports. `MLFLOW_TRACKING_URI` and `HF_TOKEN` come from environment variables.
- Data and model files never go into git, and CI checks this.

## Acceptance criteria
The project is done when every box below is checked. Each spec's own criteria must pass as well.
- [ ] From a clean clone with the raw CSV in `data/raw/`, running the 01 → 02 → 03 → 04 → 05 commands in order succeeds with exit code 0 each time.
- [ ] Each step's JSON report reconciles with the previous one (row counts and date ranges).
- [ ] Two full runs give identical parquet files, reports and metrics.
- [ ] `docs/decisions.md` records an answer for every open question below that blocks a spec, and every spec matches it.
- [ ] `reports/models/comparison.csv` shows the champion against all three baselines on validation and test. If no model is registered, the model card says so.
- [ ] The parity test passes: features from the API path, the exported history and `features.parquet` agree (floats within 1e-9, NA in the same places).
- [ ] The deployed Space passes `deploy.smoke_test`, including its 404 and 422 cases. Its `model_version`, `feature_version` and `history_snapshot_id` match the published revision.
- [ ] CI is green on `main` with all four checks: lint, tests, smoke and data-guard.
- [ ] No file under `data/` and no model file is tracked in git.

## Performance budget
These are targets for the implementation, not measured timings.

| step | current data | at scale (1M results / 5× rows) |
|---|---|---|
| 01 cleaning | < 30 s, < 2 GB | < 2 min, < 6 GB |
| 02 EDA (incl. figures) | < 60 s, < 2 GB | < 3 min, < 6 GB |
| 03 features | < 60 s, < 2 GB | < 3 min, < 6 GB |
| 05 models (full run with ablations) | < 15 min, < 4 GB; `--skip-ablations` < 5 min | < 45 min |
| 06 API | p95 < 1 s for a beach with ≤ 50 stations; `/health` < 100 ms; startup < 15 s, < 2 GB | — |
| 07 deploy | image < 1.5 GB; container start to healthy < 60 s; Space p95 < 3 s | history file ≤ 50 MB |
| 08 CI | whole workflow < 10 min | — |

## Out of scope
- External data: rainfall, tide, waves, river flow.
- Indicators other than Enterococcus as targets.
- Imputing censored values.
- Forecasts more than one day ahead.
- Scheduled data refresh or live ingestion.
- Drift monitoring and automatic retraining.
- Other model families, ensembles and SHAP beyond gain and coefficients.
- Authentication, rate limiting, a frontend, paid hosting and A/B tests.
- Official advisories or claims that a beach is safe to swim at.

## Open questions
These are the conflicts between specs, in order of urgency. Resolve each one in `docs/decisions.md` and then update the specs it names.

1. **Model handoff (05 ↔ 06 ↔ 07), which blocks deployment.**
   - Spec 06 loads the MLflow pyfunc, either from the registry or from a local MLflow model directory with `serving_metadata.json`.
   - Spec 07 still exports `bundle/pipeline.joblib` + `metadata.json`, bans MLflow from the image, uses `MODEL_BUNDLE_DIR`, and adds a `supports_no_history` key, which spec 06 forbids.
   - The proposed fix is to update spec 07 to match spec 06:
     - export the local MLflow model directory;
     - add `mlflow` (or `mlflow-skinny`) to `requirements-serve.txt`;
     - use spec 06's environment variables (`MODEL_URI`, `FEATURE_REPORT_PATH`, `HISTORY_METADATA_PATH`).
2. **Feature report missing from the Hub repo (06 ↔ 07):** spec 06 needs the exact `feature_report.json` whose SHA-256 matches the model's tag, but spec 07's Hub layout doesn't include it. The fix is to add it to the published commit and the manifest.
3. **Column names and rows in spec 04 (03 ↔ 04):** spec 04 still uses `sample_date`, `label`, `last_entero`, `roll_mean_3` and `month_sin`, and says "routine samples only." Spec 03 keeps all station-days, and its `prev_log10` is on a log10 scale. Spec 04 needs updating to spec 03's names, and its persistence rule should use `prev_exceed`. The team also needs to decide routine-only vs all samples.
4. **Who redeploys (07 ↔ 08):** spec 07 says spec 08 redeploys on merge, but spec 08 puts deployment out of scope. Either keep deployment manual and fix spec 07's wording, or add a deploy job (with the `HF_TOKEN` secret) to spec 08.
5. **CI smoke run time (05 ↔ 08):** spec 08's smoke job runs the spec 05 entry point within 10 minutes. It should call `--skip-ablations` (or a fixture-sized search). Spec 08 should also list exactly which spec 05 runs it expects.
6. **Label at exactly 104 (02 ↔ 03):** an `above_QL` value of exactly 104 counts as exceeding (≥), even when the original qualifier was `>` rather than `>=`. Keep this, or use the qualifier to mark those rows unknown? Specs 02 and 03 and their tests must change together.
7. **Test-set size (05):** spec 05 calls validation and test about 1–2k rows each. The EDA shows 2024 + 2025 ≈ 7.1k events in test, about 6.2k of them from the first four months of 2025, with a positive rate around 27%. Correct the size in spec 05 and keep the prevalence-shift question (spec 05 Q7).
8. **Refit and margins (05):** register the model trained on train only (current rule) or refit on train + validation? Are a 0.02 gain and a 0.01 tie margin right? Is precision ≥ 0.50 the right operating point? If 05 refits, spec 06's supported-date rule must change too.
9. **Static snapshot for the demo (06, 07):** without a refresh process, the app can't answer "today." Is a historical-only demo acceptable, and who records `history_observed_through`: the publisher by hand, or spec 01 at download time?
10. **Smaller open questions in each spec:**
    - **01:** `E` flag, extreme-value cap, qPCR rows, coordinates, separate stations file.
    - **02:** duplicates, other indicators, notebook vs script.
    - **03:** interval-aware worst result, history freshness, frozen-history evaluation.
    - **06:** beach summary rule, risk bands, stale marker.
    - **07:** repo visibility, data repo, cold starts.

    These stay in their own specs. Each one is resolved in `docs/decisions.md` before the spec that owns it is implemented.
