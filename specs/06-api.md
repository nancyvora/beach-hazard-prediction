# Spec: 06 – API for beach Enterococcus risk predictions

Step 6 of the pipeline. It serves the selected model through FastAPI, using the same feature calculations as step 03. Users request a beach and date and receive station predictions, a beach risk summary and the age of the available history. Model training belongs to steps 04–05; Docker and hosting belong to step 07.

## Goal and user

Beachgoers want to look up predicted Enterococcus exceedance risk before visiting a beach. County health officials are a secondary audience. The API returns an informational estimate, not a measured concentration, an official advisory or a guarantee that the water is safe.

Our team also needs a stable interface that can be tested locally and deployed without retraining a model or running the full data pipeline on each request.

## Inputs

- **Registered model from step 05:** `beach-exceedance-entero`, selected through its `champion` alias. It is an `mlflow.pyfunc` model bundling preprocessing, the classifier and the calibrator; `predict(df)` returns one calibrated exceedance probability per row. Resolve the alias once at startup and load the resulting immutable model version.
- **Model-version tags from step 05:** `feature_columns` (JSON), `threshold`, `threshold_rule`, `limit_mpn`, `train_end`, `val_end`, `test_start`, `git_commit` and `features_report_sha256`. The model's logged signature supplies the expected input schema. Also load the exact training-time `feature_report.json` whose file hash matches the tag, not a later regenerated report.
- **Station catalog:** one row per supported `station_id`, containing `station_name`, `beach_id`, `beach_name`, `county`, `lat` and `lon`. It may be exported from step 03's station metadata. Beach IDs use the source's `BeachName_id`, not names or row numbers.
- **History:** step 03's `data/processed/station_days.parquet`, or a serving export with the same history fields and semantics. It must retain unknown-target events because they can contribute concentration-history features.
- **Feature contract:** the feature builder and column definitions from spec 03, including the 2-calendar-day availability delay. Use `date`, `exceed_Enterococcus` and the named features in that spec as the canonical interface.
- **Configuration:** `MODEL_URI` (default `models:/beach-exceedance-entero@champion`), `MLFLOW_TRACKING_URI`, `FEATURE_REPORT_PATH`, `STATION_CATALOG_PATH`, `HISTORY_PATH` and `HISTORY_METADATA_PATH`, read at application startup. An explicit numeric-version URI may be used for reproducible serving. `HISTORY_METADATA_PATH` defaults to `history_metadata.json` beside the history parquet. Credentials come from environment variables; data and model artifacts stay outside git.

Spec 05 defines the model contract above. Spec 07 assigns serving export to `deploy/publish_model.py`, but its current draft assumes an older joblib-based API contract; its packaging/dependencies must be updated to the pyfunc contract here before deployment. Spec 07 explicitly defers artifact-interface choices to spec 06. Spec 05 also adopts spec 03's column names and lists the changes needed in spec 04. Use the canonical spec 03 names rather than adding API-only aliases for `sample_date`, `label` or `last_entero`.

## Outputs

1. **FastAPI application:** `src/api/main.py`, exposing `app`, with request/response schemas in `src/api/schemas.py` and artifact loading/prediction orchestration in `src/api/service.py`.
2. **Endpoints:** `GET /predict?beach=<beach_id>&date=YYYY-MM-DD` and `GET /health`. FastAPI's generated `/docs` documents the query parameters, response schemas and errors.
3. **Tests:** `tests/test_api.py`, using a tiny registered MLflow pyfunc fixture in a temporary local tracking/registry directory, station catalog and history. Tests use MLflow but do not require the team's tracking server, Hugging Face credentials, network access or the real dataset.

Local command: `uvicorn src.api.main:app --host 0.0.0.0 --port 7860`.

### Prediction request and response

Both parameters are required. `beach` is a positive integer ID; `date` is a strict ISO calendar date. Dates follow California local calendar time (`America/Los_Angeles`), consistent with spec 03's date-level calculations. A beach name alone is not accepted because names may be ambiguous.

A successful request returns HTTP 200 with the following schema. The values below are illustrative, not results from a deployed model:

```json
{
  "beach_id": 24,
  "beach_name": "Santa Monica State Beach",
  "date": "2026-11-19",
  "max_station_probability": 0.42,
  "risk_level": "elevated",
  "risk_threshold": 0.4,
  "aggregation": "max_station_probability",
  "coverage": {
    "n_stations": 2,
    "n_predicted": 2
  },
  "stations": [
    {
      "station_id": 450,
      "station_name": "SMB-3-3",
      "status": "predicted",
      "probability": 0.42,
      "risk_level": "elevated",
      "latest_history_date": "2026-11-16",
      "days_since_prev": 3,
      "n_prev_samples": 240,
      "history_status": "available"
    },
    {
      "station_id": 452,
      "station_name": "SMB-3-5",
      "status": "predicted",
      "probability": 0.12,
      "risk_level": "lower",
      "latest_history_date": null,
      "days_since_prev": null,
      "n_prev_samples": 0,
      "history_status": "none"
    }
  ],
  "model_name": "beach-exceedance-entero",
  "model_version": "7",
  "feature_version": "sha256:example-feature-report-hash",
  "threshold_rule": "example: validation precision >= 0.50",
  "history_snapshot_id": "example-2026-11-19",
  "history_observed_through": "2026-11-17",
  "history_cutoff": "2026-11-17",
  "disclaimer": "Predicted Enterococcus exceedance risk, not an official advisory or a guarantee of safe swimming. Follow local health advisories."
}
```

**Probability and beach summary**

- Each station's `probability` is the corresponding scalar returned by the registered pyfunc's `predict(feature_df)`, already calibrated for `exceed_Enterococcus == 1`. Normalize a one-dimensional array/Series or a single-column frame into one value per input row, preserving row order. Require exactly one finite probability in [0,1] per station. Do not call the wrapper's `predict_proba`, select a class column or apply another calibrator.
- `risk_level` is `elevated` when the probability is **greater than or equal to** the registered version's `threshold` tag, otherwise `lower`. Spec 05 chooses the lowest calibrated-validation threshold reaching precision 0.50, or an F1-maximizing fallback when none does. Expose the saved `threshold_rule`; do not recompute thresholds, guarantee that future precision will be 0.50, or call the lower band “safe.”
- Run the model for each station in the requested beach's catalog. The beach summary is the maximum of the available station probabilities; its risk band uses the same threshold.
- A maximum station probability is **not** the probability that any location on the whole beach exceeds. Use the explicit `max_station_probability` field rather than presenting it as an independently calibrated beach probability.
- Include every catalog station in the response, sorted by `station_id`. Spec 05 requires valid probabilities even for no-history rows and unseen counties, so do not omit such stations. Successful responses have `coverage.n_predicted == coverage.n_stations`. This only describes the catalog's monitoring locations, not every location on the beach.

### Health response

`GET /health` is a readiness check. Return HTTP 200 only after the pinned pyfunc model, its required tags, matching feature report, catalog and history have loaded and passed compatibility checks. Return `status="ok"`, model name and numeric version, feature-report hash, threshold/rule, history snapshot ID, history coverage date and the supported prediction-date range. Do not expose local filesystem paths, tracking-server credentials or secrets.

If required artifacts are absent or incompatible, or their supported date ranges do not overlap, keep prediction unavailable and return HTTP 503 with `status="not_ready"` and a stable reason code. A loaded historical snapshot may be ready for supported historical requests even when it cannot serve today's date; expose that date range rather than implying live coverage. Calculate the date bounds using today's California date at request time so they remain correct across midnight without reloading artifacts.

## Definitions and rules

**Startup and MLflow model contract**

- In registry mode, honor the configured `MODEL_URI`: resolve its alias to numeric version `N`, or use the explicitly configured numeric version directly. For the default, resolve `models:/beach-exceedance-entero@champion`, read version `N`'s tags, and load `mlflow.pyfunc.load_model("models:/beach-exceedance-entero/N")`. Keep the resolved version and tags together. If the alias changes later, existing workers keep their pinned version until restarted; they must not report new tags for an old loaded model.
- Load model, catalog and history once per worker and keep them read-only. Startup may access the configured MLflow server; requests must not. Do not retrain, refit preprocessing, contact the raw Data API or reload parquet/model files inside each request.
- Consume the complete registered pyfunc model, including preprocessing and the sigmoid calibrator selected in spec 05. Do not assume a separate `pipeline.joblib`, recreate encoders or load the uncalibrated classifier from a tuning run.
- Validate all nine required version tags listed under Inputs. Parse `feature_columns` as an ordered unique list and `threshold` as a finite number in [0,1]; require a nonempty `threshold_rule` and verify `limit_mpn` against the target in the feature report (104 under the current specs). Validate split dates and the model's signature against the ordered features and expected types.
- Verify the exact feature-report file's SHA-256 against `features_report_sha256`. Format `feature_version` as `sha256:<features_report_sha256>` in responses, and verify its target rules, transform and 2-day history delay against the spec 03 builder. `git_commit` is source provenance, not a requirement that the API's later repository commit equal the training commit.
- If spec 05 registers no model because the validation gain requirement is not met and the requested alias/version does not exist, readiness fails with 503. Do not train a model, invent a champion or silently serve a baseline. An existing explicitly selected version is identified by its actual version and dates.
- For offline hosting in spec 07, `MODEL_URI` may instead point to a local MLflow model directory containing `MLmodel`. Its `serving_metadata.json` contains `model_name`, numeric `model_version` and a `tags` object copying all nine registered-version tags. Load it with the same pyfunc loader, verify the same signature/report contract, and perform no registry access. It must reproduce the registered version's predictions. This sidecar records registry provenance; it is not a different joblib-based model. Export/download implementation belongs to spec 07.
- The JSON at `HISTORY_METADATA_PATH` contains `history_snapshot_id` and ISO `history_observed_through`, associated with the shipped history parquet. The latter is the date through which source observations have been processed, not automatically the maximum sample date: a station may simply not have been sampled recently. Reject history records after that declared coverage date, and deploy history, catalog and metadata as one consistent snapshot.
- Validate station-day uniqueness, a nonempty catalog with one beach mapping per station, known/finite location metadata, the pyfunc probability interface and model feature order. Reject missing tags, a feature-report hash mismatch or an incompatible signature; do not silently rename fields or substitute another model. Every catalog beach must have at least one station.
- For the first version, retain the complete compact Enterococcus station-day history needed for the supported dates. Shipping only the last three observations would make the expanding `station_hist_rate` and counts incorrect. A smaller export may be added later only with cumulative state and verified batch/serving equivalence.

**Feature construction and supported dates**

- Resolve the beach to its registered stations, then call the reusable spec 03 feature builder with history, query station/date pairs and known station metadata. A prediction does not require a sample or label on the requested date.
- Enforce `history_cutoff = requested_date - 2 calendar days`. Every history contribution, including expanding rates and counts, must use only same-station events on or before that cutoff. Current-day and previous-day measurements remain ineligible even if present in the snapshot.
- Build the DataFrame in the exact order of the version's `feature_columns` tag, verified against the matching feature report and logged signature. Never send the target, current lab value, qualifier, method, sample type, IDs, names or audit columns to the model as incidental columns.
- Do not impute, scale or fit encoders in API code. The pyfunc owns train-fitted imputation/scaling and county handling: spec 05 uses unknown-safe one-hot encoding for logistic regression, or train-fixed categorical levels with unseen counties mapped to NaN for LightGBM.
- Accept only dates covered by the history export and after the labels used to fit/select/calibrate the deployed model could have become available. Spec 05 registers the train-only classifier plus the validation-fit calibrator and validation-selected threshold; it does not refit on train + validation. Require `history_cutoff >= max(train_end, val_end)` using the version tags as conservative bounds. Under the current dates, the earliest eligible request is 2024-01-02. Test labels must never affect this date or the threshold.
- Historical requests before that lower bound must not apply the deployed model to its own training/selection period and describe the result as a valid past prediction. If step 05 later changes its refitting or split policy, update the eligibility bounds with that contract.
- Require `history_cutoff <= history_observed_through`. A stale source snapshot does not support today's nowcast merely because calendar features can be calculated. Reject unsupported dates with a clear error and supported date range.
- With complete compact history, the inclusive request range is `supported_start = max(train_end, val_end) + 2 calendar days` through `supported_end = min(history_observed_through + 2 calendar days, today_in_California)`. If start exceeds end, readiness is 503 with reason `no_supported_dates`. No-history support still applies to individual stations within this overall range.
- Future dates beyond today's California date are outside this version's scope. Historical responses are retrospective estimates under the team's 2-day availability assumption; actual publication-time availability is not established by sample dates alone.

**Sparse history and prediction failures**

- Pass the spec 03 no-history counts, flags and missing lags through unchanged. The spec 05 model must still return a finite probability for these rows and for a county unseen in train. Return the prediction with `history_status="none"`, null history date/age and zero counts; do not invent a new `supports_no_history` tag, drop the station or substitute an API-computed global prevalence.
- For stations with eligible history, mark it `stale` when `days_since_prev > 30`; otherwise use `available`. Stations with no eligible history retain `none`. The 30-day marker is a proposed freshness indicator, not a safety threshold or a change to spec 03's history window. Keep eligible older observations and show their age rather than quietly treating them as recent.
- Batch-score all catalog stations for a valid beach/date, including no-history stations. Missing catalog metadata is an artifact validation problem, not a reason to quietly score only a subset. Never replace missing probabilities with zero.
- An invalid pyfunc output (NaN, infinity, out-of-range probability, wrong shape or wrong row count) is an internal prediction failure. Return HTTP 503 without a fabricated probability or partial result that hides the failure.
- Serialize missing history dates and ages as JSON `null`, never NaN or Infinity. Prediction requests do not mutate the model, history or catalog.

**Error contract**

| HTTP status | code | condition |
|---|---|---|
| 422 | `invalid_request` | missing/invalid query parameters, nonpositive beach ID, invalid ISO date, future date or unexpected query parameters |
| 404 | `beach_not_found` | well-formed beach ID is not in the supported catalog |
| 422 | `unsupported_date` | date is outside the model/history-supported range |
| 503 | `service_not_ready` | required artifacts are missing/incompatible, or no supported request dates exist |
| 503 | `prediction_failed` | loaded model or feature inference fails or returns invalid probabilities |

Prediction errors use `{"detail": {"code": "...", "message": "..."}}`; date-range errors also include supported bounds. Parameter validation must use the same documented envelope. Log internal exception details server-side rather than returning tracebacks. This public read-only API accepts beach/date parameters, not arbitrary model paths, feature values or uploaded files.

## Acceptance criteria

- [ ] The local command starts the app; `/docs` lists both endpoints and their schemas. A temporary registered pyfunc with a `champion` alias gives `/health` HTTP 200 with its resolved numeric version, threshold/rule and supported dates.
- [ ] An absent champion, missing required tags, missing artifacts, incompatible signature or feature-report hash mismatch gives `/health` and `/predict` HTTP 503; no baseline or dummy probability is substituted.
- [ ] Changing the registry alias after startup leaves the running worker's loaded model, threshold and reported numeric version unchanged. Model loading and registry access happen at startup, not on each prediction request.
- [ ] An explicit numeric-version URI loads that version rather than resolving champion. A local exported MLflow model plus `serving_metadata.json` returns the same probabilities and provenance without any registry/network call; missing sidecars or history metadata make readiness fail.
- [ ] A valid beach/date request returns HTTP 200, the requested identifiers/date, station outputs, coverage, model/history provenance and the disclaimer. Every supplied probability is finite and within [0,1].
- [ ] Two station fixture predictions of 0.2 and 0.8 give `max_station_probability=0.8`; the summary is not their mean, sum or `1-(1-p1)*(1-p2)`. Exactly-at-threshold predictions use the elevated band.
- [ ] Missing/invalid inputs, ambiguous name input, invalid leap dates, future dates and extra parameters return the documented 422 envelope. Unknown numeric beach IDs return 404.
- [ ] With `val_end=2023-12-31` and an earlier `train_end`, 2024-01-01 is unsupported and 2024-01-02 is eligible if history coverage allows it. Dates after history coverage are also unsupported. Tests freeze the California current date.
- [ ] Supported date bounds follow the formulas above, include both endpoints, and advance correctly across California midnight. A snapshot with no overlap between model eligibility and coverage gives `no_supported_dates` readiness failure rather than an impossible HTTP 200 date range.
- [ ] With fixture model dates permitting the request, Jan 7 predictions can use Jan 5 history but not Jan 6 or Jan 7. Changing those ineligible values or adding later observations leaves the Jan 7 feature row and probabilities unchanged.
- [ ] The same station/date and eligible history produce the same feature values through the API and spec 03's batch path. A query with no current-day sample or target still works.
- [ ] History with more than three earlier events gives the same expanding station rate and counts as batch computation, not statistics calculated only from the last three events.
- [ ] No-history stations and unseen counties return finite probabilities, as required by spec 05. Every catalog station is scored and coverage counts agree; no-history rows retain null dates/ages and `history_status="none"`. History older than 30 days is visibly marked stale.
- [ ] The service calls the pyfunc's `predict` once per beach batch, preserves station order, and returns its calibrated values without another calibration or a class-column assumption. NaN, out-of-range, multi-column or incorrectly sized outputs return 503; successful JSON contains no NaN/Infinity values.
- [ ] Fixture API probabilities match the same registered pyfunc's direct predictions to 1e-9 on the same feature rows. The pyfunc receives only the ordered feature whitelist, with no target/current-day lab fields. No fit operation is called during startup or requests.
- [ ] Repeated requests reuse loaded artifacts, produce the same prediction payload for the same inputs/snapshot, and do not change history counts. Tests run without network access or production artifacts.

## Performance budget

- Once artifacts are loaded, a single-beach request with up to 50 catalog stations completes in **under 1 s at the 95th percentile** in a sequential local smoke benchmark. `/health` completes in under 100 ms.
- Startup with the current compact history should complete in under 15 s with peak memory under 2 GB per worker. Use a station-indexed history lookup and batch station inference; do not scan all lab results per request.
- These are proposed acceptance targets, not measured timings. Hosting resource limits and concurrent load are checked in spec 07.

## Out of scope

- Training/selecting models, choosing operating thresholds, fitting calibration or preprocessing, and changing upstream target/feature definitions.
- Docker/Hugging Face setup, CI redeployment, model download credentials and hosting configuration (specs 07–08).
- Scheduled history refresh, live ingestion, database writes, user accounts or a frontend.
- Multi-day future forecasts, predictions outside the snapshot's supported dates, or claims of official beach safety/closure status.
- A separately trained beach-level probability model, arbitrary feature-input prediction endpoints, batch uploads or exposing raw lab records.

## Open questions

1. **Spec 07 export alignment:** `deploy/publish_model.py` owns export. Its current draft must replace the old `pipeline.joblib`/`predict_proba` assumptions, metadata/version formats and no-MLflow serving dependency rule with this spec's local MLflow model directory, `serving_metadata.json`, matching feature report and path configuration. Its smoke-case date must be chosen within this spec's supported bounds. Coordinate those changes with the spec 07 owner; do not add a second inference path that bypasses the registered pyfunc.
2. **Upstream alignment:** spec 05 adopts spec 03's names, but spec 04 still needs that update and a routine-versus-all-samples decision. Resolve those upstream without API-only renaming or filtering.
3. **Beach summary:** use the maximum station probability (current proposal), or choose another validated station-to-beach rule? A true beach-level event probability needs a separately defined target and evaluation.
4. **Risk bands:** use lower/elevated with spec 05's saved threshold (current proposal), or add a third band? This spec does not reopen the precision-0.50/F1 threshold-selection rule.
5. **Freshness:** keep a 30-day stale-history marker (current proposal), or add stricter limits after validation? The no-history prediction behavior is already required by spec 05 and must not be changed only in the API.
6. **Snapshot coverage and historical access:** who produces the coverage watermark and refresh process? Retain supported retrospective requests (current proposal), or limit the public endpoint to today once fresh snapshots are reliably available?
