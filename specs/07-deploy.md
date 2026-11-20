# Spec: 07 – Deploying the model and API as a Hugging Face Docker Space

Step 7 of the pipeline. It takes the `champion` model registered in spec 05, packages it into the artifacts the spec 06 API loads (model bundle, station catalog, history snapshot), publishes them to a Hugging Face Hub model repo, and runs the spec 06 FastAPI app as a Docker Space. Training and registering the model (specs 04–05), the endpoint behavior (spec 06) and CI (spec 08) are not this step.

## Goal and user
Our 4-person team, and the people who try the app (the grader, beachgoers). The team needs one repeatable path from "`champion` in the MLflow registry" to "public URL that answers `/predict`", and a way to roll back to an earlier model. Nobody should copy model files by hand or commit them to git.

## Inputs
- **Registered model:** `REGISTERED_MODEL_NAME` from `src/models/config.py` (`beach-exceedance-entero`), alias `champion`, with the version tags from spec 05 (`feature_columns`, `threshold`, `threshold_rule`, `limit_mpn`, `train_end`, `val_end`, `test_start`, `git_commit`, `features_report_sha256`). Read through `MLFLOW_TRACKING_URI`. Note that `beach-exceedance` is the *experiment* name, not the model name.
- **Spec 05 report:** `reports/models/model_card.md`, reused as the Hub model card.
- **Spec 03 outputs:** `data/processed/station_days.parquet`, `feature_report.json` and `features.parquet` (the last one only for dtypes and the parity test).
- **Spec 06 app:** `src.api.main:app`. It reads `MODEL_BUNDLE_DIR`, `STATION_CATALOG_PATH` and `HISTORY_PATH` at startup, and defines the endpoints, response schema, error codes and disclaimer. This spec does not change them.
- **Accounts:** a Hugging Face account (or org) that owns one model repo and one Docker Space.
- **Environment variables** (never hard-coded, never committed):

| variable | where it is set | used for |
|---|---|---|
| `MLFLOW_TRACKING_URI` | your shell | reading the registered model |
| `HF_TOKEN` (write access) | your shell, later a GitHub Actions secret | publishing the model repo and pushing the Space |
| `HF_MODEL_REPO` | Space variable and shell | model repo id, e.g. `<user>/beach-exceedance-model` |
| `HF_SPACE_REPO` | shell | Space repo id |
| `MODEL_REVISION` | Space variable | full commit hash of the model repo to download |
| `MODEL_BUNDLE_DIR`, `STATION_CATALOG_PATH`, `HISTORY_PATH` | set by the Dockerfile; overridden in local runs and tests | where the spec 06 app finds its artifacts |

- **Size now:** a small model bundle and a history file of a few MB. **Size later:** support history files up to 50 MB.

**Handoff assumptions (to confirm, see Open questions 1–2).** Spec 05 registers an MLflow `pyfunc` model, while spec 06 loads `pipeline.joblib` plus `metadata.json` and must run without MLflow. This spec bridges the two: `publish_model` converts the registered version into the spec 06 bundle. That needs the fitted preprocessing, model and calibrator from spec 05's wrapper to be available as plain objects. The file and key names below are proposals that spec 06's loader has the last word on.

## Outputs
1. **`Dockerfile`** (repo root), **`.dockerignore`** and **`requirements-serve.txt`**: build the API image (see Docker rules).
2. **`deploy/publish_model.py`**: exports the `champion` model, the station catalog and the history snapshot to the Hub model repo.
3. **`deploy/fetch_artifacts.py`**: runs at container start, downloads the pinned revision from the Hub and checks file hashes.
4. **`deploy/push_space.py`**: stages the Space files and uploads them to the Space repo.
5. **`deploy/space_README.md`**: the Space README with the YAML front matter Hugging Face reads.
6. **`deploy/smoke_test.py`**: checks a running URL (local container or Space).
7. **`docs/deployment.md`**: a one-page runbook (one-time setup, publish, deploy, roll back, where each secret lives).
8. **The model repo on the Hub**:

| path | meaning |
|---|---|
| `bundle/pipeline.joblib` | the fitted inference pipeline (preprocessing + model + calibrator), loadable without MLflow |
| `bundle/metadata.json` | model metadata required by spec 06 (see below) |
| `data/station_catalog.parquet` | one row per supported station |
| `data/station_history.parquet` | the history snapshot |
| `data/history_metadata.json` | `history_snapshot_id`, `history_observed_through`, row counts |
| `manifest.json` | provenance: registry version, MLflow run id, git commit, SHA-256 of every file above, validation and test metrics copied from the run, `created_at` |
| `README.md` | the spec 05 model card plus a short deployment section |

9. **`reports/deploy/smoke_case.json`** (local, not committed): a beach id and date that are valid for this snapshot, used by the smoke test.
10. A short printed summary of each command (repo ids, revision hashes, file hashes, snapshot id).

Commands:
```
python -m deploy.publish_model --alias champion --observed-through YYYY-MM-DD [--dry-run]
python -m deploy.push_space [--dry-run]
python -m deploy.smoke_test --url https://<user>-<space>.hf.space [--benchmark]
```

Note: nothing here trains, tunes, recalibrates or re-evaluates a model. Metrics in the manifest are copied from the registered run.

## Definitions and rules
**What gets deployed:** the registry version that has the alias `champion`. If the alias is missing (spec 05 registers nothing when no model beats the baselines), `publish_model` stops with a clear message. It never picks "the latest version".

**Bundle export**
- `pipeline.joblib` contains the complete fitted pipeline with `predict_proba` and a saved class mapping, as spec 06 requires. It must **not** depend on MLflow: unpickling it in an image without MLflow must work.
- Spec 06 owns the checks on the bundle (feature contract, model feature order, probability interface). Publishing does not repeat them but runs them before upload (see "Pre-upload verification").

**`bundle/metadata.json`** has the fields spec 06 lists as required. Where each one comes from:

| metadata field (proposed key) | source |
|---|---|
| `model_version` | `<REGISTERED_MODEL_NAME>-v<registry_version>` |
| `feature_version` | `spec03-<first 12 hex of features_report_sha256>` |
| `feature_names` (ordered) and `feature_dtypes` | the `feature_columns` tag, which must equal the list in `feature_report.json`; dtypes from `features.parquet` |
| `positive_class` | `1` |
| `target_definition` | `limit_mpn`, accepted units (`MPN/100mL`, `CFU/100mL`) and the target-rule version from `feature_report.json` |
| `history_delay_days` | `feature_report.json` (2) |
| `calibration` | the `calibration` param of the registered run |
| `risk_threshold` and `threshold_rule` | the `threshold` and `threshold_rule` tags. The threshold is copied, never recomputed. |
| `latest_label_date` | the `val_end` tag. Validation is used for tuning, calibration and the threshold, so it is the latest label date used to fit or select the model. |
| `supports_no_history` | `true` only if the pipeline returns finite probabilities for a row with `no_history = 1`, checked at publish time, not assumed |
| `registry_version`, `mlflow_run_id`, `train_end`, `val_end`, `test_start`, `git_commit`, `library_versions` | tags and run info |

**Station catalog** (`data/station_catalog.parquet`): built from `station_days.parquet`, one row per `station_id`, columns exactly `station_id`, `station_name`, `beach_id`, `beach_name`, `county`, `lat`, `lon` (`beach_id` is the source's `BeachName_id`). If a station has more than one distinct beach, name or coordinates, `publish_model` stops and lists them. Following spec 03, conflicts are reported for review and not resolved by taking the latest record.

**History snapshot** (`data/station_history.parquet`):
- Built from `station_days.parquet`, **all dates, not only recent ones**, because `station_hist_rate` and the counts in spec 03 are expanding over a station's whole past.
- Columns only: `station_id`, `date`, `value`, `censored`, `exceed_Enterococcus`. Rows with an NA target are **kept**, since they still feed concentration-history features.
- Sorted by `station_id`, `date`, with unique `station_id` × `date`.
- `history_observed_through` is the date through which the source data was processed. It cannot be derived from the data (a station may simply not have been sampled recently), so `--observed-through` is **required**. It must not be earlier than the latest `date` in the history, and not later than today's California date.
- `history_snapshot_id` = `<observed_through>-<first 8 hex of the history file's SHA-256>`.

**Supported dates:** following spec 06, a request is valid only if its history cutoff (`date − 2 days`) is on or after `latest_label_date` and on or before `history_observed_through`. If no date satisfies both, `publish_model` stops: there would be nothing the app could serve. The app enforces this per request; this step only checks that the snapshot leaves a non-empty range.

**Smoke case:** the beach with the most station-days in the history, and `date = history_observed_through`. Written to `reports/deploy/smoke_case.json`.

**Pre-upload verification:** before anything is uploaded, `publish_model` loads the staged artifacts through the spec 06 loader and sends the smoke case to the app in-process. It must return HTTP 200. A bundle the API would reject (feature mismatch, bad probabilities, no supported dates) is never published.

**Hub publishing**
- The model repo is separate from the GitHub repo. Model and data artifacts are never committed to git.
- Bundle, catalog and history go in **one commit**, so a model version always comes with the history snapshot it was built for. `publish_model` prints that commit hash. That full hash is what `MODEL_REVISION` is set to.
- `--dry-run` writes the staged folder locally, runs the verification, and uploads nothing.
- The model card is `reports/models/model_card.md` from spec 05, followed by a deployment section (snapshot id, `history_observed_through`, supported date range, the disclaimer text from spec 06).

**Startup (`deploy/fetch_artifacts.py`)**
- Runs before the server starts, only when `HF_MODEL_REPO` is set.
- `MODEL_REVISION` is **required** and must be a full commit hash. There is no fallback to `main`, because spec 06 forbids a moving "latest" model. The script exits non-zero if it is missing.
- Downloads that revision, checks every file against the SHA-256 in `manifest.json`, and exits non-zero (with the name of the file) on a mismatch or a missing file. It never starts the server with partial artifacts.
- Files go to the paths in `MODEL_BUNDLE_DIR`, `STATION_CATALOG_PATH` and `HISTORY_PATH`.
- If the app is started directly with missing or incompatible artifacts, spec 06 applies: `/health` and `/predict` return 503. This step adds nothing there and substitutes no dummy model.

**Docker rules**
- The Python version matches `CLAUDE.md`. Pin the base image to an exact tag (no `latest`).
- Run as a non-root user with UID 1000 and use `COPY --chown=user`, which Hugging Face Spaces expect.
- Install from `requirements-serve.txt` only: the web framework, `huggingface_hub`, `pyarrow`, `joblib`, `pandas`, `numpy`, `scikit-learn` and `lightgbm`. **No MLflow**, no plotting or notebook packages. `scikit-learn`, `lightgbm`, `numpy` and `pandas` are pinned to the versions in `library_versions`, so the pipeline loads as it was saved.
- Copy `requirements-serve.txt` and install before copying `src/`, so rebuilds reuse the cache. Copy `src/` (the app, the spec 03 feature builder and any module the pipeline needs to load) and `deploy/fetch_artifacts.py` only.
- The image contains **no data, bundle files, `.env` or tokens**. Set the three artifact path variables with defaults under `/home/user/app/artifacts`.
- Expose port **7860**, bind `0.0.0.0`, one worker, no `--reload`. The start command runs `python -m deploy.fetch_artifacts` and then
  `uvicorn src.api.main:app --host 0.0.0.0 --port 7860`.

**Space configuration**
- `deploy/space_README.md` front matter includes `sdk: docker` and `app_port: 7860`, the same port as the Dockerfile.
- Free CPU hardware. Free Spaces sleep after about 48 hours without requests, so the first request after a sleep is slow, and the smoke test waits for it. Space storage is wiped on restart; artifacts are downloaded again each start.
- `HF_MODEL_REPO` and `MODEL_REVISION` are Space **variables**. The Space gets **no write token**. If the model repo is private, add a separate read-only token as a Space **secret**.

**Secrets:** tokens are read from environment variables only. They never appear in the Dockerfile, the image layers, any exported file, the model card or printed output. Scripts print repo ids and hashes, never token values.

**Smoke test** (`deploy/smoke_test.py`), using the spec 06 interface and the smoke case:
- `/health` returns 200 with `status="ok"`.
- `GET /predict?beach=<id>&date=<date>` returns 200. `beach_id` and `date` echo the request, `max_station_probability` is finite and in [0, 1], `risk_level` is `elevated` or `lower`, `stations` is not empty, `coverage` is present, and `disclaimer` is non-empty.
- `model_version`, `feature_version` and `history_snapshot_id` in that response equal the ones in `bundle/metadata.json` and `data/history_metadata.json` for the deployed revision.
- An unknown beach id returns 404 `beach_not_found`; an invalid date returns 422 `invalid_request`; a date before the supported range returns 422 `unsupported_date`. Error bodies have the `{"detail": {"code": ..., "message": ...}}` form.
- Waits up to 5 minutes for the first healthy response, to allow for a Space waking from sleep.
- With `--benchmark`: 20 sequential `/predict` calls for the smoke beach, then 10 parallel calls repeated 5 times. It reports p50 and p95 latency and the number of 5xx responses. This is the load and resource check that spec 06 leaves to this step.

**Rollback:** set `MODEL_REVISION` to an earlier commit hash and restart the Space. The runbook lists the steps.

## Acceptance criteria
- [ ] `docker build` succeeds from a clean checkout with no `data/`, `reports/` or artifact files present.
- [ ] The image runs as a non-root user, and contains no data folder, `*.parquet`, `*.joblib`, `.env` or MLflow package.
- [ ] No token value appears in the image history, the image's files, the exported files, the model card or printed output (checked by searching for the `HF_TOKEN` value and the `hf_` prefix).
- [ ] The Dockerfile's exposed and listening port and the `app_port` in `deploy/space_README.md` are all 7860.
- [ ] A container started with the three path variables pointing at fixture artifacts and no `HF_MODEL_REPO` reports `/health` 200 within 60 s. Started with no artifacts, `/health` and `/predict` return 503 and no probability is returned.
- [ ] `fetch_artifacts` exits non-zero when `MODEL_REVISION` is missing or is not a full commit hash, when a file is missing, and when a file's hash differs from `manifest.json`.
- [ ] `publish_model --dry-run` against a temporary local MLflow registry creates a staged folder with exactly the files in the Hub layout table (except the model card's source being `model_card.md`), runs the verification, and uploads nothing.
- [ ] `publish_model` stops with a clear message when: there is no `champion` alias; `--observed-through` is missing, earlier than the latest history date, or in the future; no supported date remains (`observed_through − 2 days` is before `latest_label_date`); the catalog has conflicting stations; or the pre-upload verification fails.
- [ ] `metadata.json` has every field in the mapping table. `feature_names` equals both the `feature_columns` tag and the list in `feature_report.json`. `risk_threshold` equals the `threshold` tag, `latest_label_date` equals `val_end`, and `supports_no_history` matches a real check on a `no_history = 1` row.
- [ ] Versions of `scikit-learn`, `lightgbm`, `numpy` and `pandas` in `requirements-serve.txt` equal those in `library_versions`.
- [ ] The station catalog has exactly the listed columns and a unique `station_id`, and its stations equal the stations in the history.
- [ ] The history snapshot has exactly the listed columns, is sorted, has unique `station_id` × `date`, has the same row count and the same number of NA-target rows as `station_days.parquet`, and is under 10 MB on today's data.
- [ ] **Parity test:** for 20 randomly chosen station-days (fixed seed), the spec 03 feature function run on the exported history and catalog gives the same values as the matching rows of `features.parquet` (equal integers, floats within 1e-9, NA in the same places).
- [ ] The smoke case's beach and date return HTTP 200 when sent to the app in-process during publishing.
- [ ] After deploying, `python -m deploy.smoke_test` passes against the Space URL, including the 404 and both 422 cases and the version and snapshot checks.
- [ ] `--benchmark` against the local container shows p95 under 1 s for `/predict` and `/health` under 100 ms (spec 06's budget). Against the Space it reports p95 under 3 s (network included) and zero 5xx responses.
- [ ] The Space answers after a restart, and after setting `MODEL_REVISION` to an earlier commit it serves that version (checked by hand once and noted in `docs/deployment.md`).
- [ ] pytest tests in `tests/test_deploy.py` cover the staging, metadata mapping, catalog, history snapshot, hashes, supported-date check, `fetch_artifacts` failures and the parity test, using small fixtures and no network. Docker and Hub checks are a manual checklist in `docs/deployment.md`, not part of the default pytest run.

## Performance budget
- Image size **under 1.5 GB**; a cold build in **under 10 min**.
- From container start to `/health` 200, including the download: **under 60 s**. Spec 06's own budget (artifacts load in under 15 s, under 2 GB peak memory per worker) applies inside that, and is checked with `docker stats` on the local container.
- Latency targets are in the smoke test's `--benchmark` criteria above.
- `publish_model` and `push_space` finish in under 5 min on a normal connection.
- These are acceptance targets, not timings already measured.

## Out of scope
- Training, evaluating, calibrating or registering models, and choosing the risk threshold (specs 04–05).
- The endpoints, response schema, error codes, validation of the feature contract and the disclaimer wording (spec 06). If the bundle or metadata layout needs to change, spec 06 and spec 05 change first.
- GitHub Actions: tests on pull requests and redeploying on merge (spec 08). The commands here must work from a command line so spec 08 can call them.
- Refreshing the history from data.ca.gov on a schedule, live ingestion, or re-running the cleaning and feature steps on the Space.
- Hosting the MLflow server (it lives outside the Space, because Space storage is wiped on restart).
- Authentication, rate limiting, custom domains, monitoring and drift dashboards.
- Paid hardware, GPUs, multiple models or A/B tests.

## Open questions
1. **Spec 05 → 06 handoff:** spec 05 specifies an MLflow `pyfunc` model, while spec 06 expects `pipeline.joblib` (with `predict_proba` and a class mapping) that loads without MLflow. This spec exports the bundle from the registered version, which needs spec 05's wrapper to expose its fitted preprocessing, model and calibrator as plain objects. Does spec 05 add an `export_bundle` function (current assumption), or should the pyfunc wrapper be dropped for a plain pipeline?
2. **Names:** confirm the file names (`pipeline.joblib`, `metadata.json`, `history_metadata.json`) and metadata keys with spec 06's loader. Where they differ, spec 06 wins and this spec follows.
3. **`--observed-through`:** supplied by hand when publishing (current rule), or recorded by spec 01 when the raw file is downloaded so it can be read automatically?
4. **Catalog conflicts:** stop on any station with more than one beach or location (current rule), or warn and allow a documented choice?
5. **Model repo visibility:** public (current rule, since the data is public and the Space then needs no token), or private with a read-only Space secret?
6. **Where the data lives:** inside the model repo so the model and history are versioned together (current rule), or in a separate Hub dataset repo?
7. **Static snapshot:** with no refresh, the app can only answer historical dates from `latest_label_date + 2 days` to `history_observed_through + 2 days`, and not "today". Is that acceptable for the final demo, or do we need a refresh process before submission? (Spec 06, open questions 5–6.)
8. **Pushing to the Space:** upload with the `huggingface_hub` library from `push_space.py` (current rule), or `git push` to the Space's remote?
9. **Cold starts:** accept the sleep and slow first request on the free tier (current rule), or add a keep-alive ping?
10. **No champion:** if spec 05 registers nothing, or registers a baseline as `champion` (its open question 5), deployment stops (current rule). Should a baseline be deployable, and through which bundle format?
