# California Beach Enterococcus Risk

[![CI](https://github.com/nancyvora/beach-hazard-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/nancyvora/beach-hazard-prediction/actions/workflows/ci.yml)

Runnable project-update scaffold implementing the core of specs 01–08. It cleans
lab results, builds station-day labels and leakage-safe history features, logs
baselines and a calibrated logistic model to local MLflow, exports a consistent
snapshot, and serves `/health` and `/predict` with FastAPI.

**The included demo generates synthetic observations. Its scores and plots check
the plumbing; they are not evidence about California beaches.** No real dataset,
model, credentials or generated reports belong in git. The badge reports the latest
GitHub Actions result; local verification is recorded separately. CI pins Ubuntu
24.04 and reads the exact Python version from `.python-version`, matching the
available GitHub Python installer for that OS rather than following `ubuntu-latest`.

## Quick start (Docker is optional)

Use Python **3.13.12** (`.python-version`). On this machine it is available as
`/opt/homebrew/opt/python@3.13/bin/python3.13`.

```bash
cd /Users/dongmeng/Desktop/DSC200/project/beach-hazard-prediction
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.demo --serve
```

The first run builds `.demo/` locally; subsequent runs reuse the verified export.
Use another `--root .demo/run2` for a fresh run. The demo uses a separate
`beach-exceedance-demo` model/alias and never promotes a production champion.

Open `http://127.0.0.1:7860/docs`, or use:

```bash
curl http://127.0.0.1:7860/health
curl 'http://127.0.0.1:7860/predict?beach=1&date=2025-04-30'
```

`demo: true` and the disclaimer identify synthetic results. The requested date is
historical because the snapshot is static. Unsampled days do not become negative
training labels. Without model artifacts the regular app still starts and returns
503 readiness, rather than fabricated probabilities:

```bash
uvicorn src.api.main:app --port 7860
```

## Real data pipeline

Official source: [Beach Water Quality Monitoring Results — Bacteria](https://data.ca.gov/@california-state-water-resources-control-board/beach-water-quality-postings-and-closures/r/7bd961cf-abe4-433b-8033-378161237ff3).
The full download is approximately 1.7 GB; these commands are explicit opt-ins:

```bash
# Optional API preview; first rows are not a representative sample.
python -m src.data.download --limit 1000
# Full resource, saved to data/raw/beach-monitoring-results.csv:
python -m src.data.download --full

python -m src.data.cleaning
python -m src.data.eda
python -m src.features.build
python -m src.models.train_baselines
python -m src.models.train_models --skip-ablations
```

Alternatively pass an existing CSV with `python -m src.data.cleaning --input /path/to/file.csv`.
All pipeline commands accept `--root PATH` for isolated inputs and outputs.
Train/validation/test dates are centralized in `src/models/config.py`; no random
split is used. Production registration requires the spec 05 validation gain over
the baselines. No qualifying model means no new `champion` and no fabricated API.
Both the gain comparison and threshold selection use the forward-calibration
validation rows; see `reports/models/selection_comparison.csv` and `selection.json`.
Rows identified as synthetic by the demo provenance marker require explicit demo
mode; they cannot silently enter production experiments or become a champion.

To run the four implemented feature ablations, omit `--skip-ablations`. The optional
LightGBM pipeline uses a small preliminary grid, **not** the full spec 05 search:

```bash
pip install -r requirements-optional.txt
python -m src.models.train_models --families logreg,lgbm --skip-ablations
```

LightGBM may also need OpenMP on macOS. It is not required by the runnable demo and
has not been verified in the initial local environment.

MLflow defaults to a local SQLite store at `artifacts/mlflow.db`; alternatively set
`MLFLOW_TRACKING_URI` explicitly. SQLite avoids MLflow 3.18's disabled-by-default
legacy filesystem store. Do not point tests or smoke runs at a team server.

## Serving and local Docker

Export a production champion locally after successful training. Supply the date
through which the source was processed; do not infer that date from the newest sample:

```bash
python -m deploy.publish_model --observed-through YYYY-MM-DD --dry-run
```

The default export directory is `artifacts/serving/`; use a new `--output` directory
instead of overwriting an existing snapshot. It contains an MLflow pyfunc directory,
version tags, the matching feature report, catalog and complete station-day history.
The exporter verifies batch/API compatibility before finishing. A manifest allows:

```bash
python -m deploy.fetch_artifacts --root .demo/artifacts/serving
docker build -t beach-risk:dev .
docker run --rm -p 7860:7860 \
  --mount type=bind,source="$(pwd)/.demo/artifacts/serving",target=/home/user/app/artifacts,readonly \
  beach-risk:dev
```

Use the same demo curl request above to check the container. It runs as UID 1000
and gets artifacts from a read-only mount; they are absent from the built image.
For a production snapshot, replace the mount source. Hub download/upload, hosted
Space deployment, rollback and the full spec 07 load test remain pending.

## Checks

```bash
ruff check . && ruff format --check .
pytest -q --cov=src
bash scripts/smoke.sh
python scripts/check_repo_files.py
```

The smoke script creates a fresh temporary synthetic run, checks MLflow baseline
and model runs and executes an in-process API request. Tests use temporary artifacts
and do not read the real `data/` directory. The data guard checks tracked Git blobs.
Use `python scripts/check_repo_files.py --include-untracked` before committing to
also check pending source files while respecting `.gitignore`.

See [implementation status and rubric mapping](docs/project_update.md) for completed
and pending scope. The eight implementation specs are **not** the rubric's required
maximum-four-page submission PDF. Team members must prepare that concise document
and write the personal reflection themselves.
