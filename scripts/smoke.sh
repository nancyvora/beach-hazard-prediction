#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/beach-smoke.XXXXXX")"
export MLFLOW_TRACKING_URI="sqlite:///$smoke_root/mlflow.db"
export MLFLOW_DISABLE_TELEMETRY=true
export MLFLOW_DISABLE_AGENT_HINT=1
python -m src.demo --root "$smoke_root/run" --skip-eda
python - "$smoke_root/run" <<'PY'
import sys
from pathlib import Path
import mlflow
from src.common import read_json
root = Path(sys.argv[1])
client = mlflow.MlflowClient()
experiment = client.get_experiment_by_name("beach-exceedance-demo")
for spec, minimum in [("04", 3), ("05", 2)]:
    runs = client.search_runs([experiment.experiment_id], f"tags.spec = '{spec}'")
    assert len(runs) >= minimum, (spec, len(runs))
assert read_json(root / "reports/demo_result.json")["prediction"]["demo"] is True
print("SMOKE PASSED: local tracking, clean/features/models/export/API; synthetic data only.")
PY
