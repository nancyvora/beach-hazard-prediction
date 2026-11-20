"""Export a pinned MLflow pyfunc snapshot locally. Hub upload is intentionally not implemented yet."""

import argparse
import shutil
import tempfile
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd

from src.api.service import PredictionService, Settings
from src.common import Paths, california_today, sha256, write_json
from src.data.events import TARGET, station_catalog
from src.models.config import REGISTERED_MODEL_NAME
from src.models.tracking import configure_tracking


def export_snapshot(paths, name, version, observed_through, output=None):
    output = Path(output or paths.artifacts / "serving")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty: {output}; choose a new directory")
    events = pd.read_parquet(paths.processed / "station_days.parquet")
    observed = date.fromisoformat(str(observed_through))
    if observed > california_today() or (events.date.dt.date > observed).any():
        raise ValueError("Observed-through must cover all source rows and not be in the future")
    client = mlflow.MlflowClient()
    registered = client.get_model_version(name, str(version))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        downloaded = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{name}/{version}", dst_path=temporary
        )
        shutil.copytree(downloaded, output / "model")
    metadata = {"model_name": name, "model_version": str(version), "tags": dict(registered.tags)}
    write_json(output / "model/serving_metadata.json", metadata)
    shutil.copy2(paths.processed / "feature_report.json", output / "feature_report.json")
    station_catalog(events).to_parquet(output / "station_catalog.parquet", index=False)
    history_file = output / "station_history.parquet"
    events[["station_id", "date", "value", "censored", TARGET]].to_parquet(
        history_file, index=False
    )
    snapshot_id = f"{observed.isoformat()}-{sha256(history_file)[:8]}"
    write_json(
        output / "history_metadata.json",
        {
            "history_snapshot_id": snapshot_id,
            "history_observed_through": observed,
            "n_rows": len(events),
            "n_unknown": int(events[TARGET].isna().sum()),
        },
    )
    settings = Settings(
        str(output / "model"),
        output / "feature_report.json",
        output / "station_catalog.parquet",
        history_file,
        output / "history_metadata.json",
    )
    service = PredictionService(settings)
    bounds = service.bounds()
    # Choose a guaranteed-supported date, rather than blindly using observed_through.
    smoke = {"beach": int(service.catalog.beach_id.iloc[0]), "date": bounds["supported_end"]}
    result = service.predict(smoke["beach"], date.fromisoformat(smoke["date"]))
    files = {
        str(file.relative_to(output)): sha256(file)
        for file in sorted(output.rglob("*"))
        if file.is_file()
    }
    write_json(
        output / "manifest.json",
        {
            "model_name": name,
            "model_version": str(version),
            "files": files,
            "smoke_case": smoke,
            "demo": result["demo"],
        },
    )
    write_json(paths.reports / "deploy/smoke_case.json", smoke)
    return settings, smoke


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--observed-through", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.dry_run:
        parser.error(
            "Initial scaffold supports --dry-run local export only; Hub publishing is pending"
        )
    paths = Paths(args.root)
    client, _ = configure_tracking(paths)
    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, args.alias)
    print(
        export_snapshot(
            paths, REGISTERED_MODEL_NAME, version.version, args.observed_through, args.output
        )
    )
