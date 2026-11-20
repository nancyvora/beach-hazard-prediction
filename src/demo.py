"""One-command synthetic demo. Never presents fixture scores as scientific results."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.main import create_app
from src.api.service import PredictionService, Settings
from src.common import Paths, read_json, write_json


def synthetic_raw():
    rng = np.random.default_rng(42)
    rows = []
    result_id = 1
    for station in [101, 102, 201, 202]:
        beach = station // 100
        for sampled in pd.date_range("2018-01-01", "2025-04-28", freq="7D"):
            wet = sampled.month in [11, 12, 1, 2, 3]
            positive = rng.random() < (0.08 + 0.32 * wet + 0.08 * (beach == 2))
            value = int(rng.integers(130, 1000) if positive else rng.integers(1, 95))
            qualifier = "<" if not positive and rng.random() < 0.2 else "="
            if qualifier == "<":
                value = 10
            rows.append(
                {
                    "RESULTS id": result_id,
                    "Station_id": station,
                    "Station_Name": f"DEMO-{station}",
                    "BeachName_id": beach,
                    "Beach_Name": f"Synthetic Beach {beach}",
                    "County": "Demo North" if beach == 1 else "Demo South",
                    "Station_UpperLat": 33.0 + beach,
                    "Station_UpperLon": -118.0 - beach,
                    "SampleDate": sampled.date().isoformat(),
                    "StartTime": "08:30:00",
                    "Parameter": "Enterococcus",
                    "Qualifier": qualifier,
                    "Result": value,
                    "Unit": "MPN/100ml",
                    "AnalysisMethod": "Enterolert",
                    "SampleType": "Results",
                    "RL": 1,
                    "MDL": 1,
                }
            )
            result_id += 1
    return pd.DataFrame(rows)


def run(root=Path(".demo"), include_eda=True):
    from fastapi.testclient import TestClient

    from deploy.publish_model import export_snapshot
    from src.data.cleaning import run as clean
    from src.data.eda import run as eda
    from src.features.build import run as features
    from src.models.train_baselines import run as baselines
    from src.models.train_models import run as train

    paths = Paths(Path(root))
    manifest = paths.artifacts / "serving/manifest.json"
    if manifest.exists():
        settings = Settings.from_root(root)
        smoke = read_json(manifest)["smoke_case"]
        PredictionService(settings)
        print("Reusing existing synthetic demo. Use another --root to build a fresh run.")
    else:
        if paths.raw.exists():
            raise FileExistsError(
                "Demo refuses to overwrite an existing CSV; choose another --root"
            )
        paths.raw.parent.mkdir(parents=True, exist_ok=True)
        synthetic_raw().to_csv(paths.raw, index=False)
        write_json(
            paths.root / "source_metadata.json",
            {
                "source_kind": "synthetic",
                "seed": 42,
                "real_observations": 0,
                "purpose": "plumbing demonstration only",
            },
        )
        print("[1/6] Cleaning SYNTHETIC observations", flush=True)
        clean(paths)
        print("[2/6] Station-days and cutoff-safe features", flush=True)
        features(paths)
        if include_eda:
            eda(paths)
        print("[3/6] Baselines and local MLflow runs", flush=True)
        baselines(paths, demo=True)
        print("[4/6] Calibrated logistic demo model (not production champion)", flush=True)
        selected = train(paths, demo=True, skip_ablations=True)
        print("[5/6] Export and verify local snapshot", flush=True)
        events = pd.read_parquet(paths.processed / "station_days.parquet")
        settings, smoke = export_snapshot(
            paths, selected["model_name"], selected["model_version"], events.date.max().date()
        )
    print("[6/6] In-process HTTP smoke test", flush=True)
    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        response = client.get("/predict", params=smoke)
        if health.status_code != 200 or response.status_code != 200:
            raise RuntimeError(f"Demo API failed: {health.text}; {response.text}")
        assert response.json()["demo"] is True
        write_json(
            paths.reports / "demo_result.json",
            {"health": health.json(), "prediction": response.json()},
        )
    print(f"SYNTHETIC DEMO OK: {paths.reports / 'demo_result.json'}", flush=True)
    print(f"GET /predict?beach={smoke['beach']}&date={smoke['date']}", flush=True)
    return settings, smoke


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".demo"))
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    settings, _ = run(args.root, include_eda=not args.skip_eda)
    if args.serve:
        import uvicorn

        uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port)
