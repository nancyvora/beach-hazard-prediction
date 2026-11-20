"""Spec 04 command: evaluate and log all three baselines."""

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from src.common import Paths, validate_source_mode, write_json
from src.data.events import TARGET
from src.models.baselines import Baseline
from src.models.config import TRAIN_END, VAL_END
from src.models.metrics import breakdown, evaluate
from src.models.split import time_split
from src.models.tracking import configure_tracking


def run(paths, demo=False):
    validate_source_mode(paths, demo)
    frame = pd.read_parquet(paths.processed / "features.parquet").dropna(subset=[TARGET])
    train, val, test = time_split(frame)
    if any(group.empty for group in [train, val, test]):
        raise ValueError("All three time partitions must contain measured, known targets")
    configure_tracking(paths, "beach-exceedance-demo" if demo else "beach-exceedance")
    root = paths.reports / "baselines"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ["persistence", "station_rate", "prevalence"]:
        model = Baseline(name).fit(train)
        with mlflow.start_run(run_name=name, tags={"spec": "04", "demo": str(demo).lower()}):
            mlflow.log_params(
                {
                    "model_type": name,
                    "train_end": TRAIN_END,
                    "val_end": VAL_END,
                    "feature_set": "canonical_spec03",
                }
            )
            for split_name, split in [("val", val), ("test", test)]:
                scores = model.predict(split)
                metrics = evaluate(split[TARGET], scores)
                mlflow.log_metrics({f"{split_name}_{key}": value for key, value in metrics.items()})
                rows.append({"model": name, "split": split_name, **metrics})
                scored = split.assign(
                    score=scores,
                    season=split.month.isin([4, 5, 6, 7, 8, 9, 10]).map(
                        {True: "swim", False: "off"}
                    ),
                )
                for by in ["county", "season"]:
                    file = root / f"{name}_{split_name}_{by}.csv"
                    breakdown(scored, by).to_csv(file, index=False)
                    mlflow.log_artifact(str(file))
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "comparison.csv", index=False)
    write_json(root / "run_summary.json", {"demo": demo, "models": 3, "rows": len(frame)})
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    run(Paths(args.root), demo=args.demo)
