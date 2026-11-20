"""Runnable first model stage: logistic tuning, time-ordered sigmoid calibration and registration.

Default production mode honors the validation gain gate. --demo registers only
an explicitly named demo model; it never creates or promotes a production champion.
"""

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.common import Paths, read_json, sha256, validate_source_mode, write_json
from src.data.events import TARGET
from src.features.build import model_input
from src.models import lgbm, logreg
from src.models.ablations import ABLATIONS
from src.models.baselines import Baseline
from src.models.calibration import crossfit_calibrate, fit_calibrator
from src.models.config import LOGREG_CS, REGISTERED_MODEL_NAME, TEST_START, TRAIN_END, VAL_END
from src.models.metrics import choose_threshold, evaluate
from src.models.pyfunc import RiskModel
from src.models.register import git_commit, register_model
from src.models.selection import select_model
from src.models.split import time_split
from src.models.tracking import configure_tracking


def candidate(train, val, columns, family="logreg", params=None):
    pipeline = (logreg if family == "logreg" else lgbm).build(params or {}, columns)
    pipeline.fit(model_input(train, columns), train[TARGET].astype(int))
    scores = pipeline.predict_proba(model_input(val, columns))[:, 1]
    oof = crossfit_calibrate(scores, val[TARGET], val.date)
    usable = np.isfinite(oof)
    if usable.sum() < 2 or val.loc[usable, TARGET].nunique() != 2:
        raise ValueError("Too little time-ordered validation data for calibrated evaluation")
    calibrator = fit_calibrator(scores, val[TARGET])
    return RiskModel(pipeline, calibrator, columns), oof, usable


def run(paths, demo=False, skip_ablations=False, families=("logreg",)):
    validate_source_mode(paths, demo)
    report_path = paths.processed / "feature_report.json"
    report = read_json(report_path)
    columns = report["feature_columns"]
    frame = pd.read_parquet(paths.processed / "features.parquet").dropna(subset=[TARGET])
    train, val, test = time_split(frame)
    if any(part.empty for part in [train, val, test]):
        raise ValueError("Each time partition needs known observations")
    if any(part[TARGET].nunique() != 2 for part in [train, val]):
        raise ValueError("Training and calibration partitions need both classes")
    baselines = pd.read_csv(paths.reports / "baselines/comparison.csv")
    configure_tracking(paths, "beach-exceedance-demo" if demo else "beach-exceedance")
    root = paths.reports / "models"
    root.mkdir(parents=True, exist_ok=True)
    models, validation, parameters, oofs, masks = {}, {}, {}, {}, {}
    for family in families:
        if family not in {"logreg", "lgbm"}:
            raise ValueError(f"Unknown model family: {family}")
        choices = (
            [{"C": c} for c in LOGREG_CS]
            if family == "logreg"
            else [{"num_leaves": 15}, {"num_leaves": 31}]
        )
        with mlflow.start_run(run_name=family, tags={"spec": "05", "demo": str(demo).lower()}):
            trials = []
            for params in choices:
                wrapper, oof, usable = candidate(train, val, columns, family, params)
                metrics = evaluate(val.loc[usable, TARGET], oof[usable])
                with mlflow.start_run(run_name="trial", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
                trials.append((metrics, wrapper, oof, usable, params))
            metrics, wrapper, oof, usable, params = max(
                trials, key=lambda item: (item[0]["pr_auc"], -item[0]["brier"])
            )
            models[family], validation[family], parameters[family] = wrapper, metrics, params
            oofs[family], masks[family] = oof, usable
            mlflow.log_params(
                {"model_type": family, "feature_set": "full", "calibration": "sigmoid"}
            )
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
    # Cross-fitting omits the seed block. Compare every candidate and baseline
    # on the SAME remaining rows rather than changing prevalence in the gate.
    shared_mask = np.logical_and.reduce(list(masks.values()))
    selection_rows = []
    for name in models:
        validation[name] = evaluate(val.loc[shared_mask, TARGET], oofs[name][shared_mask])
        selection_rows.append({"model": name, **validation[name]})
    baseline_scores = []
    for name in ["persistence", "station_rate", "prevalence"]:
        scores = Baseline(name).fit(train).predict(val.loc[shared_mask])
        metrics = evaluate(val.loc[shared_mask, TARGET], scores)
        selection_rows.append({"model": name, **metrics})
        baseline_scores.append(metrics["pr_auc"])
    best_baseline = max(baseline_scores)
    pd.DataFrame(selection_rows).to_csv(root / "selection_comparison.csv", index=False)
    selected, register = select_model(validation, best_baseline, demo=demo)
    wrapper, usable = models[selected], masks[selected]
    threshold, rule = choose_threshold(val.loc[usable, TARGET], oofs[selected][usable])
    # Selection, hyperparameters, calibration and threshold are fixed BEFORE test scoring.
    comparison = baselines.to_dict("records")
    for family, model in models.items():
        comparison.append({"model": family, "split": "val", **validation[family]})
        comparison.append(
            {
                "model": family,
                "split": "test",
                **evaluate(test[TARGET], model.predict(None, model_input(test, columns))),
            }
        )
    registered_name = "beach-exceedance-demo" if demo else REGISTERED_MODEL_NAME
    tags = {
        "feature_columns": json.dumps(columns),
        "threshold": threshold,
        "threshold_rule": rule,
        "limit_mpn": report["threshold"],
        "train_end": TRAIN_END,
        "val_end": VAL_END,
        "test_start": TEST_START,
        "git_commit": git_commit(),
        "features_report_sha256": sha256(report_path),
        "demo": str(demo).lower(),
    }
    version = None
    with mlflow.start_run(
        run_name="final", tags={"spec": "05", "demo": str(demo).lower(), "selected": selected}
    ):
        mlflow.log_params(
            {
                "model_type": selected,
                "calibration": "sigmoid",
                "threshold": threshold,
                "threshold_rule": rule,
                "registration_gate_passed": register,
                "crossfit_validation_coverage": float(usable.mean()),
            }
        )
        mlflow.log_metrics({f"val_{k}": v for k, v in validation[selected].items()})
        for row in comparison:
            if row["model"] == selected and row["split"] == "test":
                mlflow.log_metrics(
                    {f"test_{k}": row[k] for k in ["pr_auc", "brier", "recall_at_p50"]}
                )
        mlflow.log_artifact(str(report_path))
        if register:
            version = register_model(
                wrapper,
                model_input(train.head(5), columns),
                registered_name,
                tags,
                alias="demo" if demo else "champion",
            )
    pd.DataFrame(comparison).to_csv(root / "comparison.csv", index=False)
    ablations = []
    if not skip_ablations:
        for name, keep in ABLATIONS.items():
            subset = [c for c in columns if keep(c)]
            model, oof, mask = candidate(train, val, subset, selected, parameters[selected])
            val_metrics = evaluate(val.loc[mask, TARGET], oof[mask])
            test_metrics = evaluate(test[TARGET], model.predict(None, model_input(test, subset)))
            row = {
                "ablation": name,
                "val_pr_auc": val_metrics["pr_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "val_brier": val_metrics["brier"],
                "test_brier": test_metrics["brier"],
                "delta_val_pr_auc_vs_full": val_metrics["pr_auc"] - validation[selected]["pr_auc"],
            }
            ablations.append(row)
            with mlflow.start_run(
                run_name=name, tags={"spec": "05", "ablation": name, "demo": str(demo).lower()}
            ):
                mlflow.log_metrics({k: v for k, v in row.items() if k != "ablation"})
    pd.DataFrame(
        ablations,
        columns=[
            "ablation",
            "val_pr_auc",
            "test_pr_auc",
            "val_brier",
            "test_brier",
            "delta_val_pr_auc_vs_full",
        ],
    ).to_csv(root / "ablations.csv", index=False)
    result = {
        "selected": selected,
        "registered": register,
        "model_name": registered_name,
        "model_version": version,
        "demo": demo,
        "threshold": threshold,
        "threshold_rule": rule,
        "best_baseline_val_pr_auc": best_baseline,
        "selected_val_pr_auc": validation[selected]["pr_auc"],
        "crossfit_validation_coverage": float(usable.mean()),
        "implemented_ablations": list(ABLATIONS) if not skip_ablations else [],
    }
    write_json(root / "selection.json", result)
    (root / "model_card.md").write_text(
        f"# {'SYNTHETIC DEMO — ' if demo else ''}Enterococcus risk model\n\n"
        f"Selected: {selected}. Registered: {register}. Threshold: {threshold:.6f} ({rule}).\n\n"
        f"Training through {TRAIN_END}; validation through {VAL_END}; test from {TEST_START}.\n\n"
        "Reported validation calibration metrics cover only forward cross-fit blocks with both earlier classes; "
        "the seed block is excluded. Tuning still uses validation and can make these selection metrics optimistic.\n\n"
        "Initial implementation: four feature ablations, logistic grid, optional small LightGBM grid. "
        "Full 30-trial LightGBM/early stopping, other spec-05 ablations and complete calibration/subgroup plots remain pending.\n\n"
        "Missing sampling days have no target. History availability assumes a two-day lab delay. "
        "The station maximum is not a calibrated beach-level probability.\n"
    )
    print(pd.DataFrame(comparison).to_string(index=False))
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--families", default="logreg")
    parser.add_argument(
        "--demo", action="store_true", help="synthetic/local demo registry only, never champion"
    )
    args = parser.parse_args()
    run(
        Paths(args.root),
        demo=args.demo,
        skip_ablations=args.skip_ablations,
        families=tuple(args.families.split(",")),
    )
