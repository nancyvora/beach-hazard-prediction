import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss

from src.data.events import TARGET
from src.features.build import model_input
from src.models.baselines import Baseline
from src.models.calibration import crossfit_calibrate, fit_calibrator, logit
from src.models.logreg import build
from src.models.metrics import choose_threshold, evaluate
from src.models.selection import select_model
from src.models.split import time_split


def test_calendar_split_and_invalid_input():
    frame = pd.DataFrame(
        {"date": pd.to_datetime(["2021-12-31", "2022-01-01", "2023-12-31", "2024-01-01"])}
    )
    train, val, test = time_split(frame)
    assert [len(train), len(val), len(test)] == [1, 2, 1]
    assert train.date.max() < val.date.min() <= val.date.max() < test.date.min()
    with pytest.raises(ValueError):
        time_split(pd.DataFrame())


def test_metrics_match_known_values():
    y, scores = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    result = evaluate(y, scores)
    assert result["pr_auc"] == pytest.approx(average_precision_score(y, scores))
    assert result["brier"] == pytest.approx(brier_score_loss(y, scores))
    with pytest.raises(ValueError):
        evaluate(y, [np.nan] * 4)


def test_train_only_fallbacks():
    train = pd.DataFrame({"county": ["A", "A", "B"], TARGET: [0, 1, 0]})
    query = pd.DataFrame(
        {
            "county": ["A", "new"],
            "prev_exceed": [np.nan, np.nan],
            "station_hist_rate": [np.nan, np.nan],
        }
    )
    assert Baseline("station_rate").fit(train).predict(query) == pytest.approx([0.5, 1 / 3])
    assert Baseline("persistence").fit(train).predict(query) == pytest.approx([1 / 3, 1 / 3])


def test_selection_gate_and_tie():
    metrics = {"logreg": {"pr_auc": 0.70, "brier": 0.1}, "lgbm": {"pr_auc": 0.705, "brier": 0.09}}
    assert select_model(metrics, 0.60) == ("logreg", True)
    assert select_model(metrics, 0.69) == ("logreg", False)


def test_threshold_and_forward_calibration():
    y = np.tile([0, 1], 30)
    scores = np.tile([0.7, 0.95], 30)
    dates = pd.date_range("2022-01-01", periods=len(y))
    oof = crossfit_calibrate(scores, y, dates)
    assert np.isnan(oof[:10]).all() and np.isfinite(oof[20:]).all()
    calibrated = fit_calibrator(scores, y).predict_proba(logit(scores))[:, 1]
    assert brier_score_loss(y, calibrated) < brier_score_loss(y, scores)
    threshold, rule = choose_threshold(y, calibrated)
    assert 0 <= threshold <= 1 and rule == "validation_precision_ge_0.50"


def test_preprocessor_unseen_county_and_missing_lags(five_events):
    from src.data.events import station_catalog
    from src.features.build import build_features

    features = build_features(
        five_events, five_events[["station_id", "date"]], station_catalog(five_events)
    )
    x = model_input(features)
    pipeline = build().fit(x, [0, 0, 1, 1, 1])
    query = x.iloc[[0]].copy()
    query["county"] = "previously unseen"
    assert np.isfinite(pipeline.predict_proba(query)).all()


def test_synthetic_data_requires_explicit_demo_mode(tmp_path):
    from src.common import Paths, write_json
    from src.models.train_baselines import run as run_baselines
    from src.models.train_models import run as run_models

    write_json(tmp_path / "source_metadata.json", {"source_kind": "synthetic"})
    for run in [run_baselines, run_models]:
        with pytest.raises(ValueError, match="Synthetic source"):
            run(Paths(tmp_path))
