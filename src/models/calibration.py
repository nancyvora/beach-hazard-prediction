"""Sigmoid calibration with strictly earlier validation dates for cross-fit scores."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit


def logit(probabilities):
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(probabilities / (1 - probabilities)).reshape(-1, 1)


def fit_calibrator(scores, y):
    if len(np.unique(y)) != 2:
        raise ValueError("Calibration requires both classes; collect more validation data")
    return LogisticRegression(C=1e6, max_iter=2000, random_state=42).fit(
        logit(scores), np.asarray(y, dtype=int)
    )


def crossfit_calibrate(scores, y, dates, n_folds=5):
    """Initial validation block seeds calibration; it is not scored in cross-fit metrics."""
    scores, y, dates = np.asarray(scores), np.asarray(y, dtype=int), pd.to_datetime(dates)
    unique_dates = np.sort(dates.unique())
    if len(unique_dates) < 3:
        raise ValueError("At least three distinct validation dates are required")
    output = np.full(len(scores), np.nan)
    for earlier, later in TimeSeriesSplit(n_splits=min(n_folds, len(unique_dates) - 1)).split(
        unique_dates
    ):
        train_mask = dates.isin(unique_dates[earlier])
        score_mask = dates.isin(unique_dates[later])
        if len(np.unique(y[train_mask])) < 2:
            continue
        calibrator = fit_calibrator(scores[train_mask], y[train_mask])
        output[score_mask] = calibrator.predict_proba(logit(scores[score_mask]))[:, 1]
    return output
