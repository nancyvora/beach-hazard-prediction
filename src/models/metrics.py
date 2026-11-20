"""Evaluation metrics and explicit small-group skips."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from src.data.events import TARGET


def evaluate(y_true, y_score):
    y = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    if not len(y) or y.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Metrics need nonempty, finite, equally sized vectors")
    if not np.isin(y, [0, 1]).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid labels or probabilities")
    precision, recall, _ = precision_recall_curve(y, scores)
    qualifying = precision[:-1] >= 0.5
    return {
        "pr_auc": float(average_precision_score(y, scores)) if y.sum() else 0.0,
        "recall_at_p50": float(recall[:-1][qualifying].max()) if qualifying.any() else 0.0,
        "brier": float(brier_score_loss(y, scores)),
        "prevalence": float(y.mean()),
        "n_rows": len(y),
    }


def breakdown(frame, by, score_column="score"):
    rows = []
    for name, group in frame.groupby(by, observed=True, dropna=False):
        reason = "low_n" if len(group) < 30 else "no_positives" if not group[TARGET].sum() else None
        row = {by: name, "n_rows": len(group), "skipped": reason is not None, "reason": reason}
        if not reason:
            row.update(evaluate(group[TARGET], group[score_column]))
        rows.append(row)
    return pd.DataFrame(rows)


def choose_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(np.asarray(y_true, dtype=int), scores)
    eligible = np.flatnonzero(precision[:-1] >= 0.5)
    if len(eligible):
        return float(thresholds[eligible[0]]), "validation_precision_ge_0.50"
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)]), "validation_max_f1_fallback"
