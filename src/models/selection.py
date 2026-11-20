"""The registration gate uses validation metrics only."""

from src.models.config import MIN_PR_AUC_GAIN, TIE_MARGIN


def select_model(metrics, best_baseline, demo=False):
    if not metrics:
        raise ValueError("No candidate models")
    best = max(metrics, key=lambda name: (metrics[name]["pr_auc"], -metrics[name]["brier"]))
    if "logreg" in metrics and metrics[best]["pr_auc"] - metrics["logreg"]["pr_auc"] <= TIE_MARGIN:
        best = "logreg"
    clears_gate = metrics[best]["pr_auc"] >= best_baseline + MIN_PR_AUC_GAIN
    return best, bool(clears_gate or demo)
