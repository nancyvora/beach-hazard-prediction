"""Optional LightGBM pipeline. Install requirements-optional.txt to enable it."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.models.config import SEED


def build(params, columns):
    try:
        from lightgbm import LGBMClassifier
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "LightGBM is optional; install requirements-optional.txt and OpenMP"
        ) from exc
    numeric = [c for c in columns if c != "county"]
    transformers = [("numeric", "passthrough", numeric)]
    if "county" in columns:
        transformers.append(
            (
                "county",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                ["county"],
            )
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers)),
            (
                "classifier",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=100,
                    random_state=SEED,
                    verbosity=-1,
                    n_jobs=1,
                    **params,
                ),
            ),
        ]
    )
