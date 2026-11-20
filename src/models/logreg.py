"""Train-fitted numeric preprocessing and unknown-safe county encoding."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build import FEATURE_COLUMNS
from src.models.config import SEED


def build(params=None, columns=FEATURE_COLUMNS):
    params = params or {}
    numeric = [name for name in columns if name != "county"]
    transforms = [
        (
            "numeric",
            Pipeline(
                [
                    ("impute", SimpleImputer(add_indicator=True, keep_empty_features=True)),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric,
        )
    ]
    if "county" in columns:
        transforms.append(
            ("county", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["county"])
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transforms)),
            (
                "classifier",
                LogisticRegression(C=params.get("C", 1), max_iter=2000, random_state=SEED),
            ),
        ]
    )
