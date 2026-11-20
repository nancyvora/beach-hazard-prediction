"""Pinned MLflow model plus a consistent, immutable serving snapshot."""

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.common import california_today, json_value, read_json, sha256
from src.data.events import LIMIT, TARGET, TARGET_RULE_VERSION
from src.features.build import FEATURE_COLUMNS, build_features, model_input

MODEL_NAME = "beach-exceedance-entero"
DISCLAIMER = (
    "Predicted Enterococcus exceedance risk, not an official advisory or a guarantee "
    "of safe swimming. Follow local health advisories."
)
REQUIRED_TAGS = {
    "feature_columns",
    "threshold",
    "threshold_rule",
    "limit_mpn",
    "train_end",
    "val_end",
    "test_start",
    "git_commit",
    "features_report_sha256",
}


class ServiceError(Exception):
    def __init__(self, code, message, status=422, **context):
        self.code, self.message, self.status, self.context = code, message, status, context
        super().__init__(message)


@dataclass(frozen=True)
class Settings:
    model_uri: str
    feature_report: Path
    catalog: Path
    history: Path
    history_metadata: Path
    tracking_uri: str | None = None

    @classmethod
    def from_root(cls, root):
        root = Path(root)
        serving = root / "artifacts/serving"
        return cls(
            str(serving / "model"),
            serving / "feature_report.json",
            serving / "station_catalog.parquet",
            serving / "station_history.parquet",
            serving / "history_metadata.json",
        )

    @classmethod
    def from_env(cls):
        history = Path(os.getenv("HISTORY_PATH", "artifacts/serving/station_history.parquet"))
        return cls(
            os.getenv("MODEL_URI", f"models:/{MODEL_NAME}@champion"),
            Path(os.getenv("FEATURE_REPORT_PATH", "artifacts/serving/feature_report.json")),
            Path(os.getenv("STATION_CATALOG_PATH", "artifacts/serving/station_catalog.parquet")),
            history,
            Path(os.getenv("HISTORY_METADATA_PATH", str(history.parent / "history_metadata.json"))),
            os.getenv("MLFLOW_TRACKING_URI"),
        )


class PredictionService:
    def __init__(self, settings):
        self.settings = settings
        if settings.model_uri.startswith("models:/"):
            if not settings.tracking_uri:
                raise ValueError("Registry serving requires an explicit MLFLOW_TRACKING_URI")
            mlflow.set_tracking_uri(settings.tracking_uri)
            client = mlflow.MlflowClient()
            reference = settings.model_uri.removeprefix("models:/")
            if "@" in reference:
                name, alias = reference.split("@", 1)
                version = client.get_model_version_by_alias(name, alias)
            else:
                name, number = reference.rsplit("/", 1)
                version = client.get_model_version(name, number)
            metadata = {
                "model_name": name,
                "model_version": str(version.version),
                "tags": version.tags,
            }
            model_uri = f"models:/{name}/{version.version}"
        else:
            model_dir = Path(settings.model_uri)
            metadata = read_json(model_dir / "serving_metadata.json")
            model_uri = str(model_dir)
        tags = metadata["tags"]
        if REQUIRED_TAGS - set(tags):
            raise ValueError("Missing model-version tags")
        self.metadata, self.tags = metadata, dict(tags)
        self.threshold = float(tags["threshold"])
        if (
            not np.isfinite(self.threshold)
            or not 0 <= self.threshold <= 1
            or not tags["threshold_rule"]
        ):
            raise ValueError("Invalid risk threshold")
        if float(tags["limit_mpn"]) != LIMIT:
            raise ValueError("Incompatible concentration limit")
        report = read_json(settings.feature_report)
        self.columns = json.loads(tags["feature_columns"])
        if self.columns != FEATURE_COLUMNS or self.columns != report["feature_columns"]:
            raise ValueError("Incompatible feature columns/order")
        if sha256(settings.feature_report) != tags["features_report_sha256"]:
            raise ValueError("Feature report hash mismatch")
        if report["history_delay_days"] != 2 or report["transform"] != "log10(max(value,1.0))":
            raise ValueError("Incompatible feature semantics")
        if report["target_rule_version"] != TARGET_RULE_VERSION:
            raise ValueError("Incompatible target rules")
        self.model = mlflow.pyfunc.load_model(model_uri)
        signature = self.model.metadata.signature
        if signature is None or signature.inputs.input_names() != self.columns:
            raise ValueError("Incompatible model signature")
        self.catalog = pd.read_parquet(settings.catalog)
        required_catalog = [
            "station_id",
            "station_name",
            "beach_id",
            "beach_name",
            "county",
            "lat",
            "lon",
        ]
        if self.catalog.empty or set(required_catalog) - set(self.catalog):
            raise ValueError("Invalid station catalog")
        if (
            self.catalog.station_id.duplicated().any()
            or self.catalog[required_catalog].isna().any().any()
        ):
            raise ValueError("Conflicting/incomplete station catalog")
        if not np.isfinite(self.catalog[["lat", "lon"]].to_numpy(dtype=float)).all():
            raise ValueError("Invalid coordinates")
        self.history = pd.read_parquet(settings.history)
        if {"station_id", "date", "value", "censored", TARGET} - set(self.history):
            raise ValueError("Invalid history schema")
        self.history["date"] = pd.to_datetime(self.history.date).astype("datetime64[ns]")
        if not self.history.date.eq(self.history.date.dt.normalize()).all():
            raise ValueError("History must contain date-only observations")
        values = self.history.value.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("History has invalid concentrations")
        if self.history.duplicated(["station_id", "date"]).any() or self.history.date.isna().any():
            raise ValueError("Duplicate/missing history dates")
        if not self.history.station_id.isin(self.catalog.station_id).all():
            raise ValueError("History references unknown stations")
        self.snapshot = read_json(settings.history_metadata)
        self.observed_through = date.fromisoformat(self.snapshot["history_observed_through"])
        if (
            not self.snapshot["history_snapshot_id"]
            or (self.history.date.dt.date > self.observed_through).any()
        ):
            raise ValueError("Invalid coverage watermark")
        self.train_end, self.val_end = (
            date.fromisoformat(tags["train_end"]),
            date.fromisoformat(tags["val_end"]),
        )
        if not self.train_end < self.val_end < date.fromisoformat(tags["test_start"]):
            raise ValueError("Invalid model split dates")
        self.demo = str(tags.get("demo", "false")).lower() == "true"
        self.bounds()

    def bounds(self):
        first = max(self.train_end, self.val_end) + timedelta(days=2)
        last = min(self.observed_through + timedelta(days=2), california_today())
        if first > last:
            raise ServiceError(
                "service_not_ready", "No supported dates", 503, reason="no_supported_dates"
            )
        return {"supported_start": first.isoformat(), "supported_end": last.isoformat()}

    def provenance(self):
        return {
            "model_name": self.metadata["model_name"],
            "model_version": str(self.metadata["model_version"]),
            "feature_version": "sha256:" + self.tags["features_report_sha256"],
            "history_snapshot_id": self.snapshot["history_snapshot_id"],
            "history_observed_through": self.observed_through.isoformat(),
            "demo": self.demo,
        }

    def health(self):
        return {
            "status": "ok",
            **self.provenance(),
            **self.bounds(),
            "risk_threshold": self.threshold,
            "threshold_rule": self.tags["threshold_rule"],
        }

    def predict(self, beach, requested):
        if requested > california_today():
            raise ServiceError("invalid_request", "Future dates are not supported")
        bounds = self.bounds()
        catalog = self.catalog.loc[self.catalog.beach_id.eq(beach)].sort_values("station_id")
        if catalog.empty:
            raise ServiceError("beach_not_found", "Unknown beach ID", 404)
        if not bounds["supported_start"] <= requested.isoformat() <= bounds["supported_end"]:
            raise ServiceError(
                "unsupported_date", "Date is outside this model/snapshot range", **bounds
            )
        queries = catalog[["station_id"]].assign(date=pd.Timestamp(requested))
        features = build_features(self.history, queries, catalog)
        raw_scores = self.model.predict(model_input(features, self.columns))
        if isinstance(raw_scores, pd.DataFrame) and raw_scores.shape[1] == 1:
            raw_scores = raw_scores.iloc[:, 0]
        scores = np.asarray(raw_scores, dtype=float)
        if (
            scores.shape != (len(features),)
            or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()
        ):
            raise ServiceError("prediction_failed", "Invalid model probabilities", 503)
        stations = []
        for row, probability in zip(features.itertuples(index=False), scores, strict=True):
            has_history = row.n_prev_samples > 0
            stations.append(
                {
                    "station_id": row.station_id,
                    "station_name": str(row.station_name),
                    "status": "predicted",
                    "probability": float(probability),
                    "risk_level": self.risk(probability),
                    "latest_history_date": row.latest_history_date.date().isoformat()
                    if has_history
                    else None,
                    "days_since_prev": int(row.days_since_prev) if has_history else None,
                    "n_prev_samples": row.n_prev_samples,
                    "history_status": "none"
                    if not has_history
                    else "stale"
                    if row.days_since_prev > 30
                    else "available",
                }
            )
        highest = float(scores.max())
        disclaimer = ("SYNTHETIC DEMO DATA. " if self.demo else "") + DISCLAIMER
        return json_value(
            {
                "beach_id": beach,
                "beach_name": str(catalog.beach_name.iloc[0]),
                "date": requested.isoformat(),
                "max_station_probability": highest,
                "risk_level": self.risk(highest),
                "risk_threshold": self.threshold,
                "threshold_rule": self.tags["threshold_rule"],
                "aggregation": "max_station_probability",
                "coverage": {"n_stations": len(stations), "n_predicted": len(stations)},
                "stations": stations,
                **self.provenance(),
                "history_cutoff": (requested - timedelta(days=2)).isoformat(),
                "disclaimer": disclaimer,
            }
        )

    def risk(self, probability):
        return "elevated" if probability >= self.threshold else "lower"
