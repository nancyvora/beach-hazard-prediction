"""Shared filesystem and strict JSON helpers."""

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def california_today():
    return datetime.now(ZoneInfo("America/Los_Angeles")).date()


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_mode(paths, demo):
    marker = paths.root / "source_metadata.json"
    if marker.exists() and read_json(marker).get("source_kind") == "synthetic" and not demo:
        raise ValueError("Synthetic source cannot enter production runs; use explicit --demo mode")


@dataclass(frozen=True)
class Paths:
    root: Path = Path(".")

    @property
    def raw(self):
        return self.root / "data/raw/beach-monitoring-results.csv"

    @property
    def interim(self):
        return self.root / "data/interim"

    @property
    def processed(self):
        return self.root / "data/processed"

    @property
    def reports(self):
        return self.root / "reports"

    @property
    def artifacts(self):
        return self.root / "artifacts"
