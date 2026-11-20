"""Public query/response contracts; no arbitrary feature upload endpoint."""

import re
from datetime import date as calendar_date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    beach: int = Field(gt=0)
    date: str

    @field_validator("date")
    @classmethod
    def iso_date(cls, value):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Use YYYY-MM-DD")
        calendar_date.fromisoformat(value)
        return value


class StationPrediction(BaseModel):
    station_id: int
    station_name: str
    status: str = "predicted"
    probability: float = Field(ge=0, le=1)
    risk_level: str
    latest_history_date: str | None
    days_since_prev: int | None
    n_prev_samples: int
    history_status: str


class Coverage(BaseModel):
    n_stations: int
    n_predicted: int


class PredictionResponse(BaseModel):
    beach_id: int
    beach_name: str
    date: str
    max_station_probability: float = Field(ge=0, le=1)
    risk_level: str
    risk_threshold: float
    threshold_rule: str
    aggregation: str
    coverage: Coverage
    stations: list[StationPrediction]
    model_name: str
    model_version: str
    feature_version: str
    history_snapshot_id: str
    history_observed_through: str
    history_cutoff: str
    demo: bool = False
    disclaimer: str
