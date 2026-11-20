"""Three baselines; learned fallbacks are fitted on training rows only."""

import numpy as np

from src.data.events import TARGET


class Baseline:
    def __init__(self, kind):
        if kind not in ["persistence", "station_rate", "prevalence"]:
            raise ValueError(kind)
        self.kind = kind

    def fit(self, train):
        if train.empty or train[TARGET].isna().any():
            raise ValueError("Baseline needs known training labels")
        self.prevalence = float(train[TARGET].mean())
        self.county_rates = train.groupby("county", observed=True)[TARGET].mean().to_dict()
        return self

    def predict(self, frame):
        if self.kind == "persistence":
            return frame.prev_exceed.astype(float).fillna(self.prevalence).to_numpy()
        if self.kind == "station_rate":
            county = frame.county.astype(str).map(self.county_rates).astype(float)
            return (
                frame.station_hist_rate.fillna(county).fillna(self.prevalence).to_numpy(dtype=float)
            )
        return np.full(len(frame), self.prevalence)
