"""Calendar split: never shuffle observations across time."""

import pandas as pd

from src.models.config import TRAIN_END, VAL_END


def time_split(frame):
    if frame.empty or "date" not in frame:
        raise ValueError("time_split requires a nonempty frame with canonical 'date'")
    dates = pd.to_datetime(frame.date, errors="raise")
    if dates.isna().any():
        raise ValueError("Missing split dates")
    return tuple(
        frame.loc[mask].copy()
        for mask in [dates <= TRAIN_END, (dates > TRAIN_END) & (dates <= VAL_END), dates > VAL_END]
    )
