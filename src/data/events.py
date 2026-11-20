"""One shared station-day and censoring policy for EDA, features and serving."""

import numpy as np
import pandas as pd

LIMIT = 104.0
TARGET = "exceed_Enterococcus"
TARGET_RULE_VERSION = "spec02-03-boundary-v1"


def event_labels(events):
    values = events.value.to_numpy(dtype=float)
    censoring = events.censored.astype(str).to_numpy()
    labels = np.full(len(events), np.nan)
    exact = censoring == "exact"
    labels[exact] = (values[exact] > LIMIT).astype(float)
    labels[(censoring == "below_DL") & (values <= LIMIT)] = 0
    labels[(censoring == "above_QL") & (values >= LIMIT)] = 1
    return pd.array(labels, dtype="Int8")


def station_days(clean):
    df = clean.loc[clean.indicator.eq("Enterococcus")].copy()
    df["n_results"] = df.groupby(["station_id", "date"], observed=True).result_id.transform("size")
    df["_rank"] = df.censored.astype(str).map({"exact": 0, "above_QL": 1, "below_DL": 2})
    df = df.sort_values(
        ["station_id", "date", "value", "_rank", "result_id"],
        ascending=[True, True, False, True, True],
    )
    df = df.drop_duplicates(["station_id", "date"]).drop(columns="_rank")
    df[TARGET] = event_labels(df)
    return df.sort_values(["station_id", "date"]).reset_index(drop=True)


def station_catalog(events):
    columns = ["station_id", "station_name", "beach_id", "beach_name", "county", "lat", "lon"]
    catalog = events[columns].drop_duplicates()
    conflicts = catalog.loc[catalog.station_id.duplicated(False), "station_id"].unique()
    if len(conflicts):
        raise ValueError(f"Conflicting station metadata: {conflicts.tolist()}")
    if catalog.empty or catalog[columns].isna().any().any():
        raise ValueError("Station catalog must be nonempty with complete location metadata")
    return catalog.sort_values("station_id").reset_index(drop=True)
