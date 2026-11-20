"""Spec 03: vectorized per-station time-safe features."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import Paths, write_json
from src.data.events import LIMIT, TARGET, TARGET_RULE_VERSION, station_catalog, station_days

HISTORY_DELAY_DAYS = 2
ENGINEERED = [
    "month",
    "dow",
    "wet_season",
    "doy_sin",
    "doy_cos",
    "prev_log10",
    "prev_exceed",
    "days_since_prev",
    "roll3_log10",
    "roll3_max_log10",
    "n_roll3",
    "station_hist_rate",
    "n_prev_samples",
    "n_prev_known",
    "no_history",
]
FEATURE_COLUMNS = ["county", "lat", "lon", *ENGINEERED]
AUDIT_COLUMNS = ["history_cutoff", "latest_history_date"]
ID_COLUMNS = ["station_id", "beach_id", "station_name", "beach_name", "date"]


def build_features(history, queries, catalog):
    if queries.duplicated(["station_id", "date"]).any():
        raise ValueError("Duplicate query station-day")
    if history.duplicated(["station_id", "date"]).any():
        raise ValueError("Duplicate history station-day")
    if catalog.station_id.duplicated().any():
        raise ValueError("Duplicate catalog station")
    result = queries[["station_id", "date"]].copy()
    result["date"] = pd.to_datetime(result.date).dt.normalize().astype("datetime64[ns]")
    if result.date.isna().any():
        raise ValueError("Missing query date")
    result = result.merge(
        catalog, on="station_id", how="left", validate="many_to_one", indicator=True
    )
    if not result._merge.eq("both").all():
        raise ValueError("Unknown query station")
    result = (
        result.drop(columns="_merge").sort_values(["station_id", "date"]).reset_index(drop=True)
    )
    result["history_cutoff"] = result.date - pd.Timedelta(days=HISTORY_DELAY_DAYS)
    result["month"] = result.date.dt.month.astype("int8")
    result["dow"] = result.date.dt.day_of_week.astype("int8")
    result["wet_season"] = result.month.isin([11, 12, 1, 2, 3]).astype("int8")
    angle = (
        2
        * np.pi
        * (result.date.dt.day_of_year - 1)
        / np.where(result.date.dt.is_leap_year, 366, 365)
    )
    result["doy_sin"], result["doy_cos"] = np.sin(angle), np.cos(angle)
    for name in [
        "prev_log10",
        "prev_exceed",
        "days_since_prev",
        "roll3_log10",
        "roll3_max_log10",
        "station_hist_rate",
    ]:
        result[name] = np.nan
    for name in ["n_roll3", "n_prev_samples", "n_prev_known"]:
        result[name] = 0
    result["latest_history_date"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
    grouped = {
        key: group.sort_values("date")
        for key, group in history.groupby("station_id", observed=True)
    }
    for station, query_rows in result.groupby("station_id", observed=True):
        h = grouped.get(station)
        if h is None or h.empty:
            continue
        days = h.date.to_numpy(dtype="datetime64[ns]")
        cutoff = query_rows.history_cutoff.to_numpy(dtype="datetime64[ns]")
        count = np.searchsorted(days, cutoff, side="right")
        result.loc[query_rows.index, "n_prev_samples"] = count
        labels = h[TARGET].to_numpy(dtype=float, na_value=np.nan)
        known_prefix = np.r_[0, np.cumsum(~np.isnan(labels))]
        positive_prefix = np.r_[0, np.cumsum(np.nan_to_num(labels))]
        known = known_prefix[count]
        result.loc[query_rows.index, "n_prev_known"] = known
        result.loc[query_rows.index, "station_hist_rate"] = np.divide(
            positive_prefix[count], known, out=np.full(len(count), np.nan), where=known > 0
        )
        active = count > 0
        idx = query_rows.index[active]
        last = count[active] - 1
        if not len(last):
            continue
        log_values = np.log10(np.maximum(h.value.to_numpy(dtype=float), 1))
        result.loc[idx, "prev_log10"] = log_values[last]
        result.loc[idx, "prev_exceed"] = labels[last]
        result.loc[idx, "latest_history_date"] = days[last]
        result.loc[idx, "days_since_prev"] = (
            query_rows.loc[idx, "date"].to_numpy(dtype="datetime64[ns]") - days[last]
        ) / np.timedelta64(1, "D")
        indexes = last[:, None] - np.arange(3)[None, :]
        window = np.where(indexes >= 0, log_values[np.maximum(indexes, 0)], np.nan)
        result.loc[idx, "roll3_log10"] = np.nanmean(window, axis=1)
        result.loc[idx, "roll3_max_log10"] = np.nanmax(window, axis=1)
        result.loc[idx, "n_roll3"] = np.minimum(count[active], 3)
    result["no_history"] = result.n_prev_samples.eq(0).astype("int8")
    result["prev_exceed"] = result.prev_exceed.astype("Int8")
    result["n_roll3"] = result.n_roll3.astype("int8")
    result["county"] = result.county.astype("category")
    return result


def model_input(frame, columns=FEATURE_COLUMNS):
    """Normalize nullable integer/categorical dtypes to an MLflow-safe input schema."""
    result = frame[list(columns)].copy()
    for name in columns:
        if name == "county":
            result[name] = result[name].astype(str)
        else:
            result[name] = result[name].astype("float64")
    return result


def run(paths):
    clean = pd.read_parquet(paths.interim / "results_clean.parquet")
    events = station_days(clean)
    catalog = station_catalog(events)
    features = build_features(events, events[["station_id", "date"]], catalog)
    features = features.merge(
        events[["station_id", "date", TARGET]], on=["station_id", "date"], validate="one_to_one"
    )
    paths.processed.mkdir(parents=True, exist_ok=True)
    events.to_parquet(paths.processed / "station_days.parquet", index=False)
    features.to_parquet(paths.processed / "features.parquet", index=False)
    report = {
        "input_rows": len(clean),
        "enterococcus_rows": int(clean.indicator.eq("Enterococcus").sum()),
        "station_days": len(events),
        "rows_collapsed": int(events.n_results.sum() - len(events)),
        "known_targets": int(events[TARGET].notna().sum()),
        "unknown_targets": int(events[TARGET].isna().sum()),
        "positives": int(events[TARGET].sum()),
        "stations": len(catalog),
        "min_date": events.date.min(),
        "max_date": events.date.max(),
        "no_history_rows": int(features.no_history.sum()),
        "short_history_rows": int((features.n_roll3 < 3).sum()),
        "missing_per_feature": features[FEATURE_COLUMNS].isna().sum().to_dict(),
        "history_delay_days": HISTORY_DELAY_DAYS,
        "threshold": LIMIT,
        "target_rule_version": TARGET_RULE_VERSION,
        "transform": "log10(max(value,1.0))",
        "rolling_window": "up to 3 eligible station-days",
        "feature_columns": FEATURE_COLUMNS,
        "id_columns": ID_COLUMNS,
        "target_columns": [TARGET],
        "audit_columns": AUDIT_COLUMNS,
    }
    write_json(paths.processed / "feature_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(run(Paths(args.root)))


if __name__ == "__main__":
    main()
