"""Spec 02: reproducible descriptive tables and headless plots."""

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "beach-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "beach-cache"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common import Paths, write_json
from src.data.events import TARGET, station_days
from src.features.build import build_features


def grouped_summary(events, key):
    rows = []
    for name, group in events.groupby(key, observed=True, dropna=False):
        known = group[TARGET].dropna()
        rows.append(
            {
                key: name,
                "n_events": len(group),
                "n_known": len(known),
                "n_stations": group.station_id.nunique(),
                "exceed_rate": known.mean() if len(known) else np.nan,
                "low_n": len(known) < 30,
            }
        )
    return pd.DataFrame(rows)


def run(paths):
    from src.data.events import station_catalog

    clean = pd.read_parquet(paths.interim / "results_clean.parquet")
    events = station_days(clean)
    if events.empty:
        raise ValueError("EDA requires at least one Enterococcus station-day")
    root = paths.reports / "eda"
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    known = events[TARGET].dropna()
    report = {
        "n_events": len(events),
        "n_positive": int(known.sum()),
        "positive_rate": known.mean(),
        "n_unknown": int(events[TARGET].isna().sum()),
        "n_stations": events.station_id.nunique(),
        "n_beaches": events.beach_id.nunique(),
        "min_date": events.date.min(),
        "max_date": events.date.max(),
        "rows_collapsed": int(events.n_results.sum() - len(events)),
    }
    pd.DataFrame(
        [{k: report[k] for k in ["n_events", "n_positive", "positive_rate", "n_unknown"]}]
    ).to_csv(tables / "target_summary.csv", index=False)
    indicators = []
    for name, group in clean.groupby("indicator", observed=True):
        indicators.append(
            {
                "indicator": name,
                "n_results": len(group),
                "median": group.value.median(),
                "p95": group.value.quantile(0.95),
                **{
                    f"pct_{kind}": 100 * group.censored.eq(kind).mean()
                    for kind in ["exact", "below_DL", "above_QL"]
                },
            }
        )
    pd.DataFrame(indicators).to_csv(tables / "indicator_summary.csv", index=False)
    events["year"], events["month"] = events.date.dt.year, events.date.dt.month
    for key, filename in [("year", "yearly"), ("month", "monthly"), ("county", "county")]:
        summary = grouped_summary(events, key)
        summary.to_csv(tables / f"{filename}.csv", index=False)
        plot = summary.loc[~summary.low_n]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(plot[key].astype(str), plot.exceed_rate)
        ax.set(
            xlabel=key,
            ylabel="Enterococcus exceedance fraction",
            title=f"Rate by {key} (known n >= 30)",
        )
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        filename = f"events_and_rate_by_{key}" if key != "county" else "rate_by_county"
        fig.savefig(figures / f"{filename}.png")
        plt.close(fig)
    station = events.groupby("station_id", observed=True).agg(
        station_name=("station_name", "first"),
        county=("county", "first"),
        n_events=("date", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        exceed_rate=(TARGET, "mean"),
        lat=("lat", "first"),
        lon=("lon", "first"),
    )
    gaps = events.groupby("station_id", observed=True).date.diff().dt.days
    station["median_gap_days"] = gaps.groupby(events.station_id).median()
    station.to_csv(tables / "station_summary.csv")
    buckets = pd.cut(
        gaps, bins=[0, 1, 3, 7, 14, 30, np.inf], labels=["1", "2–3", "4–7", "8–14", "15–30", ">30"]
    )
    buckets.value_counts(sort=False).rename_axis("gap_bucket").rename("n_events").to_csv(
        tables / "sampling_gaps.csv"
    )
    persistence = []
    catalog = station_catalog(events)
    for delay in [0, 2]:
        # delay 0 means any strictly earlier day, not the current day's result.
        queries = events[["station_id", "date"]].copy()
        queries["date"] += pd.Timedelta(days=2 - max(delay, 1))
        history = build_features(events, queries, catalog)
        history["actual"] = events[TARGET].to_numpy()
        history = history.loc[history.prev_exceed.notna() & history.actual.notna()]
        for previous, group in history.groupby("prev_exceed", observed=True):
            persistence.append(
                {
                    "prev_exceed": previous,
                    "min_gap_days": delay,
                    "n_events": len(group),
                    "exceed_rate": group.actual.mean(),
                }
            )
    pd.DataFrame(persistence).to_csv(tables / "persistence.csv", index=False)
    splits = []
    for cutoff in [2012, 2014, 2016, 2018, 2020]:
        train, test = events.loc[events.year < cutoff], events.loc[events.year >= cutoff]
        splits.append(
            {
                "cutoff_year": cutoff,
                "n_train": len(train),
                "n_test": len(test),
                "rate_train": train[TARGET].mean(),
                "rate_test": test[TARGET].mean(),
                "n_test_stations_unseen": test.loc[
                    ~test.station_id.isin(train.station_id), "station_id"
                ].nunique(),
            }
        )
    pd.DataFrame(splits).to_csv(tables / "split_candidates.csv", index=False)
    for name in ["dist_log10_by_indicator", "sampling_gap_hist", "station_map"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        if name == "dist_log10_by_indicator":
            for indicator, group in clean.groupby("indicator", observed=True):
                ax.hist(np.log10(np.maximum(group.value, 1)), bins=30, alpha=0.5, label=indicator)
            ax.set(xlabel="log10(max(reported value /100mL, 1))", ylabel="Lab results")
            ax.legend()
        elif name == "sampling_gap_hist":
            ax.hist(gaps.dropna().clip(upper=60), bins=30)
            ax.set(xlabel="Days between samples (capped at 60)", ylabel="Intervals")
        else:
            points = ax.scatter(
                station.lon, station.lat, c=station.exceed_rate.astype(float), vmin=0, vmax=1
            )
            fig.colorbar(points, ax=ax, label="Exceedance fraction")
            ax.set(xlabel="Longitude (degrees)", ylabel="Latitude (degrees)")
        ax.set_title(name.replace("_", " "))
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png")
        plt.close(fig)
    write_json(root / "eda_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    print(run(Paths(parser.parse_args().root)))
