import numpy as np
import pandas as pd
import pytest

from src.data.events import TARGET, event_labels, station_catalog
from src.features.build import FEATURE_COLUMNS, build_features


def test_cutoff_five_event_example(five_events):
    features = build_features(
        five_events, five_events[["station_id", "date"]], station_catalog(five_events)
    )
    row = features.iloc[-1]
    assert row.prev_log10 == 3 and row.prev_exceed == 1 and row.days_since_prev == 2
    assert row.roll3_log10 == 2 and row.roll3_max_log10 == 3
    assert row.n_roll3 == row.n_prev_samples == row.n_prev_known == 3
    assert row.station_hist_rate == pytest.approx(1 / 3)
    assert features.iloc[0].no_history == 1 and pd.isna(features.iloc[0].prev_log10)


def test_future_mutation_cannot_change_features(five_events):
    catalog = station_catalog(five_events)
    query = five_events[["station_id", "date"]].iloc[[-1]]
    original = build_features(five_events, query, catalog)
    changed = five_events.copy()
    changed.loc[changed.date > "2024-01-05", "value"] = 99999999
    changed.loc[changed.date > "2024-01-05", TARGET] = 0
    pd.testing.assert_frame_equal(original, build_features(changed, query, catalog))


def test_unknown_history_denominator_and_station_isolation(five_events):
    five_events.loc[0, TARGET] = pd.NA
    another = five_events.copy().assign(station_id=999, beach_id=99, station_name="other")
    history = pd.concat([five_events, another], ignore_index=True)
    query = five_events[["station_id", "date"]].iloc[[-1]]
    row = build_features(history, query, station_catalog(history)).iloc[0]
    assert row.n_prev_samples == 3 and row.n_prev_known == 2 and row.station_hist_rate == 0.5


def test_no_current_result_needed_and_no_history(five_events):
    query = pd.DataFrame({"station_id": [101], "date": pd.to_datetime(["2023-12-30"])})
    row = build_features(five_events, query, station_catalog(five_events)).iloc[0]
    assert row.no_history == 1 and row.n_prev_samples == 0
    assert np.isnan(row.station_hist_rate)
    assert TARGET not in FEATURE_COLUMNS and "value" not in FEATURE_COLUMNS


def test_censored_label_boundaries():
    frame = pd.DataFrame(
        {
            "value": [104, 105, 10, 500, 24196, 50, 104],
            "censored": [
                "exact",
                "exact",
                "below_DL",
                "below_DL",
                "above_QL",
                "above_QL",
                "above_QL",
            ],
        }
    )
    actual = event_labels(frame)
    assert actual[:3].tolist() == [0, 1, 0]
    assert pd.isna(actual[3]) and actual[4] == 1 and pd.isna(actual[5]) and actual[6] == 1


def test_duplicate_history_rejected(five_events):
    history = pd.concat([five_events, five_events.iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate history"):
        build_features(history, five_events[["station_id", "date"]], station_catalog(five_events))
