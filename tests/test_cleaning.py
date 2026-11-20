import pandas as pd
import pytest

from src.data.cleaning import clean_results, parse_hour


@pytest.mark.parametrize(
    "raw,expected", [("09:30", 9), ("1899-12-30 09:30:00", 9), ("930", 9), ("0000", 0)]
)
def test_hour_formats(raw, expected):
    assert parse_hour(raw) == expected


def test_censoring_and_reconciliation(raw_rows):
    raw_rows["Result"] = ["<10", ">24196", "-2", "12", "15"]
    raw_rows["Qualifier"] = ["<", ">", "=", "=", "="]
    raw_rows.loc[3, "SampleDate"] = "1900-01-04"
    raw_rows.loc[4, "Unit"] = "Copies/100ml"
    clean, report = clean_results(raw_rows, run_date="2026-11-19")
    assert clean.value.tolist() == [10, 24196]
    assert clean.censored.tolist() == ["below_DL", "above_QL"]
    assert report["rows_in"] - sum(report["dropped"].values()) == report["rows_out"] == 2


def test_unknown_qualifier_not_safe(raw_rows):
    raw_rows["Qualifier"] = ["NS", "P", "NA", "CG", "ND"]
    raw_rows["Result"] = [0, 0, 0, 0, ""]
    raw_rows["RL"] = 10
    clean, _ = clean_results(raw_rows)
    assert len(clean) == 1
    assert clean.iloc[0].qualifier == "ND" and clean.iloc[0].value == 10


def test_bad_ids_duplicates_and_coordinates(raw_rows):
    raw = pd.concat([raw_rows, raw_rows.iloc[[0]]], ignore_index=True)
    raw["Station_id"] = raw["Station_id"].astype(object)
    raw.loc[1, "Station_id"] = "invalid"
    raw.loc[2, "Station_UpperLat"] = 0
    clean, report = clean_results(raw)
    assert clean.result_id.is_unique
    assert report["dropped"]["duplicate_result_id"] == 1
    assert report["dropped"]["bad_coordinates"] == 1
    assert report["dropped"]["bad_identifiers"] == 1


def test_missing_schema_raises():
    with pytest.raises(ValueError, match="Missing raw columns"):
        clean_results(pd.DataFrame({"Result": [1]}))


def test_not_analyzed_boundary_and_qc_are_not_training_samples(raw_rows):
    raw_rows["Qualifier"] = ["NA", "=", "=", "=", "="]
    raw_rows["Result"] = ["<10", 10, 10, 10, 10]
    raw_rows.loc[1, "SampleType"] = "QC Check"
    clean, report = clean_results(raw_rows)
    assert len(clean) == 3
    assert report["dropped"]["unsupported_sample_type"] == 1
    assert report["dropped"]["unusable_qualifier"] == 1
