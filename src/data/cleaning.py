"""Spec 01: typed lab results; preserve censoring boundaries."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import Paths, california_today, write_json

RENAME = {
    "RESULTS id": "result_id",
    "Station_id": "station_id",
    "Station_Name": "station_name",
    "BeachName_id": "beach_id",
    "Beach_Name": "beach_name",
    "County": "county",
    "Station_UpperLat": "lat",
    "Station_UpperLon": "lon",
    "SampleDate": "date",
    "StartTime": "hour",
    "SampleType": "sample_type",
    "Parameter": "indicator",
    "Result": "value",
    "Unit": "unit",
    "Qualifier": "qualifier",
    "AnalysisMethod": "method",
    "RL": "rl",
    "MDL": "mdl",
    "BeachType": "beach_type",
    "WaterBodyType": "water_body_type",
    "WaterBodyClass": "water_body_class",
    "WaterShedName": "water_shed_name",
    "BeachAccess": "beach_access",
    "AB411Beach": "ab411_beach",
    "Beach Length": "beach_length",
    "AttendanceSummer": "attendance_summer",
    "AttendanceWinter": "attendance_winter",
    "SwimSeasonLength": "swim_season_length",
}
REQUIRED = {
    "RESULTS id",
    "Station_id",
    "BeachName_id",
    "SampleDate",
    "Result",
    "Parameter",
    "Unit",
    "Station_UpperLat",
    "Station_UpperLon",
}
NULLS = {"", "NA", "N/A", "NULL", "NONE", "NAN", "\\N"}


def indicator_name(value):
    text = str(value).lower()
    if "entero" in text:
        return "Enterococcus"
    if "fecal" in text or "thermotolerant" in text:
        return "Fecal coliform"
    if "total" in text and "coli" in text:
        return "Total coliform"
    if any(s in text for s in ["e. coli", "e.coli", "ecoli", "escherichia"]):
        return "E. coli"
    return None


def parse_hour(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if re.fullmatch(r"\d{1,4}(?:\.0)?", text):
        number = int(float(text))
        hour, minute = divmod(number, 100)
        return hour if 0 <= hour <= 23 and 0 <= minute <= 59 else pd.NA
    match = re.search(r"(?:^|[ T])(\d{1,2}):(\d{2})(?::\d{2})?$", text)
    if match and int(match[1]) < 24 and int(match[2]) < 60:
        return int(match[1])
    return pd.NA


def clean_results(raw, run_date=None):
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")
    df = raw.reindex(columns=list(RENAME)).rename(columns=RENAME).copy()
    for col in df:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].astype("string").str.strip()
            df.loc[df[col].str.upper().isin(NULLS), col] = pd.NA
    reported = df.value.astype("string")
    # NA is a qualifier code for not analyzed, not an exact zero result.
    # Preserve the literal NA (not analyzed) qualifier so it cannot be rescued
    # by a numeric boundary in Result. Other missing tokens may use that fallback.
    qualifier = (
        raw.get("Qualifier", pd.Series(pd.NA, index=raw.index))
        .astype("string")
        .str.strip()
        .str.upper()
    )
    qualifier = qualifier.mask(qualifier.isin(NULLS - {"NA"}))
    fallback = reported.str.extract(r"^\s*(<=|>=|<|>|=)", expand=False)
    qualifier = qualifier.fillna(fallback)
    df["qualifier"] = qualifier
    df["censored"] = qualifier.map(
        {
            "=": "exact",
            "E": "exact",
            "<": "below_DL",
            "<=": "below_DL",
            "ND": "below_DL",
            ">": "above_QL",
            ">=": "above_QL",
        }
    )
    df["value"] = pd.to_numeric(reported.str.replace(r"[<>=,\s]", "", regex=True), errors="coerce")
    for col in ["rl", "mdl", "lat", "lon", "result_id", "station_id", "beach_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    nd = qualifier.eq("ND").fillna(False) & df.value.isna()
    df.loc[nd, "value"] = df.loc[nd, "rl"]
    df["date"] = (
        pd.to_datetime(df.date, format="mixed", errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    df["hour"] = df.hour.map(parse_hour).astype("Int8")
    df["indicator"] = df.indicator.map(indicator_name)
    df["unit"] = (
        df.unit.astype("string")
        .str.lower()
        .str.replace(" ", "", regex=False)
        .map({"mpn/100ml": "MPN/100mL", "cfu/100ml": "CFU/100mL", "copies/100ml": "Copies/100mL"})
    )
    report = {"rows_in": len(raw), "dropped": {}}

    def drop(rule, bad):
        nonlocal df
        bad = bad.fillna(True)
        report["dropped"][rule] = int(bad.sum())
        df = df.loc[~bad].copy()

    drop("other_parameter", df.indicator.isna())
    drop("unsupported_sample_type", ~df.sample_type.isin(["Results", "Duplicate", "AB411"]))
    drop("unusable_qualifier", df.censored.isna())
    drop("non_numeric_result", df.value.isna() | ~np.isfinite(df.value.astype(float)))
    drop("negative_value", df.value < 0)
    drop("qpcr_units", df.unit.eq("Copies/100mL"))
    drop("unsupported_unit", df.unit.isna())
    drop(
        "date_out_of_range",
        df.date.isna()
        | (df.date < "1946-01-01")
        | (df.date > pd.Timestamp(run_date or california_today())),
    )
    drop("bad_coordinates", ~df.lat.between(32, 42.1) | ~df.lon.between(-124.5, -114))
    bad_ids = pd.Series(False, index=df.index)
    for col in ["result_id", "station_id", "beach_id"]:
        bad_ids |= df[col].isna() | ~np.isfinite(df[col]) | (df[col] % 1 != 0) | (df[col] <= 0)
    drop("bad_identifiers", bad_ids)
    drop("duplicate_result_id", df.result_id.duplicated(keep="first"))
    for col in ["result_id", "station_id", "beach_id"]:
        df[col] = df[col].astype("int64")
    for col in ["value", "rl", "mdl", "lat", "lon"]:
        df[col] = df[col].astype("float64")
    for col in ["station_name", "beach_name"]:
        df[col] = df[col].astype("string")
    for col in ["county", "indicator", "sample_type", "unit", "censored", "qualifier", "method"]:
        df[col] = df[col].astype("category")
    report.update(
        rows_out=len(df),
        min_date=df.date.min(),
        max_date=df.date.max(),
        extreme_values=int((df.value > 1e6).sum()),
        duplicate_samples=int(df.sample_type.eq("Duplicate").sum()),
    )
    assert report["rows_in"] - sum(report["dropped"].values()) == len(df)
    return df.sort_values("result_id").reset_index(drop=True), report


def run(paths, source=None):
    source = Path(source or paths.raw)
    raw = pd.read_csv(
        source, dtype="string", keep_default_na=False, usecols=lambda name: name in RENAME
    )
    clean, report = clean_results(raw)
    paths.interim.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(paths.interim / "results_clean.parquet", index=False)
    write_json(paths.interim / "cleaning_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    print(run(Paths(args.root), args.input))


if __name__ == "__main__":
    main()
