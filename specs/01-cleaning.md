# Spec: 01 – Cleaning the beach monitoring results

Step 1 of the pipeline. It turns the raw data.ca.gov CSV into one clean, typed parquet file with **one row per lab result**. Aggregating to station-days, building the target, and adding lag features are later steps (02+), not this one.

## Goal and user
Our 4-person team. Every later step (EDA, features, model, API) reads this file. So each step starts from the same clean rows, and no one has to redo date parsing or censoring logic in their own notebook.

## Inputs
- **Source:** `beach-monitoring-results.csv` from data.ca.gov (resource `7bd961cf-abe4-433b-8033-378161237ff3`), saved as `data/raw/beach-monitoring-results.csv`. The raw file is never edited.
- **Format:** CSV, 93 columns, mixed types. Blank strings and `NA`/`N/A`/`NULL`/`NONE`/`NAN` mean missing.
- **Size now:** 230,268 rows, about 769 MB once loaded in pandas.
- **Size later:** the dataset is updated over time (2025 alone already has 6k+ events through April). Plan for up to 1M rows.

## Outputs
1. **`data/interim/results_clean.parquet`**: one row per kept lab result, with exactly these columns:

| column | dtype | from / meaning |
|---|---|---|
| `result_id` | int64 | `RESULTS id` (unique key) |
| `station_id` | int64 | `Station_id` |
| `station_name` | string | `Station_Name` |
| `beach_id` | int64 | `BeachName_id` |
| `beach_name` | string | `Beach_Name` |
| `county` | category | `County` |
| `lat`, `lon` | float64 | `Station_UpperLat`, `Station_UpperLon` |
| `date` | datetime64[ns] (date only, no tz) | `SampleDate` |
| `hour` | Int8 (nullable) | hour parsed from `StartTime`, 0–23, else NA |
| `sample_type` | category | `SampleType` (Results / Duplicate / AB411) |
| `indicator` | category | `Parameter` mapped to one of: Enterococcus, Fecal coliform, Total coliform, E. coli |
| `value` | float64 | number parsed from `Result` (the reported number, before any censoring adjustment) |
| `unit` | category | `Unit`, standardized to `MPN/100mL` or `CFU/100mL` |
| `censored` | category | `exact` / `below_DL` / `above_QL` (see rules) |
| `qualifier` | category | original `Qualifier` code, kept for auditing |
| `method` | category | `AnalysisMethod` |
| `rl`, `mdl` | float64 | reporting limit and method detection limit |
| static beach attributes | as loaded | `BeachType`, `WaterBodyType`, `WaterBodyClass`, `WaterShedName`, `BeachAccess`, `AB411Beach`, `Beach Length`, `AttendanceSummer`, `AttendanceWinter`, `SwimSeasonLength` (renamed to snake_case) |

2. **`data/interim/cleaning_report.json`**: rows in, rows out, and the number of rows dropped by each rule (keyed by rule name), plus the min/max `date`.
3. A short printed summary in the same format as the report.

Note: `qualifier`, `method`, `rl`, `mdl` and same-day `value`s of other indicators are kept **for cleaning and auditing only**. They are leakage if used as model features (EDA §11–12).

## Definitions and rules
**Columns dropped:** the 33 id/admin/timestamp columns and 3 agency-budget columns from EDA §2. Also the 9 field-observation columns (`Turbidity`, `WaterColor`, `Weather`, `TidalHeight`, `SurfHeight`, `StormDrainFlow`, `Odor`, `Distance`, `Direction`), which are ≥99.5% missing. Redundant copies are dropped too (`CountyName`, `Beach_County`, `*_LowerLat/Lon`, `Beach_*Lat/Lon`).

**Types**
- `date`: parse `SampleDate` with mixed formats, then normalize to midnight. Times are local California time with no time zone (we don't convert).
- `hour`: parse `StartTime` (formats vary: `09:30`, `1899-12-30 09:30:00`, `930`). If fewer than 50% of rows parse as time, fall back to `int(StartTime) // 100`. Anything outside 0–23 becomes NA.
- `value`: take `Result` as a string, strip `< > = ,` and whitespace, then convert to a number. Rows where that fails are dropped (rule `non_numeric_result`).

**Indicator mapping** (lowercase match on `Parameter`): contains `entero` → Enterococcus; `fecal` or `thermotolerant` → Fecal coliform; `total` + `coli` → Total coliform; `e. coli`/`e.coli`/`ecoli`/`escherichia` → E. coli; anything else → dropped (rule `other_parameter`).

**Censoring**: decided by `Qualifier` first, then by a leading `<`/`>` in `Result`:

| Qualifier | rows now | `censored` |
|---|---|---|
| `=` Equal to | 134,248 | exact |
| `E` Estimated | 10,974 | exact |
| `<` Less than, `<=`, `ND` Not detected | 78,598 | below_DL |
| `>` Greater than, `>=` | 6,439 | above_QL |
| `P`, `C`, `CG`, `NS`, missing (Not analyzed) | 9 | **dropped** (rule `unusable_qualifier`) |

- `value` stays **as reported**: `<10` is stored as `value=10, censored=below_DL`. This step does **not** substitute DL/2 or anything else.
- If an `ND` row has no number, it gets `value = rl` if available, otherwise it is dropped.

**Units**
- `MPN/100ml` and `CFU/100ml` are treated as equivalent; only the spelling is standardized.
- `Copies/100ml` (qPCR, 1,031 Enterococcus rows) is a different measurement and can't be compared to the 104 limit, so these rows are dropped (rule `qpcr_units`).

**Validation rules** (each failing row is dropped and counted in the report):
- `negative_value`: `value < 0` (28 rows now). `value == 0` is valid.
- `date_out_of_range`: `date` missing, earlier than **1946-01-01**, or later than the run date. This removes the 1900-01-04 placeholder dates (67 events) and the 1 unparseable date.
- `bad_coordinates`: `lat` not in [32, 42.1] or `lon` not in [-124.5, -114] (California). Currently 0 rows, but checked anyway.
- `duplicate_result_id`: if `RESULTS id` repeats, keep the first and drop the rest.

**Not dropped, only flagged in the report:** values above 1e6 (e.g. the E. coli maximum of 1e9 looks like a typo), and `SampleType == 'Duplicate'` (46 rows).

## Acceptance criteria
- [ ] Output parquet exists and has exactly the columns and dtypes in the table above.
- [ ] `result_id` is unique.
- [ ] No `value < 0` and no missing `value`.
- [ ] All `date` values are between 1946-01-01 and the run date, and none are missing.
- [ ] `indicator` takes only the 4 allowed values. `censored` takes only `exact`/`below_DL`/`above_QL`. `unit` contains no `Copies/100ml`.
- [ ] A raw `Result` of `<10` becomes `value == 10.0` and `censored == 'below_DL'`. `>24196` becomes `24196.0` / `above_QL`.
- [ ] `hour` is NA or in 0–23.
- [ ] `rows_in − sum(dropped per rule) == rows_out` in `cleaning_report.json`.
- [ ] On today's file, rows out is about 229,000 (230,239 after the EDA's drops, minus the qPCR rows and the 1900 dates). Any big change from that should be investigated.
- [ ] Running the step twice produces an identical parquet (deterministic).
- [ ] pytest tests in `tests/test_cleaning.py` run each rule on a tiny hand-made CSV.

## Performance budget
- The current file (230k rows) runs in **under 30 s** with peak memory **under 2 GB** on a laptop.
- 1M rows runs in under 2 min and under 6 GB. Read only the columns we need (`usecols`) to keep memory down.

## Out of scope
- Aggregating to station × day, choosing "worst value per day", building the target `exceed_Enterococcus`.
- Lag/history features, time-based train/test split.
- Imputing censored values (DL/2, Tobit, etc.).
- Converting CFU ↔ MPN, or using the qPCR rows.
- Adding rainfall/tide data.
- Loading into SQL / the FastAPI service (separate specs).

## Open questions
1. **`E` (Estimated) results:** treat as exact (current rule), or add an `is_estimated` flag?
2. **Extreme values:** should values above some cap (1e6? 1e7?) be dropped as typos instead of only flagged?
3. **qPCR rows:** drop (current rule), or keep with their unit so a later step can decide?
4. **Coordinates:** use station upper lat/lon (current rule), or the midpoint of upper and lower?
5. **Static beach attributes:** keep them in this file (current rule), or move them to a separate `stations.parquet` and join later, so this table stays narrow?
