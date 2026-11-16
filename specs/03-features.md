# Spec: 03 – Features for Enterococcus exceedance prediction

Step 3 of the pipeline. It reads the clean lab results from step 01, saves one Enterococcus event per station-day using the definitions in step 02, and builds lag, seasonality and station-history features. Training models, choosing the final time split and evaluating predictions are later steps (04–05), not this one.

## Goal and user

Our 4-person team. Steps 04–05 need one reproducible modeling table, and step 06 needs the same feature calculations for a requested station and date. The main signals suggested by the EDA are previous Enterococcus results, recent concentrations, station history and seasonality. They must be calculated using only information that would have been available on the prediction date.

## Inputs

- **Source:** `data/interim/results_clean.parquet` and `data/interim/cleaning_report.json` from spec 01. The raw CSV is never read here.
- **Format:** parquet, one row per lab result, with the columns and dtypes defined in spec 01. Required columns are `result_id`, `station_id`, `station_name`, `beach_id`, `beach_name`, `county`, `lat`, `lon`, `date`, `indicator`, `value`, `unit`, `censored`, `qualifier` and `sample_type`.
- **Size now:** about 229,000 clean lab results, before filtering to Enterococcus and collapsing station-days. Read the exact count from the cleaning report; the EDA summary's 85,024 Enterococcus events is a reference, not a required output count after cleaning changes.
- **Size later:** support at least 1M clean lab results. Do not hard-code the current row count or last date.
- **Decisions:** follow `docs/decisions.md` when populated. The defaults below are proposed rules for this spec; unresolved choices stay in Open questions.

## Outputs

1. **`data/processed/station_days.parquet`**: one row per `station_id` × `date` for Enterococcus, sorted by those keys:

| column | dtype | meaning |
|---|---|---|
| `station_id`, `beach_id` | int64 | station and beach identifiers |
| `station_name`, `beach_name` | string | readable names, not model features |
| `county` | category | location category |
| `lat`, `lon` | float64 | station coordinates from spec 01 |
| `date` | datetime64[ns] | prediction / sample date, normalized to midnight |
| `result_id` | int64 | selected result's ID, for auditing |
| `value`, `unit`, `censored`, `qualifier`, `sample_type` | as in spec 01 | selected result, for target construction and auditing only |
| `n_results` | int64 | Enterococcus lab results collapsed into this event |
| `exceed_Enterococcus` | Int8 (nullable) | 0 / 1 / NA using the rules below |

2. **`data/processed/features.parquet`**: the same station-day keys, identifiers, names, `date`, `county`, `lat`, `lon` and nullable target, followed by these feature columns:

| column | dtype | meaning |
|---|---|---|
| `month` | int8 | calendar month, 1–12 |
| `dow` | int8 | day of week, Monday=0 through Sunday=6 |
| `wet_season` | int8 | 1 for Nov–Mar, otherwise 0 |
| `doy_sin`, `doy_cos` | float64 | cyclic day-of-year features |
| `prev_log10` | float64 | log-transformed reported value of the latest eligible station-day |
| `prev_exceed` | Int8 (nullable) | that same event's target, possibly unknown |
| `days_since_prev` | float64 | days from prediction date to that eligible event's sample date |
| `roll3_log10` | float64 | mean of log-transformed values over up to 3 latest eligible station-days |
| `roll3_max_log10` | float64 | maximum of those log-transformed values |
| `n_roll3` | int8 | number of events used in the rolling features, 0–3 |
| `station_hist_rate` | float64 | expanding fraction of eligible station-days with known targets that exceeded |
| `n_prev_samples` | int64 | number of all eligible station-days, including unknown targets |
| `n_prev_known` | int64 | number of eligible station-days with known targets |
| `no_history` | int8 | 1 when `n_prev_samples == 0`, otherwise 0 |

Also save `history_cutoff` and `latest_history_date` as datetime audit columns. They are **not** model features. Do not copy the current event's `value`, qualifier, sample type or other lab fields into this file.

3. **`data/processed/feature_report.json`**: input rows, Enterococcus rows, station-days, rows collapsed, known/unknown targets, positives, stations, date range, rows with no eligible history, rows with fewer than 3 eligible events, and missing counts per feature. Include the 2-day delay, threshold and target-rule version, transform and rolling-window definitions, and explicit `feature_columns`, `id_columns`, `target_columns` and `audit_columns` lists.
4. A short printed summary in the same format as the report.

The default `feature_columns` list contains `county`, `lat`, `lon` and the engineered feature columns above. Identifiers, names, dates, the target and audit columns must not be selected automatically as predictors. Any later categorical encoding must be fitted on training data only.

## Definitions and rules

**Station-day events and target**

- Filter to `indicator == 'Enterococcus'`. Do not create targets from other indicators or from days without a measurement.
- Reuse the event definition from spec 02: highest reported `value` per station-day; ties prefer `exact`, then `above_QL`, then `below_DL`. Break remaining ties by the smallest `result_id`, so the result does not depend on input row order.
- Include the sample types retained by spec 01. A `Duplicate` is collapsed into its station-day, not counted as another historical event.
- Use the selected event's target rules from spec 02:

| `censored` | condition | `exceed_Enterococcus` |
|---|---|---|
| `exact` | `value > 104` | 1 |
| `exact` | `value <= 104` | 0 |
| `above_QL` | `value >= 104` | 1 |
| `above_QL` | `value < 104` | NA: the bound alone cannot establish exceedance |
| `below_DL` | `value <= 104` | 0 |
| `below_DL` | `value > 104` | NA |

The `above_QL` rule at exactly 104 follows spec 02's current convention; it does not resolve the original `>` versus `>=` distinction. Keep `qualifier` for auditing and list that boundary decision in Open questions. Estimated results follow spec 01's current mapping to `exact`. Do not silently introduce a different label definition in this step.

Keep unknown-target events in both output tables. They may contribute a reported-value history feature, but never count as negative labels or enter the denominator of `station_hist_rate`. Steps 04–05 must exclude unknown targets from supervised training and evaluation.

**Prediction time and available history**

- For a prediction on date `t`, `history_cutoff = t - 2 calendar days`. Only events at the **same station** with `date <= history_cutoff` may contribute to any history feature. The boundary is inclusive.
- This is the team's date-level assumption for the 24–48 h lab turnaround, not a measured publication timestamp. Do not use `hour` to claim more precise availability. If actual release times become available later, both training and serving must enforce them too.
- Sort station-days by sample date before looking up history. A plain `groupby().shift(1)` is insufficient: yesterday's sample is ineligible even if it is the previous row.
- The latest eligible event is the latest event satisfying the cutoff, not necessarily the immediately preceding sample. `prev_exceed` refers to that event; do not substitute an older known label if its own label is unknown.
- `days_since_prev` is measured from `t`, not from the cutoff. With history present it must be at least 2.
- Build each row using its own cutoff, including during validation and testing. Earlier validation/test measurements may enter later history once eligible, representing a system that receives new results over time. This is not a frozen-history, multi-day forecast; steps 04–05 must state that evaluation assumption.

**Concentration lags and rolling features**

- Transform eligible reported values as `log10(max(value, 1.0))`, so zero is finite and values below 1 map to 0. This is a fixed transform, not a parameter fitted on all data.
- `prev_log10` is the transformed value of the latest eligible event.
- `roll3_log10` is the arithmetic mean of the transformed values from the latest **up to 3 distinct station-days**; `roll3_max_log10` is their maximum. This is a 3-sample window, not a 3-day window, and it is not the log of the raw-value mean.
- Use one or two available events when fewer than three exist and record `n_roll3`. Do not pad or backfill the window with future events.
- For censored observations, use the reported boundary retained by spec 01. No DL/2 substitution or upper-tail estimate is introduced. These features describe reported values/bounds, not exact latent concentrations; record that limitation in the report.

**Station history rate**

- `n_prev_samples` counts all eligible station-days at the station.
- `n_prev_known` counts only those with a 0/1 target.
- `station_hist_rate = sum(eligible known targets) / n_prev_known`. Use NA when the denominator is zero.
- This is an expanding rate recomputed as of each prediction date. Never calculate a station's full-dataset mean and join it back onto past rows; that would expose future labels.
- No smoothing or global/county target prior in this step. Such a prior would need its own training-only or historical fitting rules.

**Seasonality and location**

- Derive calendar features from requested date `t`, without needing a new sample or lab result.
- Let `d` be day of year starting at 1 and `L` be 365, or 366 in a leap year. Set `doy_sin = sin(2*pi*(d-1)/L)` and `doy_cos = cos(2*pi*(d-1)/L)`.
- Use `county`, `lat` and `lon` as location features under the explicit assumption that the station's registered location is known when predicting. Do not infer location from current-day lab measurements. Conflicting station locations or beach mappings must be reported for review, not silently resolved using the latest future record.
- Keep station/beach IDs and names for grouping, joining and reporting. Do not target-encode them or include them in the default predictors.
- Exclude other beach attributes from the default feature set until the team confirms their usefulness and historical availability. In particular, undated visitor totals may describe a later period.

**Missing history, resamples and inference**

- With no eligible history: `n_prev_samples=0`, `n_prev_known=0`, `n_roll3=0`, `no_history=1`; concentration lags, `prev_exceed`, gap, rate and `latest_history_date` are NA. Calendar features remain available. Do not fill missing history with zero or an all-data average.
- If eligible events exist but all targets are unknown, concentration lags remain available and `no_history=0`, while `station_hist_rate` stays NA.
- Do not drop stations based on their total future sample count or positive rate. Small-history stations retain their counts and missing-history indicators.
- The clean schema has no confirmed resample flag. Keep eligible station-days by default; `Duplicate` does not establish resampling. Do not use current-day `sample_type`, sampling gaps that require future observations, or a guessed resample label as predictors. A confirmed routine-only policy requires an agreed input flag and matching history rules.
- Expose the feature calculation as a reusable function taking station-day history, query station/date pairs, and known station metadata. The batch command uses observed event dates as queries; the future API can request a date with no current measurement. The latter must produce the same feature schema without requiring a target or current-day result.
- No global imputation, scaling, feature selection or categorical fitting here. Those belong inside the training pipeline and are fitted on the training partition only.

## Acceptance criteria

- [ ] `python -m src.features.build` on the spec 01 outputs creates both parquet files and `feature_report.json`, prints the summary and exits with code 0.
- [ ] Both tables have the specified columns/dtypes, are sorted by station/date, and have unique `station_id` × `date` keys. Their key sets are identical.
- [ ] Enterococcus input rows equal station-days plus rows collapsed. Known-target events plus unknown-target events equal all station-days.
- [ ] Event selection and labels match spec 02 for shared fixtures; ties are deterministic after shuffling input rows. Any correction to upstream target rules is coordinated rather than made only here.
- [ ] Target fixtures cover `exact=104` → 0, `exact=105` → 1, `below_DL=10` → 0, `below_DL=500` → NA, `above_QL=24196` → 1, and `above_QL=50` → NA. The current `above_QL=104` convention is explicitly tested.
- [ ] Every nonmissing `latest_history_date <= history_cutoff`, and every nonmissing `days_since_prev >= 2`. Nothing from `t-1`, `t` or a future date enters a history feature.
- [ ] A five-event fixture at one station on Jan 1, 3, 5, 6 and 7, with exact values 10, 100, 1000, 10000 and 100000, gives the following for Jan 7: cutoff Jan 5, `prev_log10=3`, `prev_exceed=1`, `days_since_prev=2`, `roll3_log10=2`, `roll3_max_log10=3`, `n_roll3=3`, `n_prev_samples=n_prev_known=3`, and `station_hist_rate=1/3`.
- [ ] Changing Jan 6 or Jan 7 values in that fixture, or appending later events, does not change Jan 7's feature values. Changing an earlier target does not change features on dates before that target becomes available.
- [ ] A station's first event, and a query with no eligible history, have the specified NA lags, zero counts and `no_history=1`. A second event only one day after the first still has no eligible history.
- [ ] Unknown historical labels do not enter the rate denominator; an unknown current target does not remove the row. Histories from a second station never affect the first station's features.
- [ ] `station_hist_rate` is NA or within [0,1]; zero concentrations give finite log features; month and wet-season boundaries and leap-day cyclic features match their formulas.
- [ ] Building a row as an inference query without a current-day measurement gives the same predictors as the corresponding batch row when provided the same eligible history and station metadata.
- [ ] The feature whitelist excludes the target, IDs/names, audit columns, current-day lab values, qualifiers, methods, limits, sample type and same-day results of other indicators.
- [ ] Running twice produces identical table contents and JSON. pytest fixtures in `tests/test_features.py` cover the rules above without downloading real data.

## Performance budget

- The current clean file (about 229k lab results) runs in **under 60 s** with peak memory **under 2 GB** on a laptop.
- 1M clean lab results runs in under 3 min and under 6 GB. Read only needed columns; sort once, then use cumulative statistics, rolling calculations and cutoff-aware lookups. Do not rescan a station's full history for each row.
- These are implementation acceptance targets, not timings already measured for this spec.

## Out of scope

- Recleaning the raw CSV or silently fixing issues in spec 01's output.
- Changing spec 02's event-selection or target policy independently.
- The final train/validation/test split, fitting encoders/imputers/scalers, baselines, models, calibration, metrics or MLflow logging (specs 04–05).
- Current-day lab-derived features, high-cardinality ID encoding, or full-dataset target statistics.
- External rainfall, tide, wave or river-flow data; lagged results of other bacterial indicators.
- Inventing labels for unsampled dates, aggregating station predictions to a beach risk, or implementing the API (spec 06).
- Automatically excluding guessed resamples, censoring models or interpolation between sample dates.

## Open questions

1. **Target boundary:** spec 02 currently labels `above_QL=104` positive, but original `>104` and `>=104` do not mean the same thing for a strictly `>104` target. Should both steps use the retained `qualifier` to mark `>=104` unknown? Any change must update spec 02 and its tests too.
2. **Worst result:** keep spec 02's highest reported boundary (current rule), or use an interval-aware station-day outcome? For example, `<500` can outrank an exact 300 even though the exact result already establishes an exceedance.
3. **Censored / estimated concentrations:** keep reported boundaries and spec 01's treatment of estimates (current rule), or compare alternate handling in a later ablation?
4. **Resamples:** keep all cleaned station-days (current rule), or restrict history/training to confirmed routine samples once a reliable flag is available?
5. **History freshness:** allow all eligible historical results with an explicit age feature (current rule), or expire concentration lags after a maximum gap such as 30 days?
6. **Feature extensions:** add smoothed station/county priors, other-indicator lags or more static beach attributes only after agreeing on time-safe calculation and inference availability?
7. **Evaluation mode:** use the rolling availability assumption above (current rule), or additionally evaluate a forecast with history frozen at the split date? The split dates themselves remain for spec 04.
