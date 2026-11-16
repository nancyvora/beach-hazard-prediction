# Spec: 02 – EDA of the cleaned beach monitoring results

Step 2 of the pipeline. It reads the clean parquet from step 01 and produces a small, reproducible set of tables, figures and numbers that the team uses to make its modeling decisions (target, split, features, baseline). Building lag features, saving a station-day table, and modeling are later steps (03+), not this one.

## Goal and user
Our 4-person team. We need evidence for `docs/decisions.md` (is the target usable, how imbalanced is it, how should we split by time, which signals are worth building features for). The same numbers must be re-creatable with one command, not from a notebook someone ran once.

## Inputs
- **Source:** `data/interim/results_clean.parquet` and `data/interim/cleaning_report.json` from spec 01. The raw CSV is never read here.
- **Format:** parquet, one row per lab result, columns and dtypes exactly as in spec 01 ("Outputs").
- **Size now:** about 229,000 rows (see `cleaning_report.json` for the exact count).
- **Size later:** up to 1M rows.

## Outputs
Everything goes under `reports/eda/`.

1. **Tables** (`reports/eda/tables/*.csv`):

| file | one row per | columns |
|---|---|---|
| `indicator_summary.csv` | indicator | `indicator`, `n_results`, `median`, `p95`, `pct_exact`, `pct_below_DL`, `pct_above_QL` |
| `target_summary.csv` | (single row) | `n_events`, `n_positive`, `positive_rate`, `n_unknown` |
| `yearly.csv` | year | `year`, `n_events`, `n_stations`, `exceed_rate` |
| `monthly.csv` | month (1–12) | `month`, `n_events`, `exceed_rate` |
| `county.csv` | county | `county`, `n_events`, `n_stations`, `exceed_rate` |
| `station_summary.csv` | station | `station_id`, `station_name`, `county`, `n_events`, `first_date`, `last_date`, `median_gap_days`, `exceed_rate` |
| `sampling_gaps.csv` | bucket of days since previous sample | `gap_bucket`, `n_events` |
| `persistence.csv` | previous-sample outcome × minimum gap | `prev_exceed`, `min_gap_days`, `n_events`, `exceed_rate` |
| `split_candidates.csv` | candidate cutoff year | `cutoff_year`, `n_train`, `n_test`, `rate_train`, `rate_test`, `n_test_stations_unseen` |

2. **Figures** (`reports/eda/figures/*.png`, titled, axes labeled with units):
   - `dist_log10_by_indicator.png`: histogram of `log10(value)`, one panel per indicator.
   - `events_and_rate_by_year.png` and `events_and_rate_by_month.png`.
   - `rate_by_county.png` (counties with at least 30 events).
   - `sampling_gap_hist.png`.
   - `station_map.png`: lat/lon scatter coloured by station exceed rate.

3. **`reports/eda/eda_report.json`**: every headline number (events, positives, positive rate, stations, beaches, date range, unknown events, rows collapsed into events).
4. A short printed summary in the same format as the report.

Note: the station-day events table is built **in memory only**. It is not saved, and no lag feature is saved. `qualifier`, `method`, `rl`, `mdl` and same-day values of other indicators are used for **description only**; they are leakage if used as model features.

## Definitions and rules
**Scope:** the target indicator is **Enterococcus**. Other indicators appear only in `indicator_summary.csv` and the distribution figure.

**Event:** one row per `station_id` × `date` for Enterococcus. If a station has several Enterococcus results on the same day (including `sample_type == 'Duplicate'`), keep the **worst** one: the highest `value`. Ties prefer `exact`, then `above_QL`, then `below_DL`. The number of results collapsed into events is reported.

**Exceedance:** the single-sample limit is **104 per 100 mL**. An event exceeds when:
- `censored == 'exact'`: `value > 104`.
- `censored == 'above_QL'`: `value >= 104` (the true value is at least the reported number).
- `censored == 'below_DL'`: does **not** exceed if `value <= 104`. If `value > 104` the outcome is **unknown**: the event is excluded from rates and counted as `n_unknown`.

All rates are computed over events with a known outcome. Any group with fewer than 30 events is marked `low_n` in the tables and left out of the figures.

**Time:** `year` and `month` come from `date`. Wet season is Nov–Mar, dry season is Apr–Oct.

**Sampling gaps and persistence:**
- For each station, sort events by `date`. `days_since_prev` is the gap to that station's previous event, and `prev_exceed` is whether it exceeded. A station's first event has both missing and is left out of the persistence table.
- Lab results are not available the same day, so a prediction for day *t* cannot see day *t−1*. Persistence is therefore reported twice: for any earlier sample (`min_gap_days = 0`) and only for earlier samples at least 2 days back (`min_gap_days = 2`). The second is the one spec 03 may use.
- Gap buckets: 1, 2–3, 4–7, 8–14, 15–30, >30 days.

**Split candidates:** for cutoff years 2012, 2014, 2016, 2018 and 2020, train is `year < cutoff` and test is `year >= cutoff`. Report sizes, exceed rates, and how many test stations never appear in train. This informs the split decision; it does not make it.

**Missing and low-count handling:** `hour` may be NA and is reported as such. No imputation of any kind.

## Acceptance criteria
- [ ] `python -m src.data.eda` on the spec 01 output creates every table, figure and `eda_report.json` listed above and exits with code 0.
- [ ] Every table has exactly the columns listed above, and every rate is within [0, 1].
- [ ] No station-day appears twice in the events table.
- [ ] A `below_DL` result with `value = 10` is not an exceedance. A `below_DL` result with `value = 500` is `unknown` and excluded from the rates.
- [ ] An `above_QL` result with `value = 24196` is an exceedance.
- [ ] A station-day with results 50 and 300 becomes one event with worst value 300.
- [ ] `n_events`, `n_positive` and `positive_rate` are identical in `target_summary.csv` and `eda_report.json`.
- [ ] `persistence.csv` contains both `min_gap_days = 0` and `min_gap_days = 2`, and a station's first event is not in it.
- [ ] The number of Enterococcus rows in the input equals the events plus the rows collapsed, as stated in `eda_report.json`.
- [ ] Running the step twice produces identical CSVs and an identical JSON (sorted outputs, fixed seed).
- [ ] pytest tests in `tests/test_eda.py` run each rule on a tiny hand-made parquet.

## Performance budget
- The current file (about 229k rows) runs in **under 60 s** with peak memory **under 2 GB** on a laptop, including figures.
- 1M rows runs in under 3 min and under 6 GB. Read only the columns needed and aggregate with `groupby`, not row loops.

## Out of scope
- Saving a station-day table, lag features, or any model input (spec 03).
- The final train/test split, baselines, models, calibration (specs 04–05).
- Imputing censored values (DL/2, Tobit, etc.).
- Rainfall, tide or other external data.
- Single-feature screening or feature selection.
- Interactive dashboards.
- Fixing data problems. Anything wrong in the cleaned file is reported back to spec 01, not patched here.

## Open questions
1. **Duplicates and resamples:** keep the worst value per station-day (current rule), or exclude `Duplicate` samples first?
2. **Other indicators as targets:** Enterococcus only (current rule), or also summarize exceedance for fecal coliform, total coliform and E. coli? If so, we need to confirm their limits.
3. **`below_DL` above the limit:** unknown and excluded (current rule), or treated as not exceeding?
4. **`above_QL` at exactly the limit:** counts as exceed (`>=`, current rule), or not?
5. **Where it lives:** a script in `src/data/eda.py` (current rule) with a thin notebook that calls it, or a notebook only?
6. **Station mix over time:** should the yearly rate also be shown for only the stations sampled across all years, to separate real change from changes in which stations are sampled?
7. **Reports in git:** commit only `eda_report.json`, or also the figures?
