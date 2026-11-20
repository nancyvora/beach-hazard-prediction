"""Implemented feature ablations; the full spec 05 experiment matrix is future work."""

ABLATIONS = {
    "no_lags": lambda c: (
        not (
            c.startswith(("prev_", "roll3_", "n_prev_"))
            or c in {"days_since_prev", "n_roll3", "no_history"}
        )
    ),
    "no_station_rate": lambda c: c not in {"station_hist_rate", "n_prev_known"},
    "no_location": lambda c: c not in {"county", "lat", "lon"},
    "no_calendar": lambda c: c not in {"month", "dow", "wet_season", "doy_sin", "doy_cos"},
}
