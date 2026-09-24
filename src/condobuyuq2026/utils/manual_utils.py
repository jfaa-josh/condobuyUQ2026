"""Shared backend for the manual, one-off scripts in run/manual/.

Not part of the analysis package's own logic (that's the rest of condobuyuq2026) — these are
helpers for data-pulling/model-building scripts specifically: talking to external APIs (BLS,
FRED, ...), and saving what comes back into data/raw/. Keep this modular by data source (one
fetch_<source>_* function per API) so future manual scripts pulling from the same source reuse
it, rather than each script re-implementing its own request/pagination/saving logic. Every
fetch_<source>_* function returns the same row shape regardless of source -- a list of
{"year", "period" ("M01".."M12"), "periodName", "value"} dicts -- so everything downstream
(filter_monthly, save_raw_csv/load_raw_csv, to_monthly_series, and beyond that fit_annual_ou etc.
in ou_fitting.py) works identically no matter which source a given model was built from.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from condobuyuq2026.paths import DATA_RAW_DIR, MODELS_DIR
from condobuyuq2026.paths import model_fit_path as _model_fit_path

BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
# BLS API v2's per-request limits: a registration key raises them, but doesn't require them —
# a small one-off historical pull (a handful of requests) works fine without a key at all.
BLS_MAX_YEARS_WITH_KEY = 20
BLS_MAX_YEARS_WITHOUT_KEY = 10


def _year_chunks(start_year: int, end_year: int, max_span: int) -> list[tuple[int, int]]:
    """Splits [start_year, end_year] into chunks no wider than max_span years, to respect the
    BLS API's per-request span limit."""
    chunks = []
    chunk_start = start_year
    while chunk_start <= end_year:
        chunk_end = min(chunk_start + max_span - 1, end_year)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + 1
    return chunks


def fetch_bls_series(series_id: str, start_year: int, end_year: int, api_key: str | None = None) -> list[dict[str, Any]]:
    """Pulls one BLS series over [start_year, end_year], transparently chunking requests to
    respect the API's per-request year-span limit. Works with or without a registration key
    (pass api_key if you have one) — omits it from the request entirely if none is given, using
    the lower unregistered limits instead of failing. Returns every observation (all
    periodicities BLS reports for this series — filter with filter_monthly() if you only want
    monthly points), sorted chronologically ascending regardless of how BLS ordered each chunk
    internally.
    """
    max_span = BLS_MAX_YEARS_WITH_KEY if api_key else BLS_MAX_YEARS_WITHOUT_KEY
    all_points: list[dict[str, Any]] = []
    for chunk_start, chunk_end in _year_chunks(start_year, end_year, max_span):
        payload = {
            "seriesid": [series_id],
            "startyear": str(chunk_start),
            "endyear": str(chunk_end),
        }
        if api_key:
            payload["registrationkey"] = api_key
        response = requests.post(BLS_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(
                f"BLS API request failed for {series_id} {chunk_start}-{chunk_end}: {body.get('message')}"
            )
        series_list = body["Results"]["series"]
        if not series_list or not series_list[0].get("data"):
            raise RuntimeError(f"BLS API returned no data for {series_id} {chunk_start}-{chunk_end}")
        all_points.extend(series_list[0]["data"])

    all_points.sort(key=lambda p: (int(p["year"]), p["period"]))
    return all_points


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(series_id: str) -> list[dict[str, Any]]:
    """Pulls one FRED series' entire available history via its CSV endpoint -- no API key, no
    rate limit, no date-range parameter (FRED always returns everything it has; slice the result
    afterward if you need less). Returns the same year/period/periodName/value row shape
    fetch_bls_series does (see module docstring), so the rest of a manual script's pipeline
    (filter_monthly, save_raw_csv/load_raw_csv, to_monthly_series, ou_fitting.fit_annual_ou, ...)
    doesn't need to know or care which source a series came from.

    Values are kept as FRED wrote them, unconverted -- including FRED's own placeholder for a
    missing/not-yet-published observation ("."), which parse_numeric_value (used by
    to_monthly_series) already treats as NaN the same way it does BLS's "-". This keeps data/raw/'s
    CSV a faithful copy of what the source actually returned, same as for BLS.

    Only meaningfully supports series FRED reports at MONTHLY-or-finer frequency -- period is
    always written as "M<month>", so a quarterly/annual series would be mislabeled.
    """
    series = pd.read_csv(f"{FRED_CSV_URL}?id={series_id}", index_col=0, parse_dates=True).iloc[:, 0]
    if series.empty:
        raise RuntimeError(f"FRED returned no data for series {series_id!r}")
    return [
        {
            "year": str(date.year),
            "period": f"M{date.month:02d}",
            "periodName": date.strftime("%B"),
            "value": str(value),
        }
        for date, value in series.items()
    ]


def filter_monthly(data_points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keeps only monthly observations (period M01-M12), dropping annual averages (M13) and any
    other non-monthly periodicity a source might include for a given series. A no-op for sources
    that only ever report monthly periods to begin with (e.g. fetch_fred_series) -- kept in the
    pipeline anyway so every manual script follows the same fetch -> filter_monthly -> save_raw_csv
    shape regardless of source."""
    return [p for p in data_points if p["period"].startswith("M") and p["period"] != "M13"]


def raw_csv_path(model_name: str) -> Path:
    """Path to data/raw/<model_name>.csv, whether or not it's been written yet."""
    return DATA_RAW_DIR / f"{model_name}.csv"


def save_raw_csv(data_points: list[dict[str, Any]], model_name: str) -> Path:
    """Saves data_points (year/period/periodName/value rows) to data/raw/<model_name>.csv,
    creating data/raw/ if needed. Returns the path written."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = raw_csv_path(model_name)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "period", "periodName", "value"])
        writer.writeheader()
        for point in data_points:
            writer.writerow({key: point.get(key) for key in ["year", "period", "periodName", "value"]})
    return out_path


def load_raw_csv(model_name: str) -> list[dict[str, Any]]:
    """Reads data/raw/<model_name>.csv back into the same list-of-dicts shape save_raw_csv wrote
    (string-valued year/period/periodName/value, matching what fetch_bls_series returns -- including
    BLS's "-" placeholder for suppressed/missing observations, untouched). Lets a manual script work
    from an already-pulled file on later runs without calling the API again."""
    in_path = raw_csv_path(model_name)
    with in_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def model_plots_dir(model_name: str) -> Path:
    """Returns models/<model_name>/plots (creating it if needed) — where a manual script should
    save its sanity-check plots for that model."""
    path = MODELS_DIR / model_name / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_fit_path(model_name: str) -> Path:
    """Returns models/<model_name>/fit.json (creating models/<model_name>/ if needed) — where a
    manual script should save its fitted model parameters (see ou_fitting.save_fit/load_fit),
    alongside its plots. Path convention lives in condobuyuq2026.paths (shared with input_deck.py's
    read side) -- this just also ensures the directory exists, which the read side shouldn't do."""
    path = _model_fit_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_numeric_value(raw: str) -> float:
    """Parses a raw value string to float. Sources use different non-numeric placeholders for a
    suppressed/missing/not-yet-published observation (BLS: "-", FRED: "."); either way it comes
    back as NaN rather than raising, so callers can decide how to handle a gap (drop it, leave it
    as a visual gap, etc.) instead of crashing."""
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


def to_monthly_series(data_points: list[dict[str, Any]]) -> pd.Series:
    """Converts monthly data points (year/period/value dicts, from any fetch_<source>_* function
    -- see module docstring) into a float pandas Series indexed by month-start Timestamps,
    reindexed to a *complete* monthly calendar via asfreq("MS"). Any suppressed/missing value
    (parsed to NaN, see parse_numeric_value) and any real gap month the reindex inserts both come
    back as NaN.

    The reindex matters beyond just filling gaps for display: pandas' .shift(n) shifts by row
    position, not by calendar time, so a missing month would otherwise silently misalign every
    later lag/diff calculation by one month against what the calendar actually says -- reindexing
    to a complete calendar first is what makes .shift(n) mean "n months," not "n rows."
    """
    index = pd.PeriodIndex(
        [f"{p['year']}-{p['period'][1:]}" for p in data_points], freq="M"
    ).to_timestamp()
    values = [parse_numeric_value(p["value"]) for p in data_points]
    return pd.Series(values, index=index).sort_index().asfreq("MS")
