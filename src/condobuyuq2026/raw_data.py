"""Low-level storage/parsing for a founding process's raw historical data (data/raw/<model_name>.csv)
-- split out of utils/manual_utils.py so that carrying_costs.py (which needs to read raw home_price
history for market_value's own pre-closing averaging window, see that module's docstring) can import
this without a circular import back through manual_utils.py (which itself imports FROM
carrying_costs.py for the reassessment-scheduling helpers). Kept separate from utils/manual_utils.py's
own fetch_<source>_* functions, which are genuinely manual-script-only (talk to external APIs) --
this module is just data already on disk.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from condobuyuq2026.paths import DATA_RAW_DIR


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
    -- see utils.manual_utils's module docstring) into a float pandas Series indexed by month-start
    Timestamps, reindexed to a *complete* monthly calendar via asfreq("MS"). Any suppressed/missing
    value (parsed to NaN, see parse_numeric_value) and any real gap month the reindex inserts both
    come back as NaN.

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


def load_monthly_series(model_name: str) -> pd.Series:
    """load_raw_csv(model_name) -> to_monthly_series in one step -- the shape carrying_costs.py
    needs to look up a founding process's own REAL historical level at a given calendar month
    (see project_market_value_schedule's pre-closing averaging window)."""
    return to_monthly_series(load_raw_csv(model_name))
