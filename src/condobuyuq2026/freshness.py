"""Checks whether each forecast_models entry's underlying raw data is stale (behind its own
EXPECTED reporting lag), and if so, re-runs that entry's own run/manual build script to pull fresh
data and refit -- so run/main.py never silently works off stale market data, without hitting a
rate-limited API (BLS) on every single run.

MOTIVATION (the user's own words, paraphrased): manual model development (choosing/checking a
model's STRUCTURE via run/manual/build_*.py) is a deliberate, one-time act; but once a model's
structure is settled, running run/main.py's simulation shouldn't require a manual re-fetch every
time, nor should it silently keep using an old fit forever, nor should it hit a rate-limited API 100
times if main.py is run 100 times in a day. This throttles itself naturally: once a given month's
data has been pulled, every later run that same month sees it's already current and skips the live
call entirely -- the live pull only fires again once a new calendar month has fully elapsed.

Each run/manual/build_*.py script (see e.g. build_general_price_model.py) already overwrites its
own MODEL_NAME's data/raw/<MODEL_NAME>.csv and models/<MODEL_NAME>/ IN PLACE on every run --
MODEL_NAME itself doesn't change -- so re-running one here never requires updating
input_deck.yaml's forecast_models.<name>.model_name afterward.

PER-SOURCE EXPECTED LAG (see EXPECTED_LAG_MONTHS): not every source reports its own just-closed
month promptly. BLS CPI (general) and yfinance (market) do -- checked against the last FULLY
COMPLETED calendar month (1 month behind today's own month). S&P/Case-Shiller (home_price, via
FRED's DNXRNSA) does NOT: confirmed against FRED and S&P Dow Jones Indices directly (2026-09-25)
that it (a) is itself a 3-month moving average and (b) is published on the last Tuesday of the
SECOND month after the reference month closes -- e.g. July 2026's index isn't released until the
last Tuesday of September 2026. Net effect: a given month's data isn't expected until 3 months
after that month, i.e. checked against 3 months behind today's own month, not 1 -- using the
default 1-month check against home_price would flag it "stale" (and re-trigger a live FRED pull)
on almost every run, since FRED's own data is always exactly this far behind by design, not because
anything is actually wrong. (FRED has no rate limit -- see manual_utils.fetch_fred_series's own
docstring -- so this was harmless, just wasteful; now fixed.)
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from condobuyuq2026.input_deck import load_deck
from condobuyuq2026.paths import REPO_ROOT
from condobuyuq2026.raw_data import load_raw_csv
from condobuyuq2026.reporting import print_forecast_model_refresh, print_forecast_model_up_to_date

FORECAST_MODEL_BUILD_SCRIPTS: dict[str, Path] = {
    "general": REPO_ROOT / "run" / "manual" / "build_general_price_model.py",
    "home_price": REPO_ROOT / "run" / "manual" / "build_co_home_price_model.py",
    "market": REPO_ROOT / "run" / "manual" / "build_market_model.py",
}

# How many months behind TODAY'S OWN calendar month a source's data is expected to reach -- 1 (the
# last fully completed calendar month) unless listed here. See module docstring's "PER-SOURCE
# EXPECTED LAG" note for home_price's 3-month figure.
EXPECTED_LAG_MONTHS: dict[str, int] = {"home_price": 3}


def _month_offset(today: date, months_back: int) -> tuple[int, int]:
    """(year, month) for the calendar month `months_back` months before today's own month -- e.g.
    months_back=1 on 2026-09-25 gives (2026, 8): September itself isn't over yet, so it doesn't
    count, no matter how many of its days have already elapsed."""
    total = today.year * 12 + (today.month - 1) - months_back
    year, month_index0 = divmod(total, 12)
    return year, month_index0 + 1


def last_full_calendar_month(today: date | None = None) -> tuple[int, int]:
    """(year, month) of the last FULLY COMPLETED calendar month as of `today` (real today if not
    given) -- see _month_offset. This is the default expected lag (EXPECTED_LAG_MONTHS's fallback,
    1 month) -- most sources report their own just-closed month promptly."""
    return _month_offset(today or date.today(), 1)


def expected_latest_month(name: str, today: date | None = None) -> tuple[int, int]:
    """(year, month) forecast_models.<name> should have data through by now, given its own known
    reporting lag (see EXPECTED_LAG_MONTHS) -- 1 month behind (last_full_calendar_month) by
    default, or that source's own documented lag if listed."""
    return _month_offset(today or date.today(), EXPECTED_LAG_MONTHS.get(name, 1))


def is_forecast_model_stale(name: str, model_name: str, today: date | None = None) -> bool:
    """True if forecast_models.<name>'s raw data (data/raw/<model_name>.csv, see
    manual_utils.load_raw_csv -- already sorted chronologically ascending, so its last row is the
    latest observation) doesn't yet reach that source's own expected latest month (see
    expected_latest_month) -- i.e. its build script hasn't been (re-)run since new data should have
    become available."""
    last_row = load_raw_csv(model_name)[-1]
    last_point = (int(last_row["year"]), int(last_row["period"][1:]))
    return last_point < expected_latest_month(name, today)


def _run_build_script(script_path: Path) -> None:
    """Runs a run/manual/build_*.py script with this process's own Python (sys.executable -- this
    only ever runs from inside `uv run ...`, so it's already the project's venv), equivalent to
    `uv run python <script_path>` from the repo root."""
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=REPO_ROOT)


def refresh_stale_forecast_models(
    deck: dict[str, Any] | None = None,
    today: date | None = None,
    run_build_script: Callable[[Path], None] = _run_build_script,
) -> list[str]:
    """Checks every forecast_models entry (general/home_price/market) for staleness (see
    is_forecast_model_stale) and re-runs its build script (see FORECAST_MODEL_BUILD_SCRIPTS) for
    each one that is -- printing a message either way (see reporting.print_forecast_model_refresh/
    print_forecast_model_up_to_date) so it's visible whether or not a refresh happened.
    run_build_script is injectable so a test can verify which entries WOULD be refreshed without
    actually invoking the real script (see run/tests/test_freshness.py) -- this project's standing
    rule is that only the user runs those scripts for real.

    Returns the list of forecast_models keys actually refreshed (empty if everything was already
    current).
    """
    deck = deck if deck is not None else load_deck()
    refreshed = []
    for name, script_path in FORECAST_MODEL_BUILD_SCRIPTS.items():
        model_name = deck["forecast_models"][name]["model_name"]
        if not is_forecast_model_stale(name, model_name, today):
            print_forecast_model_up_to_date(name, model_name)
            continue
        print_forecast_model_refresh(name, model_name, script_path)
        run_build_script(script_path)
        refreshed.append(name)
    return refreshed
