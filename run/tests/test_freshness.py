"""Exercises freshness.refresh_stale_forecast_models -- for manual runs, not CI (no CI is set up
for this project): either `uv run pytest run/tests/test_freshness.py -v -s`, or
`uv run python run/tests/test_freshness.py` to run the checks in order as a plain script.

Deliberately never lets this touch a real API: every call here injects a fake run_build_script
(see freshness.refresh_stale_forecast_models's run_build_script parameter) so a "stale" result
never actually shells out to a run/manual/build_*.py script -- this project's standing rule is that
only the user runs those scripts for real (see .claude/implementation-plan.md's Earmarks and the
"no live API calls" memory).

Every test here prints its own important intermediate variables as
f"VARIABLE CHECK FOR {name}: {value}" -- test-script-only convention, never in main.py/backend
modules (see .claude/implementation-plan.md's "Backend build progress").
"""

from datetime import date

from condobuyuq2026.freshness import (
    EXPECTED_LAG_MONTHS,
    expected_latest_month,
    is_forecast_model_stale,
    last_full_calendar_month,
    refresh_stale_forecast_models,
)
from condobuyuq2026.input_deck import load_deck


def test_last_full_calendar_month_is_the_month_before_today_regardless_of_day():
    cases = [date(2026, 9, 25), date(2026, 9, 1), date(2026, 1, 15)]
    for today in cases:
        result = last_full_calendar_month(today)
        print(f"VARIABLE CHECK FOR last_full_calendar_month({today}): {result}")

    assert last_full_calendar_month(date(2026, 9, 25)) == (2026, 8)
    assert last_full_calendar_month(date(2026, 9, 1)) == (2026, 8)
    assert last_full_calendar_month(date(2026, 1, 15)) == (2025, 12)


def test_expected_latest_month_uses_home_price_own_longer_lag():
    today = date(2026, 9, 25)
    general_expected = expected_latest_month("general", today)
    home_price_expected = expected_latest_month("home_price", today)
    print(f"VARIABLE CHECK FOR EXPECTED_LAG_MONTHS: {EXPECTED_LAG_MONTHS}")
    print(f"VARIABLE CHECK FOR expected_latest_month('general', {today}): {general_expected}")
    print(f"VARIABLE CHECK FOR expected_latest_month('home_price', {today}): {home_price_expected}")

    assert general_expected == (2026, 8)  # default 1-month lag
    assert home_price_expected == (2026, 6)  # home_price's own 3-month lag (see EXPECTED_LAG_MONTHS)


def test_print_staleness_of_every_forecast_model():
    """Diagnostic printout -- no live calls, just checks each forecast_models entry's already-saved
    raw data (data/raw/<model_name>.csv) against its own expected latest month."""
    deck = load_deck()
    print(f"VARIABLE CHECK FOR deck['forecast_models']: {deck['forecast_models']}")
    for name, entry in deck["forecast_models"].items():
        model_name = entry["model_name"]
        stale = is_forecast_model_stale(name, model_name)
        print(f"VARIABLE CHECK FOR is_forecast_model_stale({name!r}): {stale}")
        print(f"forecast_models.{name} ({model_name}): {'STALE' if stale else 'up to date'}")


def test_refresh_stale_forecast_models_only_calls_the_fake_runner_for_stale_entries():
    deck = load_deck()
    calls = []

    refreshed = refresh_stale_forecast_models(deck, run_build_script=calls.append)

    print(f"VARIABLE CHECK FOR refreshed: {refreshed}")
    print(f"VARIABLE CHECK FOR calls: {calls}")

    # Exactly one fake call per entry staleness flagged -- never the real script.
    assert len(calls) == len(refreshed)


if __name__ == "__main__":
    test_last_full_calendar_month_is_the_month_before_today_regardless_of_day()
    test_expected_latest_month_uses_home_price_own_longer_lag()
    test_print_staleness_of_every_forecast_model()
    test_refresh_stale_forecast_models_only_calls_the_fake_runner_for_stale_entries()
    print("All freshness checks passed.")
