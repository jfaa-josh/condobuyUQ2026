"""Main entry point (as opposed to run/manual/'s one-off, source-specific build scripts).

Stage 0: check every forecast_models entry's underlying raw data against its own expected
reporting lag, re-running that entry's run/manual build script if it's behind (see
condobuyuq2026.freshness.refresh_stale_forecast_models) -- so this always works off current data
without a manual re-fetch, and without hitting a rate-limited API more than once a month.

Stage 1: combine the three fitted forecast_models (general/home_price/market) into each
derived_latents entry's model, per the deck's weights -- see
condobuyuq2026.utils.manual_utils.build_all_derived_latent_models for how, and input_deck.yaml's
derived_latents section comment for the formula being applied.

Stage 2: run every scenario (see condobuyuq2026.scenarios.get_scenario_set), pairing each buy
scenario with its matching "rent" reference case's result -- each distinct reference case run only
once, even though multiple buy scenarios may share it (see condobuyuq2026.runner.run_scenarios).
Acquisition costs, occupancy/tax classification, and financing (mortgage rate + amortization) are
actually computed today; everything else is a placeholder print, filled in as later deck sections
(carrying_costs, taxes, rent_reference, capital_markets, exit, ...) get built.

Run with: uv run python run/main.py
"""

from condobuyuq2026.freshness import refresh_stale_forecast_models
from condobuyuq2026.reporting import print_derived_latent_build_report
from condobuyuq2026.runner import run_scenarios
from condobuyuq2026.utils.manual_utils import build_all_derived_latent_models

if __name__ == "__main__":
    refresh_stale_forecast_models()

    results = build_all_derived_latent_models()
    print_derived_latent_build_report(results)

    print()
    run_scenarios()
