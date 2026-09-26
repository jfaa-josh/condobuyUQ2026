"""Main entry point (as opposed to run/manual/'s one-off, source-specific build scripts).

Stage 0: check every forecast_models entry's underlying raw data against its own expected
reporting lag, re-running that entry's run/manual build script if it's behind (see
condobuyuq2026.freshness.refresh_stale_forecast_models) -- so this always works off current data
without a manual re-fetch, and without hitting a rate-limited API more than once a month.

Stage 1: combine the three fitted forecast_models (general/home_price/market) into each
derived_latents entry's model, per the deck's weights -- see
condobuyuq2026.utils.manual_utils.build_all_derived_latent_models for how, and input_deck.yaml's
derived_latents section comment for the formula being applied.

Stage 1.5: build every carrying_costs prior's CHANGE-MAGNITUDE model (unscaled, price=1.0 basis --
see condobuyuq2026.carrying_costs' own module docstring for why): the 5 LEVEL+GROWTH entries
(hoa_dues_annual/insurance_annual/maintenance_annual/special_assessment/utilities, a hand-elicited
level + a fitted derived-latent's own growth) plus market_value's own real-data + OU-projection
build (see condobuyuq2026.carrying_costs.build_market_value_change_magnitude) -- see
condobuyuq2026.utils.manual_utils.build_all_carrying_cost_prior_models and input_deck.yaml's
carrying_costs section comment.

Stage 2: run every scenario (see condobuyuq2026.scenarios.get_scenario_set), pairing each buy
scenario with its matching "rent" reference case's result -- each distinct reference case run only
once, even though multiple buy scenarios may share it (see condobuyuq2026.runner.run_scenarios).
Acquisition costs, occupancy/tax classification, financing (mortgage rate + amortization),
carrying costs (property tax, HOA, insurance, maintenance, special assessment, utilities -- see
condobuyuq2026.carrying_costs, plus their own SCALED per-scenario result plots saved to results/,
see condobuyuq2026.results), and taxes (mortgage/depreciation/rental-income/sale effects -- see
condobuyuq2026.taxes) are actually computed today; the buy-vs-reference comparison itself is still
a placeholder print, filled in as later deck sections (rent_reference, capital_markets, exit, ...)
get built.
*** taxes.py has ONE remaining PLACEHOLDER input (gross_rent) -- see that module's docstring and
PLACEHOLDER_FIELDS constant before trusting any gross_rent-derived dollar figure it produces. ***

Run with: uv run python run/main.py
"""

from condobuyuq2026.freshness import refresh_stale_forecast_models
from condobuyuq2026.reporting import print_carrying_cost_prior_build_report, print_derived_latent_build_report
from condobuyuq2026.runner import run_scenarios
from condobuyuq2026.utils.manual_utils import build_all_carrying_cost_prior_models, build_all_derived_latent_models

if __name__ == "__main__":
    refresh_stale_forecast_models()

    results = build_all_derived_latent_models()
    print_derived_latent_build_report(results)

    carrying_cost_prior_results = build_all_carrying_cost_prior_models()
    print_carrying_cost_prior_build_report(carrying_cost_prior_results)

    print()
    run_scenarios()
