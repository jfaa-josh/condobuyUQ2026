"""Main entry point (as opposed to run/manual/'s one-off, source-specific build scripts).

Stage 0: check every forecast_models entry's underlying raw data against its own expected
reporting lag. OFF by default: only with --refresh does a stale entry get its run/manual build
script re-run (live API calls, see condobuyuq2026.freshness.refresh_stale_forecast_models). Without
--refresh, stale entries just get a one-line warning and the run continues on the saved models.

Stage 1: combine the three fitted forecast_models (general/home_price/market) into each
derived_latents entry's model, per the deck's weights -- see
condobuyuq2026.utils.manual_utils.build_all_derived_latent_models for how, and input_deck.yaml's
derived_latents section comment for the formula being applied.

Stage 1.5: build every carrying_costs prior's CHANGE-MAGNITUDE model (unscaled, price=1.0 basis --
see condobuyuq2026.carrying_costs' own module docstring for why), rebuilt automatically every run
just like Stage 1 (see condobuyuq2026.utils.prior_utils' own module docstring for why this lives
apart from utils.manual_utils' genuinely one-off external-data-pulling scripts): the UNINFORMED
LEVEL+GROWTH entries (hoa_dues_annual/insurance_annual/maintenance_annual/special_assessment/
utilities, a hand-elicited level + a fitted derived-latent's own growth) plus market_value's own
INFORMED build (real-data + OU-projection change-magnitude, PLUS a fitness check against the
property's own known past reassessment value(s) -- see
condobuyuq2026.utils.prior_utils.build_market_value_informed_model) -- see
condobuyuq2026.utils.prior_utils.build_all_carrying_cost_prior_models and input_deck.yaml's
carrying_costs section comment.

Stage 2: run every scenario (see condobuyuq2026.scenarios.get_scenario_set), pairing each buy
scenario with its matching "rent" reference case's result -- each distinct reference case run only
once, even though multiple buy scenarios may share it (see condobuyuq2026.runner.run_scenarios).
Acquisition costs, occupancy/tax classification, financing (mortgage rate + amortization),
carrying costs (property tax, HOA, insurance, maintenance, special assessment, utilities -- see
condobuyuq2026.carrying_costs, plus their own SCALED per-scenario result plots saved to results/,
see condobuyuq2026.results), rental operations (fitted occupancy/ADR priors and the per-scenario revenue
projection -- see condobuyuq2026.rental_operations), and taxes (mortgage/depreciation/rental-income/sale
effects -- see condobuyuq2026.taxes) are actually computed today; the buy-vs-reference comparison itself is still
a placeholder print, filled in as later deck sections (rent_reference, capital_markets, exit, ...)
get built.

Run with: uv run python run/main.py            (offline; saved models only)
          uv run python run/main.py --refresh  (also refreshes stale forecast_models)
"""

import argparse

from condobuyuq2026.freshness import (
    FORECAST_MODEL_BUILD_SCRIPTS,
    is_forecast_model_stale,
    refresh_stale_forecast_models,
)
from condobuyuq2026.input_deck import load_deck
from condobuyuq2026.reporting import (
    print_carrying_cost_prior_build_report,
    print_derived_latent_build_report,
    print_rental_operations_prior_build_report,
    print_stale_forecast_models_warning,
)
from condobuyuq2026.runner import run_scenarios
from condobuyuq2026.utils.manual_utils import build_all_derived_latent_models
from condobuyuq2026.utils.prior_utils import build_all_carrying_cost_prior_models
from condobuyuq2026.utils.rental_prior_utils import build_rental_operations_prior_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the condobuyUQ2026 pipeline.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run stale forecast_models' build scripts (live API calls). Off by default.",
    )
    args = parser.parse_args()

    if args.refresh:
        refresh_stale_forecast_models()
    else:
        deck = load_deck()
        stale = [
            name
            for name in FORECAST_MODEL_BUILD_SCRIPTS
            if is_forecast_model_stale(name, deck["forecast_models"][name]["model_name"])
        ]
        if stale:
            print_stale_forecast_models_warning(stale)

    results = build_all_derived_latent_models()
    print_derived_latent_build_report(results)

    # Before the carrying-cost priors: utilities reads expected rented nights from these.
    rental_prior_results = build_rental_operations_prior_models()
    print_rental_operations_prior_build_report(rental_prior_results)

    carrying_cost_prior_results = build_all_carrying_cost_prior_models()
    print_carrying_cost_prior_build_report(carrying_cost_prior_results)

    print()
    run_scenarios()
