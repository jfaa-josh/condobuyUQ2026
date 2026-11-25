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
Per buy scenario: acquisition costs, occupancy/tax classification, financing, carrying costs, rental
operations, personal use, taxes, exit (net equity at the sale), and ΔW = W_buy - W_rent against its
reference case (condobuyuq2026.delta_w: P(buy beats rent), the band, what drives it). Per reference
case: the lease (rent_reference), the invested portfolio (capital_markets) and its taxes. Each stage's
plots go to results/<scenario>/ and results/rent_ref*/.

Stage 3 (deck section 13, `run`; see condobuyuq2026.run_outputs): the cross-scenario summary
(results/summary.csv + summary.png), a warning listing results/ folders this run didn't write (never
deleted), and the run manifest results/run_info.json (when, git commit, deck fingerprint, run settings,
forecast data dates, every warning, folders written).

The model builds (stages 1 and 1.5) project as far as the sweep's longest scenario.horizon_years
(input_deck.get_build_horizon_months). Every reporting/solver setting comes from the deck's `run`
section; only --refresh (live data pulls) is a command-line switch.

Run with: uv run python run/main.py            (offline; saved models only)
          uv run python run/main.py --refresh  (also refreshes stale forecast_models)
"""

import argparse
from datetime import datetime

from condobuyuq2026.freshness import (
    FORECAST_MODEL_BUILD_SCRIPTS,
    is_forecast_model_stale,
    refresh_stale_forecast_models,
)
from condobuyuq2026.input_deck import get_build_horizon_months, load_deck
from condobuyuq2026.reporting import (
    print_carrying_cost_prior_build_report,
    print_derived_latent_build_report,
    print_rental_operations_prior_build_report,
    print_run_manifest_report,
    print_run_summary_report,
    print_stale_forecast_models_warning,
    print_stale_results_warning,
)
from condobuyuq2026.results import save_run_summary
from condobuyuq2026.run_outputs import (
    collect_warnings,
    find_stale_results_folders,
    summary_table,
    write_run_manifest,
    written_folders,
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
    started = datetime.now().astimezone()

    if args.refresh:
        refresh_stale_forecast_models()
    deck = load_deck()
    stale_forecasts = [
        name
        for name in FORECAST_MODEL_BUILD_SCRIPTS
        if is_forecast_model_stale(name, deck["forecast_models"][name]["model_name"])
    ]
    if stale_forecasts and not args.refresh:
        print_stale_forecast_models_warning(stale_forecasts)

    build_horizon_months = get_build_horizon_months(deck)  # the sweep's longest horizon
    results = build_all_derived_latent_models(deck, build_horizon_months)
    print_derived_latent_build_report(results)

    # Before the carrying-cost priors: utilities reads expected rented nights from these.
    rental_prior_results = build_rental_operations_prior_models()
    print_rental_operations_prior_build_report(rental_prior_results)

    carrying_cost_prior_results = build_all_carrying_cost_prior_models(deck, build_horizon_months)
    print_carrying_cost_prior_build_report(carrying_cost_prior_results)

    print()
    run = run_scenarios(deck)

    # Stage 3: run-level outputs (deck section 13).
    table = summary_table(run)
    summary_paths = save_run_summary(table)
    print_run_summary_report(summary_paths, len(table))
    written = written_folders(run)
    stale_folders = find_stale_results_folders(written)
    print_stale_results_warning(stale_folders)
    warnings = collect_warnings(run, stale_forecasts, rental_prior_results, carrying_cost_prior_results,
                                stale_folders, deck)
    manifest = write_run_manifest(run, deck, started, warnings, written, stale_folders, summary_paths)
    print_run_manifest_report(manifest, len(warnings))
