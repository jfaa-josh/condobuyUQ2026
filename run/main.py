"""Main entry point (as opposed to run/manual/'s one-off, source-specific build scripts).

Stage 1: combine the three fitted forecast_models (general/home_price/market) into each
derived_latents entry's model, per the deck's weights -- see
condobuyuq2026.utils.manual_utils.build_all_derived_latent_models for how, and input_deck.yaml's
derived_latents section comment for the formula being applied.

Stage 2: expand scenario's sweeps into the full set of scenarios to run (see
input_deck.get_scenarios), and compute acquisition's derived costs for each one (see
condobuyuq2026.acquisition.compute_acquisition_costs) -- the first real per-scenario backend
computation. Later deck sections (financing, carrying_costs, exit, ...) will build on the same
per-scenario pattern as they're added.

Run with: uv run python run/main.py

Requires general/home_price/market's fit.json to already exist (run the three run/manual/build_*
scripts first if not).
"""

from condobuyuq2026.acquisition import compute_acquisition_costs
from condobuyuq2026.input_deck import get_scenarios
from condobuyuq2026.reporting import print_acquisition_report, print_derived_latent_build_report
from condobuyuq2026.utils.manual_utils import build_all_derived_latent_models

if __name__ == "__main__":
    results = build_all_derived_latent_models()
    print_derived_latent_build_report(results)

    print()
    for scenario in get_scenarios():
        costs = compute_acquisition_costs(scenario)
        print_acquisition_report(scenario, costs)
