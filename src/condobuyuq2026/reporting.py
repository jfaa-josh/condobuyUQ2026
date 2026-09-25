"""Shared human-readable print formatting for backend build/compute results -- one function per
deck-section stage, called identically by run/main.py (the real build/report pipeline) and each
matching run/tests/test_*.py (the user's manual-run check for that same stage, see
.claude/implementation-plan.md's "Backend build progress"). Keeping this formatting here, in one
place, means main.py and a test script can never drift into two different versions of the same
printout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def print_derived_latent_build_report(results: dict[str, dict[str, Path]]) -> None:
    """Prints one line per derived_latents entry build result -- results as returned by
    utils.manual_utils.build_all_derived_latent_models (name -> {"fit_path", "plot_path"})."""
    print('\nSaving derived variables to //models/derived_variables/...')
    for name, _paths in results.items():
        print(f"{name}: saved fit to /{name} and plot to /{name}/plots")


def print_acquisition_report(scenario: dict[str, Any], acquisition_costs: dict[str, float]) -> None:
    """Prints one buy scenario's acquisition costs as a single, explicitly-labeled line -- scenario
    as returned by scenarios.get_scenario_set (or input_deck.get_scenarios), acquisition_costs as
    returned by acquisition.compute_acquisition_costs for that scenario."""
    print(
        f"alternative={scenario['alternative']}, horizon_years={scenario['horizon_years']}, "
        f"residency_state={scenario['residency_state']}, purchase_price=${acquisition_costs['purchase_price']:,.0f} -> "
        f"closing_costs=${acquisition_costs['closing_costs_total']:,.0f}, "
        f"loan_amount=${acquisition_costs['loan_amount']:,.0f}, "
        f"family_loan=${acquisition_costs['family_loan']:,.0f}, "
        f"total_cash_outlay=${acquisition_costs['total_cash_outlay']:,.0f}"
    )


def print_reference_report(reference_id: int, reference_case: dict[str, Any]) -> None:
    """Prints one reference case (see scenarios.get_scenario_set) as a single, explicitly-labeled
    line -- same label style as print_acquisition_report so the two are easy to compare side by
    side. No dollar figure yet: the rent branch's invested amount is capital_markets.
    initial_investment, which just imports acquisition.personal_cash_at_closing (still to be wired
    in when the capital_markets stage is built) -- deliberately NOT the buy branch's
    total_cash_outlay (see scenarios.py's module docstring for why).

    PLACEHOLDER: reference_case only carries reference/horizon_years/residency_state/property_state
    today (enough for scenarios.get_scenario_set's dedup) -- once rent_reference/capital_markets
    get built, a reference case will need more variables of its own (rent cost, invested amount,
    ...) added here before it can actually be run.
    """
    print(
        f"reference_id={reference_id}: reference={reference_case['reference']}, "
        f"horizon_years={reference_case['horizon_years']}, "
        f"residency_state={reference_case['residency_state']} (reference case)"
    )
    print("  [PLACEHOLDER] more variables needed here for renting (rent_reference/capital_markets inputs)")


def print_reference_cache_hit(reference_id: int) -> None:
    """Prints that reference_id's result is being reused rather than run again -- see
    runner.run_scenarios, which runs each distinct reference case only once even though multiple
    buy scenarios may share it."""
    print(f"reference_id={reference_id}: reusing already-run result (not re-run)")


def print_scenario_comparison_placeholder(scenario: dict[str, Any], reference_id: int) -> None:
    """Placeholder for the eventual buy-vs-reference comparison (ΔW = W_buy - W_rent, see
    .claude/implementation-plan.md's Goal) -- prints which buy scenario would be compared against
    which reference case's result, once both sides actually compute a result to compare. See
    runner.run_scenarios."""
    print(
        f"[PLACEHOLDER] would compare this buy scenario's result against reference_id={reference_id}'s "
        "result here"
    )


def print_forecast_model_refresh(name: str, model_name: str, script_path: Path) -> None:
    """Prints that forecast_models.<name>'s data was stale and its build script is being re-run to
    refresh it -- see freshness.refresh_stale_forecast_models."""
    print(f"forecast_models.{name} ({model_name}) data is stale -- refreshing via {script_path.name}...")


def print_forecast_model_up_to_date(name: str, model_name: str) -> None:
    """Prints that forecast_models.<name>'s data was checked and is already current -- see
    freshness.refresh_stale_forecast_models."""
    print(f"forecast_models.{name} ({model_name}) checked -- up to date")
