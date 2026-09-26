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


def print_carrying_cost_prior_build_report(results: dict[str, dict[str, Path]]) -> None:
    """Prints one line per carrying_costs LEVEL+GROWTH entry build result -- results as returned by
    utils.manual_utils.build_all_carrying_cost_prior_models (name -> {"fit_path", "plot_path"})."""
    print("\nSaving carrying-cost priors to //models/carrying_cost_priors/...")
    for name, paths in results.items():
        print(f"{name}: saved fit to {paths['fit_path']}, plot to {paths['plot_path']}")


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


def print_classification_report(classification: dict[str, Any]) -> None:
    """Prints one scenario's occupancy classification (see
    classification.compute_occupancy_classification) as a single line -- loan_occupancy_class,
    tax_use_class, and the annual day counts tax_use_class was decided from."""
    print(
        f"loan_occupancy_class={classification['loan_occupancy_class']}, "
        f"tax_use_class={classification['tax_use_class']} "
        f"(annual_rented_days={classification['annual_rented_days']:.0f}, "
        f"annual_personal_days={classification['annual_personal_days']:.0f})"
    )


def print_financing_report(financing: dict[str, Any]) -> None:
    """Prints one scenario's financing results (see financing.compute_financing) as a single line
    -- the mortgage rate actually applied, and the loan's cost/payoff over the scenario's horizon.
    Not the full amortization_schedule (too verbose for a normal run) -- see a test script's own
    VARIABLE CHECK print for that."""
    print(
        f"mortgage_rate={financing['mortgage_rate']:.4%} ({financing['loan_occupancy_class']}), "
        f"loan_term_years={financing['loan_term_years']} -> "
        f"total_interest_paid=${financing['total_interest_paid']:,.0f}, "
        f"ending_balance=${financing['ending_balance']:,.0f}"
    )


def print_carrying_costs_report(carrying_costs: dict[str, Any]) -> None:
    """Prints one buy scenario's carrying costs (see carrying_costs.compute_carrying_costs) as a
    single line -- total over the horizon, plus year-1's own breakdown so each prior's level is
    visible at a glance -- and a second line with property tax's own year-1 median/lo/hi band
    (carrying_costs.compute_property_tax_schedule -- the one carrying-cost figure that carries a
    real distribution, not just a point estimate, per the user's explicit request)."""
    year_1 = carrying_costs["annual_schedule"].loc[1]
    property_tax_year_1 = carrying_costs["property_tax_schedule"].loc[1]
    print(
        f"total_carrying_costs=${carrying_costs['total_carrying_costs']:,.0f} over the horizon "
        f"(year 1: property_tax=${year_1['property_tax']:,.0f}, hoa_dues=${year_1['hoa_dues']:,.0f}, "
        f"insurance=${year_1['insurance']:,.0f}, maintenance=${year_1['maintenance']:,.0f}, "
        f"special_assessment_expected=${year_1['special_assessment_expected']:,.0f}, "
        f"utilities=${year_1['utilities']:,.0f})"
    )
    print(
        f"  property_tax year 1 distribution: assessed_value=${property_tax_year_1['assessed_value_median']:,.0f} "
        f"(${property_tax_year_1['assessed_value_lo']:,.0f}-${property_tax_year_1['assessed_value_hi']:,.0f}), "
        f"tax=${property_tax_year_1['tax_median']:,.0f} "
        f"(${property_tax_year_1['tax_lo']:,.0f}-${property_tax_year_1['tax_hi']:,.0f})"
    )


def print_carrying_cost_result_plots_report(result_plots: dict[str, Path]) -> None:
    """Prints where one scenario's own SCALED carrying-cost plots were saved (see
    results.save_carrying_cost_result_plots) -- results/<scenario>/carrying_costs/, one line total
    rather than one per plot (the folder itself is the useful pointer; the report's own year-1
    breakdown line already shows the numbers)."""
    folder = next(iter(result_plots.values())).parent
    print(f"  saved {len(result_plots)} carrying-cost result plots to {folder}")


def print_taxes_report(taxes: dict[str, Any]) -> None:
    """Prints one buy scenario's tax results (see taxes.compute_owner_taxes) as a single line, plus
    a LOUD placeholder warning (taxes.PLACEHOLDER_FIELDS) and any residency flags (taxes.
    compute_owner_taxes' "flags") on their own lines so neither is easy to miss."""
    print(f"total_tax=${taxes['total_tax']:,.0f} over the horizon (annual + sale)")
    if taxes["placeholder_fields"]:
        fields = ", ".join(taxes["placeholder_fields"])
        print(f"  [PLACEHOLDER] uses simplified, un-grown estimates for: {fields} -- see taxes.py's module docstring")
    for flag in taxes["flags"]:
        print(f"  {flag}")


def print_renter_taxes_report(taxes: dict[str, Any]) -> None:
    """Prints a reference case's tax results (see taxes.compute_renter_taxes) -- always $0, a real
    confirmed result, not a placeholder (renting has no real-estate tax effect)."""
    print(f"total_tax=${taxes['total_tax']:,.0f} (renting has no real-estate tax effect)")


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
