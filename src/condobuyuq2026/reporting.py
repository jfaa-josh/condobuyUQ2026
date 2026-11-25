"""Shared human-readable print formatting for backend build/compute results -- one function per
deck-section stage, called by run/main.py (the real build/report pipeline). Keeping all printout
formatting here, in one place, keeps it out of main.py and the backend modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def print_rental_operations_prior_build_report(results: dict[str, dict[str, Any]]) -> None:
    """Prints the rental_operations prior build (utils.rental_prior_utils.build_rental_operations_prior_models):
    where each prior was saved, its fitted annual median and band, your annual rented nights, and any fit
    warnings."""
    print("\nSaving rental_operations priors to //models/rental_operations_priors/...")
    for name, result in results.items():
        print(f"{name}: saved fit to {result['fit_path']}, plot to {result['plot_path']}")
    occupancy, adr = results["occupancy"]["annual_band"], results["adr"]["annual_band"]
    nights = results["occupancy"]["annual_rented_nights"]
    print(
        f"  fitted from {results['occupancy']['source_count']} included source(s): annual occupancy "
        f"{occupancy['median']:.1%} ({occupancy['lo']:.1%}-{occupancy['hi']:.1%}), ADR ${adr['median']:,.0f} "
        f"(${adr['lo']:,.0f}-${adr['hi']:,.0f}); your rented nights/yr: {nights['mean']:.0f} expected, "
        f"{nights['median_sum']:.0f} by monthly medians"
    )
    for warning in results["occupancy"]["warnings"]:
        print(f"  WARNING: {warning}")


def print_derived_latent_build_report(results: dict[str, dict[str, Path]]) -> None:
    """Prints one line per derived_latents entry build result -- results as returned by
    utils.manual_utils.build_all_derived_latent_models (name -> {"fit_path", "plot_path"})."""
    print('\nSaving derived variables to //models/derived_variables/...')
    for name, _paths in results.items():
        print(f"{name}: saved fit to /{name} and plot to /{name}/plots")


def print_carrying_cost_prior_build_report(results: dict[str, dict[str, Any]]) -> None:
    """Prints one line per carrying_costs prior build result -- results as returned by
    utils.prior_utils.build_all_carrying_cost_prior_models (name -> {"fit_path", "plot_path", ...}).
    For market_value (the one INFORMED entry, see carrying_costs.INFORMED_CARRYING_COST_PRIORS), if
    at least 2 real observations exist, also prints the fitted Denver-to-Keystone calibration line
    (utils.prior_utils.fit_market_value_calibration's own slope/intercept) plus one line per
    observation showing how far the fitted line's own prediction is from what was actually recorded
    -- so it's visible on every run how well the line fits the known data, not just in the saved
    plot."""
    print("\nSaving carrying-cost priors to //models/carrying_cost_priors/...")
    for name, paths in results.items():
        print(f"{name}: saved fit to {paths['fit_path']}, plot to {paths['plot_path']}")
        if "calibration_report" in paths:
            calibration = paths["calibration_report"]
            print(
                f"  market_value calibration saved to {paths['calibration_plot_path']}: "
                f"slope={calibration['slope']:,.1f}, intercept={calibration['intercept']:,.0f}"
            )
            for observation in calibration["observations"]:
                predicted = calibration["slope"] * observation["denver_window_avg"] + calibration["intercept"]
                residual = predicted - observation["keystone_actual"]
                print(
                    f"    {observation['effective_year']}: actual=${observation['keystone_actual']:,.0f}, "
                    f"fitted=${predicted:,.0f} (residual ${residual:+,.0f})"
                )
        if "utilities_report" in paths:
            print_utilities_build_report(paths["utilities_report"])


def print_utilities_build_report(report: dict[str, Any]) -> None:
    """Prints utilities' three-category build details (utils.prior_utils.build_utilities_prior_model):
    the RESOLVED heating type (so a wrong heating_type code is obvious on every run), the three
    calibrated pools and their rates, year 1 excluding add-ons (which the build already checked equals
    12 x the observed average) with the add-on total reported SEPARATELY (add-ons intentionally raise
    the total above the previous owner's bill), and a WARNING per month whose occupied days had to be
    capped at the days in that month."""
    heating = report["heating"]
    calibration = report["calibration"]
    if heating["carrier"] is None:
        heating_line = f"type {heating['heating_type']} = {heating['name']} (no heating on this bill)"
    else:
        shares = ", ".join(
            f"{name} {fraction * 100:.1f}%" for name, fraction in heating["heating_fraction_by_component"].items() if fraction > 0
        )
        cop = f"COP {heating['cop']:g}" if heating["cop"] is not None else "COP n/a (gas)"
        heating_line = (
            f"type {heating['heating_type']} = {heating['name']}, carrier {heating['carrier']}, {cop}, "
            f"heating_share {heating['heating_share']:g} -> effective heating share of bill: {shares}"
        )
    print(f"  utilities heating: {heating_line}")
    print(
        f"  utilities calibration ({report['calibration_source']}, {sum(report['calibration_occupied_days'].values()):.0f} "
        f"occupied days/yr, observed ${calibration['annual_observed_total']:,.0f}/yr): "
        f"flat ${calibration['flat_pool']:,.0f}/yr (${calibration['flat_monthly']:,.2f}/month), "
        f"occupancy ${calibration['occupancy_pool']:,.0f}/yr (${calibration['cost_per_occupied_day']:,.2f}/occupied day), "
        f"heating ${calibration['heating_pool']:,.0f}/yr (${calibration['cost_per_heating_degree_day']:,.4f}/heating degree-day)"
    )
    print(
        f"  utilities year 1 (before growth): ${report['year_1_total_excluding_add_ons']:,.0f} excluding add-ons, "
        f"+ ${report['year_1_add_ons_total']:,.0f} add-ons (new costs on top of the previous owner's bill)"
    )
    for warning in report["cap_warnings"]:
        print(f"  WARNING: utilities {warning}")


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
    Not the full amortization_schedule (too verbose for a normal run)."""
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
        f"utilities=${year_1['utilities']:,.0f})"
    )
    print(
        f"  property_tax year 1 distribution: assessed_value=${property_tax_year_1['assessed_value_median']:,.0f} "
        f"(${property_tax_year_1['assessed_value_lo']:,.0f}-${property_tax_year_1['assessed_value_hi']:,.0f}), "
        f"tax=${property_tax_year_1['tax_median']:,.0f} "
        f"(${property_tax_year_1['tax_lo']:,.0f}-${property_tax_year_1['tax_hi']:,.0f})"
    )


def print_rental_operations_result_plots_report(result_plots: dict[str, Path]) -> None:
    """One line: where this scenario's rental result plots went (results.save_rental_operations_result_plots)."""
    print(f"  saved {len(result_plots)} rental result plots to {result_plots['occupancy'].parent} "
          f"(+ {result_plots['net_rental_income'].name} one level up)")


def print_rental_operations_report(rental_operations: dict[str, Any]) -> None:
    """Prints one buy scenario's rental projection (rental_operations.compute_rental_operations): horizon
    gross and net (mean, median and band), then year 1's nights, gross and fees."""
    horizon, year_1, fees = rental_operations["horizon"], rental_operations["annual"].loc[1], rental_operations["fees"]
    ci = rental_operations["ci"]
    band = f"{ci[0]:.0%}-{ci[1]:.0%}"
    print(
        f"rental_operations: gross=${horizon['gross_mean']:,.0f} mean over the horizon (median ${horizon['gross_median']:,.0f}, "
        f"{band} ${horizon['gross_lo']:,.0f}-${horizon['gross_hi']:,.0f}); net=${horizon['net_mean']:,.0f} "
        f"after channel commissions {fees['channel_commission_fraction']:.1%} and management fee "
        f"{fees['mgmt_fee_fraction']:.1%}"
    )
    print(
        f"  year 1: {year_1['nights_mean']:.0f} nights rented (expected), gross=${year_1['gross_mean']:,.0f} "
        f"({band} ${year_1['gross_lo']:,.0f}-${year_1['gross_hi']:,.0f}), commissions=${year_1['channel_commissions_mean']:,.0f}, "
        f"management fee=${year_1['mgmt_fee_mean']:,.0f}, net=${year_1['net_mean']:,.0f}"
    )


def print_personal_use_report(personal_use: dict[str, Any]) -> None:
    """Prints one buy scenario's alternate-lodging cost (personal_use.compute_personal_use): horizon total
    (mean, median and band), then year 1's total and your personal days."""
    horizon, year_1 = personal_use["horizon"], personal_use["annual"].loc[1]
    ci = personal_use["ci"]
    band = f"{ci[0]:.0%}-{ci[1]:.0%}"
    latent = personal_use["inputs"]["alternate_lodging_growth_latent"]
    print(
        f"personal_use: alternate lodging=${horizon['lodging_mean']:,.0f} mean over the horizon (median "
        f"${horizon['lodging_median']:,.0f}, {band} ${horizon['lodging_lo']:,.0f}-${horizon['lodging_hi']:,.0f}), "
        f"charged as entered, grown by {latent}; personal cost, not deductible, not in carrying-cost totals"
    )
    print(
        f"  year 1: {year_1['personal_days']:.0f} personal days, alternate lodging=${year_1['lodging_mean']:,.0f} "
        f"({band} ${year_1['lodging_lo']:,.0f}-${year_1['lodging_hi']:,.0f})"
    )


def print_personal_use_result_plots_report(result_plots: dict[str, Path]) -> None:
    """One line: where this scenario's personal_use plot went (results.save_personal_use_result_plots)."""
    print(f"  saved {len(result_plots)} personal_use result plot to {result_plots['alternate_lodging_cost'].parent}")


SPECIAL_ASSESSMENT_FOLDED_TAIL_WARNING_THRESHOLD = 0.01  # warn when P(N > k_max) exceeds 1%


def print_special_assessment_report(special_assessment: dict[str, Any]) -> None:
    """Prints one buy scenario's special-assessment model (see carrying_costs.
    compute_special_assessment) on its own line, labeled as EXCLUDED from the totals above so the
    two are never read as one number. Adds a WARNING line when the chance of more than k_max
    assessments -- which the model folds into the k_max outcome, slightly understating cost there
    -- is above 1%, since that's the signal to raise k_max in the deck."""
    sa = special_assessment
    if not sa["enabled"]:
        print("  special_assessment: off (annual_rate: 0) -- no tail-risk overlay")
        return
    k_max = sa["k_max"]
    counts = ", ".join(
        f"P(N{'>=' if int(k) == k_max else '='}{int(k)})={p * 100:.1f}%" for k, p in sa["count_probabilities"].items()
    )
    sizes = ", ".join(f"{label}=${size:,.0f}" for label, size in sa["sizes_nominal_usd"].items())
    print(
        # plain "delta-W", not the Greek letter: the Windows console's default encoding can't print it
        f"  special_assessment (EXCLUDED from totals; tail-risk overlay on delta-W later): "
        f"expected count={sa['expected_count']:.2f}, {counts}; sizes ({sa['timing']} timing): {sizes}; "
        f"expected total=${sa['expected_total_cost_usd']:,.0f}"
    )
    if sa["folded_tail_probability"] > SPECIAL_ASSESSMENT_FOLDED_TAIL_WARNING_THRESHOLD:
        print(
            f"  WARNING: special_assessment P(more than k_max={k_max} assessments)="
            f"{sa['folded_tail_probability'] * 100:.1f}% (> 1%) is folded into the N>={k_max} outcome, "
            f"understating cost there -- raise carrying_costs.special_assessment.k_max"
        )


def print_carrying_cost_result_plots_report(result_plots: dict[str, Path]) -> None:
    """Prints where one scenario's own SCALED carrying-cost plots were saved (see
    results.save_carrying_cost_result_plots) -- results/<scenario>/carrying_costs/, one line total
    rather than one per plot (the folder itself is the useful pointer; the report's own year-1
    breakdown line already shows the numbers)."""
    folder = next(iter(result_plots.values())).parent
    print(f"  saved {len(result_plots)} carrying-cost result plots to {folder}")


def print_taxes_report(taxes: dict[str, Any]) -> None:
    """Prints one buy scenario's tax results (see taxes.compute_owner_taxes): the jurisdiction layers
    used, the derived rules applied, then the totals; residency flags on their own lines so they're hard
    to miss."""
    schedule = taxes["annual_schedule"]
    rules = taxes["rules"]
    jurisdiction = taxes["jurisdiction"]
    sale = taxes["sale"]
    residence = jurisdiction.residence_state
    print(
        f"tax jurisdiction: {jurisdiction.label} (data as of {jurisdiction.as_of}); residence {residence.name} "
        f"({residence.describe_income_tax(int(schedule['tax_year'].iloc[0]))}); STR guest taxes "
        f"{jurisdiction.guest_tax_total():.3%} (guest-paid, not income)"
    )
    passive = "" if rules["passive_status"] == "n/a" else f", losses {rules['passive_status']}"
    print(
        f"  rules: use class {rules['tax_use_class']} (rental share {rules['rental_share']:.0%}); depreciation "
        f"{rules['depreciation_years']:g} yr on {rules['building_fraction']:.0%} of basis (avg guest stay "
        f"{rules['average_guest_stay_nights']:g} nights){passive}; itemizes in {int(schedule['itemizes'].sum())} of "
        f"{len(schedule)} years"
    )
    print(
        f"total_tax=${taxes['total_tax']:,.0f} over the horizon (annual ${schedule['tax'].sum():,.0f} + sale "
        f"${sale['tax']:,.0f}); federal ${schedule['federal'].sum() + sale['federal']:,.0f}, "
        f"{jurisdiction.property_state.name} ${schedule['property_state'].sum() + sale['property_state']:,.0f}; "
        f"expected gross rent ${schedule['gross_rent'].sum():,.0f}, taxable rental income "
        f"${schedule['taxable_rental'].sum():,.0f}"
    )
    print(
        f"  sale: gain ${sale['gain']:,.0f} = recapture ${sale['recapture']:,.0f} + long-term ${sale['ltcg']:,.0f} "
        f"+ excluded ${sale['excluded_121']:,.0f}; released passive losses ${sale['released_passive']:,.0f}; lost 280A "
        f"carryforward ${sale['lost_280a_carry']:,.0f}"
    )
    for flag in taxes["flags"]:
        print(f"  {flag}")


def print_tax_result_plots_report(result_plots: dict[str, Path]) -> None:
    """One line: where this scenario's taxes plot went (results.save_tax_result_plots)."""
    print(f"  saved {len(result_plots)} taxes result plot to {result_plots['tax_effects_by_year'].parent}")


def print_renter_taxes_report(taxes: dict[str, Any]) -> None:
    """Prints a reference case's tax results (taxes.compute_renter_taxes): the portfolio's taxes only
    (renting itself has no tax effect). Developer/testing output."""
    yearly = float(taxes["annual_schedule"]["tax"].sum())
    print(f"  renter taxes: total=${taxes['total_tax']:,.0f} (portfolio: yearly interest/dividends "
          f"${yearly:,.0f}, sale ${taxes['sale']['tax']:,.0f}; rent itself isn't deductible)")


def print_capital_markets_report(capital_markets: dict[str, Any]) -> None:
    """One developer/testing line for the rent branch's portfolio (capital_markets.compute_capital_markets)."""
    horizon, sale, inputs = capital_markets["horizon"], capital_markets["sale"], capital_markets["inputs"]
    ci = capital_markets["ci"]
    allocation = inputs["allocation"]
    print(
        f"capital_markets: invested ${horizon['contributions']:,.0f} (market {allocation['market']:.0%} / cash "
        f"{allocation['cash']:.0%}) -> after-tax ${horizon['after_tax_mean']:,.0f} mean at the horizon "
        f"({ci[0]:.0%}-{ci[1]:.0%} ${horizon['after_tax_lo']:,.0f}-${horizon['after_tax_hi']:,.0f}); "
        f"sale gain ${sale['gain']:,.0f} taxed at {sale['effective_rate']:.1%}"
    )


def print_capital_markets_result_plots_report(result_plots: dict[str, Path]) -> None:
    """One line: where this reference case's capital_markets plot went (results.save_capital_markets_result_plots)."""
    print(f"  saved {len(result_plots)} capital_markets result plot to {result_plots['portfolio'].parent}")


def print_reference_report(reference_id: int, reference_case: dict[str, Any]) -> None:
    """Prints one reference case (see scenarios.get_scenario_set) as a single, explicitly-labeled
    line -- same label style as print_acquisition_report so the two are easy to compare side by
    side. No dollar figure yet: the rent branch's invested amount is capital_markets.
    initial_investment, which just imports acquisition.personal_cash_at_closing (still to be wired
    in when the capital_markets stage is built) -- deliberately NOT the buy branch's
    total_cash_outlay (see scenarios.py's module docstring for why).

    The lease cost and the portfolio are printed right after (print_rent_reference_report,
    print_capital_markets_report).
    """
    print(
        f"reference_id={reference_id}: reference={reference_case['reference']}, "
        f"horizon_years={reference_case['horizon_years']}, "
        f"residency_state={reference_case['residency_state']} (reference case)"
    )


def print_rent_reference_report(rent_reference: dict[str, Any]) -> None:
    """Prints one reference case's lease cost (rent_reference.compute_rent_reference): horizon total
    (mean, median and band), then year 1's total (exactly as entered: rent steps up once a year)."""
    horizon, year_1 = rent_reference["horizon"], rent_reference["annual"].loc[1]
    ci = rent_reference["ci"]
    band = f"{ci[0]:.0%}-{ci[1]:.0%}"
    latent = rent_reference["inputs"]["lease_growth_latent"]
    print(
        f"rent_reference: lease=${horizon['lease_mean']:,.0f} mean over the horizon (median "
        f"${horizon['lease_median']:,.0f}, {band} ${horizon['lease_lo']:,.0f}-${horizon['lease_hi']:,.0f}), "
        f"charged as entered, stepped up once a year by {latent}; not deductible"
    )
    print(f"  year 1: lease=${year_1['lease_mean']:,.0f} (as entered; first increase at the 1st anniversary of closing)")


def print_rent_reference_result_plots_report(result_plots: dict[str, Path]) -> None:
    """One line: where this reference case's rent_reference plot went (results.save_rent_reference_result_plots)."""
    print(f"  saved {len(result_plots)} rent_reference result plot to {result_plots['alt_lease_cost'].parent}")


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


def print_stale_forecast_models_warning(stale: list[str]) -> None:
    """One-line warning that these forecast_models entries are stale but were NOT refreshed (run/main.py
    without --refresh) -- the run continues on their saved models."""
    print(
        f"WARNING: stale forecast_models not refreshed ({', '.join(stale)}) -- "
        "using saved models; pass --refresh to update"
    )
