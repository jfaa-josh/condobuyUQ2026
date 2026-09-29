"""Exercises utils.prior_utils.build_all_carrying_cost_prior_models and carrying_costs.py directly
(the actual backend logic, not a copy of it) -- for manual runs, not CI (no CI is set up for this
project): either `uv run pytest run/tests/test_carrying_costs.py -v -s`, or
`uv run python run/tests/test_carrying_costs.py` to run the checks in order as a plain script
(prints unconditionally, no -s needed).

Every test here prints its own important intermediate variables as
f"VARIABLE CHECK FOR {name}: {value}" -- test-script-only convention, never in main.py/backend
modules (see .claude/implementation-plan.md's "Backend build progress").

Order matters: the build test runs first (both here and in __main__) because every later test reads
models/carrying_cost_priors/*/fit.json, which it creates -- same dependency
test_derived_latents.py/test_financing.py already have on their own upstream fits.
"""

import pandas as pd
import pytest

import condobuyuq2026.carrying_costs as carrying_costs_module
from condobuyuq2026.acquisition import compute_acquisition_costs
from condobuyuq2026.carrying_costs import (
    compute_carrying_costs,
    compute_carrying_costs_schedule,
    compute_property_tax_schedule,
    load_market_value_change_magnitude,
    project_condo_value_schedule,
    project_market_value_schedule,
)
from condobuyuq2026.input_deck import (
    get_closing_date,
    get_hoa_dues_config,
    get_insurance_config,
    get_maintenance_config,
    get_market_value_config,
    get_property_tax_inputs,
    get_scenarios,
    get_utilities_prior_config,
    load_deck,
)
from condobuyuq2026.reporting import print_carrying_cost_prior_build_report
from condobuyuq2026.results import save_carrying_cost_result_plots
from condobuyuq2026.utils.prior_utils import (
    build_all_carrying_cost_prior_models,
    fit_market_value_calibration,
    load_market_value_history,
    months_to_first_reassessment,
    reassessment_boundary_months,
)


def _build_scenario_with_acquisition(deck, min_horizon_years: int = 1):
    """Runs just enough of the real pipeline (acquisition) for carrying_costs.py's own inputs -- it
    doesn't need classification/financing at all. min_horizon_years picks a scenario long enough to
    exercise multiple market_value reassessment periods."""
    scenario = next(s for s in get_scenarios(deck) if s["horizon_years"] >= min_horizon_years)
    scenario["acquisition"] = compute_acquisition_costs(scenario, deck)
    return scenario


def test_build_all_carrying_cost_prior_models_saves_fit_and_plot_for_every_entry():
    deck = load_deck()
    results = build_all_carrying_cost_prior_models(deck)
    print(f"VARIABLE CHECK FOR results: {results}")

    assert set(results.keys()) == {
        "hoa_dues_annual",
        "insurance_annual",
        "maintenance_annual",
        "market_value",
        "special_assessment",
        "utilities",
    }
    for name, paths in results.items():
        assert paths["fit_path"].exists() and paths["fit_path"].stat().st_size > 0, name
        assert paths["plot_path"].exists() and paths["plot_path"].stat().st_size > 0, name
        assert paths["fit_path"].name == "fit.json"
        assert paths["plot_path"].parent.name == "plots"

    print_carrying_cost_prior_build_report(results)


def test_fit_market_value_calibration_fits_a_line_through_known_observations():
    """load_market_value_history combines the deck's own validation_data_source CSV (currently one
    real historical row, 2023-2024) with an automatically-synthesized CURRENT-cycle record built
    from initial_price/first_reassessment_year/reassessment_frequency_years (never hand-duplicated
    in the CSV -- see that function's own docstring) -- so at least 2 observations should always be
    available here. fit_market_value_calibration should fit a line (slope+intercept) through their
    (denver_window_avg, keystone_actual) pairs -- with exactly 2 points, EXACTLY (zero residual for
    both) -- using each observation's own real averaging window (no fitted OU parameter -- see that
    function's own docstring for why)."""
    deck = load_deck()
    history = load_market_value_history(deck)
    assert len(history) >= 2

    calibration = fit_market_value_calibration(deck, history)
    print(f"VARIABLE CHECK FOR fit_market_value_calibration: {calibration}")

    assert len(calibration["observations"]) == len(history)
    for observation in calibration["observations"]:
        assert observation["denver_window_avg"] > 0
        assert observation["keystone_actual"] > 0
        predicted = calibration["slope"] * observation["denver_window_avg"] + calibration["intercept"]
        if len(history) == 2:
            # A line is exactly determined by 2 points -- zero residual for both.
            assert abs(predicted - observation["keystone_actual"]) < 1e-6


def test_market_value_calibration_is_saved_against_known_history():
    """With >= 2 known historical observations, build_market_value_informed_model should save a
    validation plot/report showing the fitted line against the real observations that informed it --
    without the report itself altering anything (see that function's own docstring)."""
    deck = load_deck()
    history = load_market_value_history(deck)
    assert len(history) >= 2

    results = build_all_carrying_cost_prior_models(deck)
    market_value_result = results["market_value"]
    print(f"VARIABLE CHECK FOR market_value calibration_report: {market_value_result.get('calibration_report')}")

    assert "calibration_report" in market_value_result
    calibration = market_value_result["calibration_report"]
    assert "calibration_plot_path" in market_value_result
    assert market_value_result["calibration_plot_path"].exists()
    assert market_value_result["calibration_plot_path"].stat().st_size > 0
    assert market_value_result["calibration_plot_path"].name == "calibration_fit.png"
    assert len(calibration["observations"]) == len(history)
    for observation in calibration["observations"]:
        assert observation["keystone_actual"] > 0


def test_hoa_insurance_maintenance_utilities_configs_match_deck():
    """hoa_dues_annual/utilities have a KNOWN starting value (0% uncertain); insurance_annual/
    maintenance_annual have a hand-elicited GUESS (moved insurance_annual here 2026-09-30 -- a
    current premium turned out to be an estimate too, not known truth data) -- each its own
    dedicated getter, no shared generic one."""
    deck = load_deck()

    hoa = get_hoa_dues_config(deck)
    print(f"VARIABLE CHECK FOR get_hoa_dues_config: {hoa}")
    assert hoa["initial_annual_dues"] > 0
    assert hoa["initial_annual_association_dues"] >= 0
    assert hoa["latent"]  # a non-empty derived_latents name

    insurance = get_insurance_config(deck)
    print(f"VARIABLE CHECK FOR get_insurance_config: {insurance}")
    assert insurance["initial_guess_premium"] > 0
    assert insurance["latent"]

    maintenance = get_maintenance_config(deck)
    print(f"VARIABLE CHECK FOR get_maintenance_config: {maintenance}")
    assert maintenance["initial_guess_mean"] > 0
    assert maintenance["latent"]

    utilities = get_utilities_prior_config(deck)
    print(f"VARIABLE CHECK FOR get_utilities_prior_config: {utilities}")
    assert utilities["initial_annual_average_monthly_cost"] > 0
    assert utilities["seasonal_data_source"]
    assert utilities["latent"]


def test_market_value_config_matches_deck():
    """market_value has its own dedicated shape (initial_price/reassessment_frequency_years/
    first_reassessment_year/latent), not the generic LEVEL+GROWTH prior shape -- see
    get_market_value_config's own docstring for why."""
    deck = load_deck()
    config = get_market_value_config(deck)
    print(f"VARIABLE CHECK FOR get_market_value_config: {config}")

    assert config["initial_price"] > 0
    assert config["reassessment_frequency_years"] > 0
    assert isinstance(config["first_reassessment_year"], int)
    assert config["first_reassessment_year"] >= 1
    assert config["latent"]  # a non-empty founding-process or derived_latents name


def test_closing_date_matches_deck():
    deck = load_deck()
    closing = get_closing_date(deck)
    print(f"VARIABLE CHECK FOR get_closing_date: {closing}")
    assert 1 <= closing["closing_month"] <= 12
    assert closing["closing_year"] > 2000


def test_project_condo_value_schedule_starts_at_purchase_price():
    deck = load_deck()
    scenario = _build_scenario_with_acquisition(deck)
    schedule = project_condo_value_schedule(scenario, deck)
    print(f"VARIABLE CHECK FOR condo_value_schedule: {schedule.to_dict()}")

    assert len(schedule) == scenario["horizon_years"] + 1
    assert abs(schedule.loc[0] - scenario["acquisition"]["purchase_price"]) < 1e-6


def test_current_cycle_observation_is_synthesized_not_duplicated_in_the_csv():
    """The CSV (data/manual/<validation_data_source>) must NOT contain a row for the CURRENT cycle
    (the one initial_price itself reflects) -- load_market_value_history synthesizes that one
    directly from initial_price/first_reassessment_year/reassessment_frequency_years so it can never
    drift out of sync with initial_price. Given this deck's own example values (closing Oct 2026,
    first_reassessment_year=1, reassessment_frequency_years=2), the current cycle's own averaging
    window must be Jan 2023-Jun 2024 (the window the user says actually produced initial_price)."""
    deck = load_deck()
    config = get_market_value_config(deck)
    history = load_market_value_history(deck)
    print(f"VARIABLE CHECK FOR load_market_value_history: {history}")

    current_cycle_records = [r for r in history if r["assessed_value"] == config["initial_price"]]
    assert len(current_cycle_records) == 1
    current = current_cycle_records[0]
    assert current["window_start"] == pd.Timestamp("2023-01-01")
    assert current["window_end"] == pd.Timestamp("2024-06-01")


def test_reassessment_boundary_months_matches_manual_calculation():
    """Worked example from input_deck.yaml's own market_value comment: closing in October,
    first_reassessment_year=2 -> 15 months to the first reassessment (Nov+Dec of the closing year,
    all of the next year, plus January of the year after -- 24 - 9 = 15)."""
    assert months_to_first_reassessment(closing_month=10, first_reassessment_year=2) == 15
    assert months_to_first_reassessment(closing_month=1, first_reassessment_year=1) == 12

    boundaries = reassessment_boundary_months(
        closing_month=10, first_reassessment_year=2, reassessment_frequency_years=2, num_periods=3
    )
    print(f"VARIABLE CHECK FOR reassessment_boundary_months: {boundaries}")
    assert boundaries == [0, 15, 39, 63]


def test_market_value_change_magnitude_periods_are_deterministic_before_first_reassessment():
    """Before the first reassessment, the period must reproduce market_value.initial_price EXACTLY,
    already in USD, DIRECTLY -- not derived from the fitted calibration line at all (projected_value=0,
    growth_sigma=0 -- known, constant, zero variance -- see build_market_value_change_magnitude's
    docstring for why this is an unconditional guarantee, independent of the calibration)."""
    deck = load_deck()
    build_all_carrying_cost_prior_models(deck)  # ensures the fit exists for this test's own read
    change_magnitude = load_market_value_change_magnitude()
    print(f"VARIABLE CHECK FOR market_value change_magnitude periods: {change_magnitude['periods']}")

    first_period = change_magnitude["periods"][0]
    assert first_period["start_month"] == 0
    assert first_period["projected_value"] == 0.0
    assert first_period["growth_sigma"] == 0.0
    initial_price = get_market_value_config(deck)["initial_price"]
    assert first_period["known_value"] == initial_price

    # Every later period's own uncertainty should GROW the further out it is (each window is
    # further along the latent's own widening projection).
    later_sigmas = [p["growth_sigma"] for p in change_magnitude["periods"][1:]]
    assert later_sigmas == sorted(later_sigmas)
    assert later_sigmas[-1] > 0


def test_market_value_schedule_steps_and_matches_the_saved_change_magnitude():
    """Consecutive years share a value exactly when they resolve to the same precomputed
    reassessment period -- and the scaled median must equal known_value+projected_value for that
    period, directly from the saved (already-USD) change-magnitude model, not recomputed
    independently."""
    deck = load_deck()
    scenario = _build_scenario_with_acquisition(deck, min_horizon_years=10)
    schedule = project_market_value_schedule(scenario, deck)
    print(f"VARIABLE CHECK FOR market_value_schedule:\n{schedule}")

    change_magnitude = load_market_value_change_magnitude()
    periods = change_magnitude["periods"]

    def _period_for(h_months):
        for period in periods:
            if h_months >= period["start_month"] and (period["end_month"] is None or h_months < period["end_month"]):
                return period
        raise AssertionError(f"no period covers h_months={h_months}")

    horizon_years = scenario["horizon_years"]
    expected_periods = [_period_for(year * 12) for year in range(1, horizon_years + 1)]
    for year in range(2, horizon_years + 1):
        same_expected = expected_periods[year - 1] is expected_periods[year - 2]
        same_actual = schedule.loc[year, "median"] == schedule.loc[year - 1, "median"]
        assert same_actual == same_expected, (year, expected_periods[year - 1], expected_periods[year - 2])

    year_1_period = expected_periods[0]
    expected_median = year_1_period["known_value"] + year_1_period["projected_value"]
    assert abs(schedule.loc[1, "median"] - expected_median) < 1e-6
    # Year 1 must fall before the first reassessment given this deck's example values -- known
    # exactly, zero variance.
    assert schedule.loc[1, "lo"] == schedule.loc[1, "median"] == schedule.loc[1, "hi"]


def test_property_tax_schedule_matches_manual_calculation_from_deck_rates():
    deck = load_deck()
    scenario = _build_scenario_with_acquisition(deck, min_horizon_years=5)
    schedule = compute_property_tax_schedule(scenario, deck)
    print(f"VARIABLE CHECK FOR property_tax_schedule:\n{schedule}")

    inputs = get_property_tax_inputs(deck)
    print(f"VARIABLE CHECK FOR get_property_tax_inputs: {inputs}")
    market_value_schedule = project_market_value_schedule(scenario, deck)
    combined_assessed_rate = 0.90 * inputs["assessment_rate_local"] + 1.00 * inputs["assessment_rate_school"]
    combined_tax_rate = (
        0.90 * inputs["assessment_rate_local"] * inputs["mill_levy_local"]
        + 1.00 * inputs["assessment_rate_school"] * inputs["mill_levy_school"]
    )

    year = 5  # by year 5, market_value carries a real band (past the first reassessment) -- a
    # sharper check than year 1, which this deck's example values leave at zero variance.
    market_value_year = market_value_schedule.loc[year, "median"]
    assert abs(schedule.loc[year, "assessed_value_median"] - market_value_year * combined_assessed_rate) < 1e-6
    assert abs(schedule.loc[year, "tax_median"] - market_value_year * combined_tax_rate) < 1e-6
    assert schedule.loc[year, "assessed_value_lo"] < schedule.loc[year, "assessed_value_median"] < schedule.loc[year, "assessed_value_hi"]
    assert schedule.loc[year, "tax_lo"] < schedule.loc[year, "tax_median"] < schedule.loc[year, "tax_hi"]


def test_property_tax_raises_for_an_unimplemented_classification():
    """classification is hardcoded to "residential" (see carrying_costs.PROPERTY_TAX_CLASSIFICATION's
    own comment) -- anything else has no defined local/school split yet."""
    deck = load_deck()
    scenario = _build_scenario_with_acquisition(deck)
    original = carrying_costs_module.PROPERTY_TAX_CLASSIFICATION
    carrying_costs_module.PROPERTY_TAX_CLASSIFICATION = "lodging"
    try:
        with pytest.raises(NotImplementedError, match="lodging"):
            carrying_costs_module.compute_property_tax_schedule(scenario, deck)
    finally:
        carrying_costs_module.PROPERTY_TAX_CLASSIFICATION = original


def test_carrying_costs_schedule_has_all_columns_and_positive_totals():
    deck = load_deck()
    scenario = _build_scenario_with_acquisition(deck)
    schedule = compute_carrying_costs_schedule(scenario, deck)
    print(f"VARIABLE CHECK FOR carrying_costs_schedule:\n{schedule}")

    expected_columns = {"property_tax", "hoa_dues", "insurance", "maintenance", "special_assessment_expected", "utilities", "total"}
    assert set(schedule.columns) == expected_columns
    assert len(schedule) == scenario["horizon_years"]
    assert (schedule["total"] > 0).all()
    # "total" really is the sum of the other six, not something separately computed.
    assert (abs(schedule["total"] - schedule.drop(columns="total").sum(axis=1)) < 1e-6).all()


def test_compute_carrying_costs_end_to_end():
    deck = load_deck()
    scenario = _build_scenario_with_acquisition(deck)
    carrying_costs = compute_carrying_costs(scenario, deck)
    print(f"VARIABLE CHECK FOR total_carrying_costs: {carrying_costs['total_carrying_costs']}")

    assert set(carrying_costs.keys()) == {"annual_schedule", "property_tax_schedule", "total_carrying_costs"}
    assert abs(carrying_costs["total_carrying_costs"] - carrying_costs["annual_schedule"]["total"].sum()) < 1e-6
    # annual_schedule's own "property_tax" column is property_tax_schedule's median, not something
    # separately computed.
    assert (
        abs(carrying_costs["annual_schedule"]["property_tax"] - carrying_costs["property_tax_schedule"]["tax_median"]) < 1e-6
    ).all()


def test_save_carrying_cost_result_plots_saves_every_expected_plot():
    deck = load_deck()
    scenario = _build_scenario_with_acquisition(deck, min_horizon_years=5)
    paths = save_carrying_cost_result_plots(scenario, deck)
    print(f"VARIABLE CHECK FOR result plot paths: {paths}")

    expected_names = {
        "hoa_dues",
        "insurance",
        "maintenance",
        "special_assessment_expected",
        "utilities",
        "property_tax",
        "total_carrying_costs",
    }
    assert set(paths.keys()) == expected_names
    for name, path in paths.items():
        assert path.exists() and path.stat().st_size > 0, name
        if name == "total_carrying_costs":
            # the one exception: saved in the PARENT results/<scenario>/ folder, not carrying_costs/
            assert path.parent.name != "carrying_costs"
        else:
            assert path.parent.name == "carrying_costs"


if __name__ == "__main__":
    test_build_all_carrying_cost_prior_models_saves_fit_and_plot_for_every_entry()
    test_fit_market_value_calibration_fits_a_line_through_known_observations()
    test_market_value_calibration_is_saved_against_known_history()
    test_hoa_insurance_maintenance_utilities_configs_match_deck()
    test_market_value_config_matches_deck()
    test_closing_date_matches_deck()
    test_current_cycle_observation_is_synthesized_not_duplicated_in_the_csv()
    test_project_condo_value_schedule_starts_at_purchase_price()
    test_reassessment_boundary_months_matches_manual_calculation()
    test_market_value_change_magnitude_periods_are_deterministic_before_first_reassessment()
    test_market_value_schedule_steps_and_matches_the_saved_change_magnitude()
    test_property_tax_schedule_matches_manual_calculation_from_deck_rates()
    test_property_tax_raises_for_an_unimplemented_classification()
    test_carrying_costs_schedule_has_all_columns_and_positive_totals()
    test_compute_carrying_costs_end_to_end()
    test_save_carrying_cost_result_plots_saves_every_expected_plot()
    print("All carrying_costs checks passed.")
