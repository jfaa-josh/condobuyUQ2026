"""Exercises the same scenario-running orchestration run/main.py uses -- for manual runs, not CI
(no CI is set up for this project): either `uv run pytest run/tests/test_acquisition.py -v -s`
(the -s is needed to see the printed output -- pytest hides stdout on a pass otherwise), or
`uv run python run/tests/test_acquisition.py` to run the checks in order as a plain script (prints
unconditionally, no -s needed).

Every test here prints its own important intermediate variables as
f"VARIABLE CHECK FOR {name}: {value}" -- test-script-only convention, never in main.py/backend
modules (see .claude/implementation-plan.md's "Backend build progress").
"""

import pytest

from condobuyuq2026.acquisition import compute_acquisition_costs
from condobuyuq2026.runner import run_scenarios


def test_run_scenarios_pairs_every_buy_scenario_with_a_cached_reference_result():
    result = run_scenarios()
    reference_cases, reference_results, scenarios = (
        result["reference_cases"],
        result["reference_results"],
        result["scenarios"],
    )

    print(f"VARIABLE CHECK FOR reference_cases: {reference_cases}")
    print(f"VARIABLE CHECK FOR reference_results: {reference_results}")
    print(f"VARIABLE CHECK FOR scenarios: {scenarios}")

    assert scenarios  # the deck's sweeps should always produce at least one buy scenario
    # Every distinct reference case actually got run exactly once (never re-run, never skipped).
    assert set(reference_results.keys()) == set(reference_cases.keys())
    # Every buy scenario's reference_id points at a real reference case.
    assert all(scenario["reference_id"] in reference_cases for scenario in scenarios)
    # Each stage's computed values are actually STORED on the scenario, not just printed and
    # discarded -- see runner.run_scenarios' module docstring.
    for scenario in scenarios:
        assert "acquisition" in scenario, scenario
        assert set(scenario["acquisition"].keys()) >= {"loan_amount", "total_cash_outlay", "closing_costs_total"}


def test_compute_acquisition_costs_raises_when_purchase_price_cant_cover_total_cash_outlay():
    """Can't borrow negative money -- see acquisition.compute_acquisition_costs's loan_amount<0
    guardrail."""
    tiny_purchase_scenario = {"purchase_price": 1000}
    print(f"VARIABLE CHECK FOR tiny_purchase_scenario: {tiny_purchase_scenario}")

    with pytest.raises(ValueError, match="negative loan_amount"):
        compute_acquisition_costs(tiny_purchase_scenario)


if __name__ == "__main__":
    test_run_scenarios_pairs_every_buy_scenario_with_a_cached_reference_result()
    test_compute_acquisition_costs_raises_when_purchase_price_cant_cover_total_cash_outlay()
    print("All acquisition checks passed.")
