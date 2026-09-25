"""Exercises the same acquisition-cost computation run/main.py calls, one scenario at a time -- for
manual runs, not CI (no CI is set up for this project): either
`uv run pytest run/tests/test_acquisition.py -v -s` (the -s is needed to see the printed costs --
pytest hides stdout on a pass otherwise), or `uv run python run/tests/test_acquisition.py` to run
the checks in order as a plain script (prints unconditionally, no -s needed).
"""

from condobuyuq2026.acquisition import compute_acquisition_costs
from condobuyuq2026.input_deck import get_scenarios
from condobuyuq2026.reporting import print_acquisition_report


def test_compute_acquisition_costs_for_every_scenario():
    scenarios = get_scenarios()
    assert scenarios  # the deck's sweeps should always produce at least one scenario

    for scenario in scenarios:
        costs = compute_acquisition_costs(scenario)

        # Cheap invariants -- the money in equals the money out, however it's split.
        assert abs(
            (costs["loan_amount"] + costs["total_cash_at_closing"])
            - (costs["purchase_price"] + costs["closing_costs_total"])
        ) < 1e-9
        assert abs(costs["land_value"] + costs["building_value"] - costs["purchase_price"]) < 1e-9

        print_acquisition_report(scenario, costs)


if __name__ == "__main__":
    test_compute_acquisition_costs_for_every_scenario()
