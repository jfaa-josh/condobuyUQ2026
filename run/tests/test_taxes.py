"""Exercises taxes.py directly (the actual backend logic, not a copy of it) -- for manual runs, not
CI (no CI is set up for this project): either `uv run pytest run/tests/test_taxes.py -v -s`, or
`uv run python run/tests/test_taxes.py` to run the checks in order as a plain script (prints
unconditionally, no -s needed).

*** taxes.py has ONE remaining PLACEHOLDER input (gross_rent) -- see its module docstring and
PLACEHOLDER_FIELDS constant. These tests check the MACHINERY is wired correctly, not that the
dollar figures are realistic. ***

Every test here prints its own important intermediate variables as
f"VARIABLE CHECK FOR {name}: {value}" -- test-script-only convention, never in main.py/backend
modules (see .claude/implementation-plan.md's "Backend build progress").
"""

from condobuyuq2026.acquisition import compute_acquisition_costs
from condobuyuq2026.carrying_costs import compute_carrying_costs
from condobuyuq2026.classification import compute_occupancy_classification
from condobuyuq2026.financing import compute_financing
from condobuyuq2026.input_deck import get_scenarios, load_deck
from condobuyuq2026.taxes import (
    PLACEHOLDER_FIELDS,
    compute_owner_tax_schedule,
    compute_owner_taxes,
    compute_renter_taxes,
    placeholder_gross_rent,
)


def _build_real_scenario(deck):
    """Runs the real pipeline (acquisition -> classification -> financing -> carrying_costs) on one
    real scenario, exactly like runner.run_scenarios does, so taxes.py sees the same shape it does
    for real."""
    scenario = get_scenarios(deck)[0]
    scenario["acquisition"] = compute_acquisition_costs(scenario, deck)
    scenario["classification"] = compute_occupancy_classification(scenario, deck)
    scenario["financing"] = compute_financing(scenario, deck)
    scenario["carrying_costs"] = compute_carrying_costs(scenario, deck)
    return scenario


def test_placeholder_gross_rent_is_positive_and_clearly_documented():
    deck = load_deck()
    gross_rent = placeholder_gross_rent(deck)
    print(f"VARIABLE CHECK FOR placeholder_gross_rent: {gross_rent}")
    print(f"VARIABLE CHECK FOR PLACEHOLDER_FIELDS: {PLACEHOLDER_FIELDS}")

    assert gross_rent > 0
    assert set(PLACEHOLDER_FIELDS.keys()) == {"gross_rent"}


def test_owner_tax_schedule_accumulates_depreciation_monotonically():
    deck = load_deck()
    scenario = _build_real_scenario(deck)
    schedule = compute_owner_tax_schedule(scenario, deck)
    print(f"VARIABLE CHECK FOR annual_schedule:\n{schedule}")

    assert len(schedule) == scenario["horizon_years"]
    # Cumulative depreciation never decreases year over year.
    assert (schedule["cumulative_depr_taken"].diff().dropna() >= -1e-9).all()
    assert set(schedule["use_class"].unique()) == {scenario["classification"]["tax_use_class"]}
    # property_tax/shared_expenses are now REAL (carrying_costs.py), not flat placeholders --
    # property_tax should vary with the condo's own appreciating value.
    assert schedule["property_tax"].iloc[0] > 0


def test_compute_owner_taxes_end_to_end_for_a_real_scenario():
    deck = load_deck()
    scenario = _build_real_scenario(deck)
    print(f"VARIABLE CHECK FOR scenario['classification']: {scenario['classification']}")
    print(f"VARIABLE CHECK FOR scenario['carrying_costs']['total_carrying_costs']: {scenario['carrying_costs']['total_carrying_costs']}")

    taxes = compute_owner_taxes(scenario, deck)
    print(f"VARIABLE CHECK FOR taxes['sale']: {taxes['sale']}")
    print(f"VARIABLE CHECK FOR taxes['total_tax']: {taxes['total_tax']}")
    print(f"VARIABLE CHECK FOR taxes['flags']: {taxes['flags']}")
    print(f"VARIABLE CHECK FOR taxes['placeholder_fields']: {taxes['placeholder_fields']}")

    assert set(taxes.keys()) == {"annual_schedule", "sale", "total_tax", "flags", "placeholder_fields"}
    assert taxes["placeholder_fields"] == ["gross_rent"]
    # Sale-year math is internally consistent: recapture never exceeds the gain, and the pieces of
    # the gain that get excluded/taxed as LTCG are both non-negative.
    sale = taxes["sale"]
    assert sale["recapture"] <= max(sale["gain"], 0.0) + 1e-6
    assert sale["excluded_121"] >= 0 and sale["ltcg"] >= 0


def test_owner_tax_schedule_branches_match_tax_use_class():
    """Directly exercises all three tax_use_class branches with the SAME scenario, varying only
    the classification -- personal_use must always show $0 taxable_rental; mixed_use/rental should
    generally differ once there's actual rental activity."""
    deck = load_deck()
    base_scenario = _build_real_scenario(deck)

    results = {}
    for tax_use_class in ("personal_use", "mixed_use", "rental"):
        scenario = dict(base_scenario)
        scenario["classification"] = dict(base_scenario["classification"], tax_use_class=tax_use_class)
        results[tax_use_class] = compute_owner_tax_schedule(scenario, deck)
        print(f"VARIABLE CHECK FOR {tax_use_class} schedule['taxable_rental']: "
              f"{results[tax_use_class]['taxable_rental'].tolist()}")

    assert (results["personal_use"]["taxable_rental"] == 0).all()
    assert (results["personal_use"]["cumulative_depr_taken"] == 0).all()


def test_compute_renter_taxes_is_always_zero():
    taxes = compute_renter_taxes({"horizon_years": 5, "residency_state": "FL"})
    print(f"VARIABLE CHECK FOR renter taxes: {taxes}")
    assert taxes["total_tax"] == 0.0
    assert taxes["sale"]["tax"] == 0.0


if __name__ == "__main__":
    test_placeholder_gross_rent_is_positive_and_clearly_documented()
    test_owner_tax_schedule_accumulates_depreciation_monotonically()
    test_compute_owner_taxes_end_to_end_for_a_real_scenario()
    test_owner_tax_schedule_branches_match_tax_use_class()
    test_compute_renter_taxes_is_always_zero()
    print("All taxes checks passed.")
