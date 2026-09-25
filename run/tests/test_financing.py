"""Exercises classification.py/financing.py directly (the actual backend logic, not a copy of it)
-- for manual runs, not CI (no CI is set up for this project): either
`uv run pytest run/tests/test_financing.py -v -s`, or `uv run python run/tests/test_financing.py`
to run the checks in order as a plain script (prints unconditionally, no -s needed).

Every test here prints its own important intermediate variables as
f"VARIABLE CHECK FOR {name}: {value}" -- test-script-only convention, never in main.py/backend
modules (see .claude/implementation-plan.md's "Backend build progress").
"""

from condobuyuq2026.classification import (
    get_annual_personal_days,
    get_annual_rented_days,
    get_loan_occupancy_class,
    get_tax_use_class,
)
from condobuyuq2026.financing import amortize_loan, compute_financing, get_mortgage_rate
from condobuyuq2026.input_deck import get_financing_inputs, get_scenarios, load_deck


def test_get_loan_occupancy_class_follows_residency_then_management():
    """Residency decides primary vs not; management decides second_home vs investment when not --
    see input_deck.yaml's financing section comment for the three example rows this mirrors."""
    deck_self_managed = load_deck()
    deck_self_managed["rental_operations"]["managed_by"]["value"] = "self"
    deck_company_managed = load_deck()
    deck_company_managed["rental_operations"]["managed_by"]["value"] = "company"

    primary_scenario = {"residency_state": "CO", "property_state": "CO"}
    non_primary_scenario = {"residency_state": "FL", "property_state": "CO"}

    primary_self = get_loan_occupancy_class(primary_scenario, deck_self_managed)
    primary_company = get_loan_occupancy_class(primary_scenario, deck_company_managed)
    second_home = get_loan_occupancy_class(non_primary_scenario, deck_self_managed)
    investment = get_loan_occupancy_class(non_primary_scenario, deck_company_managed)
    print(f"VARIABLE CHECK FOR primary_self: {primary_self}")
    print(f"VARIABLE CHECK FOR primary_company: {primary_company}")
    print(f"VARIABLE CHECK FOR second_home: {second_home}")
    print(f"VARIABLE CHECK FOR investment: {investment}")

    # Residency always wins, regardless of management.
    assert primary_self == "primary_residence"
    assert primary_company == "primary_residence"
    # Not primary: management decides.
    assert second_home == "second_home"
    assert investment == "investment"


def test_get_tax_use_class_matches_an_independent_recomputation_of_the_day_count_rule():
    deck = load_deck()
    annual_rented_days = get_annual_rented_days(deck)
    annual_personal_days = get_annual_personal_days(deck)
    print(f"VARIABLE CHECK FOR annual_rented_days: {annual_rented_days}")
    print(f"VARIABLE CHECK FOR annual_personal_days: {annual_personal_days}")

    if annual_rented_days <= 14:
        expected = "personal_use"
    elif annual_personal_days > 14 or annual_personal_days > 0.10 * annual_rented_days:
        expected = "mixed_use"
    else:
        expected = "rental"

    tax_use_class = get_tax_use_class(deck)
    print(f"VARIABLE CHECK FOR tax_use_class: {tax_use_class}")
    assert tax_use_class == expected


def test_get_mortgage_rate_resolves_base_rate_plus_non_warrantable_premium():
    deck = load_deck()
    inputs = get_financing_inputs(deck)
    print(f"VARIABLE CHECK FOR inputs: {inputs}")

    for loan_occupancy_class in ("primary_residence", "second_home", "investment"):
        rate = get_mortgage_rate(loan_occupancy_class, deck)
        expected = inputs[f"mortgage_rate_{loan_occupancy_class}"] + inputs["non_warrantable_premium"]
        print(f"VARIABLE CHECK FOR get_mortgage_rate({loan_occupancy_class!r}): {rate}")
        assert abs(rate - expected) < 1e-12


def test_amortize_loan_basic_invariants():
    loan_amount, annual_rate, loan_term_years, horizon_years = 300_000, 0.06, 30, 5
    schedule = amortize_loan(loan_amount, annual_rate, loan_term_years, horizon_years)
    print(f"VARIABLE CHECK FOR schedule.head():\n{schedule.head()}")
    print(f"VARIABLE CHECK FOR schedule.tail():\n{schedule.tail()}")

    assert len(schedule) == horizon_years * 12 + 1
    assert schedule["balance"].iloc[0] == loan_amount
    assert (schedule["balance"].diff().dropna() <= 1e-9).all()  # never increases
    assert schedule["balance"].iloc[-1] < loan_amount  # some principal paid down by year 5
    # First month's interest should be close to loan_amount * monthly_rate.
    assert abs(schedule["interest"].iloc[1] - loan_amount * annual_rate / 12) < 1e-6


def test_compute_financing_for_a_real_scenario():
    deck = load_deck()
    scenario = get_scenarios(deck)[0]
    print(f"VARIABLE CHECK FOR scenario: {scenario}")

    financing = compute_financing(scenario, deck)
    print(f"VARIABLE CHECK FOR financing: { {k: v for k, v in financing.items() if k != 'amortization_schedule'} }")
    print(f"VARIABLE CHECK FOR financing['amortization_schedule'].tail():\n{financing['amortization_schedule'].tail()}")

    assert financing["loan_occupancy_class"] in {"primary_residence", "second_home", "investment"}
    assert financing["mortgage_rate"] > 0
    assert 0 <= financing["ending_balance"] <= financing["amortization_schedule"]["balance"].iloc[0]
    assert financing["total_interest_paid"] > 0


if __name__ == "__main__":
    test_get_loan_occupancy_class_follows_residency_then_management()
    test_get_tax_use_class_matches_an_independent_recomputation_of_the_day_count_rule()
    test_get_mortgage_rate_resolves_base_rate_plus_non_warrantable_premium()
    test_amortize_loan_basic_invariants()
    test_compute_financing_for_a_real_scenario()
    print("All financing checks passed.")
