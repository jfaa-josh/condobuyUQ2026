"""Determines the two DERIVED occupancy classifications every buy scenario needs but no deck field
sets directly (see input_deck.yaml's "derived variables" schema-reference note):

- loan_occupancy_class (primary_residence | second_home | investment): what the LENDER cares about
  -- drives financing.mortgage_rate's applicable rate (see financing.py). Residency decides primary
  vs not (a CO resident living in the CO property IS a primary residence; FL residency can NEVER
  be, however many days spent there -- see financing section's own deck comment); if not primary,
  rental_operations.managed_by decides second_home (self-managed) vs investment (run through a
  property manager).
- tax_use_class (personal_use | mixed_use | rental): what the IRS cares about -- day-count based
  (IRC S280A "vacation home" rules, thresholds read from data/tax_jurisdictions/federal.yaml's
  vacation_home block), independent of loan_occupancy_class and of the passive-activity rules (a
  DIFFERENT test -- average guest-stay length, see taxes.py). Consumed by taxes.py.

Both live in one module (rather than loan_occupancy_class in financing.py and tax_use_class in a
future taxes.py) because they're determined from the same category of scenario/deck inputs
(residency, property management, day counts) even though different downstream stages consume them.
"""

from __future__ import annotations

from typing import Any

from condobuyuq2026.input_deck import (
    get_annual_personal_days,
    get_rental_operations_inputs,
    load_deck,
)
from condobuyuq2026.jurisdictions import load_federal_layer
from condobuyuq2026.rental_operations import rented_nights_by_month


def get_annual_rented_days(deck: dict[str, Any] | None = None) -> float:
    """Annual rented days for the day-count rules: the fitted MEDIAN rented nights per month (capped at
    the nights you leave available), summed -- a threshold rule, so the median, not the mean (see
    rental_operations.rented_nights_by_month). Uncertain occupancy could flip the class in a given year;
    not modeled."""
    return sum(rented_nights_by_month(deck, "median").values())


def get_loan_occupancy_class(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> str:
    """Resolves loan_occupancy_class for one scenario (see module docstring). scenario must carry
    "residency_state"/"property_state" (see input_deck.get_scenarios)."""
    deck = deck if deck is not None else load_deck()
    if scenario["residency_state"] == scenario["property_state"]:
        return "primary_residence"
    managed_by = get_rental_operations_inputs(deck)["managed_by"]
    return "investment" if managed_by == "company" else "second_home"


def get_tax_use_class(
    deck: dict[str, Any] | None = None,
    annual_rented_days: float | None = None,
    annual_personal_days: float | None = None,
) -> str:
    """Resolves tax_use_class from the deck's own annual day counts (see module docstring) --
    doesn't depend on the scenario at all (residency/purchase_price don't affect day counts).
    annual_rented_days/annual_personal_days are computed from deck if not given -- pass them
    through if a caller (compute_occupancy_classification) already has them, to avoid recomputing.
    """
    deck = deck if deck is not None else load_deck()
    if annual_rented_days is None:
        annual_rented_days = get_annual_rented_days(deck)
    if annual_personal_days is None:
        annual_personal_days = get_annual_personal_days(deck)

    rule = load_federal_layer().vacation_home
    if annual_rented_days <= rule["max_excluded_rented_days"]:
        return "personal_use"
    if annual_personal_days > max(rule["personal_days_floor"], rule["personal_fraction_of_rented_days"] * annual_rented_days):
        return "mixed_use"
    return "rental"


def compute_occupancy_classification(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Computes both classifications for one scenario, plus the annual day counts tax_use_class was
    decided from (for transparency/printing -- see reporting.print_classification_report). Returns
    {"loan_occupancy_class", "tax_use_class", "annual_rented_days", "annual_personal_days"}."""
    deck = deck if deck is not None else load_deck()
    annual_rented_days = get_annual_rented_days(deck)
    annual_personal_days = get_annual_personal_days(deck)
    return {
        "loan_occupancy_class": get_loan_occupancy_class(scenario, deck),
        "tax_use_class": get_tax_use_class(deck, annual_rented_days, annual_personal_days),
        "annual_rented_days": annual_rented_days,
        "annual_personal_days": annual_personal_days,
    }
