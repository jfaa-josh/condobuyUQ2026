"""Resolves the applicable mortgage rate for one scenario (via classification.
get_loan_occupancy_class) and amortizes acquisition.compute_acquisition_costs' loan_amount over
that scenario's own horizon -- see input_deck.yaml's financing section.

Deliberately NOT part of input_deck.py (deck loading/validation only) or classification.py
(occupancy/tax classification only) -- this is the loan's own cost, the first of this deck section's
computations. Kept separate from classification.py even though it immediately consumes
get_loan_occupancy_class, the same "one computation module per deck section" pattern acquisition.py
established -- see .claude/implementation-plan.md's "Backend build progress".
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from condobuyuq2026.acquisition import compute_acquisition_costs
from condobuyuq2026.classification import get_loan_occupancy_class
from condobuyuq2026.input_deck import get_financing_inputs, load_deck


def get_mortgage_rate(loan_occupancy_class: str, deck: dict[str, Any] | None = None) -> float:
    """Resolves financing.mortgage_rate.<loan_occupancy_class> (one of "primary_residence",
    "second_home", "investment" -- see classification.get_loan_occupancy_class) plus
    financing.non_warrantable_premium on top (0.0 by default -- see that field's own deck
    comment)."""
    deck = deck if deck is not None else load_deck()
    inputs = get_financing_inputs(deck)
    base_rate = inputs[f"mortgage_rate_{loan_occupancy_class}"]
    return base_rate + inputs["non_warrantable_premium"]


def amortize_loan(
    loan_amount: float, annual_rate: float, loan_term_years: int | float, horizon_years: int | float
) -> pd.DataFrame:
    """Standard fixed-rate monthly amortization schedule for loan_amount at annual_rate over
    loan_term_years, truncated to horizon_years (the scenario's own horizon -- a sale doesn't wait
    for the mortgage to fully amortize, and horizon_years is always <= loan_term_years for every
    scenario.horizon_years value used so far, but this doesn't assume that).

    loan_term_years/horizon_years accept int or float (YAML/dict-sourced values aren't strictly
    typed) but are cast to int below -- range() needs a real int, and a fractional number of years
    wouldn't mean anything here anyway.

    Returns a DataFrame indexed by month offset h = 0..min(horizon_years, loan_term_years)*12 with
    columns "payment", "interest", "principal", "balance" -- h=0 is the moment the loan originates
    (payment/interest/principal all 0.0, balance = loan_amount); each later row is that month's
    actual cash flows and the balance remaining at the END of that month.
    """
    monthly_rate = annual_rate / 12
    n_payments = int(loan_term_years * 12)
    months_to_run = int(min(horizon_years, loan_term_years) * 12)

    if monthly_rate == 0:
        payment = loan_amount / n_payments if n_payments else 0.0
    else:
        payment = loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** (-n_payments))

    payments, interests, principals, balances = [0.0], [0.0], [0.0], [loan_amount]
    balance = loan_amount
    for _ in range(months_to_run):
        interest = balance * monthly_rate
        principal = payment - interest
        balance = max(balance - principal, 0.0)
        payments.append(payment)
        interests.append(interest)
        principals.append(principal)
        balances.append(balance)

    return pd.DataFrame(
        {"payment": payments, "interest": interests, "principal": principals, "balance": balances},
        index=pd.Index(range(months_to_run + 1), name="h_months"),
    )


def compute_financing(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Computes one scenario's financing-stage results: reuses scenario["classification"]'s
    loan_occupancy_class if runner.run_scenarios already set it, otherwise resolves it fresh (see
    classification.get_loan_occupancy_class); either way, resolves the applicable mortgage_rate
    (see get_mortgage_rate), then amortizes acquisition.compute_acquisition_costs' loan_amount over
    that scenario's own horizon_years (see amortize_loan).

    Returns:
    - loan_occupancy_class / mortgage_rate: as resolved above.
    - loan_term_years: financing.loan_term_years, as given.
    - amortization_schedule: the full month-by-month DataFrame from amortize_loan -- this IS the
      loan's cost propagated over the scenario's horizon; kept in full (not just a total) so any
      later stage needing a specific month's balance/interest (e.g. exit's payoff amount, or a
      mortgage-interest tax deduction by year) doesn't need to recompute it.
    - total_interest_paid: amortization_schedule's "interest" column summed over the horizon.
    - ending_balance: amortization_schedule's last "balance" -- what's still owed if the property
      sells at horizon_years (needed by the future exit stage to compute net sale proceeds).
    """
    deck = deck if deck is not None else load_deck()
    inputs = get_financing_inputs(deck)
    classification = scenario.get("classification") or {}
    loan_occupancy_class = classification.get("loan_occupancy_class") or get_loan_occupancy_class(scenario, deck)
    mortgage_rate = get_mortgage_rate(loan_occupancy_class, deck)

    acquisition = scenario.get("acquisition") or compute_acquisition_costs(scenario, deck)
    loan_amount = acquisition["loan_amount"]
    amortization_schedule = amortize_loan(
        loan_amount, mortgage_rate, inputs["loan_term_years"], scenario["horizon_years"]
    )

    return {
        "loan_occupancy_class": loan_occupancy_class,
        "mortgage_rate": mortgage_rate,
        "loan_term_years": inputs["loan_term_years"],
        "amortization_schedule": amortization_schedule,
        "total_interest_paid": amortization_schedule["interest"].sum(),
        "ending_balance": amortization_schedule["balance"].iloc[-1],
    }
