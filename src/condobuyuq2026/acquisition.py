"""Computes `acquisition`'s derived dollar amounts for one scenario -- purchase closing costs, the
amount financed, and the total cash outlay -- see input_deck.yaml's acquisition section and its
"COMPOSITE COST FORMULAS" schema-reference note.

Deliberately NOT part of input_deck.py: that module's job is loading/validating/normalizing the
deck (see its own docstring), not computing derived business quantities from it. This is meant to
be the first of one such computation module per deck section (financing, carrying_costs, exit,
...) as the backend gets built out stage by stage -- see .claude/implementation-plan.md.

NOTE on land/building value: acquisition.land_fraction (see input_deck.get_acquisition_inputs) is
just the assessor's land-vs-improvement RATIO -- on its own it's not an acquisition-stage cost, it
only matters once something actually depreciates the building over time (the future taxes/exit
stage). Not computed here; revisit alongside that stage instead of guessing its shape now.
"""

from __future__ import annotations

from typing import Any

from condobuyuq2026.input_deck import get_acquisition_inputs, load_deck


def compute_acquisition_costs(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, float]:
    """Computes one scenario's (see input_deck.get_scenarios) acquisition-stage dollar amounts,
    using scenario["purchase_price"] as the actual input for acquisition.purchase_price's
    discoverability stub (see the deck's "SEE scenario.X" note).

    Returns:
    - purchase_price: as given by the scenario.
    - closing_costs_fixed_fees / closing_costs_percent_of_price / closing_costs_total: from
      acquisition.purchase_closing_costs, total = fixed_fees + percent_of_price * purchase_price
      (the deck's "COMPOSITE COST FORMULAS" pattern).
    - family_loan: acquisition.family_loan, as given (a 0%-interest loan, so it's treated as
      ordinary cash here too -- no separate interest-cost modeling needed).
    - total_cash_outlay: DERIVED, acquisition.personal_cash_at_closing + family_loan -- ALL cash put
      in at closing. This is the single reported cash figure; used by
      scenarios.get_scenario_set only to dedup how many distinct "rent" reference cases are needed,
      never as a value that scenario actually invests (see that module's own docstring).
    - loan_amount: DERIVED, purchase_price + closing_costs_total - total_cash_outlay -- the amount
      financed by the mortgage. Raises ValueError if this comes out negative -- purchase_price too
      low (or total_cash_outlay too high) to make sense as a mortgage; can't borrow negative money.
    - initial_furnishing: acquisition.initial_furnishing, as given (a cash outlay, not
      depreciated, and NOT folded into total_cash_outlay above -- that's specifically the
      at-closing figure).
    """
    deck = deck if deck is not None else load_deck()
    inputs = get_acquisition_inputs(deck)

    purchase_price = scenario["purchase_price"]
    closing_costs_total = inputs["closing_costs_fixed_fees"] + inputs["closing_costs_percent_of_price"] * purchase_price
    total_cash_outlay = inputs["personal_cash_at_closing"] + inputs["family_loan"]
    loan_amount = purchase_price + closing_costs_total - total_cash_outlay
    if loan_amount < 0:
        raise ValueError(
            f"scenario purchase_price=${purchase_price:,.0f} is too low for "
            f"total_cash_outlay=${total_cash_outlay:,.0f} (plus closing costs "
            f"${closing_costs_total:,.0f}) -- this would require a negative loan_amount "
            f"(${loan_amount:,.0f}); can't borrow negative money. Lower personal_cash_at_closing/"
            "family_loan, raise this purchase_price, or drop it from scenario.purchase_price's "
            "sweep."
        )

    return {
        "purchase_price": purchase_price,
        "closing_costs_fixed_fees": inputs["closing_costs_fixed_fees"],
        "closing_costs_percent_of_price": inputs["closing_costs_percent_of_price"],
        "closing_costs_total": closing_costs_total,
        "loan_amount": loan_amount,
        "family_loan": inputs["family_loan"],
        "initial_furnishing": inputs["initial_furnishing"],
        "total_cash_outlay": total_cash_outlay,
    }
