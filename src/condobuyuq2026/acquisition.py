"""Computes `acquisition`'s derived dollar amounts for one scenario -- purchase closing costs, the
amount financed, and the land/building split used later for depreciation -- see input_deck.yaml's
acquisition section and its "COMPOSITE COST FORMULAS" schema-reference note.

Deliberately NOT part of input_deck.py: that module's job is loading/validating/normalizing the
deck (see its own docstring), not computing derived business quantities from it. This is meant to
be the first of one such computation module per deck section (financing, carrying_costs, exit,
...) as the backend gets built out stage by stage -- see .claude/implementation-plan.md.
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
    - total_cash_at_closing: acquisition.total_cash_at_closing, as given (down payment + any
      gift/family money) -- deliberately not derived from anything else here.
    - loan_amount: DERIVED, purchase_price + closing_costs_total - total_cash_at_closing (see
      acquisition.total_cash_at_closing's own deck comment for this formula) -- the amount
      financed by the mortgage.
    - land_value / building_value: purchase_price split via acquisition.land_fraction (land isn't
      depreciable; the building is -- see that field's own deck comment).
    - initial_furnishing: acquisition.initial_furnishing, as given (a cash outlay, not
      depreciated).
    - total_cash_outlay: total_cash_at_closing + initial_furnishing -- all of the buyer's own cash
      going out at acquisition (closing costs are already netted into loan_amount above, not paid
      separately out of pocket -- total_cash_at_closing's own comment is explicit that it's "ALL
      cash you put in at closing").
    """
    deck = deck if deck is not None else load_deck()
    inputs = get_acquisition_inputs(deck)

    purchase_price = scenario["purchase_price"]
    closing_costs_total = inputs["closing_costs_fixed_fees"] + inputs["closing_costs_percent_of_price"] * purchase_price
    loan_amount = purchase_price + closing_costs_total - inputs["total_cash_at_closing"]
    land_value = inputs["land_fraction"] * purchase_price
    building_value = purchase_price - land_value

    return {
        "purchase_price": purchase_price,
        "closing_costs_fixed_fees": inputs["closing_costs_fixed_fees"],
        "closing_costs_percent_of_price": inputs["closing_costs_percent_of_price"],
        "closing_costs_total": closing_costs_total,
        "total_cash_at_closing": inputs["total_cash_at_closing"],
        "loan_amount": loan_amount,
        "land_value": land_value,
        "building_value": building_value,
        "initial_furnishing": inputs["initial_furnishing"],
        "total_cash_outlay": inputs["total_cash_at_closing"] + inputs["initial_furnishing"],
    }
