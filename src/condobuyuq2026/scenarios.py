"""Assembles the full set of scenarios to run: input_deck.get_scenarios' buy-side sweep, PLUS one
reference case per distinct (horizon_years, residency_state, total_cash_outlay) combination those
buy scenarios' own acquisition costs actually produce -- keyed by a plain generated id (1, 2, 3,
...) rather than inlined as its own bare, always-"rent" scenario dict.

Why a separate id/lookup instead of just another entry in the scenario list (the first version of
this module): every buy scenario dict used to carry a "reference": "rent" field that was the same
literal string on every single entry, while a handful of OTHER entries in the same list also had
"alternative": "rent" -- correct in spirit (every buy case is compared against a reference baseline)
but confusing and clunky as data: nothing tied a specific buy scenario to which of the (deduplicated)
reference cases it should actually be compared against, and skimming the list gave no sense of "6
buy cases, but only 3 distinct reference cases are actually needed."

Now: each buy scenario carries "reference_id" (e.g. 2), a key into a SEPARATE reference_cases dict
returned alongside it. This is what makes runner.run_scenarios' run-once-per-distinct-reference /
cache-and-pair strategy possible -- see that module. reference_id is a plain integer, not "rent_2"
-- "rent" is scenario.reference's value TODAY, but nothing here should assume it's the only
reference option a future deck revision might add.

NOT the rent branch's invested amount: total_cash_outlay is used ONLY to decide how many distinct
reference cases are needed (see below), never as a value to invest -- input_deck.yaml's
capital_markets.initial_investment (which imports acquisition.personal_cash_at_closing -- a
SEPARATE input from acquisition.compute_acquisition_costs' total_cash_outlay, see both fields' own
deck comments) is what the future capital_markets stage will actually read for that.

Buy scenarios that only differ in purchase_price but land on the same total_cash_outlay (e.g. a
flat acquisition.personal_cash_at_closing/family_loan, as in the current example deck -- every
purchase_price sweep value produces the identical total_cash_outlay) share ONE reference case
rather than one per purchase_price.
"""

from __future__ import annotations

from typing import Any

from condobuyuq2026.acquisition import compute_acquisition_costs
from condobuyuq2026.input_deck import get_scenarios, load_deck


def get_scenario_set(
    deck: dict[str, Any] | None = None,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Returns (reference_cases, scenarios):

    - reference_cases: {1: {...}, 2: {...}, ...} -- one entry per distinct (horizon_years,
      residency_state, total_cash_outlay) combination among the buy scenarios (see module
      docstring), keyed by a plain generated id in first-seen order. Each value has no
      purchase_price and no dollar figure -- just reference/horizon_years/residency_state/
      property_state. TODO: only these four fields exist today (enough for acquisition-stage
      dedup) -- once rent_reference/capital_markets get built, a reference case will need more
      variables of its own (rent cost, invested amount, ...) to actually be run.
    - scenarios: input_deck.get_scenarios' buy-side sweep, each dict carrying "reference_id" (one
      of reference_cases' own keys) in place of a bare, always-"rent" "reference" field -- look its
      matching reference case up via reference_cases[scenario["reference_id"]] (see
      runner.run_scenarios).
    """
    deck = deck if deck is not None else load_deck()
    buy_scenarios = get_scenarios(deck)

    reference_cases: dict[int, dict[str, Any]] = {}
    ids_by_key: dict[tuple[Any, Any, float], int] = {}
    scenarios = []
    for scenario in buy_scenarios:
        total_cash_outlay = compute_acquisition_costs(scenario, deck)["total_cash_outlay"]
        key = (scenario["horizon_years"], scenario["residency_state"], total_cash_outlay)
        if key not in ids_by_key:
            reference_id = len(reference_cases) + 1
            ids_by_key[key] = reference_id
            reference_cases[reference_id] = {
                "reference": scenario["reference"],
                "horizon_years": scenario["horizon_years"],
                "residency_state": scenario["residency_state"],
                "property_state": scenario["property_state"],
            }
        scenarios.append(
            {
                "alternative": scenario["alternative"],
                "horizon_years": scenario["horizon_years"],
                "residency_state": scenario["residency_state"],
                "property_state": scenario["property_state"],
                "purchase_price": scenario["purchase_price"],
                "reference_id": ids_by_key[key],
            }
        )
    return reference_cases, scenarios
