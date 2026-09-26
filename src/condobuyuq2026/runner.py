"""Runs every scenario in scenarios.get_scenario_set, pairing each buy scenario with its matching
reference (rent) case's result -- running a given reference case's own computation exactly ONCE
even though multiple buy scenarios may share it (see scenarios.py's module docstring for why that
sharing happens), by caching each reference_id's result the first time it's needed.

STORAGE, not just printing: each stage's computed values are kept, not only printed and discarded.
A buy scenario dict gets a new key per stage as it's computed -- so far "acquisition"
(acquisition.compute_acquisition_costs), "classification"
(classification.compute_occupancy_classification), "financing" (financing.compute_financing),
"carrying_costs" (carrying_costs.compute_carrying_costs), and "taxes" (taxes.compute_owner_taxes,
which reuses "carrying_costs" for its own property_tax/shared_expenses -- see that module's own
docstring) -- so by the time run_scenarios returns, every scenario in its "scenarios" list carries
everything computed for it, not just what got printed along the way. reference_results[
reference_id] is, symmetrically, a dict a future reference-side stage adds its own key to -- so far
just "taxes" (taxes.compute_renter_taxes, a real confirmed $0 result, not a placeholder -- renting
has no real-estate tax effect at all).

*** taxes.py has ONE remaining PLACEHOLDER input (gross_rent) -- see that module's docstring and
PLACEHOLDER_FIELDS constant before trusting any gross_rent-derived dollar figure it produces. ***

The eventual buy-vs-reference comparison (ΔW = W_buy - W_rent, see .claude/implementation-plan.md's
Goal) is still a PLACEHOLDER (see reporting.print_scenario_comparison_placeholder) -- as later
stages (rent_reference, capital_markets, exit) get built, this is where their per-scenario
computation gets wired in -- and stored, per the paragraph above -- on both the reference and buy
sides.
"""

from __future__ import annotations

from typing import Any

from condobuyuq2026.acquisition import compute_acquisition_costs
from condobuyuq2026.carrying_costs import compute_carrying_costs
from condobuyuq2026.classification import compute_occupancy_classification
from condobuyuq2026.financing import compute_financing
from condobuyuq2026.reporting import (
    print_acquisition_report,
    print_carrying_cost_result_plots_report,
    print_carrying_costs_report,
    print_classification_report,
    print_financing_report,
    print_reference_cache_hit,
    print_reference_report,
    print_renter_taxes_report,
    print_scenario_comparison_placeholder,
    print_taxes_report,
)
from condobuyuq2026.results import save_carrying_cost_result_plots
from condobuyuq2026.scenarios import get_scenario_set
from condobuyuq2026.taxes import compute_owner_taxes, compute_renter_taxes


def run_scenarios(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Iterates every buy scenario (see scenarios.get_scenario_set), running its matching reference
    case the FIRST time that reference_id is seen (cached after that -- see module docstring), then
    immediately following it with that scenario's own buy-side computation (acquisition,
    classification, financing, carrying_costs, taxes today; more as later stages are built), so a
    buy scenario and its reference are always handled as a pair.

    Returns {"reference_cases": ..., "reference_results": ..., "scenarios": ...}:
    - "scenarios": scenarios.get_scenario_set's buy scenarios, each one now ALSO carrying
      "acquisition", "classification", "financing", "carrying_costs", "taxes" (each stage's own
      full result dict for that scenario) -- see module docstring's STORAGE note. Later stages add
      their own key the same way.
    - "reference_cases" / "reference_results": as scenarios.get_scenario_set built them, plus
      "taxes" (compute_renter_taxes' real $0 result) stored into
      reference_results[reference_id] the first time each reference case is seen.
    """
    reference_cases, scenarios = get_scenario_set(deck)
    reference_results: dict[int, dict[str, Any]] = {}

    for scenario in scenarios:
        reference_id = scenario["reference_id"]
        if reference_id in reference_results:
            print_reference_cache_hit(reference_id)
        else:
            print_reference_report(reference_id, reference_cases[reference_id])
            renter_taxes = compute_renter_taxes(reference_cases[reference_id])
            reference_results[reference_id] = {"taxes": renter_taxes}
            print_renter_taxes_report(renter_taxes)

        # scenarios.get_scenario_set already computed this (to dedup reference cases) and stored it
        # on the scenario -- reuse it instead of recomputing.
        acquisition_costs = scenario.get("acquisition") or compute_acquisition_costs(scenario, deck)
        scenario["acquisition"] = acquisition_costs
        print_acquisition_report(scenario, acquisition_costs)

        classification = compute_occupancy_classification(scenario, deck)
        scenario["classification"] = classification
        print_classification_report(classification)

        financing = compute_financing(scenario, deck)
        scenario["financing"] = financing
        print_financing_report(financing)

        carrying_costs = compute_carrying_costs(scenario, deck)
        scenario["carrying_costs"] = carrying_costs
        print_carrying_costs_report(carrying_costs)

        result_plots = save_carrying_cost_result_plots(scenario, carrying_costs, deck)
        print_carrying_cost_result_plots_report(result_plots)

        owner_taxes = compute_owner_taxes(scenario, deck)
        scenario["taxes"] = owner_taxes
        print_taxes_report(owner_taxes)

        print_scenario_comparison_placeholder(scenario, reference_id)
        print()  # blank line -- groups this scenario's reference/alternative/result together

    return {"reference_cases": reference_cases, "reference_results": reference_results, "scenarios": scenarios}
