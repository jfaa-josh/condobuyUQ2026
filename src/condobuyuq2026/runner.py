"""Runs every scenario in scenarios.get_scenario_set, pairing each buy scenario with its matching
reference (rent) case's result -- running a given reference case's own computation exactly ONCE
even though multiple buy scenarios may share it (see scenarios.py's module docstring for why that
sharing happens), by caching each reference_id's result the first time it's needed.

STORAGE, not just printing: each stage's computed values are kept, not only printed and discarded.
A buy scenario dict gets a new key per stage as it's computed -- so far "acquisition"
(acquisition.compute_acquisition_costs), "classification"
(classification.compute_occupancy_classification), and "financing" (financing.compute_financing) --
so by the time run_scenarios returns, every scenario in its "scenarios" list carries everything
computed for it, not just what got printed along the way. reference_results[reference_id] is,
symmetrically, a dict a future reference-side stage adds its own key to (e.g.
reference_results[id]["rent_reference"] = ... once that stage exists) -- empty today since only the
buy side computes anything yet.

The eventual buy-vs-reference comparison (ΔW = W_buy - W_rent, see .claude/implementation-plan.md's
Goal) is a PLACEHOLDER for now (see reporting.print_scenario_comparison_placeholder) -- as later
stages (carrying_costs, taxes, rent_reference, capital_markets, exit) get built, this is where
their per-scenario computation gets wired in -- and stored, per the paragraph above -- on both the
reference and buy sides.
"""

from __future__ import annotations

from typing import Any

from condobuyuq2026.acquisition import compute_acquisition_costs
from condobuyuq2026.classification import compute_occupancy_classification
from condobuyuq2026.financing import compute_financing
from condobuyuq2026.reporting import (
    print_acquisition_report,
    print_classification_report,
    print_financing_report,
    print_reference_cache_hit,
    print_reference_report,
    print_scenario_comparison_placeholder,
)
from condobuyuq2026.scenarios import get_scenario_set


def run_scenarios(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Iterates every buy scenario (see scenarios.get_scenario_set), running its matching reference
    case the FIRST time that reference_id is seen (cached after that -- see module docstring), then
    immediately following it with that scenario's own buy-side computation (acquisition,
    classification, financing today; more as later stages are built), so a buy scenario and its
    reference are always handled as a pair.

    Returns {"reference_cases": ..., "reference_results": ..., "scenarios": ...}:
    - "scenarios": scenarios.get_scenario_set's buy scenarios, each one now ALSO carrying
      "acquisition", "classification", "financing" (each stage's own full result dict for that
      scenario) -- see module docstring's STORAGE note. Later stages add their own key the same way.
    - "reference_cases" / "reference_results": as scenarios.get_scenario_set built them, plus
      whatever a future reference-side stage stores into reference_results[reference_id] (empty
      dicts today -- see module docstring).
    """
    reference_cases, scenarios = get_scenario_set(deck)
    reference_results: dict[int, dict[str, Any]] = {}

    for scenario in scenarios:
        reference_id = scenario["reference_id"]
        if reference_id in reference_results:
            print_reference_cache_hit(reference_id)
        else:
            print_reference_report(reference_id, reference_cases[reference_id])
            reference_results[reference_id] = {}  # PLACEHOLDER -- a future reference-side stage stores its own result here

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

        print_scenario_comparison_placeholder(scenario, reference_id)
        print()  # blank line -- groups this scenario's reference/alternative/result together

    return {"reference_cases": reference_cases, "reference_results": reference_results, "scenarios": scenarios}
