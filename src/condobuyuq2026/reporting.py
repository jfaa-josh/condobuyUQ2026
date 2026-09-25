"""Shared human-readable print formatting for backend build/compute results -- one function per
deck-section stage, called identically by run/main.py (the real build/report pipeline) and each
matching run/tests/test_*.py (the user's manual-run check for that same stage, see
.claude/implementation-plan.md's "Backend build progress"). Keeping this formatting here, in one
place, means main.py and a test script can never drift into two different versions of the same
printout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def print_derived_latent_build_report(results: dict[str, dict[str, Path]]) -> None:
    """Prints one line per derived_latents entry build result -- results as returned by
    utils.manual_utils.build_all_derived_latent_models (name -> {"fit_path", "plot_path"})."""
    for name, paths in results.items():
        print(f"{name}: saved fit to {paths['fit_path']}, plot to {paths['plot_path']}")


def print_acquisition_report(scenario: dict[str, Any], costs: dict[str, float]) -> None:
    """Prints one scenario's acquisition costs as a single line -- scenario as returned by
    input_deck.get_scenarios, costs as returned by acquisition.compute_acquisition_costs for that
    scenario."""
    print(
        f"{scenario['alternative']} | {scenario['horizon_years']}yr | {scenario['residency_state']} | "
        f"purchase_price=${costs['purchase_price']:,.0f} -> "
        f"closing_costs=${costs['closing_costs_total']:,.0f}, loan_amount=${costs['loan_amount']:,.0f}, "
        f"land_value=${costs['land_value']:,.0f}, building_value=${costs['building_value']:,.0f}, "
        f"total_cash_outlay=${costs['total_cash_outlay']:,.0f}"
    )
