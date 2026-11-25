"""personal_use runtime (deck section 8): the buy branch's alternate-lodging cost over the horizon.

MODEL, per scenario month h (calendar month m = MONTH_KEYS[(closing_month - 1 + h) % 12]):
    cost_h = L_m x G_h
where L_m = personal_use.alternate_lodging_cost_by_month[m] (closing-date dollars, KNOWN: no level
uncertainty, it's the user's own estimate) and G_h = the alternate_lodging_growth_latent's growth factor
from closing (lognormal, from that latent's fitted OU process; h = 0 -> 1). The price moves every month
(flexible lodging); compare rent_reference's lease, which resets once a year.

NO PRORATION (user decision, 2026-11-23): L_m is charged in full every month, whatever the personal-use
days or bookings. The user works out what they'd really pay (a full-month lease while still staying in
the unit a night or two, a 50/50 split, ...) and enters that.

The math (Fenton-Wilkinson yearly/horizon totals) is shared with rent_reference: see
grown_cost.project_grown_monthly_cost.

This is a PERSONAL cost: not tax-deductible, not in carrying_costs' totals. It enters the buy branch at the
final buy-vs-rent (ΔW) stage. It shares the `adr` latent with rental revenue by default, so the two are
positively correlated (a natural hedge); that covariance is handled at the ΔW stage.
"""

from __future__ import annotations

from typing import Any

from condobuyuq2026.grown_cost import project_grown_monthly_cost
from condobuyuq2026.input_deck import get_personal_use_inputs, load_deck


def compute_personal_use(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Projects one scenario's alternate-lodging cost over its horizon (see the module docstring).
    Returns:
      "monthly": DataFrame indexed by scenario month h (0 = closing month) with calendar_month, year,
        personal_days, lodging_<band> for <band> in median, lo, hi, lo_outer, hi_outer, mean;
      "annual": DataFrame indexed by scenario year with personal_days, lodging_mean, lodging_sd and the
        lodging_<band> bands;
      "horizon": {"lodging_mean", "lodging_sd", lodging_<band>};
      "inputs": get_personal_use_inputs' result; "ci", "outer_ci"."""
    deck = deck if deck is not None else load_deck()
    inputs = get_personal_use_inputs(deck)
    personal = inputs["personal_use_days_by_month"]
    projection = project_grown_monthly_cost(
        inputs["alternate_lodging_cost_by_month"],
        inputs["alternate_lodging_growth_latent"],
        int(scenario["horizon_years"]),
        "lodging",
        deck,
    )
    monthly, annual = projection["monthly"], projection["annual"]
    monthly.insert(2, "personal_days", monthly["calendar_month"].map(personal))
    annual.insert(0, "personal_days", monthly.groupby("year")["personal_days"].sum().astype(float))
    return {**projection, "inputs": inputs}
