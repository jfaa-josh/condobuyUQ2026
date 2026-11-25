"""rent_reference runtime (deck section 10): the RENT branch's lodging cost over the horizon.

MODEL, per scenario month h (calendar month m = MONTH_KEYS[(closing_month - 1 + h) % 12]):
    cost_h = L_m x G_s(h),   s(h) = 12 x floor(h / 12)
where L_m = rent_reference.lease_cost_by_month[m] (closing-date dollars, all-in, KNOWN: the user's own
estimate) and G = the lease_growth_latent's growth factor from closing (lognormal; G_0 = 1).

RENT GOES UP ONCE A YEAR (user decision, 2026-11-25): nobody raises rent every month, so the price is
reset at each anniversary of closing, like a lease renewal, and holds for 12 months. Year 1 is exactly
the entered amounts; the month-to-month pattern is kept, only its level steps up. The entered values can
describe any arrangement (12-month lease, month to month, a winter seasonal lease with 0 elsewhere).

NO PRORATION: each month is charged in full as entered (same rule as personal_use's alternate lodging).
A refundable security deposit is not a cost (user, 2026-11-25), so there's no field for it.

Not tax-deductible (federal or Colorado), so taxes.compute_renter_taxes stays $0. It enters the rent
branch at the final buy-vs-rent (ΔW) stage. By default it shares the `adr` latent with the buy branch's
rental revenue and alternate lodging, so those terms are correlated ACROSS the two branches; that
covariance is handled at the ΔW stage. The math is shared with personal_use: see
grown_cost.project_grown_monthly_cost.
"""

from __future__ import annotations

from typing import Any

from condobuyuq2026.grown_cost import project_grown_monthly_cost
from condobuyuq2026.input_deck import get_rent_reference_inputs, load_deck

LEASE_RENEWAL_MONTHS = 12


def compute_rent_reference(reference_case: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Projects one reference case's lease cost over its horizon (see the module docstring). Returns:
      "monthly": DataFrame indexed by scenario month h (0 = closing month) with calendar_month, year,
        lease_<band> for <band> in median, lo, hi, lo_outer, hi_outer, mean;
      "annual": DataFrame indexed by scenario year with lease_mean, lease_sd and the lease_<band> bands;
      "horizon": {"lease_mean", "lease_sd", lease_<band>};
      "inputs": get_rent_reference_inputs' result; "renew_every_months"; "ci", "outer_ci"."""
    deck = deck if deck is not None else load_deck()
    inputs = get_rent_reference_inputs(deck)
    projection = project_grown_monthly_cost(
        inputs["lease_cost_by_month"],
        inputs["lease_growth_latent"],
        int(reference_case["horizon_years"]),
        "lease",
        deck,
        renew_every_months=LEASE_RENEWAL_MONTHS,
    )
    return {**projection, "inputs": inputs, "renew_every_months": LEASE_RENEWAL_MONTHS}
