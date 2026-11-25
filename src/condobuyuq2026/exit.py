"""exit runtime (deck section 12): the BUY branch's money at the sale, as a distribution -- the mirror of
capital_markets' after-tax portfolio for the rent branch, so the ΔW stage can compare the two.

    net equity = P - selling costs(P) - mortgage payoff - family loan - sale tax(P)

- P, the sale price: lognormal (carrying_costs.project_sale_price: purchase_price grown by
  forecast_models.home_price, proportionally).
- selling costs(P) = exit.selling_costs.fixed_fees + percent_of_price x P, so net proceeds
  N(P) = (1 - pct) x P - fixed.
- mortgage payoff: the loan balance left at the horizon (scenario["financing"]["ending_balance"]), known.
- family loan: acquisition.family_loan if scenario["family_loan_repaid_at_sale"] (a scenario sweep, user
  decision 2026-11-25), else 0. Known.
- sale tax(P): taxes.compute_owner_sale_tax (recapture, capital gain, released passive losses, NIIT,
  state) at the EXPECTED price E[P], linearized around it with a SECANT marginal rate m measured between the
  report-band prices (lo, hi): tax(P) ~ tax(E[P]) + m x (N(P) - N(E[P])). A secant (not section 11's
  through-the-basis rate) because the condo's sale tax has pieces that don't scale with price (released
  passive losses, recapture taxed first).
  SALE AT A LOSS: taxed $0 (taxes.py's KNOWN LIMITATION): the rental share's section 1231 ordinary-loss
  deduction isn't modeled, so where the band is all losses m = 0 and the tax is $0 across it, and net
  equity is UNDERSTATED by about (rental share of the loss) x (marginal ordinary rate). The 1-year horizon
  is also still taxed under the long-term rules (held exactly 12 months). Both are flagged in
  scenario["taxes"]["flags"].

So net equity = a x P + b with a = (1 - m)(1 - pct), b = known terms: a shifted lognormal, exact mean and
quantiles (monotone while m < 1).
"""

from __future__ import annotations

import math
from typing import Any

from condobuyuq2026.carrying_costs import project_sale_price
from condobuyuq2026.input_deck import get_acquisition_inputs, get_exit_inputs, load_deck
from condobuyuq2026.taxes import compute_owner_sale_tax, compute_owner_taxes

BAND_NAMES = ("median", "lo", "hi", "lo_outer", "hi_outer")


def compute_exit(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """The buy branch's net equity at the sale (see the module docstring). Needs scenario["acquisition"],
    ["classification"], ["financing"] and ["taxes"] (compute_owner_taxes, computed here if missing).
    Returns:
      "sale_price": project_sale_price's result (mu, sigma, median, mean, sd, bands);
      "selling_costs", "mortgage_payoff", "family_loan_repaid", "sale_tax": each at the EXPECTED price;
      "marginal_sale_tax_rate": the secant rate m; "family_loan_repaid_at_sale": the switch;
      "net_equity": {"mean", "sd", <band>} with the same bands as sale_price;
      "linear": {"a", "b"} (net equity = a x P + b)."""
    deck = deck if deck is not None else load_deck()
    taxes = scenario.get("taxes") or compute_owner_taxes(scenario, deck)
    exit_inputs = get_exit_inputs(deck)
    fixed, pct = exit_inputs["selling_costs_fixed_fees"], exit_inputs["selling_costs_percent_of_price"]

    price = project_sale_price(scenario, deck)
    mean_price = price["mean"]

    def net_proceeds(p: float) -> float:
        return (1 - pct) * p - fixed

    tax_at_mean = taxes["sale"]["tax"]
    tax_lo = compute_owner_sale_tax(scenario, taxes["annual_schedule"], deck, sale_price=price["lo"])["tax"]
    tax_hi = compute_owner_sale_tax(scenario, taxes["annual_schedule"], deck, sale_price=price["hi"])["tax"]
    spread = net_proceeds(price["hi"]) - net_proceeds(price["lo"])
    marginal = (tax_hi - tax_lo) / spread if spread > 0 else 0.0

    payoff = float(scenario["financing"]["ending_balance"])
    repaid = bool(scenario["family_loan_repaid_at_sale"])
    family_loan = float(get_acquisition_inputs(deck)["family_loan"]) if repaid else 0.0

    a = (1 - marginal) * (1 - pct)
    b = -(1 - marginal) * fixed + marginal * net_proceeds(mean_price) - tax_at_mean - payoff - family_loan
    if a <= 0:
        raise ValueError(f"exit: marginal sale-tax rate {marginal:.1%} leaves no equity sensitivity to price")
    net_equity = {
        "mean": a * mean_price + b,
        "sd": a * price["sd"],
        **{band: a * price[band] + b for band in BAND_NAMES},
    }
    return {
        "sale_price": price,
        "selling_costs": fixed + pct * mean_price,
        "mortgage_payoff": payoff,
        "family_loan_repaid": family_loan,
        "family_loan_repaid_at_sale": repaid,
        "sale_tax": tax_at_mean,
        "marginal_sale_tax_rate": marginal,
        "net_equity": net_equity,
        "linear": {"a": a, "b": b},
    }


def net_equity_quantile(exit_result: dict[str, Any], z: float) -> float:
    """Net equity at standard-normal quantile z (a x P_z + b, P lognormal) -- for later stages (ΔW)."""
    price, linear = exit_result["sale_price"], exit_result["linear"]
    return linear["a"] * math.exp(price["mu"] + z * price["sigma"]) + linear["b"]
