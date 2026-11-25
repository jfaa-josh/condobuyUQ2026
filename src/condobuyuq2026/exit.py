"""exit runtime (deck section 12): the BUY branch's money at the sale, as a distribution -- the mirror of
capital_markets' after-tax portfolio for the rent branch, so the ΔW stage can compare the two.

    net equity(P) = P - selling costs(P) - mortgage payoff - family loan - sale tax(P)

- P, the sale price: lognormal (carrying_costs.project_sale_price: purchase_price grown by
  forecast_models.home_price, proportionally).
- selling costs(P) = exit.selling_costs.fixed_fees + percent_of_price x P, so net proceeds
  N(P) = (1 - pct) x P - fixed.
- mortgage payoff: the loan balance left at the horizon (scenario["financing"]["ending_balance"]), known.
- family loan: acquisition.family_loan if scenario["family_loan_repaid_at_sale"] (a scenario sweep, user
  decision 2026-11-25), else 0. Known.
- sale tax(P): taxes.owner_sale_tax_function (recapture, long- or short-term gain, the rental part's
  ordinary loss, released passive losses, NIIT, state; see taxes.py's SALE notes), evaluated at EACH price.
  A sale at a loss can make it negative (a tax saving).

EXACT (since 2026-11-25, ΔW plan Step 0): net equity rises with P (every marginal rate is below 100%), so
its quantiles are net equity at the sale price's quantiles, each with its own sale tax. Its mean and sd
come from a Gauss-Hermite rule over log P (GAUSS_HERMITE_NODES points), so the sale tax's kinks (loss vs
gain, brackets, NIIT) are averaged properly instead of linearized.

For the ΔW stage only: "linear" = {a, b} with net equity ~ a x P + b, the slope from the SECANT marginal
sale-tax rate m between the report-band prices (lo, hi), a = (1 - m)(1 - pct), and b set so the line's
mean equals the exact mean.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from condobuyuq2026.carrying_costs import project_sale_price
from condobuyuq2026.input_deck import get_acquisition_inputs, get_exit_inputs, load_deck
from condobuyuq2026.taxes import compute_owner_taxes, owner_sale_tax_function

BAND_NAMES = ("median", "lo", "hi", "lo_outer", "hi_outer")
GAUSS_HERMITE_NODES = 40


def compute_exit(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """The buy branch's net equity at the sale (see the module docstring). Needs scenario["acquisition"],
    ["classification"], ["financing"] and ["taxes"] (compute_owner_taxes, computed here if missing).
    Returns:
      "sale_price": project_sale_price's result (mu, sigma, median, mean, sd, bands);
      "selling_costs": at the expected price; "mortgage_payoff", "family_loan_repaid": known;
      "sale_tax": the EXPECTED sale tax E[tax(P)] (so the waterfall adds up to net equity's mean);
      "sale_tax_at_expected_price": tax(E[P]) (= scenario["taxes"]["sale"]["tax"]);
      "sale_tax_by_band": {<band>: tax at that band's price};
      "marginal_sale_tax_rate": the secant rate m; "family_loan_repaid_at_sale": the switch;
      "net_equity": {"mean", "sd", <band>} with the same bands as sale_price (exact);
      "linear": {"a", "b"} (net equity ~ a x P + b, for ΔW)."""
    deck = deck if deck is not None else load_deck()
    taxes = scenario.get("taxes") or compute_owner_taxes(scenario, deck)
    exit_inputs = get_exit_inputs(deck)
    fixed, pct = exit_inputs["selling_costs_fixed_fees"], exit_inputs["selling_costs_percent_of_price"]
    sale_tax = owner_sale_tax_function(scenario, taxes["annual_schedule"], deck)

    price = project_sale_price(scenario, deck)
    payoff = float(scenario["financing"]["ending_balance"])
    repaid = bool(scenario["family_loan_repaid_at_sale"])
    family_loan = float(get_acquisition_inputs(deck)["family_loan"]) if repaid else 0.0

    def net_proceeds(p: float) -> float:
        return (1 - pct) * p - fixed

    def equity(p: float, tax: float) -> float:
        return net_proceeds(p) - tax - payoff - family_loan

    tax_by_band = {band: sale_tax(price[band])["tax"] for band in BAND_NAMES}

    nodes, weights = np.polynomial.hermite_e.hermegauss(GAUSS_HERMITE_NODES)
    weights = weights / math.sqrt(2 * math.pi)
    node_prices = np.exp(price["mu"] + price["sigma"] * nodes)
    node_taxes = np.array([sale_tax(float(p))["tax"] for p in node_prices])
    node_equity = np.array([equity(float(p), t) for p, t in zip(node_prices, node_taxes, strict=True)])
    mean_tax = float(weights @ node_taxes)
    mean_equity = float(weights @ node_equity)
    sd_equity = math.sqrt(max(0.0, float(weights @ (node_equity - mean_equity) ** 2)))

    spread = net_proceeds(price["hi"]) - net_proceeds(price["lo"])
    marginal = (tax_by_band["hi"] - tax_by_band["lo"]) / spread if spread > 0 else 0.0
    a = (1 - marginal) * (1 - pct)
    if a <= 0:
        raise ValueError(f"exit: marginal sale-tax rate {marginal:.1%} leaves no equity sensitivity to price")
    b = mean_equity - a * price["mean"]

    return {
        "sale_price": price,
        "selling_costs": fixed + pct * price["mean"],
        "mortgage_payoff": payoff,
        "family_loan_repaid": family_loan,
        "family_loan_repaid_at_sale": repaid,
        "sale_tax": mean_tax,
        "sale_tax_at_expected_price": float(taxes["sale"]["tax"]),
        "sale_tax_by_band": tax_by_band,
        "marginal_sale_tax_rate": marginal,
        "net_equity": {
            "mean": mean_equity,
            "sd": sd_equity,
            **{band: equity(price[band], tax_by_band[band]) for band in BAND_NAMES},
        },
        "linear": {"a": a, "b": b},
    }


def net_equity_quantile(exit_result: dict[str, Any], z: float) -> float:
    """Net equity at standard-normal quantile z via the "linear" approximation (a x P_z + b, P lognormal) --
    for later stages (ΔW). The report bands in exit_result["net_equity"] are exact; this is the line."""
    price, linear = exit_result["sale_price"], exit_result["linear"]
    return linear["a"] * math.exp(price["mu"] + z * price["sigma"]) + linear["b"]
