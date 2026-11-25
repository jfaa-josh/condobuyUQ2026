"""capital_markets runtime (deck section 11): the RENT branch's invested portfolio over the horizon.

ACCOUNTS (buy and hold, no rebalancing; user decision 2026-11-25, "too complicated for cut 1"):
each contribution is split by capital_markets.allocation into
  - cash:   grows at cash_yield (monthly compounding), deterministic;
  - market: grows with forecast_models.market's PRICE growth G (lognormal, from its fitted OU term
            structure) plus dividends at d = dividends.paying_fraction x dividends.yield per year, paid
            monthly on the account's value and reinvested.
For a cut-1 portfolio the only contribution is capital_markets.initial_investment
(= acquisition.personal_cash_at_closing) at closing. project_portfolio also takes a whole monthly
contribution schedule: the ΔW stage will feed it the monthly cash-flow differences (user decision
2026-11-25: budget-matched, the branch that spends less each month invests the difference).

TAXES (taxes.investment_income_tax / investment_sale_tax: the section 9 with-minus-without engine, residence
state; never property-source income):
  - each scenario year: interest (ordinary) and qualified dividends (long-term rates + NIIT; ordinary in CO),
    on that year's EXPECTED amounts (point estimates, like the owner's taxes). Interest tax comes out of
    the cash account; dividend tax out of the market account (only the after-tax dividends stay
    reinvested and join the cost basis);
  - at the horizon the market account is sold: long-term gain = E[value] - basis, taxed as an increment on
    the final year's return. Its effective rate t is then applied linearly to the whole distribution,
    after-tax value = (1 - t) x value + t x basis, which keeps the lognormal's shape (a loss would get a
    full-rate benefit this way; the real $3,000/yr limit isn't modeled).

MARKET VALUE DISTRIBUTION: value at month e = sum_j m_j x K_j(e) x G_e / G_j over contributions m_j (start
of month j), with K_j(e) the deterministic dividend-reinvestment and tax factors between j and e. Exact
mean; variance with Cov(log G_e/G_i, log G_e/G_j) = Var(log G_e) - Var(log G_max(i,j)) (the
independent-increments approximation used elsewhere), matched to a lognormal (Fenton-Wilkinson).

KNOWN LIMITATION (flagged for after the GUI is wired, user 2026-11-25): the market fit understates stock
risk vs its own history (1-yr log-sd 0.088 vs 0.155; 10-yr 0.342 vs 0.456) and starts hot (i0 14%). See
the implementation plan's "To raise once the GUI is wired" list.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from condobuyuq2026.input_deck import (
    get_capital_markets_inputs,
    get_closing_date,
    get_growth_fit,
    get_report_quantiles,
    get_taxes_inputs,
    load_deck,
)
from condobuyuq2026.rental_operations import _lognormal_from_moments, _lognormal_quantiles, _sum_moments
from condobuyuq2026.taxes import get_scenario_tax_jurisdiction, investment_income_tax, investment_sale_tax
from condobuyuq2026.utils.ou_fitting import project_term_structure

MARKET_LATENT = "market"


def project_portfolio(
    contributions: np.ndarray,
    case: dict[str, Any],
    deck: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Projects a portfolio fed by `contributions` (USD at the START of each scenario month h = 0 ..
    12 x horizon_years - 1, split by capital_markets.allocation) to the horizon, with yearly taxes and the
    sale (see the module docstring). `case` is a reference case or scenario dict (horizon_years,
    residency_state). Returns:
      "annual": DataFrame indexed by scenario year: tax_year, contributions, interest, dividends,
        interest_tax, dividend_tax, federal, property_state, residence_state, tax, cash_end, basis_end,
        market_end_mean and market_end_<band> (pre-sale);
      "sale": {market_value_mean, basis, gain, effective_rate, federal, property_state, residence_state, tax};
      "horizon": cash, pre_tax_mean / pre_tax_<band> (cash + market, before the sale tax), after_tax_mean /
        after_tax_<band>, contributions, income_tax (all yearly taxes), total_tax (yearly + sale);
      "inputs", "ci", "outer_ci"."""
    deck = deck if deck is not None else load_deck()
    inputs = get_capital_markets_inputs(deck)
    taxes_inputs = get_taxes_inputs(deck)
    jurisdiction = get_scenario_tax_jurisdiction(case, deck)
    closing_year = get_closing_date(deck)["closing_year"]
    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])
    outer_ci = (0.01, 0.99)
    quantiles = {"median": 0.5, "lo": ci[0], "hi": ci[1], "lo_outer": outer_ci[0], "hi_outer": outer_ci[1]}

    horizon_years = int(case["horizon_years"])
    horizon_months = horizon_years * 12
    contributions = np.asarray(contributions, dtype=float)
    if contributions.shape != (horizon_months,) or (contributions < 0).any():
        raise ValueError(f"contributions must be {horizon_months} monthly amounts >= 0")

    growth = project_term_structure(get_growth_fit(MARKET_LATENT, deck), 1.0, horizon_months, ci=ci)
    mu = np.log(growth["median"].to_numpy())[: horizon_months + 1]
    var = ((np.log(growth["hi"].to_numpy() / growth["median"].to_numpy()) / norm.ppf(ci[1])) ** 2)[: horizon_months + 1]
    monthly_price_growth = np.exp(np.diff(mu) + np.diff(var) / 2)  # E[G_(h+1) / G_h]

    cash_share, market_share = inputs["allocation"]["cash"], inputs["allocation"]["market"]
    cash_rate = inputs["cash_yield"] / 12
    dividend_rate = inputs["dividend_paying_fraction"] * inputs["dividend_yield"] / 12
    market_in = market_share * contributions
    # log of the deterministic factor each month's market contribution has picked up by month e:
    # dividend reinvestment (1 + dividend_rate per month) and the year-end dividend-tax haircuts.
    log_haircut_after = np.zeros(horizon_months + 1)  # cumulative log haircut at the END of each month

    cash = market = basis = 0.0
    rows = []
    interest = dividends = contributed = 0.0
    for h in range(horizon_months):
        cash += cash_share * contributions[h]
        market += market_in[h]
        basis += market_in[h]
        contributed += contributions[h]
        interest += cash * cash_rate
        cash *= 1 + cash_rate
        market *= monthly_price_growth[h] * (1 + dividend_rate)
        month_dividend = market * dividend_rate / (1 + dividend_rate)
        dividends += month_dividend
        basis += month_dividend
        log_haircut_after[h + 1] = log_haircut_after[h]
        if (h + 1) % 12 == 0:
            year = (h + 1) // 12
            tax_year = closing_year + year - 1
            tax = investment_income_tax(jurisdiction, taxes_inputs, tax_year, interest, dividends)
            cash -= tax["interest_tax"]
            haircut = 1 - tax["dividend_tax"] / market if market > 0 else 1.0
            market *= haircut
            basis -= tax["dividend_tax"]
            log_haircut_after[h + 1] += math.log(haircut) if haircut > 0 else -np.inf
            rows.append({"year": year, "tax_year": tax_year, "contributions": contributed, "interest": interest,
                         "dividends": dividends, **tax, "tax": tax["total"], "cash_end": cash, "basis_end": basis})
            interest = dividends = contributed = 0.0
    annual = pd.DataFrame(rows).set_index("year").drop(columns="total")

    def market_distribution(end: int) -> tuple[float, float]:
        """(mean, variance) of the market account's value at the end of month end-1 (month index `end`)."""
        index = np.flatnonzero(market_in[:end] > 0)
        if index.size == 0:
            return 0.0, 0.0
        log_x = (np.log(market_in[index]) + (end - index) * math.log1p(dividend_rate)
                 + log_haircut_after[end] - log_haircut_after[index] + mu[end] - mu[index])
        later = np.maximum.outer(index, index)
        cov = var[end] - var[later]
        return _sum_moments(log_x, cov, np.arange(index.size))

    def bands(mean: float, variance: float) -> dict[str, float]:
        if mean <= 0:
            return dict.fromkeys(quantiles, 0.0)
        return _lognormal_quantiles(*_lognormal_from_moments(mean, variance), quantiles)

    market_end = []
    for year in annual.index:
        mean, variance = market_distribution(year * 12)
        market_end.append({"market_end_mean": mean, **{f"market_end_{k}": v for k, v in bands(mean, variance).items()}})
    annual = annual.join(pd.DataFrame(market_end, index=annual.index))

    final = annual.iloc[-1]
    market_mean, market_var = market_distribution(horizon_months)
    gain = market_mean - basis
    sale_tax = investment_sale_tax(jurisdiction, taxes_inputs, int(final["tax_year"]), gain,
                                   float(final["interest"]), float(final["dividends"]))
    rate = sale_tax["total"] / gain if gain > 0 else 0.0
    sale = {"market_value_mean": market_mean, "basis": basis, "gain": gain, "effective_rate": rate,
            **{line: sale_tax[line] for line in ("federal", "property_state", "residence_state")},
            "tax": sale_tax["total"]}

    market_q = bands(market_mean, market_var)
    income_tax = float(annual["tax"].sum())
    horizon = {
        "cash": cash,
        "pre_tax_mean": cash + market_mean,
        **{f"pre_tax_{k}": cash + v for k, v in market_q.items()},
        "after_tax_mean": cash + (1 - rate) * market_mean + rate * basis,
        **{f"after_tax_{k}": cash + (1 - rate) * v + rate * basis for k, v in market_q.items()},
        "contributions": float(contributions.sum()),
        "income_tax": income_tax,
        "total_tax": income_tax + sale["tax"],
    }
    return {"annual": annual, "sale": sale, "horizon": horizon, "inputs": inputs, "ci": ci, "outer_ci": outer_ci}


def compute_capital_markets(reference_case: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Cut 1: the rent branch invests capital_markets.initial_investment (= acquisition.
    personal_cash_at_closing) once, at closing, and holds it to the horizon. See project_portfolio."""
    deck = deck if deck is not None else load_deck()
    horizon_months = int(reference_case["horizon_years"]) * 12
    contributions = np.zeros(horizon_months)
    contributions[0] = get_capital_markets_inputs(deck)["initial_investment"]
    return project_portfolio(contributions, reference_case, deck)
