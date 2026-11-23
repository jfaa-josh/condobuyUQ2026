"""personal_use runtime (deck section 8): the buy branch's alternate-lodging cost over the horizon.

MODEL, per scenario month h (calendar month m = MONTH_KEYS[(closing_month - 1 + h) % 12]):
    cost_h = L_m x G_h
where L_m = personal_use.alternate_lodging_cost_by_month[m] (closing-date dollars, KNOWN: no level
uncertainty, it's the user's own estimate) and G_h = the alternate_lodging_growth_latent's growth factor
from closing (lognormal, from that latent's fitted OU process; h = 0 -> 1).

NO PRORATION (user decision, 2026-11-23): L_m is charged in full every month, whatever the personal-use
days or bookings. The user works out what they'd really pay (a full-month lease while still staying in
the unit a night or two, a 50/50 split, ...) and enters that.

Yearly and horizon totals: exact mean and variance of the sum of correlated lognormals, with
Cov(log G_i, log G_j) ~= Var(log G at the earlier month) (the same independent-increments approximation
rental_operations uses), then matched to a lognormal for quantiles (Fenton-Wilkinson).

This is a PERSONAL cost: not tax-deductible, not in carrying_costs' totals. It enters the buy branch at the
final buy-vs-rent (ΔW) stage. It shares the `adr` latent with rental revenue by default, so the two are
positively correlated (a natural hedge); that covariance is handled at the ΔW stage.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from condobuyuq2026.input_deck import (
    MONTH_KEYS,
    get_closing_date,
    get_growth_fit,
    get_personal_use_inputs,
    get_report_quantiles,
    load_deck,
)
from condobuyuq2026.rental_operations import (
    BAND_NAMES,
    _lognormal_from_moments,
    _lognormal_quantiles,
    _sum_moments,
)
from condobuyuq2026.utils.ou_fitting import project_term_structure


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
    costs = inputs["alternate_lodging_cost_by_month"]
    personal = inputs["personal_use_days_by_month"]
    closing_month = get_closing_date(deck)["closing_month"]
    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])
    outer_ci = (0.01, 0.99)
    quantiles = {"median": 0.5, "lo": ci[0], "hi": ci[1], "lo_outer": outer_ci[0], "hi_outer": outer_ci[1]}

    horizon_years = int(scenario["horizon_years"])
    horizon_months = horizon_years * 12
    growth_fit = get_growth_fit(inputs["alternate_lodging_growth_latent"], deck)
    growth = project_term_structure(growth_fit, 1.0, horizon_months, ci=ci)
    growth_mu = np.log(growth["median"].to_numpy())[:horizon_months]
    growth_var = ((np.log(growth["hi"].to_numpy() / growth["median"].to_numpy()) / norm.ppf(ci[1])) ** 2)[:horizon_months]

    years = np.arange(horizon_months) // 12 + 1
    mu = np.full(horizon_months, -np.inf)
    rows = []
    for h in range(horizon_months):
        m = MONTH_KEYS[(closing_month - 1 + h) % 12]
        row: dict[str, Any] = {"h": h, "calendar_month": m, "year": int(years[h]), "personal_days": personal[m]}
        if costs[m] > 0:
            mu[h] = math.log(costs[m]) + growth_mu[h]
            sigma = math.sqrt(growth_var[h])
            row.update({f"lodging_{k}": v for k, v in _lognormal_quantiles(mu[h], sigma, quantiles).items()})
            row["lodging_mean"] = math.exp(mu[h] + growth_var[h] / 2)
        else:
            row.update({f"lodging_{band}": 0.0 for band in (*BAND_NAMES, "mean")})
        rows.append(row)
    monthly = pd.DataFrame(rows).set_index("h")

    active = np.isfinite(mu)
    earlier = np.minimum.outer(np.arange(horizon_months), np.arange(horizon_months))
    cov = growth_var[earlier]
    mu_safe = np.where(active, mu, 0.0)

    def summarize(index: np.ndarray) -> dict[str, float]:
        mean, variance = _sum_moments(mu_safe, cov, index[active[index]])
        out: dict[str, float] = {"lodging_mean": mean, "lodging_sd": math.sqrt(variance)}
        if mean > 0:
            q = _lognormal_quantiles(*_lognormal_from_moments(mean, variance), quantiles)
        else:
            q = dict.fromkeys(quantiles, 0.0)
        out.update({f"lodging_{k}": v for k, v in q.items()})
        return out

    annual_rows = []
    for year in range(1, horizon_years + 1):
        index = np.flatnonzero(years == year)
        summary = summarize(index)
        summary.update({"year": year, "personal_days": float(monthly.loc[index, "personal_days"].sum())})
        annual_rows.append(summary)
    annual = pd.DataFrame(annual_rows).set_index("year")

    return {
        "monthly": monthly,
        "annual": annual,
        "horizon": summarize(np.arange(horizon_months)),
        "inputs": inputs,
        "ci": ci,
        "outer_ci": outer_ci,
    }
