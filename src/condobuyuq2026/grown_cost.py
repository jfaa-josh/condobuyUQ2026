"""A user-entered monthly cost schedule grown by a latent over the horizon: the shared math behind
personal_use's alternate lodging (section 8) and rent_reference's lease (section 10).

MODEL, per scenario month h (calendar month m = MONTH_KEYS[(closing_month - 1 + h) % 12]):
    cost_h = L_m x G_s(h)
where L_m = the entered amount for month m (closing-date dollars, KNOWN: no level uncertainty, it's the
user's own estimate) and G = the latent's growth factor from closing (lognormal, from that latent's
fitted OU process; G_0 = 1). s(h) is when the price was last reset:
  - renew_every_months=1 (alternate lodging): s(h) = h, the price moves every month;
  - renew_every_months=12 (lease): s(h) = 12 x floor(h / 12), the price is set at each anniversary of
    closing and holds for 12 months, like a lease renewal. Year 1 is exactly the entered amounts.

Each month is charged in full as entered (no proration). Yearly and horizon totals: exact mean and
variance of the sum of correlated lognormals, with Cov(log G_s(i), log G_s(j)) ~= Var(log G at the
earlier reset) (the independent-increments approximation rental_operations uses), then matched to a
lognormal for quantiles (Fenton-Wilkinson).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from condobuyuq2026.input_deck import MONTH_KEYS, get_closing_date, get_growth_fit, get_report_quantiles
from condobuyuq2026.rental_operations import (
    BAND_NAMES,
    _lognormal_from_moments,
    _lognormal_quantiles,
    _sum_moments,
)
from condobuyuq2026.utils.ou_fitting import project_term_structure


def project_grown_monthly_cost(
    costs_by_month: dict[str, float],
    growth_latent: str,
    horizon_years: int,
    prefix: str,
    deck: dict[str, Any],
    renew_every_months: int = 1,
) -> dict[str, Any]:
    """Projects one monthly cost schedule over the horizon (see the module docstring). Returns:
      "monthly": DataFrame indexed by scenario month h (0 = closing month) with calendar_month, year and
        <prefix>_<band> for <band> in median, lo, hi, lo_outer, hi_outer, mean;
      "annual": DataFrame indexed by scenario year with <prefix>_mean, <prefix>_sd and the <prefix>_<band>s;
      "horizon": {<prefix>_mean, <prefix>_sd, <prefix>_<band>};
      "ci", "outer_ci"."""
    closing_month = get_closing_date(deck)["closing_month"]
    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])
    outer_ci = (0.01, 0.99)
    quantiles = {"median": 0.5, "lo": ci[0], "hi": ci[1], "lo_outer": outer_ci[0], "hi_outer": outer_ci[1]}

    horizon_months = int(horizon_years) * 12
    growth = project_term_structure(get_growth_fit(growth_latent, deck), 1.0, horizon_months, ci=ci)
    growth_mu = np.log(growth["median"].to_numpy())[:horizon_months]
    growth_var = ((np.log(growth["hi"].to_numpy() / growth["median"].to_numpy()) / norm.ppf(ci[1])) ** 2)[:horizon_months]
    reset = (np.arange(horizon_months) // renew_every_months) * renew_every_months
    growth_mu, growth_var = growth_mu[reset], growth_var[reset]

    years = np.arange(horizon_months) // 12 + 1
    mu = np.full(horizon_months, -np.inf)
    rows = []
    for h in range(horizon_months):
        m = MONTH_KEYS[(closing_month - 1 + h) % 12]
        row: dict[str, Any] = {"h": h, "calendar_month": m, "year": int(years[h])}
        if costs_by_month[m] > 0:
            mu[h] = math.log(costs_by_month[m]) + growth_mu[h]
            sigma = math.sqrt(growth_var[h])
            row.update({f"{prefix}_{k}": v for k, v in _lognormal_quantiles(mu[h], sigma, quantiles).items()})
            row[f"{prefix}_mean"] = math.exp(mu[h] + growth_var[h] / 2)
        else:
            row.update({f"{prefix}_{band}": 0.0 for band in (*BAND_NAMES, "mean")})
        rows.append(row)
    monthly = pd.DataFrame(rows).set_index("h")

    active = np.isfinite(mu)
    # reset is non-decreasing in h, so the earlier month's reset is the earlier reset.
    earlier = np.minimum.outer(np.arange(horizon_months), np.arange(horizon_months))
    cov = growth_var[earlier]
    mu_safe = np.where(active, mu, 0.0)

    def summarize(index: np.ndarray) -> dict[str, float]:
        mean, variance = _sum_moments(mu_safe, cov, index[active[index]])
        out: dict[str, float] = {f"{prefix}_mean": mean, f"{prefix}_sd": math.sqrt(variance)}
        if mean > 0:
            q = _lognormal_quantiles(*_lognormal_from_moments(mean, variance), quantiles)
        else:
            q = dict.fromkeys(quantiles, 0.0)
        out.update({f"{prefix}_{k}": v for k, v in q.items()})
        return out

    annual = pd.DataFrame(
        [{"year": year, **summarize(np.flatnonzero(years == year))} for year in range(1, int(horizon_years) + 1)]
    ).set_index("year")

    return {
        "monthly": monthly,
        "annual": annual,
        "horizon": summarize(np.arange(horizon_months)),
        "ci": ci,
        "outer_ci": outer_ci,
    }
