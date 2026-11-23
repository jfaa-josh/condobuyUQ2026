"""rental_operations runtime: turns the fitted occupancy/ADR priors (utils.rental_prior_utils) into
nights rented and revenue for THIS deck's own personal use.

BUILD STATUS (2026-11-22): steps 1-3 of the section 7 plan. Built so far: reading the saved priors
(load_rental_priors), rentable nights, the capped occupancy moments, expected/median rented nights by
calendar month (the ONE place utilities and classification read rented nights from), and the
per-scenario revenue projection compute_rental_operations. Not yet built: taxes' real gross_rent
(step 4), results plots (step 5). See .claude/implementation-plan.md, "Section 7 plan".

CAP: occupancy is a RATE of nights available to rent, and you can't rent more nights than you leave
available, so rented nights = min(O, c) with O = rate x c lognormal and c = days in month - personal
days. min(O, c) has exact closed-form moments (capped_lognormal_moments); its median is simply
min(median O, c).
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from condobuyuq2026.input_deck import (
    DAYS_IN_MONTH,
    MONTH_KEYS,
    get_closing_date,
    get_growth_fit,
    get_personal_use_days_by_month,
    get_rental_operations_inputs,
    get_report_quantiles,
    load_deck,
)
from condobuyuq2026.paths import model_fit_path
from condobuyuq2026.utils.ou_fitting import project_term_structure


def load_rental_priors() -> dict[str, Any]:
    """Reads the saved priors written by utils.rental_prior_utils.build_rental_operations_prior_models
    (called by run/main.py before anything else needs them). Returns {"occupancy": {month: fit},
    "adr": {month: fit}} -- the same per-month shape fit_rental_priors returns."""
    priors = {}
    for name in ("occupancy", "adr"):
        path = model_fit_path(f"rental_operations_priors/{name}")
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"rental_operations_priors/{name}/fit.json doesn't exist or is empty -- run "
                "utils.rental_prior_utils.build_rental_operations_prior_models (run/main.py does this) first."
            )
        with path.open(encoding="utf-8") as f:
            priors[name] = json.load(f)["months"]
    return priors


def load_history_occupied_days() -> dict[str, Any] | None:
    """The previous owner's occupied days per calendar month, as saved in the occupancy fit.json by
    utils.rental_prior_utils.history_occupied_days_by_month ({"days", "sources"}, or None if no history
    source gives owner nights)."""
    path = model_fit_path("rental_operations_priors/occupancy")
    with path.open(encoding="utf-8") as f:
        return json.load(f).get("history_occupied_days_by_month")


def rentable_nights_by_month(deck: dict[str, Any] | None = None) -> dict[str, float]:
    """Nights YOU leave available to rent per calendar month: days in month - personal_use days."""
    personal = get_personal_use_days_by_month(deck)
    return {m: DAYS_IN_MONTH[m] - personal[m] for m in MONTH_KEYS}


def capped_lognormal_moments(median: float, sigma_log: float, cap: float) -> tuple[float, float]:
    """(E[min(O, cap)], E[min(O, cap)^2]) for O ~ lognormal(ln median, sigma_log), exactly:
    E[min] = E[O] Phi((ln c - mu - s^2)/s) + c (1 - Phi((ln c - mu)/s)),
    E[min^2] = e^(2mu + 2s^2) Phi((ln c - mu - 2s^2)/s) + c^2 (1 - Phi((ln c - mu)/s)).
    sigma_log = 0 (or median 0) -> the deterministic min(median, cap)."""
    if median <= 0 or cap <= 0:
        return 0.0, 0.0
    if sigma_log == 0:
        value = min(median, cap)
        return value, value**2
    mu, s = math.log(median), sigma_log
    z = (math.log(cap) - mu) / s
    above = 1 - norm.cdf(z)
    first = math.exp(mu + s**2 / 2) * norm.cdf(z - s) + cap * above
    second = math.exp(2 * mu + 2 * s**2) * norm.cdf(z - 2 * s) + cap**2 * above
    return first, second


def rented_nights_by_month(
    deck: dict[str, Any] | None = None, statistic: str = "mean", fit: dict[str, Any] | None = None
) -> dict[str, float]:
    """Rented nights per calendar month for YEAR 1 (no growth applies to nights), under this deck's
    personal use. statistic: "mean" (capped expectation -- for anything LINEAR in nights, e.g.
    utilities) or "median" (min(median, cap) -- for threshold rules, e.g. tax classification).
    Uses the PREDICTIVE lognormal (year-to-year + persistent spread, centered so that before the cap
    its mean is the sources' plain average) -- the same one compute_rental_operations uses."""
    deck = deck if deck is not None else load_deck()
    fit = fit if fit is not None else load_rental_priors()
    rentable = rentable_nights_by_month(deck)
    nights = {}
    for m in MONTH_KEYS:
        month_fit = fit["occupancy"][m]
        median = month_fit["predictive_median"] * rentable[m]
        if statistic == "median":
            nights[m] = min(median, rentable[m])
        elif statistic == "mean":
            nights[m] = capped_lognormal_moments(median, month_fit["predictive_sigma"], rentable[m])[0]
        else:
            raise ValueError(f"statistic must be 'mean' or 'median', got {statistic!r}")
    return nights


def expected_occupied_days_by_month(deck: dict[str, Any] | None = None) -> dict[str, float]:
    """EXPECTED occupied days per calendar month = expected (capped) rented nights + your personal-use
    days. Utilities' forward occupancy (linear in occupied days, so the mean is the right statistic)."""
    deck = deck if deck is not None else load_deck()
    personal = get_personal_use_days_by_month(deck)
    rented = rented_nights_by_month(deck, "mean")
    return {m: rented[m] + personal[m] for m in MONTH_KEYS}


# ---------------------------------------------------------------------------------------------------
# Per-scenario revenue projection (section 7 plan, build step 3)
# ---------------------------------------------------------------------------------------------------
#
# MODEL, per scenario month h (calendar month m, scenario year y = h // 12 + 1):
#   nights N_h = min(O, c_m), O lognormal around median rate_m x c_m (c_m = nights you leave available);
#   ADR    A_h = median ADR_m x G_h x (ADR shocks), G_h = the `adr` latent's growth factor from closing;
#   gross  R_h = N_h x A_h.
# The capped nights are moment-matched back to an "effective" lognormal (flagged approximation), so R_h is
# lognormal. Log-space shocks, each a standard normal scaled by that month's fitted sigma:
#   - occupancy year-to-year: one shock per scenario year, shared by all 12 months (within-year corr 1);
#   - occupancy level (persistent): ONE shock for the whole horizon, shared by all months (conservative:
#     a source bias in January is likely in February too);
#   - ADR year-to-year and ADR level: the same structure;
#   - occupancy vs ADR year-to-year shocks: correlation rental_operations.occupancy_adr_correlation;
#   - growth: Cov(log G_i, log G_j) ~= Var(log G at the earlier month) (independent-increments
#     approximation of the integrated OU process).
# Yearly and horizon sums: exact mean and variance of the sum of correlated lognormals, then matched to a
# lognormal for quantiles (Fenton-Wilkinson). Net = gross x (1 - channel_commission_fraction) x
# (1 - mgmt_fee_fraction): a constant scale, so net quantiles are gross quantiles x that keep fraction.

BAND_NAMES = ("median", "lo", "hi", "lo_outer", "hi_outer")


def _lognormal_quantiles(mu: float, sigma: float, quantiles: dict[str, float]) -> dict[str, float]:
    return {band: float(math.exp(mu + norm.ppf(p) * sigma)) for band, p in quantiles.items()}


def _lognormal_from_moments(mean: float, variance: float) -> tuple[float, float]:
    """(mu, sigma) of the lognormal with this mean and variance (Fenton-Wilkinson match)."""
    if mean <= 0:
        return -math.inf, 0.0
    sigma2 = math.log1p(variance / mean**2)
    return math.log(mean) - sigma2 / 2, math.sqrt(sigma2)


def _capped_effective(median: float, sigma_yy: float, sigma_level: float, cap: float) -> dict[str, float]:
    """Capped nights min(O, cap) matched to an effective lognormal: returns {mean, var, mu, sigma,
    sigma_yy, sigma_level} with both sigma parts scaled by sigma_eff / sigma_total (the cap shrinks
    the spread; the split between the parts is kept)."""
    sigma_total = math.hypot(sigma_yy, sigma_level)
    first, second = capped_lognormal_moments(median, sigma_total, cap)
    variance = max(second - first**2, 0.0)
    mu, sigma = _lognormal_from_moments(first, variance)
    scale = sigma / sigma_total if sigma_total > 0 else 0.0
    return {"mean": first, "var": variance, "mu": mu, "sigma": sigma,
            "sigma_yy": sigma_yy * scale, "sigma_level": sigma_level * scale}


def _sum_moments(mu: np.ndarray, cov: np.ndarray, index: np.ndarray) -> tuple[float, float]:
    """Mean and variance of sum_i exp(X_i) over `index`, X ~ multivariate normal(mu, cov)."""
    if index.size == 0:
        return 0.0, 0.0
    m = mu[index]
    c = cov[np.ix_(index, index)]
    e = np.exp(m + np.diag(c) / 2)
    mean = float(e.sum())
    variance = float((np.outer(e, e) * np.expm1(c)).sum())
    return mean, max(variance, 0.0)


def compute_rental_operations(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Projects one scenario's rental nights, ADR, gross revenue, fees and net rental income over its
    horizon (see the MODEL comment above). Returns:
      "monthly": DataFrame indexed by scenario month h (0 = closing month) with calendar_month, year,
        nights_* / adr_* / gross_* for * in median, lo, hi, lo_outer, hi_outer, mean;
      "annual": DataFrame indexed by scenario year with nights_mean, gross_mean, gross_sd, gross_* bands,
        channel_commissions_mean, mgmt_fee_mean, net_mean and net_* bands;
      "horizon": {"gross_mean", "gross_sd", "net_mean", gross_*/net_* bands};
      "fees": {"channel_commission_fraction", "mgmt_fee_fraction", "keep_fraction"};
      "ci", "outer_ci"."""
    deck = deck if deck is not None else load_deck()
    inputs = get_rental_operations_inputs(deck)
    priors = load_rental_priors()
    rentable = rentable_nights_by_month(deck)
    closing_month = get_closing_date(deck)["closing_month"]
    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])
    outer_ci = (0.01, 0.99)
    quantiles = {"median": 0.5, "lo": ci[0], "hi": ci[1], "lo_outer": outer_ci[0], "hi_outer": outer_ci[1]}
    rho = inputs["occupancy_adr_correlation"]

    horizon_months = int(scenario["horizon_years"]) * 12
    # The adr latent's growth factor from closing (h = 0 -> 1): median path and log-variance by month.
    growth = project_term_structure(get_growth_fit(inputs["adr_growth_latent"], deck), 1.0, horizon_months, ci=ci)
    growth_mu = np.log(growth["median"].to_numpy())
    growth_var = (np.log(growth["hi"].to_numpy() / growth["median"].to_numpy()) / norm.ppf(ci[1])) ** 2

    # Year-1 capped nights per calendar month (no growth on nights).
    nights_effective: dict[str, dict[str, float] | None] = {}
    for m in MONTH_KEYS:
        occupancy = priors["occupancy"][m]
        if occupancy["hard_zero"] or rentable[m] <= 0:
            nights_effective[m] = None
            continue
        nights_effective[m] = _capped_effective(
            occupancy["predictive_median"] * rentable[m], occupancy["sigma_log"], occupancy["level_sigma_log"], rentable[m]
        )

    rows = []
    a = np.zeros(horizon_months)  # occupancy year-to-year sigma (effective)
    b = np.zeros(horizon_months)  # occupancy level sigma (effective)
    c = np.zeros(horizon_months)  # ADR year-to-year sigma
    d = np.zeros(horizon_months)  # ADR level sigma
    mu = np.full(horizon_months, -np.inf)
    active = np.zeros(horizon_months, dtype=bool)
    years = np.arange(horizon_months) // 12 + 1
    for h in range(horizon_months):
        m = MONTH_KEYS[(closing_month - 1 + h) % 12]
        nights = nights_effective[m]
        adr = priors["adr"][m]
        row: dict[str, Any] = {"h": h, "calendar_month": m, "year": int(years[h])}
        if nights is None or adr["hard_zero"]:
            row.update({f"{k}_{band}": 0.0 for k in ("nights", "gross") for band in (*BAND_NAMES, "mean")})
            row.update({f"adr_{band}": math.nan for band in (*BAND_NAMES, "mean")})
            rows.append(row)
            continue
        adr_mu = math.log(adr["predictive_median"]) + growth_mu[h]
        adr_sigma = math.sqrt(adr["sigma_log"] ** 2 + adr["level_sigma_log"] ** 2 + growth_var[h])
        a[h], b[h] = nights["sigma_yy"], nights["sigma_level"]
        c[h], d[h] = adr["sigma_log"], adr["level_sigma_log"]
        gross_var = nights["sigma"] ** 2 + adr_sigma**2 + 2 * rho * a[h] * c[h]
        mu[h] = nights["mu"] + adr_mu
        active[h] = True
        nights_q = _lognormal_quantiles(nights["mu"], nights["sigma"], quantiles)
        adr_q = _lognormal_quantiles(adr_mu, adr_sigma, quantiles)
        gross_q = _lognormal_quantiles(mu[h], math.sqrt(gross_var), quantiles)
        row.update({f"nights_{k}": min(v, rentable[m]) for k, v in nights_q.items()})
        row["nights_mean"] = nights["mean"]
        row.update({f"adr_{k}": v for k, v in adr_q.items()})
        row["adr_mean"] = math.exp(adr_mu + adr_sigma**2 / 2)
        row.update({f"gross_{k}": v for k, v in gross_q.items()})
        row["gross_mean"] = math.exp(mu[h] + gross_var / 2)
        rows.append(row)
    monthly = pd.DataFrame(rows).set_index("h")

    # Log-space covariance of monthly gross revenue (see MODEL comment).
    same_year = years[:, None] == years[None, :]
    earlier = np.minimum.outer(np.arange(horizon_months), np.arange(horizon_months))
    cov = (
        same_year * (np.outer(a, a) + np.outer(c, c) + rho * (np.outer(a, c) + np.outer(c, a)))
        + np.outer(b, b)
        + np.outer(d, d)
        + growth_var[earlier]
    )
    mu_safe = np.where(active, mu, 0.0)

    keep = (1 - inputs["channel_commission_fraction"]) * (1 - inputs["mgmt_fee_fraction"])

    def summarize(index: np.ndarray) -> dict[str, float]:
        mean, variance = _sum_moments(mu_safe, cov, index[active[index]])
        out: dict[str, float] = {"gross_mean": mean, "gross_sd": math.sqrt(variance)}
        if mean > 0:
            q = _lognormal_quantiles(*_lognormal_from_moments(mean, variance), quantiles)
        else:
            q = dict.fromkeys(quantiles, 0.0)
        out.update({f"gross_{k}": v for k, v in q.items()})
        out.update({f"net_{k}": v * keep for k, v in q.items()})
        out["net_mean"] = mean * keep
        return out

    annual_rows = []
    for year in range(1, int(scenario["horizon_years"]) + 1):
        index = np.flatnonzero(years == year)
        summary = summarize(index)
        gross_mean = summary["gross_mean"]
        commissions = gross_mean * inputs["channel_commission_fraction"]
        summary.update({
            "year": year,
            "nights_mean": float(monthly.loc[index, "nights_mean"].sum()),
            "channel_commissions_mean": commissions,
            "mgmt_fee_mean": (gross_mean - commissions) * inputs["mgmt_fee_fraction"],
        })
        annual_rows.append(summary)
    annual = pd.DataFrame(annual_rows).set_index("year")

    return {
        "monthly": monthly,
        "annual": annual,
        "horizon": summarize(np.arange(horizon_months)),
        "fees": {
            "channel_commission_fraction": inputs["channel_commission_fraction"],
            "mgmt_fee_fraction": inputs["mgmt_fee_fraction"],
            "keep_fraction": keep,
        },
        "ci": ci,
        "outer_ci": outer_ci,
    }
