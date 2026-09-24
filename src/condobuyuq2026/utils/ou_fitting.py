"""Fits a continuous-time Ornstein-Uhlenbeck (mean-reverting) model to an annualized growth rate
implied by a monthly price/index series, and projects a closed-form median + confidence-band term
structure from it.

Written for forecast_models.general (CPI), but the math only assumes "a monthly index whose
year-over-year log change mean-reverts" -- nothing CPI-specific -- so it should apply unchanged to
future home_price/market fitting scripts too (see .claude/implementation-plan.md's Earmarks).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def fit_annual_ou(index_series: pd.Series) -> dict[str, float]:
    """Fits a continuous-time OU model to index_series' annual log growth rate.

    index_series must be a monthly-frequency Series (see manual_utils.to_monthly_series) with no
    calendar gaps -- gaps silently misalign the lag-12 comparisons below.

    Method: let a_t = log(index_t / index_(t-12)), the annual log growth rate as of month t
    (monthly-sampled, so consecutive a_t are ~92% overlapping). Fitting a lag-1 AR(1) directly on
    a_t would pick up that overlap as spurious autocorrelation. Instead, regress a_(t+12) on a_t
    (OLS) -- two genuinely non-overlapping annual windows, twelve months apart, while still using
    every month as an observation rather than one point per calendar year:

        a_(t+12) = alpha + beta * a_t + resid

    Read a continuous-time OU calibration off that discrete annual-step fit:
    - i_inf = alpha / (1 - beta): the long-run mean annual growth rate the process reverts to.
    - tau = -12 / ln(beta): the mean-reversion time constant, in months (beta is the process's
      correlation over a 12-month step, so beta = exp(-12/tau)).
    - phi_monthly = beta ** (1/12) = exp(-1/tau): the equivalent monthly AR(1) persistence -- this
      is what forecast_models' phi means elsewhere in this project.
    - sig = resid.std() / sqrt(12): the annual regression residual's volatility, scaled down to an
      implied monthly rate by simple linear variance scaling. NOTE: this is an approximation, not
      the exact OU discretization variance for a 12-month step (which would also depend on tau and
      beta) -- consistent with this project's preference for named, closed-form approximations
      over exact-but-unwieldy formulas.
    - i0: the most recently observed a_t, i.e. where the process starts from for projection.

    Returns a dict with i_inf, tau, sig, i0, phi_monthly, alpha, beta.
    """
    a = np.log(index_series / index_series.shift(12))
    paired = pd.concat([a, a.shift(-12)], axis=1).dropna()
    beta, alpha = np.polyfit(paired.iloc[:, 0], paired.iloc[:, 1], 1)
    resid = paired.iloc[:, 1] - (alpha + beta * paired.iloc[:, 0])

    return {
        "alpha": alpha,
        "beta": beta,
        "i_inf": alpha / (1 - beta),
        "tau": -12 / np.log(beta),
        "sig": resid.std() / np.sqrt(12),
        "i0": a.dropna().iloc[-1],
        "phi_monthly": beta ** (1 / 12),
    }


def project_term_structure(
    fit: dict[str, float], p0: float, horizon_months: int, ci: tuple[float, float] = (0.05, 0.95)
) -> pd.DataFrame:
    """Projects the index's median path and a (ci[0], ci[1]) confidence band forward from p0,
    given an OU fit from fit_annual_ou. Returns a DataFrame indexed by month offset h = 0..
    horizon_months (h=0 is p0 itself) with columns "median", "lo", "hi".

    D(h) is the closed-form integral of the expected instantaneous annual growth rate,
    i_inf + (i0 - i_inf) * exp(-t/tau), from t=0 to t=h months, converted from "rate-months" to
    "rate-years" (dividing by 12) since i_inf/i0 are annualized rates:

        D(h) = (i_inf * h + (i0 - i_inf) * tau * (1 - exp(-h/tau))) / 12

    median = p0 * exp(D(h)); the band multiplies in +/- z * sig * sqrt(h) on the log scale (sig*
    sqrt(h) treats each month's shock as i.i.d., which is the same "grows like a random walk"
    approximation used for sig itself -- see fit_annual_ou's docstring).
    """
    h = np.arange(horizon_months + 1)
    drift = (fit["i_inf"] * h + (fit["i0"] - fit["i_inf"]) * fit["tau"] * (1 - np.exp(-h / fit["tau"]))) / 12
    median = p0 * np.exp(drift)
    z_lo, z_hi = norm.ppf(ci)
    lo = p0 * np.exp(drift + z_lo * fit["sig"] * np.sqrt(h))
    hi = p0 * np.exp(drift + z_hi * fit["sig"] * np.sqrt(h))
    return pd.DataFrame({"median": median, "lo": lo, "hi": hi}, index=pd.Index(h, name="h_months"))


def project_monthly_rate(
    fit: dict[str, float], horizon_months: int, ci: tuple[float, float] = (0.05, 0.95)
) -> pd.DataFrame:
    """Projects the EXPECTED MONTHLY fractional price increase implied by the annualized rate's own
    mean-reverting path -- "how much does the index actually grow from one month to the next,"
    as opposed to project_term_structure's cumulative price LEVEL. Needs no starting price (p0):
    a fraction doesn't depend on the index's level, only on the annualized rate's own path.

    This is the annualized rate's own uncertainty CONE -- widens from i0's known starting point,
    then FLATTENS at the rate's marginal/stationary spread as h grows (see "Temporal model" in
    .claude/implementation-plan.md). That's a fundamentally different shape from
    project_term_structure's cumulative-price band, which keeps widening indefinitely because it
    integrates the rate's cone over time rather than looking at the rate's own level:

        mean(h) = i_inf + (i0 - i_inf) * phi_monthly**h        -- expected annualized rate at h
        var(h)  = sig**2 * (1 - phi_monthly**(2*h)) / (1 - phi_monthly**2)   -- its variance at h

    (fit["sig"] is treated here as the monthly AR(1) innovation's std -- the same approximation
    project_term_structure's band already relies on.) Converts the resulting annualized-rate
    distribution to an actual monthly fractional change via exp(rate / 12) - 1.

    Returns a DataFrame indexed by month offset h = 0..horizon_months with columns "median", "lo",
    "hi", each a fraction (e.g. 0.0025 = 0.25% that month).
    """
    h = np.arange(horizon_months + 1)
    mean = fit["i_inf"] + (fit["i0"] - fit["i_inf"]) * fit["phi_monthly"] ** h
    var = fit["sig"] ** 2 * (1 - fit["phi_monthly"] ** (2 * h)) / (1 - fit["phi_monthly"] ** 2)
    sd = np.sqrt(var)
    z_lo, z_hi = norm.ppf(ci)
    median = np.exp(mean / 12) - 1
    lo = np.exp((mean + z_lo * sd) / 12) - 1
    hi = np.exp((mean + z_hi * sd) / 12) - 1
    return pd.DataFrame({"median": median, "lo": lo, "hi": hi}, index=pd.Index(h, name="h_months"))


def save_fit(fit: dict[str, float], path: Path) -> Path:
    """Saves a fit dict (from fit_annual_ou) to a JSON file at path, creating parent directories as
    needed. Returns the path written. Reload with load_fit to reuse a fitted model for prediction
    (project_term_structure / project_monthly_rate) without re-fitting from raw data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({key: float(value) for key, value in fit.items()}, f, indent=2)
    return path


def load_fit(path: Path) -> dict[str, float]:
    """Loads a fit dict saved by save_fit -- e.g. fit = load_fit(...); project_term_structure(fit,
    p0=my_price, horizon_months=60)."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
