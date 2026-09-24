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
    - sig: the process's own continuous-time diffusion coefficient (time measured in months), read
      EXACTLY off the regression rather than approximated. For an OU process observed T months
      apart, Var(step) = sig^2 * tau/2 * (1 - exp(-2T/tau)); with T=12 that's exactly
      resid.var() = sig^2 * tau/2 * (1 - beta^2), so sig = sqrt(2 * resid.var() / (tau * (1 -
      beta^2))). This replaces an earlier `resid.std() / sqrt(12)` approximation that implicitly
      assumed variance keeps accumulating at a fixed rate regardless of how fast the process
      mean-reverts -- fine when tau is large relative to the horizons of interest (true for CPI),
      badly wrong when tau is small (a weakly-autocorrelated series like stock returns fits to a
      SHORT tau, and the old formula overstated the resulting price-level band by multiples --
      see project_term_structure).
    - i0: the most recently observed a_t, i.e. where the process starts from for projection.

    Returns a dict with i_inf, tau, sig, i0, phi_monthly, alpha, beta.
    """
    a = np.log(index_series / index_series.shift(12))
    paired = pd.concat([a, a.shift(-12)], axis=1).dropna()
    beta, alpha = np.polyfit(paired.iloc[:, 0], paired.iloc[:, 1], 1)
    resid = paired.iloc[:, 1] - (alpha + beta * paired.iloc[:, 0])
    tau = -12 / np.log(beta)

    return {
        "alpha": alpha,
        "beta": beta,
        "i_inf": alpha / (1 - beta),
        "tau": tau,
        "sig": np.sqrt(2 * resid.var() / (tau * (1 - beta**2))),
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

    median = p0 * exp(D(h)); the band is +/- z * sd(h) around D(h) on the log scale, where sd(h) is
    the EXACT std of that same integral (cumulative log-price growth is literally the integral of
    the OU-modeled rate, so its variance is the integral's variance, not something separately
    approximated). This is the same "integrated interest rate" variance used to price zero-coupon
    bonds under the Vasicek model (same SDE): with x = h/tau,

        Var(h) = sig^2 * tau^3 * (x - 1.5 + 2*exp(-x) - 0.5*exp(-2x)) / 144

    (the /144 converts the integral's own months^2-and-rate^2 units down to the annualized-rate-per-
    year units i_inf/sig are already in, matching the /12 already used for the drift D(h) above).
    For h >> tau this grows ~linearly (unboundedly, as expected for a cumulative price level);
    for h << tau it's small, correctly reflecting that a fast-mean-reverting rate (small tau)
    barely has time to move the cumulative total before reverting back.
    """
    h = np.arange(horizon_months + 1)
    tau = fit["tau"]
    drift = (fit["i_inf"] * h + (fit["i0"] - fit["i_inf"]) * tau * (1 - np.exp(-h / tau))) / 12
    median = p0 * np.exp(drift)
    x = h / tau
    variance = fit["sig"] ** 2 * tau**3 * (x - 1.5 + 2 * np.exp(-x) - 0.5 * np.exp(-2 * x)) / 144
    sd = np.sqrt(np.clip(variance, 0, None))  # clip: only ever-so-slightly negative from float error near h=0
    z_lo, z_hi = norm.ppf(ci)
    lo = p0 * np.exp(drift + z_lo * sd)
    hi = p0 * np.exp(drift + z_hi * sd)
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
        var(h)  = sig**2 * (tau/2) * (1 - phi_monthly**(2*h))  -- its variance at h (exact, given
                  the OU model -- this is the standard OU/Vasicek "variance of the level at time h,
                  started from a known point" formula: sig^2*tau/2 is the stationary/marginal
                  variance the process approaches as h grows, per fit_annual_ou's docstring)

    Converts the resulting annualized-rate distribution to an actual monthly fractional change via
    exp(rate / 12) - 1.

    Returns a DataFrame indexed by month offset h = 0..horizon_months with columns "median", "lo",
    "hi", each a fraction (e.g. 0.0025 = 0.25% that month).
    """
    h = np.arange(horizon_months + 1)
    mean = fit["i_inf"] + (fit["i0"] - fit["i_inf"]) * fit["phi_monthly"] ** h
    var = fit["sig"] ** 2 * (fit["tau"] / 2) * (1 - fit["phi_monthly"] ** (2 * h))
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
