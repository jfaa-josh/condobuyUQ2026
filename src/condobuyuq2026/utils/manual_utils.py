"""Shared backend for the manual, one-off scripts in run/manual/, and (via
build_derived_latent_model/build_all_derived_latent_models) for run/main.py's first real build
step.

Not part of the analysis package's own logic (that's the rest of condobuyuq2026) — these are
helpers for data-pulling/model-building scripts specifically: talking to external APIs (BLS,
FRED, Yahoo Finance, ...), and saving what comes back into data/raw/. Keep this modular by data
source (one
fetch_<source>_* function per API) so future manual scripts pulling from the same source reuse
it, rather than each script re-implementing its own request/pagination/saving logic. Every
fetch_<source>_* function returns the same row shape regardless of source -- a list of
{"year", "period" ("M01".."M12"), "periodName", "value"} dicts -- so everything downstream
(filter_monthly here, then save_raw_csv/load_raw_csv/to_monthly_series in raw_data.py, and beyond
that fit_annual_ou etc. in ou_fitting.py) works identically no matter which source a given model
was built from. raw_data.py holds the disk-storage/parsing half of this pipeline, not this module
-- split out so carrying_costs.py can read raw historical data without a circular import back
through here (this module already imports FROM carrying_costs.py, see below).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from condobuyuq2026.carrying_costs import build_market_value_change_magnitude
from condobuyuq2026.input_deck import (
    get_carrying_cost_prior_config,
    get_derived_latents,
    get_forecast_model_general,
    get_forecast_model_home_price,
    get_forecast_model_market,
    get_growth_fit,
    get_market_value_config,
    get_report_quantiles,
    get_special_assessment_config,
    get_utilities_prior_config,
)
from condobuyuq2026.input_deck import load_deck as _load_deck
from condobuyuq2026.paths import MODELS_DIR
from condobuyuq2026.paths import model_fit_path as _model_fit_path
from condobuyuq2026.plotting.forecast_plots import (
    plot_derived_latent_model,
    plot_market_value_periods,
    plot_price_level_projection,
)
from condobuyuq2026.utils.ou_fitting import project_monthly_rate, project_term_structure, save_fit

BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
# BLS API v2's per-request limits: a registration key raises them, but doesn't require them —
# a small one-off historical pull (a handful of requests) works fine without a key at all.
BLS_MAX_YEARS_WITH_KEY = 20
BLS_MAX_YEARS_WITHOUT_KEY = 10


def _year_chunks(start_year: int, end_year: int, max_span: int) -> list[tuple[int, int]]:
    """Splits [start_year, end_year] into chunks no wider than max_span years, to respect the
    BLS API's per-request span limit."""
    chunks = []
    chunk_start = start_year
    while chunk_start <= end_year:
        chunk_end = min(chunk_start + max_span - 1, end_year)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + 1
    return chunks


def fetch_bls_series(series_id: str, start_year: int, end_year: int, api_key: str | None = None) -> list[dict[str, Any]]:
    """Pulls one BLS series over [start_year, end_year], transparently chunking requests to
    respect the API's per-request year-span limit. Works with or without a registration key
    (pass api_key if you have one) — omits it from the request entirely if none is given, using
    the lower unregistered limits instead of failing. Returns every observation (all
    periodicities BLS reports for this series — filter with filter_monthly() if you only want
    monthly points), sorted chronologically ascending regardless of how BLS ordered each chunk
    internally.
    """
    max_span = BLS_MAX_YEARS_WITH_KEY if api_key else BLS_MAX_YEARS_WITHOUT_KEY
    all_points: list[dict[str, Any]] = []
    for chunk_start, chunk_end in _year_chunks(start_year, end_year, max_span):
        payload = {
            "seriesid": [series_id],
            "startyear": str(chunk_start),
            "endyear": str(chunk_end),
        }
        if api_key:
            payload["registrationkey"] = api_key
        response = requests.post(BLS_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(
                f"BLS API request failed for {series_id} {chunk_start}-{chunk_end}: {body.get('message')}"
            )
        series_list = body["Results"]["series"]
        if not series_list or not series_list[0].get("data"):
            raise RuntimeError(f"BLS API returned no data for {series_id} {chunk_start}-{chunk_end}")
        all_points.extend(series_list[0]["data"])

    all_points.sort(key=lambda p: (int(p["year"]), p["period"]))
    return all_points


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(series_id: str) -> list[dict[str, Any]]:
    """Pulls one FRED series' entire available history via its CSV endpoint -- no API key, no
    rate limit, no date-range parameter (FRED always returns everything it has; slice the result
    afterward if you need less). Returns the same year/period/periodName/value row shape
    fetch_bls_series does (see module docstring), so the rest of a manual script's pipeline
    (filter_monthly, save_raw_csv/load_raw_csv, to_monthly_series, ou_fitting.fit_annual_ou, ...)
    doesn't need to know or care which source a series came from.

    Values are kept as FRED wrote them, unconverted -- including FRED's own placeholder for a
    missing/not-yet-published observation ("."), which parse_numeric_value (used by
    to_monthly_series) already treats as NaN the same way it does BLS's "-". This keeps data/raw/'s
    CSV a faithful copy of what the source actually returned, same as for BLS.

    Only meaningfully supports series FRED reports at MONTHLY-or-finer frequency -- period is
    always written as "M<month>", so a quarterly/annual series would be mislabeled.
    """
    series = pd.read_csv(f"{FRED_CSV_URL}?id={series_id}", index_col=0, parse_dates=True).iloc[:, 0]
    if series.empty:
        raise RuntimeError(f"FRED returned no data for series {series_id!r}")
    return [
        {
            "year": str(date.year),
            "period": f"M{date.month:02d}",
            "periodName": date.strftime("%B"),
            "value": str(value),
        }
        for date, value in series.items()
    ]


def fetch_yfinance_series(ticker: str, start: str) -> list[dict[str, Any]]:
    """Pulls one ticker's monthly closing price history via yfinance, from `start` (an ISO
    "YYYY-MM-DD" date) through today -- no API key, no rate limit, one request regardless of span.
    Returns the same year/period/periodName/value row shape fetch_bls_series/fetch_fred_series do
    (see module docstring), so the rest of a manual script's pipeline doesn't need to know or care
    which source a series came from.

    Normalizes yfinance's own bar-date index (a monthly bar's date is whatever day yfinance
    happened to stamp it, not necessarily the 1st) to month-start Timestamps before reindexing to
    a complete monthly calendar via asfreq("MS") -- same reasoning as to_monthly_series's own
    reindex: keeps a later .shift(12) meaning "12 months," not "12 rows."

    Uses raw (auto_adjust=False) Close, not dividend/split-adjusted -- for an index ticker like
    "^GSPC" this is just the index's own price level, not a total-return series (no dividend
    stream is associated with the index itself for yfinance to adjust). If you need a dividends-
    reinvested return, pull a total-return ticker (e.g. "^SP500TR") instead -- that's a modeling
    choice for the caller, not something this function decides.
    """
    close = yf.download(ticker, start=start, interval="1mo", auto_adjust=False, progress=False)["Close"].squeeze()
    close.index = close.index.to_period("M").to_timestamp()
    close = close.asfreq("MS")
    if close.empty:
        raise RuntimeError(f"yfinance returned no data for ticker {ticker!r}")
    return [
        {
            "year": str(date.year),
            "period": f"M{date.month:02d}",
            "periodName": date.strftime("%B"),
            "value": str(value),
        }
        for date, value in close.items()
    ]


def filter_monthly(data_points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keeps only monthly observations (period M01-M12), dropping annual averages (M13) and any
    other non-monthly periodicity a source might include for a given series. A no-op for sources
    that only ever report monthly periods to begin with (e.g. fetch_fred_series, fetch_yfinance_
    series with interval="1mo") -- kept in the pipeline anyway so every manual script follows the
    same fetch -> filter_monthly -> save_raw_csv shape regardless of source."""
    return [p for p in data_points if p["period"].startswith("M") and p["period"] != "M13"]


def model_plots_dir(model_name: str) -> Path:
    """Returns models/<model_name>/plots (creating it if needed) — where a manual script should
    save its sanity-check plots for that model."""
    path = MODELS_DIR / model_name / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_fit_path(model_name: str) -> Path:
    """Returns models/<model_name>/fit.json (creating models/<model_name>/ if needed) — where a
    manual script should save its fitted model parameters (see ou_fitting.save_fit/load_fit),
    alongside its plots. Path convention lives in condobuyuq2026.paths (shared with input_deck.py's
    read side) -- this just also ensures the directory exists, which the read side shouldn't do."""
    path = _model_fit_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def fit_effective_ou(component_fits: dict[str, dict[str, float]], weights: dict[str, float]) -> dict[str, float]:
    """Moment-matches the three founding processes' OWN OU fits (see ou_fitting.fit_annual_ou) into
    a single "effective" OU fit for a derived latent's weighted blend -- so a derived latent's
    fit.json is a MODEL (same {alpha, beta, i_inf, tau, sig, i0, phi_monthly} shape as
    general/home_price/market's own fit.json), not a precomputed table of monthly distributions.
    This matters for the plan: other variables' growth propagation (still to build) calls
    project_term_structure/project_monthly_rate on a fit dict at whatever horizon/quantile it
    needs -- a fixed table would only ever answer the exact (horizon, quantile) pairs baked in when
    it was built, while a parametric fit answers any of them, exactly like a founding process's own.

    The blend combined_t = sum(w_k * component_k_t) is itself linear in each component's OU-modeled
    annual rate, so two of its moments are EXACT, not approximated:
    - i_inf_eff = sum(w_k * i_inf_k): a weighted sum of long-run means is exactly the long-run mean
      of the weighted sum.
    - i0_eff = sum(w_k * i0_k): same identity, at the starting point.

    tau/sig/phi/beta/alpha have no single exact closed form for a sum of OU processes with
    different taus (the sum isn't itself a plain OU process in general), so these five are
    recovered by matching the blend's actual variance curve at its two most informationally
    distinctive points, then reading an OU fit off that -- the same "back out phi/tau from two
    moments" style project_monthly_rate's own docstring already uses:
    - Initial slope: d/dh[Var(h)]|_(h=0) = sig^2 for a single OU process (immediate, from
      project_monthly_rate's var(h) formula). Assuming independent idiosyncratic components (no
      correlation term needed -- each w_k*component_k contributes its own variance),
      sig_eff^2 = sum(w_k^2 * sig_k^2) matches the blend's own initial slope exactly.
    - Asymptotic level: each component's own stationary/marginal variance is sig_k^2*tau_k/2 (see
      fit_annual_ou's docstring); the blend's is V_inf_eff = sum(w_k^2 * sig_k^2*tau_k/2) by the
      same independence assumption (deliberately excluding each entry's own idiosyncratic share --
      it has no natural real-unit scale here, consistent with build_derived_latent_model's plot
      already treating the blend as real-unit, not standardized).
    - Solving var(h) = sig_eff^2*(tau_eff/2)*(1 - phi_eff^(2h)) for tau_eff against both matched
      points gives tau_eff = 2*V_inf_eff / sig_eff^2, then phi_eff = exp(-1/tau_eff),
      beta_eff = phi_eff^12, alpha_eff = i_inf_eff*(1 - beta_eff) (backed out purely for interface
      completeness -- project_monthly_rate/project_term_structure only ever read i_inf/tau/sig/i0/
      phi_monthly, never alpha/beta directly).

    Returns a fit dict in exactly fit_annual_ou's own shape, usable everywhere a founding process's
    fit is (project_term_structure, project_monthly_rate, save_fit/load_fit).
    """
    i_inf_eff = sum(weights[key] * component_fits[key]["i_inf"] for key in weights)
    i0_eff = sum(weights[key] * component_fits[key]["i0"] for key in weights)
    sig_sq_eff = sum(weights[key] ** 2 * component_fits[key]["sig"] ** 2 for key in weights)
    v_inf_eff = sum(
        weights[key] ** 2 * component_fits[key]["sig"] ** 2 * component_fits[key]["tau"] / 2 for key in weights
    )
    tau_eff = 2 * v_inf_eff / sig_sq_eff
    sig_eff = np.sqrt(sig_sq_eff)
    phi_eff = np.exp(-1 / tau_eff)
    beta_eff = phi_eff**12
    alpha_eff = i_inf_eff * (1 - beta_eff)
    return {
        "alpha": alpha_eff,
        "beta": beta_eff,
        "i_inf": i_inf_eff,
        "tau": tau_eff,
        "sig": sig_eff,
        "i0": i0_eff,
        "phi_monthly": phi_eff,
    }


def build_derived_latent_model(
    name: str, deck: dict[str, Any] | None = None, horizon_months: int = 120
) -> dict[str, Path]:
    """Builds one derived_latents entry's model: moment-matches the three fitted forecast_models'
    (general/home_price/market) OWN OU fits into a single effective OU fit (see fit_effective_ou)
    for that entry's weights (input_deck.get_derived_latents), then saves that fit -- a MODEL, in
    the same {alpha, beta, i_inf, tau, sig, i0, phi_monthly} shape as any founding process's own
    fit.json -- rather than a table of precomputed monthly distributions. Other variables'
    propagation (still to build) can then call project_term_structure/project_monthly_rate on it
    directly, at whatever horizon/quantile they need, exactly like a founding process's fit.

    Saves models/derived_variables/<name>/fit.json (via ou_fitting.save_fit) and a sanity-check
    plot to models/derived_variables/<name>/plots/model.png: each component's own REAL-UNIT
    monthly-rate curve (ou_fitting.project_monthly_rate, unweighted -- the same curve as that
    component's own monthly_rate.png) at an alpha proportional to its weight for this entry (a
    0-weight component is invisible; a high-weight one is prominent), superimposed with the
    effective fit's own projection as a fully-opaque thick black line -- see
    plotting.forecast_plots.plot_derived_latent_model. Components won't necessarily "line up" with
    each other or the combined line (they're different real quantities, only linearly blended for
    comparison) -- that's expected, not a bug.

    Returns {"fit_path": ..., "plot_path": ...}.
    """
    deck = deck if deck is not None else _load_deck()
    weights = get_derived_latents(deck)[name]
    component_fits = {
        "market": get_forecast_model_market(deck),
        "home_price": get_forecast_model_home_price(deck),
        "general": get_forecast_model_general(deck),
    }

    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])
    outer_ci = (0.01, 0.99)  # wide reference band, same convention as the other build scripts

    effective_fit = fit_effective_ou(component_fits, weights)

    component_projections = {key: project_monthly_rate(fit, horizon_months, ci) for key, fit in component_fits.items()}
    combined_projection = project_monthly_rate(effective_fit, horizon_months, ci)
    combined_outer_projection = project_monthly_rate(effective_fit, horizon_months, outer_ci)

    fit_path = model_fit_path(f"derived_variables/{name}")
    save_fit(effective_fit, fit_path)

    plot_path = model_plots_dir(f"derived_variables/{name}") / "model.png"
    plot_derived_latent_model(
        component_projections,
        weights,
        combined_projection,
        combined_outer_projection,
        title=f"{name} — derived latent model ({int(ci[0] * 100)}-{int(ci[1] * 100)}% band)",
        save_path=plot_path,
    )
    return {"fit_path": fit_path, "plot_path": plot_path}


def build_all_derived_latent_models(
    deck: dict[str, Any] | None = None, horizon_months: int = 120
) -> dict[str, dict[str, Path]]:
    """Calls build_derived_latent_model for every entry in derived_latents. Returns {name:
    {"fit_path": ..., "plot_path": ...}}."""
    deck = deck if deck is not None else _load_deck()
    names = get_derived_latents(deck)
    return {name: build_derived_latent_model(name, deck, horizon_months) for name in names}


_CARRYING_COST_PRIOR_OUTER_CI = (0.01, 0.99)  # wide reference band, same convention as other build scripts


def build_carrying_cost_prior_model(
    name: str, deck: dict[str, Any] | None = None, horizon_months: int = 120, ylabel: str = "USD/year"
) -> dict[str, Path]:
    """Builds one of carrying_costs' LEVEL+GROWTH entries (hoa_dues_annual/insurance_annual/
    maintenance_annual/property_tax.market_value -- NOT special_assessment or utilities, see
    build_special_assessment_prior_model/build_utilities_prior_model for their own shapes) into a
    saved projection: the named growth latent's own fit (already built) applied to the level's own
    median via ou_fitting.project_term_structure, with the level's own cv folded in as a constant
    extra_log_variance (see that parameter's own docstring, and input_deck.yaml's carrying_costs
    section header comment for why this replaces a separately hand-set growth rate).

    No new "fitting" happens here -- this is a formula, not information learned from data -- so
    fit.json just echoes the resolved deck config back out ({"level_median", "level_cv", "latent"}),
    and the real value of this build step is the plot: a sanity check while tuning the level prior
    by hand, the same workflow build_derived_latent_model already gives derived_latents. ylabel
    defaults to "USD/year" (an annual cost) -- market_value passes "USD" instead (a value, not a
    yearly flow).

    Saves models/carrying_cost_priors/<name>/fit.json and .../plots/model.png. Returns
    {"fit_path": ..., "plot_path": ...}.
    """
    deck = deck if deck is not None else _load_deck()
    config = get_carrying_cost_prior_config(name, deck)
    latent_fit = get_growth_fit(config["latent"], deck)
    extra_log_variance = np.log(1 + config["level_cv"] ** 2)

    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])

    projection = project_term_structure(
        latent_fit, p0=config["level_median"], horizon_months=horizon_months, ci=ci, extra_log_variance=extra_log_variance
    )
    outer_projection = project_term_structure(
        latent_fit,
        p0=config["level_median"],
        horizon_months=horizon_months,
        ci=_CARRYING_COST_PRIOR_OUTER_CI,
        extra_log_variance=extra_log_variance,
    )

    fit_path = model_fit_path(f"carrying_cost_priors/{name}")
    with fit_path.open("w", encoding="utf-8") as f:
        json.dump({"level_median": config["level_median"], "level_cv": config["level_cv"], "latent": config["latent"]}, f, indent=2)

    plot_path = model_plots_dir(f"carrying_cost_priors/{name}") / "model.png"
    plot_price_level_projection(
        projection,
        outer_projection,
        title=f"{name} — carrying-cost prior ({int(ci[0] * 100)}-{int(ci[1] * 100)}% band)",
        save_path=plot_path,
        ylabel=ylabel,
    )
    return {"fit_path": fit_path, "plot_path": plot_path}


def build_special_assessment_prior_model(
    deck: dict[str, Any] | None = None, horizon_months: int = 120
) -> dict[str, Path]:
    """Builds carrying_costs.special_assessment's own projection: SAME mechanism as
    build_carrying_cost_prior_model, applied to the mixture's severity.median/cv (p, the annual
    probability, is NOT grown -- see that field's own deck comment -- so it's just echoed into
    fit.json as-is for carrying_costs.py to read). Saves models/carrying_cost_priors/
    special_assessment/fit.json + plot."""
    deck = deck if deck is not None else _load_deck()
    config = get_special_assessment_config(deck)
    latent_fit = get_growth_fit(config["latent"], deck)
    extra_log_variance = np.log(1 + config["severity_cv"] ** 2)

    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])

    projection = project_term_structure(
        latent_fit,
        p0=config["severity_median"],
        horizon_months=horizon_months,
        ci=ci,
        extra_log_variance=extra_log_variance,
    )
    outer_projection = project_term_structure(
        latent_fit,
        p0=config["severity_median"],
        horizon_months=horizon_months,
        ci=_CARRYING_COST_PRIOR_OUTER_CI,
        extra_log_variance=extra_log_variance,
    )

    fit_path = model_fit_path("carrying_cost_priors/special_assessment")
    with fit_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "annual_probability": config["annual_probability"],
                "severity_median": config["severity_median"],
                "severity_cv": config["severity_cv"],
                "latent": config["latent"],
            },
            f,
            indent=2,
        )

    plot_path = model_plots_dir("carrying_cost_priors/special_assessment") / "model.png"
    plot_price_level_projection(
        projection,
        outer_projection,
        title=f"special_assessment severity — carrying-cost prior ({int(ci[0] * 100)}-{int(ci[1] * 100)}% band)",
        save_path=plot_path,
        ylabel="USD (if it occurs)",
    )
    return {"fit_path": fit_path, "plot_path": plot_path}


def build_utilities_prior_model(deck: dict[str, Any] | None = None, horizon_months: int = 120) -> dict[str, Path]:
    """Builds carrying_costs.utilities' own projection. Unlike the other four entries, utilities
    has 12 separate monthly levels (its seasonal shape), not one -- fit.json keeps all 12 medians/
    cvs as given (carrying_costs.py projects each month independently at USE time, via
    ou_fitting.project_term_structure with that month's own median as p0 -- h_months=0 is "now," so
    h_months=12*n correctly lands on that same calendar month n years out).

    The saved PLOT is a sanity check only, on the ANNUAL TOTAL (sum of all 12 months' medians,
    projected using a weighted-average cv across months as one representative extra_log_variance --
    an approximation ONLY for this plot; carrying_costs.py's real per-scenario computation always
    uses each month's own exact projection, never this blended approximation).
    """
    deck = deck if deck is not None else _load_deck()
    config = get_utilities_prior_config(deck)
    latent_fit = get_growth_fit(config["latent"], deck)
    medians_by_month = config["level_medians_by_month"]
    cvs_by_month = config["level_cvs_by_month"]

    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])

    annual_total_year0 = sum(medians_by_month.values())
    weighted_cv = sum(medians_by_month[m] * cvs_by_month[m] for m in medians_by_month) / annual_total_year0
    extra_log_variance = np.log(1 + weighted_cv**2)

    annual_projection = project_term_structure(
        latent_fit, p0=annual_total_year0, horizon_months=horizon_months, ci=ci, extra_log_variance=extra_log_variance
    )
    annual_outer_projection = project_term_structure(
        latent_fit,
        p0=annual_total_year0,
        horizon_months=horizon_months,
        ci=_CARRYING_COST_PRIOR_OUTER_CI,
        extra_log_variance=extra_log_variance,
    )

    fit_path = model_fit_path("carrying_cost_priors/utilities")
    with fit_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"level_medians_by_month": medians_by_month, "level_cvs_by_month": cvs_by_month, "latent": config["latent"]},
            f,
            indent=2,
        )

    plot_path = model_plots_dir("carrying_cost_priors/utilities") / "model.png"
    plot_price_level_projection(
        annual_projection,
        annual_outer_projection,
        title=f"utilities (annual total) — carrying-cost prior ({int(ci[0] * 100)}-{int(ci[1] * 100)}% band)",
        save_path=plot_path,
        ylabel="USD/year",
    )
    return {"fit_path": fit_path, "plot_path": plot_path}


def build_market_value_prior_model(deck: dict[str, Any] | None = None, horizon_months: int = 120) -> dict[str, Path]:
    """Builds carrying_costs.property_tax.market_value's own CHANGE-MAGNITUDE model (see
    carrying_costs.build_market_value_change_magnitude for the actual computation -- real historical
    data + OU projections + the pre-closing/projected averaging blend, done ONCE here) and a
    sanity-check plot. Unlike the other four LEVEL+GROWTH entries, this fit.json is genuinely a
    MODEL that needs real computation to produce (not just an echo of deck config) -- but it's still
    UNSCALED (price=1.0 basis, see that function's own docstring), so a scenario run still never
    repeats this work, just scales the saved periods by its own live initial_price.

    The saved plot draws each reassessment period as its OWN independent object (see
    plotting.forecast_plots.plot_market_value_periods) -- a clean vertical break at each
    reassessment instead of a diagonal line connecting periods, since the value doesn't actually
    move between reassessments. Saves models/carrying_cost_priors/market_value/fit.json +
    plots/model.png.
    """
    deck = deck if deck is not None else _load_deck()
    change_magnitude = build_market_value_change_magnitude(deck, horizon_months)
    initial_price = get_market_value_config(deck)["initial_price"]
    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])

    fit_path = model_fit_path("carrying_cost_priors/market_value")
    with fit_path.open("w", encoding="utf-8") as f:
        json.dump(change_magnitude, f, indent=2)

    plot_path = model_plots_dir("carrying_cost_priors/market_value") / "model.png"
    plot_market_value_periods(
        change_magnitude["periods"],
        initial_price,
        ci,
        _CARRYING_COST_PRIOR_OUTER_CI,
        horizon_months=horizon_months,
        title=f"market_value (stair-stepped) — carrying-cost prior ({int(ci[0] * 100)}-{int(ci[1] * 100)}% band)",
        save_path=plot_path,
    )
    return {"fit_path": fit_path, "plot_path": plot_path}


def build_all_carrying_cost_prior_models(
    deck: dict[str, Any] | None = None, horizon_months: int = 120
) -> dict[str, dict[str, Path]]:
    """Builds every carrying_costs LEVEL+GROWTH entry's projection (hoa_dues_annual,
    insurance_annual, maintenance_annual, special_assessment, utilities), plus property_tax.
    market_value's own (differently-shaped, see build_market_value_prior_model) projection. Returns
    {name: {"fit_path": ..., "plot_path": ...}}."""
    deck = deck if deck is not None else _load_deck()
    results = {
        name: build_carrying_cost_prior_model(name, deck, horizon_months)
        for name in ("hoa_dues_annual", "insurance_annual", "maintenance_annual")
    }
    results["market_value"] = build_market_value_prior_model(deck, horizon_months)
    results["special_assessment"] = build_special_assessment_prior_model(deck, horizon_months)
    results["utilities"] = build_utilities_prior_model(deck, horizon_months)
    return results
