"""Computes one scenario's per-year carrying costs: property tax (REAL, using Colorado's actual
two-step assessment_rate x mill_levy mechanic, local AND school components separately -- see
input_deck.yaml's carrying_costs.property_tax comment for the research behind it), plus HOA dues,
insurance, maintenance, special assessment (expected value), and utilities -- each projected from
its own pre-built carrying_cost_prior model (see manual_utils.build_all_carrying_cost_prior_models,
wired into run/main.py's build stage, the same way derived_latents are).

Also owns project_condo_value_schedule -- the condo's own CONTINUOUSLY appreciating value over the
horizon (via forecast_models.home_price) -- since it's fundamentally a "what does this property
carry/hold cost" question; taxes.py imports it from here for its own sale-price calculation instead
of duplicating it. NOT the same thing as project_market_value_schedule below (the ASSESSOR's own,
only-periodically-updated determination used for property tax specifically) -- see that function's
own docstring for why they're different.

CHANGE-MAGNITUDE MODELS: every carrying_cost_prior (including market_value) is built and saved
UNSCALED (see manual_utils.build_all_carrying_cost_prior_models) -- a scenario run never re-fits
anything, it just scales the saved change-magnitude model by that scenario's own price/horizon.
market_value's own build step (build_market_value_change_magnitude, below) is the expensive one: it
resolves real historical data + OU projections into one set of per-reassessment-period parameters
{growth_a, growth_b, growth_sigma} ONCE; project_market_value_schedule then just looks up the right
period for each requested year and scales by initial_price -- see that function's own docstring for
the exact math.

Deliberately NOT part of input_deck.py (deck loading/validation only) -- this is the section's own
per-scenario computation, the same "one computation module per deck section" pattern
acquisition.py/financing.py established.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from condobuyuq2026.input_deck import (
    DERIVED_LATENT_COMPONENTS,
    get_closing_date,
    get_forecast_model_home_price,
    get_forecast_model_name,
    get_growth_fit,
    get_market_value_config,
    get_property_tax_inputs,
    get_report_quantiles,
    load_deck,
)
from condobuyuq2026.paths import model_fit_path
from condobuyuq2026.raw_data import load_monthly_series
from condobuyuq2026.utils.ou_fitting import load_fit, project_term_structure

# classification/lodging assessment rate are HARDCODED here (moved out of the deck 2026-09-26) --
# NOT a deck field, since no current CO law ties property-tax classification to rental activity
# (two bills that would have both died in committee in 2024 -- see the "CO property tax STR
# classification" memory and input_deck.yaml's own property_tax comment for the research).
# "residential" is simply the only currently-real option; revisit this constant (and the
# NotImplementedError below) if that law ever changes.
PROPERTY_TAX_CLASSIFICATION = "residential"
LODGING_ASSESSMENT_RATE = 0.279  # example -- CO's ~2026 commercial/lodging rate; reference only,
                                  # unused while PROPERTY_TAX_CLASSIFICATION == "residential" (no
                                  # lodging-specific local/school mill-levy split is defined below)

# What FRACTION of the assessor's market_value each component's assessment_rate applies to --
# Colorado's own rule, hardcoded (not deck-configurable) same reasoning as CO_STATUTORY_
# RESIDENCY_DAYS in taxes.py.
LOCAL_ASSESSED_VALUE_FRACTION = 0.90
SCHOOL_ASSESSED_VALUE_FRACTION = 1.00


REASSESSMENT_AVERAGING_WINDOW_MONTHS = 18  # 1.5 years -- Colorado's own real assessment-cycle
                                            # convention (see input_deck.yaml's market_value comment
                                            # for the worked Jan2023/Jun2024/Jan2025 example)


def months_to_first_reassessment(closing_month: int, first_reassessment_year: int) -> int:
    """Whole months from closing to January 1 of (closing_year + first_reassessment_year) -- e.g.
    closing_month=10 (October), first_reassessment_year=2 gives 15 (Nov+Dec of closing_year, all of
    year closing_year+1, plus Jan of closing_year+2 -- 12*2 - (10-1) = 15). closing_year itself
    cancels out of this formula (first_reassessment_year is already relative to it), so only
    closing_month matters here."""
    return first_reassessment_year * 12 - (closing_month - 1)


def reassessment_boundary_months(closing_month: int, first_reassessment_year: int, reassessment_frequency_years: int, num_periods: int) -> list[int]:
    """Returns the month offsets (from closing, t=0) where market_value's assessed value changes:
    [0, M_1, M_1+P, M_1+2P, ..., M_1+(num_periods-1)*P] -- period k spans [boundaries[k],
    boundaries[k+1]) (the last period is open-ended). Period 0 (before the first reassessment) is
    the deterministic initial_price itself; every period from 1 onward is an AVERAGED value -- see
    project_market_value_schedule's own docstring for the full mechanism."""
    period_months = reassessment_frequency_years * 12
    m1 = months_to_first_reassessment(closing_month, first_reassessment_year)
    return [0] + [m1 + k * period_months for k in range(num_periods)]


def historical_growth_factor(monthly_series: pd.Series, closing_calendar_date: pd.Timestamp, offset_months: int) -> float:
    """The REAL, already-recorded ratio of the market's own index level at (closing + offset_months)
    to its level AT closing -- offset_months must be <= 0 (a calendar month at or before closing;
    see project_market_value_schedule's docstring for why only pre-closing months use real data
    instead of a projection). Uses Series.asof, which returns the last valid (non-NaN) value AT OR
    BEFORE a given timestamp -- this both skips real gap months in the raw data AND, if closing
    itself is a near-future month the raw data doesn't extend to yet, pins to the latest actually-
    recorded value rather than needing a separate "today vs. closing" special case."""
    target_date = closing_calendar_date + pd.DateOffset(months=offset_months)
    index_at_closing = monthly_series.asof(closing_calendar_date)
    index_at_target = monthly_series.asof(target_date)
    if pd.isna(index_at_closing) or pd.isna(index_at_target):
        raise ValueError(
            f"No historical data at or before {target_date.date()} or {closing_calendar_date.date()} "
            "in this founding process's own raw series -- closing_month/closing_year (or a "
            "reassessment window that reaches further back) predates all available historical data."
        )
    return index_at_target / index_at_closing


def build_market_value_change_magnitude(deck: dict[str, Any] | None = None, horizon_months: int = 120) -> dict[str, Any]:
    """Builds market_value's own CHANGE-MAGNITUDE model: unscaled (price=1.0 basis), independent of
    any particular scenario's horizon_years or purchase_price -- the expensive part (real historical
    lookups + OU projections + the pre-closing/projected averaging blend, see
    project_market_value_schedule's own docstring for the full per-reassessment mechanism) done ONCE
    here, so a scenario run never repeats it (see this module's own docstring). Called by
    manual_utils.build_market_value_prior_model (the build step) -- NOT by
    project_market_value_schedule, which just loads the SAVED result and scales it.

    horizon_months: the "extrapolation limit" -- how far out to precompute reassessment periods for.
    Must cover whatever scenario.horizon_years the deck's sweep might need; defaults to 120 (10
    years), matching every other carrying_cost_prior build function's own default.

    Returns {"closing_month", "first_reassessment_year", "reassessment_frequency_years", "latent",
    "periods": [{"start_month", "end_month" (None for the last, open-ended period), "growth_a",
    "growth_b", "growth_sigma"}, ...]}. A period's SCALED median/lo/hi (given a live initial_price
    and confidence level) is:
        median = initial_price * (growth_a + growth_b)
        lo/hi  = initial_price * (growth_a + growth_b * exp(z_lo/hi * growth_sigma))
    (growth_a is the window's known/recorded contribution, growth_b its projected contribution,
    growth_sigma the projected portion's own representative log-space uncertainty -- see this
    function's body for the derivation. growth_sigma is deliberately NOT resolved to a lo/hi at any
    particular confidence level here, so the SAME saved model answers any report_quantiles the deck
    is set to at use time, without rebuilding.)
    """
    deck = deck if deck is not None else load_deck()
    config = get_market_value_config(deck)
    if config["latent"] not in DERIVED_LATENT_COMPONENTS:
        raise NotImplementedError(
            f"market_value.latent={config['latent']!r} isn't a founding process -- the pre-closing "
            "averaging window needs REAL historical market data, which only exists for "
            "forecast_models' three founding processes (general/home_price/market), not a synthetic "
            "derived_latents entry."
        )
    latent_fit = get_growth_fit(config["latent"], deck)
    monthly_series = load_monthly_series(get_forecast_model_name(config["latent"], deck))

    closing = get_closing_date(deck)
    closing_calendar_date = pd.Timestamp(year=closing["closing_year"], month=closing["closing_month"], day=1)

    reassessment_frequency_months = config["reassessment_frequency_years"] * 12
    m1 = months_to_first_reassessment(closing["closing_month"], config["first_reassessment_year"])
    num_periods = max(1, -(-(horizon_months - m1) // reassessment_frequency_months) + 1)  # ceil division
    boundaries = reassessment_boundary_months(
        closing["closing_month"], config["first_reassessment_year"], config["reassessment_frequency_years"], num_periods
    )

    # Each reassessment k's window looks BACKWARD from its own start: [boundaries[k] - P, boundaries[k]
    # - P + REASSESSMENT_AVERAGING_WINDOW_MONTHS) -- see this module's docstring for the worked
    # example. Collect every window up front so the OU projection below is computed ONCE, far enough
    # to cover the furthest-out month any window actually needs.
    windows = [
        list(range(boundaries[k] - reassessment_frequency_months, boundaries[k] - reassessment_frequency_months + REASSESSMENT_AVERAGING_WINDOW_MONTHS))
        for k in range(1, len(boundaries))
    ]
    projected_months = sorted({m for window in windows for m in window if m > 0})
    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])
    z_hi = norm.ppf(ci[1])
    max_projected_h = max(projected_months) if projected_months else 1
    projection = project_term_structure(latent_fit, p0=1.0, horizon_months=max_projected_h, ci=ci)
    # Recover each h's own log-space sigma from the projection's own median/hi (CI-independent once
    # divided back out by z_hi) -- so it can be re-combined with any ci at USE time, not just the
    # one read here at build time.
    sigma_at_h = np.log(projection["hi"] / projection["median"]) / z_hi

    periods: list[dict[str, Any]] = [
        {"start_month": 0, "end_month": boundaries[1], "growth_a": 1.0, "growth_b": 0.0, "growth_sigma": 0.0}
    ]
    for k in range(1, len(boundaries)):
        window_months = windows[k - 1]
        known_months = [m for m in window_months if m <= 0]
        window_projected_months = [m for m in window_months if m > 0]
        known_sum = sum(historical_growth_factor(monthly_series, closing_calendar_date, m) for m in known_months)
        if window_projected_months:
            projected_median_sum = sum(projection.loc[m, "median"] for m in window_projected_months)
            representative_sigma = sum(sigma_at_h.loc[m] for m in window_projected_months) / len(window_projected_months)
        else:
            projected_median_sum = 0.0
            representative_sigma = 0.0
        n_total = len(window_months)
        period_end = boundaries[k + 1] if k + 1 < len(boundaries) else None
        periods.append(
            {
                "start_month": boundaries[k],
                "end_month": period_end,
                "growth_a": known_sum / n_total,
                "growth_b": projected_median_sum / n_total,
                "growth_sigma": representative_sigma,
            }
        )

    return {
        "closing_month": closing["closing_month"],
        "first_reassessment_year": config["first_reassessment_year"],
        "reassessment_frequency_years": config["reassessment_frequency_years"],
        "latent": config["latent"],
        "periods": periods,
    }


def _resolve_market_value_period(periods: list[dict[str, Any]], h_months: int) -> dict[str, Any]:
    """Finds the period (see build_market_value_change_magnitude) covering h_months."""
    for period in periods:
        if h_months >= period["start_month"] and (period["end_month"] is None or h_months < period["end_month"]):
            return period
    raise ValueError(
        f"h_months={h_months} isn't covered by any precomputed market_value period -- rebuild via "
        "manual_utils.build_market_value_prior_model with a larger horizon_months."
    )


def project_condo_value_schedule(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> pd.Series:
    """REAL: projects the condo's own value at the end of each year of scenario["horizon_years"],
    via forecast_models.home_price's fit applied to purchase_price as p0 (see
    ou_fitting.project_term_structure -- exactly the use this deck's own home_price comment
    anticipated: "Used directly (unweighted) to appreciate acquisition.purchase_price to its sale
    value at exit"). Returns a Series indexed by year (0..horizon_years; year 0 is purchase_price
    itself) of the projected MEDIAN value -- only the median path is propagated (this is the sale-
    price estimate; property tax uses project_market_value_schedule instead, which DOES carry a
    full band -- see that function's own docstring for why the two are different quantities)."""
    deck = deck if deck is not None else load_deck()
    purchase_price = scenario["acquisition"]["purchase_price"]
    horizon_years = scenario["horizon_years"]
    home_price_fit = get_forecast_model_home_price(deck)
    projection = project_term_structure(home_price_fit, p0=purchase_price, horizon_months=horizon_years * 12)
    year_end_months = [year * 12 for year in range(horizon_years + 1)]
    return projection.loc[year_end_months, "median"].set_axis(range(horizon_years + 1))


def project_market_value_schedule(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> pd.DataFrame:
    """Projects the COUNTY ASSESSOR'S OWN market_value determination for each year of
    scenario["horizon_years"] -- NOT the same thing as project_condo_value_schedule (this model's
    own continuous appreciation estimate). Anchored to scenario.closing_month/closing_year
    (TIMESTEP 0), not "today," and stair-stepped at REAL reassessment years (see
    get_market_value_config), each new step an AVERAGE rather than a single point:

    - From closing until January of (closing_year + first_reassessment_year): the value is exactly
      initial_price -- known, constant, zero variance (no reassessment has happened yet).
    - At that reassessment, and every reassessment_frequency_years after: the new value is the
      AVERAGE of the market's own growth over the first REASSESSMENT_AVERAGING_WINDOW_MONTHS (18,
      i.e. 1.5 years) of the reassessment_frequency_years-long window ENDING at this reassessment --
      e.g. reassessment_frequency_years=2, reassessing Jan 2025 uses the average of Jan 2023-Jun
      2024, then holds that value until the Jan 2027 reassessment. Whatever part of that window
      falls BEFORE closing uses REAL historical data (historical_growth_factor); whatever part falls
      at or after closing uses `latent`'s own projected distribution -- so a window that's e.g. 80%
      already-recorded history only gets ~20% of a fully-projected window's uncertainty (see
      build_market_value_change_magnitude's docstring for the exact math).

    All of the actual computation (real-data lookups, OU projections, the historical/projected
    blend) happens ONCE at BUILD time (manual_utils.build_market_value_prior_model ->
    build_market_value_change_magnitude) -- this function just loads that saved, UNSCALED
    (price=1.0 basis) result and does two cheap things per requested year: find which precomputed
    period it falls in, and scale by the deck's CURRENT initial_price and report_quantiles (both
    read live, so changing either doesn't require a rebuild).

    Returns a DataFrame indexed by year (1..horizon_years) with columns "median"/"lo"/"hi" -- the
    FULL band, not just the median (unlike project_condo_value_schedule): this is exactly what's
    meant to flow through into property tax's own assessed_value/tax, per the user's explicit
    request that these come out as a real distribution, not a point estimate.
    """
    deck = deck if deck is not None else load_deck()
    config = get_market_value_config(deck)
    change_magnitude = _load_carrying_cost_prior_fit("market_value")
    report_quantiles = get_report_quantiles(deck)
    z_lo, z_hi = norm.ppf((report_quantiles[0], report_quantiles[-1]))
    initial_price = config["initial_price"]

    horizon_years = scenario["horizon_years"]
    years = list(range(1, horizon_years + 1))
    medians, los, his = [], [], []
    for year in years:
        period = _resolve_market_value_period(change_magnitude["periods"], year * 12)
        a, b, sigma = period["growth_a"], period["growth_b"], period["growth_sigma"]
        medians.append(initial_price * (a + b))
        los.append(initial_price * (a + b * np.exp(z_lo * sigma)))
        his.append(initial_price * (a + b * np.exp(z_hi * sigma)))
    return pd.DataFrame({"median": medians, "lo": los, "hi": his}, index=pd.Index(years))


def compute_property_tax_schedule(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> pd.DataFrame:
    """REAL: assessed_value = LOCAL_ASSESSED_VALUE_FRACTION (90%) of market_value at LOCAL's
    assessment_rate, PLUS SCHOOL_ASSESSED_VALUE_FRACTION (100%) of market_value at SCHOOL's
    assessment_rate; tax = each component's assessed value x its own mill_levy, summed. Since both
    components are simple constant multiples of the SAME market_value draw (not independent random
    variables), assessed_value/tax end up as market_value's own median/lo/hi band, just linearly
    rescaled -- see get_property_tax_inputs for the local/school rates and mill levies, and
    project_market_value_schedule for market_value's own band.

    Returns a DataFrame indexed by year (1..horizon_years) with columns "assessed_value_median",
    "assessed_value_lo", "assessed_value_hi", "tax_median", "tax_lo", "tax_hi" -- ONLY the combined
    total (local + school collapsed into one) is ever exposed, per the user's explicit request; the
    local/school split itself is an internal computation detail.
    """
    deck = deck if deck is not None else load_deck()
    if PROPERTY_TAX_CLASSIFICATION != "residential":
        raise NotImplementedError(
            f"PROPERTY_TAX_CLASSIFICATION={PROPERTY_TAX_CLASSIFICATION!r} has no defined local/school "
            "assessment_rate/mill_levy split -- only 'residential' is implemented (see that "
            "constant's own comment for why 'lodging' isn't reachable today anyway)."
        )
    inputs = get_property_tax_inputs(deck)
    combined_assessed_rate = (
        LOCAL_ASSESSED_VALUE_FRACTION * inputs["assessment_rate_local"]
        + SCHOOL_ASSESSED_VALUE_FRACTION * inputs["assessment_rate_school"]
    )
    combined_tax_rate = (
        LOCAL_ASSESSED_VALUE_FRACTION * inputs["assessment_rate_local"] * inputs["mill_levy_local"]
        + SCHOOL_ASSESSED_VALUE_FRACTION * inputs["assessment_rate_school"] * inputs["mill_levy_school"]
    )

    market_value_schedule = project_market_value_schedule(scenario, deck)
    return pd.DataFrame(
        {
            "assessed_value_median": market_value_schedule["median"] * combined_assessed_rate,
            "assessed_value_lo": market_value_schedule["lo"] * combined_assessed_rate,
            "assessed_value_hi": market_value_schedule["hi"] * combined_assessed_rate,
            "tax_median": market_value_schedule["median"] * combined_tax_rate,
            "tax_lo": market_value_schedule["lo"] * combined_tax_rate,
            "tax_hi": market_value_schedule["hi"] * combined_tax_rate,
        }
    )


def load_market_value_change_magnitude() -> dict[str, Any]:
    """Public wrapper around _load_carrying_cost_prior_fit("market_value") -- for a caller (e.g.
    results.py's own per-scenario plot) that needs the saved periods directly, not just
    project_market_value_schedule's per-year median/lo/hi."""
    return _load_carrying_cost_prior_fit("market_value")


def _load_carrying_cost_prior_fit(name: str) -> dict[str, Any]:
    """Loads models/carrying_cost_priors/<name>/fit.json (see
    manual_utils.build_carrying_cost_prior_model/build_special_assessment_prior_model/
    build_utilities_prior_model) -- a plain resolved-config dict, not an OU fit itself (no new
    fitting happens when these are built -- see those functions' own docstrings), so this doesn't
    reuse input_deck._load_checked_fit's OU-key validation."""
    path = model_fit_path(f"carrying_cost_priors/{name}")
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"carrying_cost_priors/{name}/fit.json doesn't exist or is empty -- run "
            "manual_utils.build_all_carrying_cost_prior_models (called automatically by "
            "run/main.py) first."
        )
    return load_fit(path)


def _project_carrying_cost_prior(name: str, horizon_years: int, deck: dict[str, Any] | None = None) -> pd.Series:
    """Projects one of the simple LEVEL+GROWTH priors (hoa_dues_annual/insurance_annual/
    maintenance_annual) to a per-year Series (1..horizon_years), using its saved config + the named
    latent's own fit."""
    deck = deck if deck is not None else load_deck()
    config = _load_carrying_cost_prior_fit(name)
    latent_fit = get_growth_fit(config["latent"], deck)
    extra_log_variance = np.log(1 + config["level_cv"] ** 2)
    projection = project_term_structure(
        latent_fit, p0=config["level_median"], horizon_months=horizon_years * 12, extra_log_variance=extra_log_variance
    )
    year_end_months = [year * 12 for year in range(1, horizon_years + 1)]
    return projection.loc[year_end_months, "median"].set_axis(range(1, horizon_years + 1)).rename(name)


def _project_special_assessment_expected(horizon_years: int, deck: dict[str, Any] | None = None) -> pd.Series:
    """Expected annual special-assessment cost = annual_probability x projected severity median
    (severity grows via its own latent; annual_probability doesn't -- see that field's own deck
    comment)."""
    deck = deck if deck is not None else load_deck()
    config = _load_carrying_cost_prior_fit("special_assessment")
    latent_fit = get_growth_fit(config["latent"], deck)
    extra_log_variance = np.log(1 + config["severity_cv"] ** 2)
    projection = project_term_structure(
        latent_fit,
        p0=config["severity_median"],
        horizon_months=horizon_years * 12,
        extra_log_variance=extra_log_variance,
    )
    year_end_months = [year * 12 for year in range(1, horizon_years + 1)]
    severity = projection.loc[year_end_months, "median"].set_axis(range(1, horizon_years + 1))
    return (severity * config["annual_probability"]).rename("special_assessment_expected")


def _project_utilities_annual(horizon_years: int, deck: dict[str, Any] | None = None) -> pd.Series:
    """Annual utilities cost = sum of all 12 months' own projected median, each month projected
    INDEPENDENTLY from its own level/cv via the utilities latent (h_months=0 is "now" for every
    month's own projection, so h_months=12*year correctly lands on that same calendar month year
    years out -- see manual_utils.build_utilities_prior_model's own docstring)."""
    deck = deck if deck is not None else load_deck()
    config = _load_carrying_cost_prior_fit("utilities")
    latent_fit = get_growth_fit(config["latent"], deck)
    medians_by_month = config["level_medians_by_month"]
    cvs_by_month = config["level_cvs_by_month"]
    year_end_months = [year * 12 for year in range(1, horizon_years + 1)]

    monthly_projections = [
        project_term_structure(
            latent_fit,
            p0=medians_by_month[month],
            horizon_months=horizon_years * 12,
            extra_log_variance=np.log(1 + cvs_by_month[month] ** 2),
        ).loc[year_end_months, "median"]
        for month in medians_by_month
    ]
    return sum(monthly_projections).set_axis(range(1, horizon_years + 1)).rename("utilities")


def compute_carrying_costs_schedule(
    scenario: dict[str, Any], deck: dict[str, Any] | None = None, property_tax_schedule: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Computes one scenario's full per-year carrying-cost schedule. property_tax_schedule (see
    compute_property_tax_schedule) is computed fresh if not passed in -- accept it as a parameter so
    compute_carrying_costs can reuse the one call it already made instead of computing it twice.

    Returns a DataFrame indexed by year (1..horizon_years) with columns: property_tax (the MEDIAN
    tax figure only -- see compute_carrying_costs' "property_tax_schedule" for the full band),
    hoa_dues, insurance, maintenance, special_assessment_expected, utilities, total (the sum of the
    other six, using property_tax's median)."""
    deck = deck if deck is not None else load_deck()
    horizon_years = scenario["horizon_years"]
    property_tax_schedule = (
        property_tax_schedule if property_tax_schedule is not None else compute_property_tax_schedule(scenario, deck)
    )

    schedule = pd.DataFrame(
        {
            "property_tax": property_tax_schedule["tax_median"],
            "hoa_dues": _project_carrying_cost_prior("hoa_dues_annual", horizon_years, deck),
            "insurance": _project_carrying_cost_prior("insurance_annual", horizon_years, deck),
            "maintenance": _project_carrying_cost_prior("maintenance_annual", horizon_years, deck),
            "special_assessment_expected": _project_special_assessment_expected(horizon_years, deck),
            "utilities": _project_utilities_annual(horizon_years, deck),
        }
    )
    schedule["total"] = schedule.sum(axis=1)
    return schedule


def compute_carrying_costs(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Top-level entry point for a buy scenario. Returns {"annual_schedule" (DataFrame, see
    compute_carrying_costs_schedule), "property_tax_schedule" (DataFrame, see
    compute_property_tax_schedule -- the FULL assessed_value/tax median/lo/hi band, not just the
    median folded into annual_schedule's own "property_tax" column), "total_carrying_costs"
    (annual_schedule's "total" column summed over the horizon)}."""
    deck = deck if deck is not None else load_deck()
    property_tax_schedule = compute_property_tax_schedule(scenario, deck)
    annual_schedule = compute_carrying_costs_schedule(scenario, deck, property_tax_schedule)
    return {
        "annual_schedule": annual_schedule,
        "property_tax_schedule": property_tax_schedule,
        "total_carrying_costs": annual_schedule["total"].sum(),
    }
