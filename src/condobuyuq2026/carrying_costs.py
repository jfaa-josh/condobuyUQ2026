"""Computes one scenario's per-year carrying costs: property tax (REAL, using Colorado's actual
two-step assessment_rate x mill_levy mechanic, local AND school components separately -- see
input_deck.yaml's carrying_costs.property_tax comment for the research behind it), plus HOA dues,
insurance, maintenance, special assessment (expected value), and utilities -- each projected from
its own pre-built carrying_cost_prior model (see utils.prior_utils.build_all_carrying_cost_prior_models,
wired into run/main.py's build stage, the same way derived_latents are).

Also owns project_condo_value_schedule -- the condo's own CONTINUOUSLY appreciating value over the
horizon (via forecast_models.home_price) -- since it's fundamentally a "what does this property
carry/hold cost" question; taxes.py imports it from here for its own sale-price calculation instead
of duplicating it. NOT the same thing as project_market_value_schedule below (the ASSESSOR's own,
only-periodically-updated determination used for property tax specifically) -- see that function's
own docstring for why they're different.

RUNTIME ONLY: this module is per-scenario load-and-scale logic, never fitting/building. Every
carrying_cost_prior (including market_value) is built and saved UNSCALED by utils.prior_utils --
see that module's own docstring for why prior-BUILDING lives there, not here, and for
INFORMED_CARRYING_COST_PRIORS/carrying_cost_prior_path below, which market_value's own build step
also uses to know where to save (a single source of truth for the read/write path scheme, shared
between the two modules without a circular import -- prior_utils.py imports FROM here, not the
other way around).

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
    get_forecast_model_home_price,
    get_property_tax_inputs,
    get_report_quantiles,
    load_deck,
)
from condobuyuq2026.paths import model_fit_path
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


# Which carrying_cost_prior entries are "informed" (checked against REAL historical data for that
# specific variable, not just a pure growth/derived-latent extrapolation -- see
# utils.prior_utils.build_market_value_informed_model) vs "uninformed" (everything else: a
# hand-elicited level scaled by a growth latent, no variable-specific real data behind it).
INFORMED_CARRYING_COST_PRIORS = frozenset({"market_value"})


def carrying_cost_prior_path(name: str) -> str:
    """models/carrying_cost_priors/<category>/<name> -- category is "informed" or "uninformed" (see
    INFORMED_CARRYING_COST_PRIORS). Single source of truth for this path scheme, shared between the
    read side here (_load_carrying_cost_prior_fit) and the write side (utils.prior_utils's own
    build_* functions) so they can never drift apart."""
    category = "informed" if name in INFORMED_CARRYING_COST_PRIORS else "uninformed"
    return f"carrying_cost_priors/{category}/{name}"


def resolve_period(periods: list[dict[str, Any]], h_months: int) -> dict[str, Any]:
    """Finds the period ({"start_month", "end_month", "known_value", "projected_value",
    "growth_sigma"} -- see utils.prior_utils.build_market_value_change_magnitude for the shape every
    carrying_cost_prior's own saved `periods` list now uses) covering h_months. Public (not
    `_`-prefixed), and generic over ANY prior's periods (not just market_value's, despite the name of
    the concept it originated from -- renamed from resolve_market_value_period 2026-09-29 once every
    carrying_cost_prior started saving this same shape): every `_project_*` read-side function below,
    plus results.py's own per-year distribution plot, resolves periods this exact same way."""
    for period in periods:
        if h_months >= period["start_month"] and (period["end_month"] is None or h_months < period["end_month"]):
            return period
    raise ValueError(
        f"h_months={h_months} isn't covered by any precomputed period -- rebuild via "
        "utils.prior_utils.build_all_carrying_cost_prior_models with a larger horizon_months."
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
    input_deck.get_market_value_config), each new step an AVERAGE rather than a single point:

    - From closing until January of (closing_year + first_reassessment_year): the value is exactly
      initial_price -- known, constant, zero variance (no reassessment has happened yet).
    - At that reassessment, and every reassessment_frequency_years after: the new value is the
      AVERAGE of the market's own growth over the first 18 months (1.5 years) of the
      reassessment_frequency_years-long window ENDING at this reassessment -- e.g.
      reassessment_frequency_years=2, reassessing Jan 2025 uses the average of Jan 2023-Jun 2024,
      then holds that value until the Jan 2027 reassessment. Whatever part of that window falls
      BEFORE closing uses REAL historical data; whatever part falls at or after closing uses
      `latent`'s own projected distribution -- so a window that's e.g. 80% already-recorded history
      only gets ~20% of a fully-projected window's uncertainty.

    All of the actual computation (real-data lookups, OU projections, the historical/projected
    blend, and the Denver-to-Keystone dollar calibration -- see utils.prior_utils.
    fit_market_value_calibration) happens ONCE at BUILD time (utils.prior_utils.
    build_market_value_informed_model -> build_market_value_change_magnitude) -- this function just
    loads that saved result, ALREADY IN USD, and does two cheap things per requested year: find
    which precomputed period it falls in, and apply the deck's CURRENT report_quantiles (read live,
    so changing quantiles doesn't require a rebuild). No further scaling happens here -- the saved
    periods need no additional conversion.

    Returns a DataFrame indexed by year (1..horizon_years) with columns "median"/"lo"/"hi" -- the
    FULL band, not just the median (unlike project_condo_value_schedule): this is exactly what's
    meant to flow through into property tax's own assessed_value/tax, per the user's explicit
    request that these come out as a real distribution, not a point estimate.
    """
    deck = deck if deck is not None else load_deck()
    change_magnitude = _load_carrying_cost_prior_fit("market_value")
    report_quantiles = get_report_quantiles(deck)
    z_lo, z_hi = norm.ppf((report_quantiles[0], report_quantiles[-1]))

    horizon_years = scenario["horizon_years"]
    years = list(range(1, horizon_years + 1))
    medians, los, his = [], [], []
    for year in years:
        period = resolve_period(change_magnitude["periods"], year * 12)
        known, projected, sigma = period["known_value"], period["projected_value"], period["growth_sigma"]
        medians.append(known + projected)
        los.append(known + projected * np.exp(z_lo * sigma))
        his.append(known + projected * np.exp(z_hi * sigma))
    return pd.DataFrame({"median": medians, "lo": los, "hi": his}, index=pd.Index(years))


def get_combined_property_tax_rate(deck: dict[str, Any] | None = None) -> float:
    """The single combined LOCAL+SCHOOL rate that converts a market_value dollar into a property_tax
    dollar (assessment_rate x mill_levy for each of LOCAL/SCHOOL, weighted by their own
    ASSESSED_VALUE_FRACTION and summed -- see compute_property_tax_schedule for the full derivation).
    Exposed separately (not just inlined there) so a caller needing market_value's own saved PERIOD
    STRUCTURE rescaled into tax dollars (see market_value_periods_to_tax_periods) can reuse this exact
    rate, rather than recomputing the LOCAL/SCHOOL combination a second time."""
    deck = deck if deck is not None else load_deck()
    if PROPERTY_TAX_CLASSIFICATION != "residential":
        raise NotImplementedError(
            f"PROPERTY_TAX_CLASSIFICATION={PROPERTY_TAX_CLASSIFICATION!r} has no defined local/school "
            "assessment_rate/mill_levy split -- only 'residential' is implemented (see that "
            "constant's own comment for why 'lodging' isn't reachable today anyway)."
        )
    inputs = get_property_tax_inputs(deck)
    return (
        LOCAL_ASSESSED_VALUE_FRACTION * inputs["assessment_rate_local"] * inputs["mill_levy_local"]
        + SCHOOL_ASSESSED_VALUE_FRACTION * inputs["assessment_rate_school"] * inputs["mill_levy_school"]
    )


def market_value_periods_to_tax_periods(
    periods: list[dict[str, Any]], deck: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Rescales market_value's own saved periods (see utils.prior_utils.
    build_market_value_change_magnitude) into property_tax dollar terms: known_value/projected_value
    scaled by get_combined_property_tax_rate (growth_sigma is a unitless log-space ratio, untouched by
    the rescale -- same reasoning project_market_value_schedule itself relies on). Exposes the PERIOD
    STRUCTURE itself, not just compute_property_tax_schedule's own per-year band, for a caller that
    needs to group by underlying period the way results.py's own distribution plot does."""
    rate = get_combined_property_tax_rate(deck)
    return [
        {**period, "known_value": period["known_value"] * rate, "projected_value": period["projected_value"] * rate}
        for period in periods
    ]


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
    inputs = get_property_tax_inputs(deck)
    combined_assessed_rate = (
        LOCAL_ASSESSED_VALUE_FRACTION * inputs["assessment_rate_local"]
        + SCHOOL_ASSESSED_VALUE_FRACTION * inputs["assessment_rate_school"]
    )
    combined_tax_rate = get_combined_property_tax_rate(deck)

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


def load_carrying_cost_prior_fit(name: str) -> dict[str, Any]:
    """Public wrapper around _load_carrying_cost_prior_fit -- for a caller (e.g. results.py's own
    per-scenario plots) that needs one of the six carrying_cost_priors' saved `periods`/config
    directly, not just this module's own per-scenario Series/DataFrame projection of them."""
    return _load_carrying_cost_prior_fit(name)


def load_market_value_change_magnitude() -> dict[str, Any]:
    """Public wrapper around load_carrying_cost_prior_fit("market_value") -- kept as its own named
    function since market_value's own callers (project_market_value_schedule's own docstring,
    results.py) read more naturally this way than the generic name."""
    return load_carrying_cost_prior_fit("market_value")


def _load_carrying_cost_prior_fit(name: str) -> dict[str, Any]:
    """Loads models/carrying_cost_priors/<category>/<name>/fit.json (see carrying_cost_prior_path
    for the category, utils.prior_utils's own build_* functions for what writes it) -- a plain
    resolved-config dict, not an OU fit itself (no new fitting happens when these are built -- see
    those functions' own docstrings), so this doesn't reuse input_deck._load_checked_fit's OU-key
    validation."""
    path = model_fit_path(carrying_cost_prior_path(name))
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"{carrying_cost_prior_path(name)}/fit.json doesn't exist or is empty -- run "
            "utils.prior_utils.build_all_carrying_cost_prior_models (called automatically by "
            "run/main.py) first."
        )
    return load_fit(path)


def _project_carrying_cost_prior(name: str, horizon_years: int, deck: dict[str, Any] | None = None) -> pd.Series:
    """Projects insurance_annual/maintenance_annual to a per-year Series (1..horizon_years) -- a pure
    LOOKUP against its saved FULL-MONTHLY-RESOLUTION `periods` (see
    utils.prior_utils.bake_monthly_periods), no re-projection at scenario-run time (standardized
    2026-09-29, matching project_market_value_schedule's own "build once, read many times"
    convention). Each of these represents a CONTINUOUSLY-evolving annual-cost estimate (not a real
    monthly bill), so a single lookup at the year's own end is the right annual figure -- unlike
    hoa_dues_annual/utilities, which are genuine monthly bills and need _project_hoa_dues_annual/
    _project_utilities_annual's own SUM instead. Queries year*12 - 1 (the period covering that month
    holds the value AT h=year*12, its own period's end -- see bake_monthly_periods's own docstring)."""
    config = _load_carrying_cost_prior_fit(name)
    periods_by_year = [resolve_period(config["periods"], year * 12 - 1) for year in range(1, horizon_years + 1)]
    values = [period["known_value"] + period["projected_value"] for period in periods_by_year]
    return pd.Series(values, index=pd.Index(range(1, horizon_years + 1)), name=name)


def _project_hoa_dues_annual(horizon_years: int, deck: dict[str, Any] | None = None) -> pd.Series:
    """Annual HOA total = sum of the 12 MONTHLY periods falling in that year (see
    utils.prior_utils.bake_hoa_monthly_periods -- each month's own payment, including January's
    association-fee spike, already correctly split) -- hoa_dues_annual is a genuine monthly bill, so
    (like utilities) its annual total needs a SUM, not the single year-end lookup
    _project_carrying_cost_prior uses for insurance_annual/maintenance_annual's own continuously-
    evolving estimates."""
    config = _load_carrying_cost_prior_fit("hoa_dues_annual")
    periods = config["periods"]
    totals = [
        sum(
            period["known_value"] + period["projected_value"]
            for period in periods
            if (year - 1) * 12 <= period["start_month"] < year * 12
        )
        for year in range(1, horizon_years + 1)
    ]
    return pd.Series(totals, index=pd.Index(range(1, horizon_years + 1)), name="hoa_dues")


def _project_special_assessment_expected(horizon_years: int, deck: dict[str, Any] | None = None) -> pd.Series:
    """Expected annual special-assessment cost -- a pure LOOKUP against its saved FULL-MONTHLY-
    RESOLUTION `periods` (already annual_probability x projected severity median, see
    utils.prior_utils.build_special_assessment_prior_model), same convention as
    _project_carrying_cost_prior (a continuously-evolving estimate, not a real monthly bill)."""
    config = _load_carrying_cost_prior_fit("special_assessment")
    periods_by_year = [resolve_period(config["periods"], year * 12 - 1) for year in range(1, horizon_years + 1)]
    values = [period["known_value"] + period["projected_value"] for period in periods_by_year]
    return pd.Series(values, index=pd.Index(range(1, horizon_years + 1)), name="special_assessment_expected")


def _project_utilities_annual(horizon_years: int, deck: dict[str, Any] | None = None) -> pd.Series:
    """Annual utilities cost = sum of the 12 MONTHLY periods falling in that year (see
    utils.prior_utils.bake_calendar_month_periods) -- a pure LOOKUP, no re-projection at scenario-run
    time, same convention as _project_hoa_dues_annual."""
    config = _load_carrying_cost_prior_fit("utilities")
    periods = config["periods"]
    totals = [
        sum(
            period["known_value"] + period["projected_value"]
            for period in periods
            if (year - 1) * 12 <= period["start_month"] < year * 12
        )
        for year in range(1, horizon_years + 1)
    ]
    return pd.Series(totals, index=pd.Index(range(1, horizon_years + 1)), name="utilities")


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
            "hoa_dues": _project_hoa_dues_annual(horizon_years, deck),
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


def compute_total_carrying_costs_schedule(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> pd.DataFrame:
    """Combines property_tax + hoa_dues + insurance + maintenance + utilities into ONE MONTHLY total
    cost distribution (1..horizon_months). DELIBERATELY EXCLUDES special_assessment -- its own rare,
    spiky risk is a separate, deferred "what-if" concern (see results.py's own module docstring and
    the "carrying-costs-known-value-redesign" project notes): folding a small-probability,
    large-severity mixture into this sum would misleadingly smear a spiky tail risk into an
    ordinary-looking smooth confidence band, which is exactly what the user asked NOT to do. If this
    exclusion is ever revisited, it must be a deliberate decision, not something silently reintroduced
    here.

    property_tax/insurance/maintenance are saved as ANNUAL-cadence quantities (one saved period
    represents the FULL YEAR's cost, evaluated continuously) -- divided by 12 here to get each one's
    own MONTHLY contribution. hoa_dues/utilities are already genuine MONTHLY bills, used as-is (scale
    1.0). Getting this scale factor right per variable is exactly the "some are yearly, some are
    monthly" bookkeeping the user asked to take care with.

    Each contributing period is `known_value + projected_value*exp(Z*growth_sigma)` -- i.e. a
    deterministic `known_value` plus a lognormal-ish `projected_value` component. Summing several such
    (assumed INDEPENDENT -- each driven by a DIFFERENT fitted latent process: construction_cost,
    insurance's own, maintenance's own, utilities' own, home_price for property_tax) components has no
    closed form in general, so this uses the same MOMENT-MATCHING approximation this codebase already
    relies on elsewhere (e.g. combining derived_latents components): sum each component's own real
    MEAN and VARIANCE in closed form, then approximate the total as a single lognormal with that same
    mean/variance (solving for the equivalent median/sigma) -- closed-form throughout, no Monte Carlo.

    Returns a DataFrame indexed by month (1..horizon_months) with columns "median", "lo", "hi".
    """
    deck = deck if deck is not None else load_deck()
    horizon_months = scenario["horizon_years"] * 12
    report_quantiles = get_report_quantiles(deck)
    z_lo, z_hi = norm.ppf((report_quantiles[0], report_quantiles[-1]))

    tax_periods = market_value_periods_to_tax_periods(load_market_value_change_magnitude()["periods"], deck)
    contributions = [
        (tax_periods, 1 / 12),
        (_load_carrying_cost_prior_fit("hoa_dues_annual")["periods"], 1.0),
        (_load_carrying_cost_prior_fit("insurance_annual")["periods"], 1 / 12),
        (_load_carrying_cost_prior_fit("maintenance_annual")["periods"], 1 / 12),
        (_load_carrying_cost_prior_fit("utilities")["periods"], 1.0),
    ]

    medians, los, his = [], [], []
    for month in range(horizon_months):
        combined_mean = 0.0
        combined_var = 0.0
        for periods, scale in contributions:
            period = resolve_period(periods, month)
            known = period["known_value"] * scale
            projected = period["projected_value"] * scale
            sigma = period["growth_sigma"]
            combined_mean += known + projected * np.exp(sigma**2 / 2)
            combined_var += (projected**2) * np.exp(sigma**2) * (np.exp(sigma**2) - 1)

        if combined_var < 1e-12 or combined_mean <= 0:
            medians.append(combined_mean)
            los.append(combined_mean)
            his.append(combined_mean)
            continue
        cv_sq = combined_var / combined_mean**2
        sigma_eff = float(np.sqrt(np.log(1 + cv_sq)))
        median_eff = combined_mean * np.exp(-(sigma_eff**2) / 2)
        medians.append(median_eff)
        los.append(median_eff * np.exp(z_lo * sigma_eff))
        his.append(median_eff * np.exp(z_hi * sigma_eff))

    return pd.DataFrame({"median": medians, "lo": los, "hi": his}, index=pd.Index(range(1, horizon_months + 1), name="month"))
