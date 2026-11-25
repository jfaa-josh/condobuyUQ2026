"""Computes one scenario's per-year carrying costs: property tax (REAL, using the property state's own
mechanic from the selected tax jurisdiction -- for Colorado the two-step assessment-rate x mill-levy
system, local AND school components separately; see jurisdictions/property_tax.py and
data/tax_jurisdictions/CO/state.yaml -- with the parcel's own mill levies from the deck), plus HOA dues,
insurance, maintenance, and utilities -- each projected from its own pre-built carrying_cost_prior
model (see utils.prior_utils.build_all_carrying_cost_prior_models, wired into run/main.py's build
stage, the same way derived_latents are). Special assessments get their own separate frequency x
size model (compute_special_assessment) that is NEVER part of any carrying-cost total.

Also owns project_sale_price -- the condo's sale price at the horizon as a lognormal (purchase_price
grown via forecast_models.home_price) -- used by taxes.py (sale tax at the EXPECTED price) and exit.py
(net equity at the sale). NOT the same thing as project_market_value_schedule below (the ASSESSOR's own,
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

import math
from itertools import combinations_with_replacement
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from condobuyuq2026.input_deck import (
    get_closing_date,
    get_forecast_model_home_price,
    get_property_tax_inputs,
    get_report_quantiles,
    get_tax_jurisdiction_selection,
    load_deck,
)
from condobuyuq2026.jurisdictions import load_tax_jurisdiction
from condobuyuq2026.paths import model_fit_path
from condobuyuq2026.utils.ou_fitting import load_fit, project_term_structure

# Heating systems for carrying_costs.utilities.heating_type (2026-11-22). Hardcoded reference data, not
# deck fields: these are equipment physics, not property-specific inputs. `carrier` is the utility
# component whose bill holds the heating cost; `cop` (coefficient of performance: heat delivered per unit
# of electricity) scales an electric-resistance-equivalent heating_share down for heat pumps.
#   1: electric resistance converts electricity to heat 1:1 by definition (COP 1.0).
#   2: cold-climate heat pumps: NEEP's ccASHP specification requires COP >= 1.75 at 5°F; 2.0 is a
#      conservative SEASONAL average for a high-altitude Colorado winter (much of it far colder than the
#      mild-climate ratings, ~3+).
#   3: a standard heat pump leaning on resistance backup in deep cold averages well below its rating:
#      1.5 sits between pure resistance (1.0) and the cold-climate unit above.
#   4: gas furnace/boiler: the gas bill carries the heat (heating_share applies to it directly), plus
#      the blower fan's electricity -- NYSERDA measured ~5-9 kWh of blower electricity per MMBtu of
#      furnace heat; modeled simply as GAS_FURNACE_BLOWER_ELECTRIC_SHARE of the ELECTRIC bill (user
#      decision 2026-11-22: avoids needing gas/electric prices to convert).
#   0: heat supplied by the HOA/common system -- nothing on this unit's own bill.
HEATING_TYPES: dict[int, dict[str, Any]] = {
    0: {"name": "HOA/common heat", "carrier": None, "cop": None},
    1: {"name": "electric resistance (baseboard/wall/radiant)", "carrier": "electric", "cop": 1.0},
    2: {"name": "cold-climate heat pump", "carrier": "electric", "cop": 2.0},
    3: {"name": "standard heat pump with resistance backup", "carrier": "electric", "cop": 1.5},
    4: {"name": "gas furnace/boiler", "carrier": "gas", "cop": None},
}
GAS_FURNACE_BLOWER_ELECTRIC_SHARE = 0.05


def resolve_heating_type(heating_type: int, heating_share: float, components: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Resolves carrying_costs.utilities' heating_type code + heating_share into how much of each
    component's bill is space heating. Raises ValueError for an unknown code, or a heating carrier
    whose weight is 0 (e.g. code 4 with gas weight 0 -- the heat would have nowhere to come from).

    For electric codes (1-3), heating_share S is stated as if heated by electric RESISTANCE, so a heat
    pump with coefficient of performance COP uses S/COP for the same heat while the non-heating
    1 - S is unchanged. Re-expressed as a share of the (smaller) actual electric bill:
        effective share s = (S/COP) / (S/COP + 1 - S)
    For gas (code 4): s = S of the gas bill, plus GAS_FURNACE_BLOWER_ELECTRIC_SHARE of the electric bill.

    Returns {"heating_type", "name", "carrier", "cop", "heating_share" (as entered),
    "heating_fraction_by_component" ({component: share of THAT component's bill that is heating})}."""
    if heating_type not in HEATING_TYPES:
        raise ValueError(
            f"carrying_costs.utilities.heating_type must be one of {sorted(HEATING_TYPES)}, got {heating_type!r}"
        )
    spec = HEATING_TYPES[heating_type]
    carrier, cop = spec["carrier"], spec["cop"]
    fractions = {name: 0.0 for name in components}
    if carrier is not None:
        if components[carrier]["weight"] == 0:
            raise ValueError(
                f"carrying_costs.utilities.heating_type {heating_type} ({spec['name']}) heats with {carrier}, "
                f"but components.{carrier}.weight is 0 -- give {carrier} a weight or pick another heating_type"
            )
        if cop is None:  # gas furnace/boiler
            fractions[carrier] = heating_share
            fractions["electric"] = GAS_FURNACE_BLOWER_ELECTRIC_SHARE
        else:
            per_cop = heating_share / cop
            fractions[carrier] = per_cop / (per_cop + 1 - heating_share)
    return {
        "heating_type": heating_type,
        "name": spec["name"],
        "carrier": carrier,
        "cop": cop,
        "heating_share": heating_share,
        "heating_fraction_by_component": fractions,
    }


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


def project_sale_price(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, float]:
    """The condo's sale price at the horizon (month 12 x horizon_years) as a LOGNORMAL: purchase_price grown
    by forecast_models.home_price's fit (ou_fitting.project_term_structure with p0 = purchase_price),
    proportionally -- the user chose this (2026-11-25) over reusing the property-tax market_value line
    calibration (fitted from only 2 points). Returns {"mu", "sigma"} (log scale), "median", "mean", "sd",
    and the report bands "lo"/"hi" (report_quantiles) and "lo_outer"/"hi_outer" (1%/99%). Not the same
    quantity as project_market_value_schedule (the assessor's periodic determination, for property tax)."""
    deck = deck if deck is not None else load_deck()
    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])
    horizon_months = int(scenario["horizon_years"]) * 12
    projection = project_term_structure(
        get_forecast_model_home_price(deck), p0=scenario["acquisition"]["purchase_price"], horizon_months=horizon_months,
        ci=ci,
    )
    median = float(projection.loc[horizon_months, "median"])
    sigma = float(math.log(projection.loc[horizon_months, "hi"] / median) / norm.ppf(ci[1]))
    mu = math.log(median)
    mean = math.exp(mu + sigma**2 / 2)
    bands = {name: math.exp(mu + norm.ppf(p) * sigma)
             for name, p in (("lo", ci[0]), ("hi", ci[1]), ("lo_outer", 0.01), ("hi_outer", 0.99))}
    return {"mu": mu, "sigma": sigma, "median": median, "mean": mean, "sd": mean * math.sqrt(math.expm1(sigma**2)),
            **bands}


def project_market_value_schedule(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> pd.DataFrame:
    """Projects the COUNTY ASSESSOR'S OWN market_value determination for each year of
    scenario["horizon_years"] -- NOT the same thing as project_sale_price (this model's
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
    FULL band per year (project_sale_price only covers the sale): this is exactly what's
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


def property_tax_calendar_year(h_months: int, deck: dict[str, Any] | None = None) -> int:
    """The calendar (tax) year that scenario month h_months (0 = closing) falls in."""
    closing = get_closing_date(deck)
    return closing["closing_year"] + (closing["closing_month"] - 1 + h_months) // 12


def property_tax_for_value(
    market_value: Any, tax_year: int, deck: dict[str, Any] | None = None
) -> dict[str, np.ndarray]:
    """Assessed value and tax for an assessor market value (scalar or array) in a tax year: the selected
    jurisdiction's property-tax model (see jurisdictions.TaxJurisdiction.property_tax_model) applied with
    the deck's mill levies (input_deck.get_property_tax_inputs). Returns {"assessed_value", "tax"} --
    the combined local+school totals only (the component split is internal)."""
    deck = deck if deck is not None else load_deck()
    model = load_tax_jurisdiction(get_tax_jurisdiction_selection(deck)).property_tax_model()
    mill_levies = get_property_tax_inputs(deck)["mill_levies"]
    unknown = set(mill_levies) - set(model.components)
    if unknown:
        raise ValueError(
            f"carrying_costs.property_tax.mill_levy has component(s) {sorted(unknown)} the property state's model "
            f"doesn't use (it expects {list(model.components)})"
        )
    return {
        "assessed_value": model.assessed_value(market_value, tax_year),
        "tax": model.tax(market_value, mill_levies, tax_year),
    }


def market_value_periods_to_tax_periods(
    periods: list[dict[str, Any]], deck: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Rescales market_value's own saved periods (see utils.prior_utils.
    build_market_value_change_magnitude) into property_tax dollar terms. Each period's known_value/
    projected_value is scaled by its EFFECTIVE tax rate (tax / value at the period's median value, in
    the calendar year the period starts); growth_sigma is a unitless log-space ratio, untouched. Exact
    wherever the model is linear in value (CO: below the $700k point where the local reduction caps),
    a close linearization above it. Exposes the PERIOD STRUCTURE itself, not just
    compute_property_tax_schedule's per-year band, for a caller that groups by underlying period the way
    results.py's own distribution plot does."""
    deck = deck if deck is not None else load_deck()
    rescaled = []
    for period in periods:
        median = period["known_value"] + period["projected_value"]
        tax_year = property_tax_calendar_year(period["start_month"], deck)
        rate = float(property_tax_for_value(median, tax_year, deck)["tax"]) / median if median > 0 else 0.0
        rescaled.append(
            {**period, "known_value": period["known_value"] * rate, "projected_value": period["projected_value"] * rate}
        )
    return rescaled


def compute_property_tax_schedule(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> pd.DataFrame:
    """REAL: the selected jurisdiction's property-tax model (see property_tax_for_value) applied to
    market_value's own median/lo/hi band each year, in the calendar year that scenario year starts. Every
    model is monotone in value, so the band maps straight through -- see project_market_value_schedule
    for market_value's own band.

    Returns a DataFrame indexed by year (1..horizon_years) with columns "assessed_value_median",
    "assessed_value_lo", "assessed_value_hi", "tax_median", "tax_lo", "tax_hi" -- ONLY the combined
    total (local + school collapsed into one) is ever exposed, per the user's explicit request; the
    component split itself is an internal computation detail.
    """
    deck = deck if deck is not None else load_deck()
    market_value_schedule = project_market_value_schedule(scenario, deck)
    columns: dict[str, list[float]] = {
        f"{what}_{band}": [] for what in ("assessed_value", "tax") for band in ("median", "lo", "hi")
    }
    for year in market_value_schedule.index:
        tax_year = property_tax_calendar_year((year - 1) * 12, deck)
        for band in ("median", "lo", "hi"):
            result = property_tax_for_value(market_value_schedule.loc[year, band], tax_year, deck)
            columns[f"assessed_value_{band}"].append(float(result["assessed_value"]))
            columns[f"tax_{band}"].append(float(result["tax"]))
    return pd.DataFrame(columns, index=market_value_schedule.index)


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


def special_assessment_count_probabilities(annual_rate: float, horizon_years: int, k_max: int) -> pd.Series:
    """P(N = k) for k = 0..k_max, where N ~ Poisson(annual_rate x horizon_years) is the number of
    special assessments over the horizon.

    THE CUTOFF, IN PLAIN ENGLISH: only the outcomes 0, 1, ..., k_max are listed. The small chance of
    MORE than k_max assessments is not dropped -- it's FOLDED INTO the k_max entry, so the last entry
    means "k_max OR MORE" (P(N >= k_max)). That keeps the probabilities summing to exactly 1, at the
    cost of slightly understating the cost in those rare cases (e.g. 4 assessments get costed as 3).
    See special_assessment_folded_tail_probability for how big that folded-in chance is, which
    reporting.print_special_assessment_report warns about above 1%.

    Shared by the per-scenario model (compute_special_assessment) and the build-time model.png
    (utils.prior_utils.build_special_assessment_prior_model), so both always agree."""
    expected_count = annual_rate * horizon_years
    k = np.arange(k_max + 1)
    if expected_count == 0:
        probabilities = np.zeros(k_max + 1)
        probabilities[0] = 1.0
    else:
        probabilities = poisson.pmf(k, expected_count)
        probabilities[-1] = poisson.sf(k_max - 1, expected_count)  # P(N >= k_max): the fold-in
    return pd.Series(probabilities, index=pd.Index(k, name="k"), name="probability")


def special_assessment_folded_tail_probability(annual_rate: float, horizon_years: int, k_max: int) -> float:
    """P(N > k_max): the chance of MORE assessments than the model lists separately -- the amount
    folded into the k_max outcome (see special_assessment_count_probabilities)."""
    expected_count = annual_rate * horizon_years
    return 0.0 if expected_count == 0 else float(poisson.sf(k_max, expected_count))


def _special_assessment_size_multiplier(size_growth_multiplier: list[float], timing: str, horizon_months: int) -> float:
    """How much ONE assessment's today's-dollar size grows by the time it's paid, given `timing`:
    - "worst_case_year1": paid right at closing (month 0), so no growth: 1.0.
    - "uniform": any month 0..horizon_months-1 equally likely, so the AVERAGE of the saved monthly
      growth multiplier over those months. APPROXIMATION (flagged here so it doesn't read as exact):
      this replaces each assessment's random payment month with the average multiplier, so the cost
      distribution stays a short list of exact points instead of becoming a continuous spread. The
      growth over a 1-10 year horizon is small next to the size_scenarios' own spread, so the effect
      on the distribution's shape is minor.
    `size_growth_multiplier` is the latent's median growth path from today, one value per month
    (see utils.prior_utils.build_special_assessment_prior_model)."""
    if timing == "worst_case_year1":
        return 1.0
    if horizon_months > len(size_growth_multiplier):
        raise ValueError(
            f"horizon_months={horizon_months} exceeds the saved special_assessment growth path -- rebuild via "
            "utils.prior_utils.build_all_carrying_cost_prior_models with a larger horizon_months."
        )
    return float(np.mean(size_growth_multiplier[:horizon_months]))


def compute_special_assessment(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """One scenario's special-assessment model: how many assessments (N ~ Poisson(annual_rate x
    horizon_years), see special_assessment_count_probabilities) and the resulting EXACT discrete
    distribution of their total cost, in nominal dollars at payment time (sizes grown per `timing`,
    see _special_assessment_size_multiplier). Reads the saved build (fit.json), not the deck, same as
    every other carrying cost here.

    DELIBERATELY KEPT OUT OF EVERY CARRYING-COST TOTAL: compute_carrying_costs stores this under its
    own "special_assessment" key, never in annual_schedule, so total_carrying_costs,
    compute_total_carrying_costs_schedule and taxes.py's expenses never see it. A rare, large bill
    averaged into a smooth yearly cost would hide exactly the risk it represents. It's applied later
    as a tail-risk overlay on ΔW instead (see .claude/implementation-plan.md's "Final reporting:
    special-assessment overlay").

    NOT compounded to the horizon here -- the overlay does that (at expected market growth, for the
    opportunity cost of the cash), since that's a ΔW-level question, not a carrying-cost one.

    EXACT, no Monte Carlo: given k assessments, each independently one of the size_scenarios, every
    possible combination is listed (itertools.combinations_with_replacement -- order doesn't change
    the total) with its multinomial probability, times P(N = k). With k_max ~3 and a few sizes that's
    a few dozen rows at most.

    Returns {"enabled" (False when annual_rate == 0 -- the deck's off switch), "annual_rate", "k_max",
    "timing", "latent", "expected_count" (annual_rate x horizon_years), "count_probabilities"
    (Series, k -> P), "folded_tail_probability" (P(N > k_max), see the fold-in note above),
    "size_multiplier", "sizes_nominal_usd" ({label: grown size}), "outcomes" (DataFrame: k,
    combination, total_cost_usd, probability -- one row per distinct combination),
    "cost_distribution" (DataFrame: total_cost_usd, probability -- outcomes merged by total),
    "expected_total_cost_usd"}."""
    fit = _load_carrying_cost_prior_fit("special_assessment")
    horizon_years = scenario["horizon_years"]
    annual_rate, k_max, scenarios = fit["annual_rate"], fit["k_max"], fit["size_scenarios"]

    count_probabilities = special_assessment_count_probabilities(annual_rate, horizon_years, k_max)
    size_multiplier = _special_assessment_size_multiplier(fit["size_growth_multiplier"], fit["timing"], horizon_years * 12)
    sizes = [s["amount_today_usd"] * size_multiplier for s in scenarios]

    rows = []
    # The index is exactly 0..k_max (special_assessment_count_probabilities builds it that way), so the
    # position IS k -- enumerate avoids Series.items()' untyped Hashable labels.
    for k, p_k in enumerate(count_probabilities.to_numpy()):
        for combo in combinations_with_replacement(range(len(scenarios)), k):
            counts = [combo.count(i) for i in range(len(scenarios))]
            ways = math.factorial(k) // math.prod(math.factorial(c) for c in counts)
            p_combo = ways * math.prod(s["probability"] ** c for s, c in zip(scenarios, counts, strict=True))
            rows.append(
                {
                    "k": k,
                    "combination": "+".join(scenarios[i]["label"] for i in combo) or "none",
                    "total_cost_usd": sum(sizes[i] for i in combo),
                    "probability": float(p_k) * p_combo,
                }
            )
    outcomes = pd.DataFrame(rows)
    outcomes = outcomes[outcomes["probability"] > 0].reset_index(drop=True)
    cost_distribution = (
        outcomes.groupby(outcomes["total_cost_usd"].round(2))["probability"]
        .sum()
        .reset_index()
        .sort_values("total_cost_usd", ignore_index=True)
    )

    return {
        "enabled": annual_rate > 0,
        "annual_rate": annual_rate,
        "k_max": k_max,
        "timing": fit["timing"],
        "latent": fit["latent"],
        "expected_count": annual_rate * horizon_years,
        "count_probabilities": count_probabilities,
        "folded_tail_probability": special_assessment_folded_tail_probability(annual_rate, horizon_years, k_max),
        "size_multiplier": size_multiplier,
        "sizes_nominal_usd": {s["label"]: size for s, size in zip(scenarios, sizes, strict=True)},
        "outcomes": outcomes,
        "cost_distribution": cost_distribution,
        "expected_total_cost_usd": float((outcomes["total_cost_usd"] * outcomes["probability"]).sum()),
    }


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
    hoa_dues, insurance, maintenance, utilities, total (the sum of the other five, using
    property_tax's median). No special-assessment column: it's excluded from every total (see
    compute_special_assessment) -- taxes.py sums this frame's columns, so adding one here would
    silently pull it back in."""
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
    (annual_schedule's "total" column summed over the horizon), "special_assessment" (see
    compute_special_assessment -- kept as its own separate result, NOT part of annual_schedule or
    total_carrying_costs)}."""
    deck = deck if deck is not None else load_deck()
    property_tax_schedule = compute_property_tax_schedule(scenario, deck)
    annual_schedule = compute_carrying_costs_schedule(scenario, deck, property_tax_schedule)
    return {
        "annual_schedule": annual_schedule,
        "property_tax_schedule": property_tax_schedule,
        "total_carrying_costs": annual_schedule["total"].sum(),
        "special_assessment": compute_special_assessment(scenario, deck),
    }


def compute_total_carrying_costs_schedule(
    scenario: dict[str, Any], deck: dict[str, Any] | None = None, outer_ci: tuple[float, float] = (0.01, 0.99)
) -> pd.DataFrame:
    """Combines property_tax + hoa_dues + insurance + maintenance + utilities into ONE MONTHLY total
    cost distribution (1..horizon_months). DELIBERATELY EXCLUDES special_assessment -- its own rare,
    spiky risk is modeled separately (compute_special_assessment) and applied later as a tail-risk
    overlay on ΔW: folding a small-probability, large-size event into this sum would misleadingly
    smear a spiky tail risk into an
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

    Returns a DataFrame indexed by month (1..horizon_months) with columns "median", "lo", "hi" (the
    deck's report quantiles) and "lo_outer", "hi_outer" (outer_ci, default 1-99% -- the wide reference
    band the results plots show alongside the main one).
    """
    deck = deck if deck is not None else load_deck()
    horizon_months = scenario["horizon_years"] * 12
    report_quantiles = get_report_quantiles(deck)
    z_lo, z_hi = norm.ppf((report_quantiles[0], report_quantiles[-1]))
    z_lo_outer, z_hi_outer = norm.ppf(outer_ci)

    tax_periods = market_value_periods_to_tax_periods(load_market_value_change_magnitude()["periods"], deck)
    contributions = [
        (tax_periods, 1 / 12),
        (_load_carrying_cost_prior_fit("hoa_dues_annual")["periods"], 1.0),
        (_load_carrying_cost_prior_fit("insurance_annual")["periods"], 1 / 12),
        (_load_carrying_cost_prior_fit("maintenance_annual")["periods"], 1 / 12),
        (_load_carrying_cost_prior_fit("utilities")["periods"], 1.0),
    ]

    medians, los, his, los_outer, his_outer = [], [], [], [], []
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
            los_outer.append(combined_mean)
            his_outer.append(combined_mean)
            continue
        cv_sq = combined_var / combined_mean**2
        sigma_eff = float(np.sqrt(np.log(1 + cv_sq)))
        median_eff = combined_mean * np.exp(-(sigma_eff**2) / 2)
        medians.append(median_eff)
        los.append(median_eff * np.exp(z_lo * sigma_eff))
        his.append(median_eff * np.exp(z_hi * sigma_eff))
        los_outer.append(median_eff * np.exp(z_lo_outer * sigma_eff))
        his_outer.append(median_eff * np.exp(z_hi_outer * sigma_eff))

    return pd.DataFrame(
        {"median": medians, "lo": los, "hi": his, "lo_outer": los_outer, "hi_outer": his_outer},
        index=pd.Index(range(1, horizon_months + 1), name="month"),
    )
