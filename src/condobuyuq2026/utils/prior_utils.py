"""Builds every carrying_cost_prior model AUTOMATICALLY from the current deck + already-built
growth/derived-latent fits -- rebuilt on every run/main.py run (Stage 1.5), never a one-off manual
script. Split out from utils/manual_utils.py (2026-09-28) specifically because of this distinction:
manual_utils.py is for genuinely manual, one-off scripts that pull from EXTERNAL data sources (BLS,
FRED, Yahoo Finance) -- a human decision, run deliberately, not something a scenario run redoes.
Everything here, by contrast, depends ONLY on the deck (never on an external API) and is rebuilt
unconditionally every run, exactly like utils.manual_utils.build_all_derived_latent_models already
was -- so it belongs with the "automatic" build stage, not the "manual" one.

Imports FROM carrying_costs.py (INFORMED_CARRYING_COST_PRIORS/carrying_cost_prior_path, the shared
path scheme) and FROM manual_utils.py (model_fit_path/model_plots_dir, generic filesystem helpers)
-- never the other way around, so there's no cycle.

TWO CLASSES OF PRIOR, now distinguished explicitly (see carrying_costs.INFORMED_CARRYING_COST_PRIORS):
- UNINFORMED (hoa_dues_annual/insurance_annual/maintenance_annual/special_assessment/utilities): a
  purely PROJECTED distribution grown by a growth or derived-latent model (special_assessment is the
  odd one out: a count x discrete-size model, see build_special_assessment_prior_model), with no real historical
  data for the carrying cost itself feeding the fit. As of the 2026-09-29 redesign, three of these
  (hoa_dues_annual/insurance_annual/utilities) start from a REAL KNOWN current value (0% uncertain at
  h=0, same role as market_value.initial_price), not a hand-elicited level+cv -- only
  maintenance_annual still has a genuine hand-elicited GUESS at its own current value (see
  build_maintenance_prior_model for how its own point-uncertainty is derived from the latent's own
  process instead of a separately hand-picked cv). Saved under
  models/carrying_cost_priors/uninformed/<name>/.
- INFORMED (market_value): fits the Denver-metro home_price trend (with its own propagated
  uncertainty) once from real Denver data, then CALIBRATES it to actual Keystone dollars via a
  simple linear fit (slope + intercept, least squares) against however many of this property's own
  known real reassessment values exist (deck field market_value.validation_data_source names a
  data/manual/ CSV of them -- see load_market_value_history and fit_market_value_calibration) --
  USED DIRECTLY in the saved forward model (build_market_value_change_magnitude), not just reported
  as a diagnostic. Saved periods are in ACTUAL USD -- no further scaling needed downstream (see
  build_market_value_change_magnitude's own docstring for why). Saved under
  models/carrying_cost_priors/informed/<name>/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import norm

from condobuyuq2026.carrying_costs import (
    carrying_cost_prior_path,
    resolve_heating_type,
    special_assessment_count_probabilities,
)
from condobuyuq2026.input_deck import (
    DERIVED_LATENT_COMPONENTS,
    MONTH_KEYS,
    SPECIAL_ASSESSMENT_NO_GROWTH,
    UTILITIES_CALIBRATION_OCCUPANCY_FROM_DECK,
    get_closing_date,
    get_expected_occupied_days_by_month,
    get_forecast_model_name,
    get_growth_fit,
    get_hoa_dues_config,
    get_insurance_config,
    get_maintenance_config,
    get_market_value_config,
    get_report_quantiles,
    get_special_assessment_config,
    get_utilities_prior_config,
)
from condobuyuq2026.input_deck import load_deck as _load_deck
from condobuyuq2026.paths import DATA_MANUAL_DIR
from condobuyuq2026.plotting.forecast_plots import (
    plot_market_value_calibration,
    plot_market_value_correction_diagnostic,
    plot_periods_projection,
    plot_periods_stairstep,
    plot_special_assessment_count_by_horizon,
    plot_utilities_year1_breakdown,
)
from condobuyuq2026.raw_data import load_monthly_series
from condobuyuq2026.utils.manual_utils import model_fit_path, model_plots_dir
from condobuyuq2026.utils.ou_fitting import project_term_structure

_CARRYING_COST_PRIOR_OUTER_CI = (0.01, 0.99)  # wide reference band, same convention as other build scripts

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


def reassessment_boundary_months(
    closing_month: int, first_reassessment_year: int, reassessment_frequency_years: int, num_periods: int
) -> list[int]:
    """Returns the month offsets (from closing, t=0) where market_value's assessed value changes:
    [0, M_1, M_1+P, M_1+2P, ..., M_1+(num_periods-1)*P] -- period k spans [boundaries[k],
    boundaries[k+1]) (the last period is open-ended). Period 0 (before the first reassessment) is
    the deterministic initial_price itself; every period from 1 onward is an AVERAGED value -- see
    carrying_costs.project_market_value_schedule's own docstring for the full mechanism."""
    period_months = reassessment_frequency_years * 12
    m1 = months_to_first_reassessment(closing_month, first_reassessment_year)
    return [0] + [m1 + k * period_months for k in range(num_periods)]


def historical_index_level(monthly_series: pd.Series, closing_calendar_date: pd.Timestamp, offset_months: int) -> float:
    """The REAL, already-recorded RAW index level at (closing + offset_months) -- offset_months
    should be <= 0 (a calendar month at or before closing; see
    carrying_costs.project_market_value_schedule's docstring for why only pre-closing months use
    real data instead of a projection). Uses Series.asof, which returns the last valid (non-NaN)
    value AT OR BEFORE a given timestamp -- this both skips real gap months in the raw data AND, if
    closing itself is a near-future month the raw data doesn't extend to yet, pins to the latest
    actually-recorded value rather than needing a separate "today vs. closing" special case.

    Returns the RAW index level (not a ratio to closing) -- market_value's own forward model scales
    by a combined Denver-to-Keystone conversion factor (see compute_market_value_offset), not by a
    growth-factor-relative-to-closing the way an earlier version of this mechanism did.
    """
    target_date = closing_calendar_date + pd.DateOffset(months=offset_months)
    # Series.asof's stub return type is broader than what it actually returns for a scalar `where`
    # (a single float, or NaN) -- cast so downstream arithmetic sees a plain float, not the stub's
    # own overly-wide union (same reasoning as ou_fitting.py's own np.log cast).
    value = cast(float, monthly_series.asof(target_date))
    if math.isnan(value):
        raise ValueError(
            f"No historical data at or before {target_date.date()} in this founding process's own "
            "raw series -- closing_month/closing_year (or a reassessment window that reaches "
            "further back) predates all available historical data."
        )
    return value


def _window_average_index(monthly_series: pd.Series, window_start: pd.Timestamp, window_end: pd.Timestamp) -> float:
    """Real average index level over [window_start, window_end] (inclusive, monthly) -- e.g. the
    18-month averaging window a real reassessment was actually based on."""
    months = pd.date_range(window_start, window_end, freq="MS")
    values = [cast(float, monthly_series.asof(month)) for month in months]
    return sum(values) / len(values)


def fit_market_value_calibration(
    deck: dict[str, Any] | None = None, history: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """FITTING AND VALIDATION APPROACH, in brief: home_price's own OU fit already captures Denver-
    metro market TREND and uncertainty from real Denver data; this function calibrates that trend
    into actual Keystone dollars via a plain least-squares LINE (slope + intercept) between Denver's
    own real average, over each of this property's known real reassessment windows, and what this
    property was ACTUALLY assessed at over that same window. Every number here is in real units:
    Denver averages are index points, assessed values and the fitted slope/intercept convert index
    points to USD.

    Per the user's own correction (2026-09-28): this is NOT "average several ratios into one scalar"
    (a prior pass did that, and the user rightly pointed out it adds nothing a plain ratio didn't
    already do). A LINE (not a single ratio) is fit instead, because a ratio forces the relationship
    through the origin -- a line lets slope and intercept separately absorb (a) how much Keystone's
    OWN price level differs from Denver's index units and (b) any additional flat gap between the two
    markets that a pure scale factor can't express. With exactly 2 known points (the common case
    today) the line passes through both EXACTLY, matching them with zero error; with more points
    later, it becomes a genuine least-squares regression that can average out noise across cycles;
    with exactly 1 point (no real history yet, just the current one), slope/intercept can't both be
    identified, so this falls back to a pure proportional line through the origin (intercept=0),
    degenerating gracefully to the single-point case.

    CRITICALLY: `market_value.initial_price` is used as one of the fitting points (it IS a real,
    0%-uncertainty known value, valuable for calibrating the line), but the model's own "right now"
    period is NEVER derived from this fitted line -- see build_market_value_change_magnitude, which
    always reports initial_price directly and exactly regardless of what this function returns. This
    function's job is ONLY to calibrate how the model should extrapolate FUTURE reassessments.

    Each observation's own Denver-side value is the Denver-metro index's real average over THAT
    exact averaging window (window_start through window_end, matching what the assessor actually
    used for that record -- see load_market_value_history). load_market_value_history always returns
    at least ONE record (see that function's own docstring), so this always has something to fit
    even before any real data/manual/<...>.csv exists.

    Returns {"slope", "intercept", "observations": [{"effective_year", "denver_window_avg",
    "keystone_actual"}, ...]}.
    """
    deck = deck if deck is not None else _load_deck()
    config = get_market_value_config(deck)
    monthly_series = load_monthly_series(get_forecast_model_name(config["latent"], deck))
    history = history if history is not None else load_market_value_history(deck)

    observations = [
        {
            "effective_year": record["effective_year"],
            "denver_window_avg": _window_average_index(monthly_series, record["window_start"], record["window_end"]),
            "keystone_actual": record["assessed_value"],
        }
        for record in history
    ]

    denver_values = [o["denver_window_avg"] for o in observations]
    keystone_values = [o["keystone_actual"] for o in observations]
    if len(observations) == 1:
        # Can't identify both slope and intercept from a single point -- fall back to a pure
        # proportional relationship (a line through the origin AND that one point), the same
        # degenerate case a plain ratio would give.
        slope = keystone_values[0] / denver_values[0]
        intercept = 0.0
    else:
        slope, intercept = np.polyfit(denver_values, keystone_values, 1)

    return {"slope": float(slope), "intercept": float(intercept), "observations": observations}


def _month_index(date: pd.Timestamp) -> int:
    """Whole calendar months since year 0 -- lets month-since-closing arithmetic (e.g. placing a
    real observation's window midpoint on a "months from closing" x-axis) use plain integer
    subtraction instead of pandas Period gymnastics."""
    return date.year * 12 + date.month


def annotate_observation_timing(
    record: dict[str, Any], closing_calendar_date: pd.Timestamp, reassessment_frequency_years: int
) -> dict[str, Any]:
    """Adds a history record's (see load_market_value_history) own timing, all in whole
    months-since-closing, for plotting.forecast_plots.plot_market_value_correction_diagnostic's
    scatter + reference lines:
    - "month_since_closing": the averaging window's own MIDPOINT -- where the "Actuals" dot is drawn.
    - "window_start_month"/"window_end_month": the averaging window itself (window_start through
      window_end, real dates from the record) -- for the thin red "Averaging Window" line.
    - "period_start_month"/"period_end_month": the reassessment_frequency_years-long period, starting
      January 1 of the record's own effective_year, that this assessed value actually HELD for
      real-world tax purposes -- for the right panel's thin black "Assessment Period" line. Distinct
      from the averaging window: the window is what the value was CALCULATED from, the period is when
      it was IN EFFECT (the two don't overlap -- the period starts only after the window ends).
    """
    closing_month_index = _month_index(closing_calendar_date)
    window_start_month = _month_index(pd.Timestamp(record["window_start"])) - closing_month_index
    window_end_month = _month_index(pd.Timestamp(record["window_end"])) - closing_month_index
    period_start_month = _month_index(pd.Timestamp(year=record["effective_year"], month=1, day=1)) - closing_month_index
    return {
        **record,
        "month_since_closing": (window_start_month + window_end_month) // 2,
        "window_start_month": window_start_month,
        "window_end_month": window_end_month,
        "period_start_month": period_start_month,
        "period_end_month": period_start_month + reassessment_frequency_years * 12,
    }


def build_market_value_naive_reference(
    deck: dict[str, Any] | None = None, history: list[dict[str, Any]] | None = None, horizon_months: int = 120
) -> pd.DataFrame:
    """MODEL-DIAGNOSTIC ONLY -- never read by any scenario run, and NOT used by
    build_market_value_change_magnitude itself. Projects home_price's raw OU fit as ONE continuous
    trajectory, scaled by a PLAIN proportional factor (initial_price / denver_level_at_closing, no
    intercept) so it passes through initial_price exactly at closing -- i.e. what
    fit_market_value_calibration's line would degenerate to if its intercept were forced to 0. Used
    by build_market_value_informed_model to draw plotting.forecast_plots.
    plot_market_value_correction_diagnostic, so it's visible at a glance whether a plain
    unit-conversion scalar already passes near this property's real known reassessment values, or
    whether (motivating the fitted intercept) it systematically misses them.

    Returns a DataFrame indexed by month-since-closing (negative = real Denver history, via
    historical_index_level; 0..horizon_months = project_term_structure's own projection, evaluated at
    both the deck's report_quantiles and the wide (0.01, 0.99) reference band) with columns
    "median"/"lo"/"hi"/"lo_outer"/"hi_outer", all pre-scaled to USD by the plain factor.
    """
    deck = deck if deck is not None else _load_deck()
    config = get_market_value_config(deck)
    latent_fit = get_growth_fit(config["latent"], deck)
    monthly_series = load_monthly_series(get_forecast_model_name(config["latent"], deck))
    history = history if history is not None else load_market_value_history(deck)

    closing = get_closing_date(deck)
    closing_calendar_date = pd.Timestamp(year=closing["closing_year"], month=closing["closing_month"], day=1)
    closing_month_index = _month_index(closing_calendar_date)
    denver_level_at_closing = cast(float, monthly_series.asof(closing_calendar_date))
    scale = config["initial_price"] / denver_level_at_closing

    earliest_month = min([0] + [_month_index(pd.Timestamp(r["window_start"])) - closing_month_index for r in history])
    history_months = list(range(earliest_month, 0))
    history_levels = [historical_index_level(monthly_series, closing_calendar_date, m) * scale for m in history_months]

    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])
    projection = project_term_structure(latent_fit, p0=denver_level_at_closing, horizon_months=horizon_months, ci=ci)
    outer_projection = project_term_structure(
        latent_fit, p0=denver_level_at_closing, horizon_months=horizon_months, ci=_CARRYING_COST_PRIOR_OUTER_CI
    )

    months = history_months + list(projection.index)
    return pd.DataFrame(
        {
            "median": history_levels + list(projection["median"] * scale),
            "lo": history_levels + list(projection["lo"] * scale),
            "hi": history_levels + list(projection["hi"] * scale),
            "lo_outer": history_levels + list(outer_projection["lo"] * scale),
            "hi_outer": history_levels + list(outer_projection["hi"] * scale),
        },
        index=pd.Index(months, name="month"),
    )


def build_market_value_change_magnitude(deck: dict[str, Any] | None = None, horizon_months: int = 120) -> dict[str, Any]:
    """Builds market_value's own CHANGE-MAGNITUDE model, already in ACTUAL USD -- no further
    scaling needed downstream (see fit_market_value_calibration's own docstring for the fitting/
    validation approach, and the "no more scaling" rationale below) -- independent of any particular
    scenario's horizon_years or purchase_price. The expensive part (real historical lookups + OU
    projections + the pre-closing/projected averaging blend, see
    carrying_costs.project_market_value_schedule's own docstring for the full per-reassessment
    mechanism) is done ONCE here, so a scenario run never repeats it. Called by
    build_market_value_informed_model (the build step) -- NOT by
    carrying_costs.project_market_value_schedule, which just loads the SAVED result directly.

    horizon_months: the "extrapolation limit" -- how far out to precompute reassessment periods for.
    Must cover whatever scenario.horizon_years the deck's sweep might need; defaults to 120 (10
    years), matching every other carrying_cost_prior build function's own default.

    Returns {"closing_month", "first_reassessment_year", "reassessment_frequency_years", "latent",
    "calibration" (see fit_market_value_calibration -- kept here so the validation plot/report don't
    need to recompute it), "periods": [{"start_month", "end_month" (None for the last, open-ended
    period), "known_value", "projected_value", "growth_sigma"}, ...]}. `known_value`/`projected_value`
    are ALREADY IN USD (unlike an earlier version of this mechanism, which stored raw Denver-index
    units plus a separate scale factor applied at use time) -- this module's job is done producing
    them; nothing downstream needs to know Denver index units exist at all. A period's median/lo/hi
    at a given confidence level is:
        median = known_value + projected_value
        lo/hi  = known_value + projected_value * exp(z_lo/hi * growth_sigma)
    (`known_value` is the window's known/recorded contribution, already converted to USD via the
    fitted calibration line; `projected_value` is the projected contribution, also in USD;
    `growth_sigma` the projected portion's own representative log-space uncertainty -- unaffected by
    the USD conversion, since it's a ratio. `growth_sigma` is deliberately NOT resolved to a lo/hi at
    any particular confidence level here, so the SAME saved model answers any report_quantiles the
    deck is set to at use time, without rebuilding.)

    Period 0 (before the first reassessment) is special-cased to `known_value = initial_price`,
    `projected_value = 0`, `growth_sigma = 0` DIRECTLY -- market_value.initial_price is known exactly
    from the deck (0% uncertainty, the user's own explicit requirement), never re-derived from the
    fitted calibration line even though that line is itself informed by this same value (see
    fit_market_value_calibration's own docstring for why those are different things).
    """
    deck = deck if deck is not None else _load_deck()
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
    denver_level_at_closing = cast(float, monthly_series.asof(closing_calendar_date))

    calibration = fit_market_value_calibration(deck)
    slope, intercept = calibration["slope"], calibration["intercept"]

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
    # p0 = the REAL Denver index level as of closing (not 1.0) -- this projection stays in raw
    # Denver-index units; the calibration's slope/intercept convert to USD only at the very end
    # (see docstring) so growth_sigma (recovered just below) stays a clean, unitless ratio.
    projection = project_term_structure(latent_fit, p0=denver_level_at_closing, horizon_months=max_projected_h, ci=ci)
    # Recover each h's own log-space sigma from the projection's own median/hi (CI-independent once
    # divided back out by z_hi, and scale-invariant -- unaffected by using a raw Denver level instead
    # of 1.0 as p0) -- so it can be re-combined with any ci at USE time, not just the one read here
    # at build time. np.log on a Series returns a Series at runtime (numpy ufuncs respect pandas'
    # __array_ufunc__), but numpy's stubs don't know that and declare NDArray -- cast so type
    # checkers see the real runtime type (same reasoning as ou_fitting.py's own cast).
    sigma_at_h = cast(pd.Series, np.log(projection["hi"] / projection["median"]) / z_hi)

    periods: list[dict[str, Any]] = [
        {
            "start_month": 0,
            "end_month": boundaries[1],
            "known_value": config["initial_price"],
            "projected_value": 0.0,
            "growth_sigma": 0.0,
        }
    ]
    for k in range(1, len(boundaries)):
        window_months = windows[k - 1]
        known_months = [m for m in window_months if m <= 0]
        window_projected_months = [m for m in window_months if m > 0]
        known_sum = sum(historical_index_level(monthly_series, closing_calendar_date, m) for m in known_months)
        if window_projected_months:
            projected_median_sum = sum(projection.loc[m, "median"] for m in window_projected_months)
            representative_sigma = sum(sigma_at_h.loc[m] for m in window_projected_months) / len(window_projected_months)
        else:
            projected_median_sum = 0.0
            representative_sigma = 0.0
        n_total = len(window_months)
        denver_a = known_sum / n_total
        denver_b = projected_median_sum / n_total
        period_end = boundaries[k + 1] if k + 1 < len(boundaries) else None
        periods.append(
            {
                "start_month": boundaries[k],
                "end_month": period_end,
                # Converted to USD here via the fitted calibration line -- see this function's own
                # docstring for the known_value/projected_value split and why intercept belongs
                # entirely with known_value (it's a deterministic term, unaffected by projection
                # uncertainty).
                "known_value": slope * denver_a + intercept,
                "projected_value": slope * denver_b,
                "growth_sigma": representative_sigma,
            }
        )

    return {
        "closing_month": closing["closing_month"],
        "first_reassessment_year": config["first_reassessment_year"],
        "reassessment_frequency_years": config["reassessment_frequency_years"],
        "latent": config["latent"],
        "calibration": {"slope": slope, "intercept": intercept, "observations": calibration["observations"]},
        "periods": periods,
    }


def _current_cycle_observation(deck: dict[str, Any]) -> dict[str, Any]:
    """Synthesizes THIS cycle's own known-real-value observation from initial_price/
    first_reassessment_year/reassessment_frequency_years directly, rather than requiring it to be
    hand-duplicated as a row in the history CSV (which would risk drifting out of sync with
    initial_price if one gets edited and not the other). "This cycle" means the reassessment
    effective at or before closing -- i.e. the one initial_price itself reflects (see
    market_value's own deck comment for the "reassessment = effective date" convention this
    depends on): effective_year = closing_year + first_reassessment_year - reassessment_frequency_years
    (one cycle earlier than the NEXT reassessment first_reassessment_year points to), and its own
    averaging window is the same first-18-months-of-the-preceding-cycle rule every other
    reassessment uses.

    Returns {"effective_year", "window_start", "window_end", "assessed_value": initial_price} in the
    same shape load_market_value_history's own CSV-sourced records use.
    """
    config = get_market_value_config(deck)
    closing = get_closing_date(deck)
    effective_year = closing["closing_year"] + config["first_reassessment_year"] - config["reassessment_frequency_years"]
    window_start = pd.Timestamp(year=effective_year - config["reassessment_frequency_years"], month=1, day=1)
    window_end = window_start + pd.DateOffset(months=REASSESSMENT_AVERAGING_WINDOW_MONTHS - 1)
    return {
        "effective_year": effective_year,
        "window_start": window_start,
        "window_end": window_end,
        "assessed_value": config["initial_price"],
    }


def load_market_value_history(deck: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Loads market_value.validation_data_source (a filename under data/manual/ -- see that field's
    own deck comment and data/manual/README.md's entry for the file format) -- the county assessor's
    own ACTUAL past reassessment records for this property -- PLUS an automatically-synthesized
    record for the CURRENT cycle (see _current_cycle_observation; this is what "add the current
    initial price to the same variable" means -- initial_price is never hand-duplicated into the
    CSV, it's derived from the deck fields that already exist).

    Returns a list of {"effective_year": int, "window_start"/"window_end": pd.Timestamp,
    "assessed_value": float} dicts, sorted chronologically. If validation_data_source is blank/
    absent or the file doesn't exist, returns just the synthesized current-cycle record (so this
    function ALWAYS returns at least one observation) -- compute_market_value_offset's own
    single-point fallback then degenerates to using that one record, equivalent to the model's
    original initial_price-only anchor.
    """
    deck = deck if deck is not None else _load_deck()
    config = get_market_value_config(deck)
    records: list[dict[str, Any]] = [_current_cycle_observation(deck)]

    filename = config["validation_data_source"]
    if filename:
        path = DATA_MANUAL_DIR / filename
        if path.exists():
            df = pd.read_csv(path)
            records.extend(
                {
                    "effective_year": int(row["effective_year"]),
                    "window_start": pd.Timestamp(f"{row['window_start']}-01"),
                    "window_end": pd.Timestamp(f"{row['window_end']}-01"),
                    "assessed_value": float(row["assessed_value"]),
                }
                for _, row in df.iterrows()
            )

    return sorted(records, key=lambda r: cast(pd.Timestamp, r["window_start"]))


def bake_annual_periods(projection: pd.DataFrame, ci: tuple[float, float], horizon_months: int) -> list[dict[str, Any]]:
    """Bakes a continuous project_term_structure-shaped projection (median/lo/hi indexed by month,
    p0 at h=0) into ANNUAL periods -- the same {start_month, end_month, known_value, projected_value,
    growth_sigma} shape build_market_value_change_magnitude's own periods use, one period per year,
    each holding the value sampled at that year's own END (h=year*12) -- the exact sample points
    carrying_costs.py's own per-scenario projections have always used for these entries, just
    precomputed HERE instead of redone at every scenario run (standardized 2026-09-29 so every
    carrying_cost_prior follows the same "build once, read many times" convention market_value
    already did -- see carrying_costs.py's own module docstring).

    known_value is always 0.0 here: unlike market_value's pre-reassessment periods, nothing about
    these entries is ever "already known" -- they're projected from h=0 immediately, every dollar is
    `projected_value`. growth_sigma is reconstructed from that year's own real lo/hi
    (log(hi/median)/z_hi), the SAME CI-independent trick build_market_value_change_magnitude's own
    sigma_at_h uses, so it stays usable at any report_quantiles at READ time, not baked to the one
    band `ci` happened to be when this was built.
    """
    z_hi = norm.ppf(ci[1])
    horizon_years = horizon_months // 12
    periods = []
    for year in range(1, horizon_years + 1):
        h = year * 12
        median = projection.loc[h, "median"]
        hi = projection.loc[h, "hi"]
        sigma = float(np.log(hi / median) / z_hi) if hi > median else 0.0
        periods.append(
            {
                "start_month": (year - 1) * 12,
                "end_month": year * 12,
                "known_value": 0.0,
                "projected_value": float(median),
                "growth_sigma": sigma,
            }
        )
    return periods


def bake_monthly_periods(projection: pd.DataFrame, ci: tuple[float, float], horizon_months: int) -> list[dict[str, Any]]:
    """Bakes a continuous project_term_structure-shaped projection into FULL MONTHLY RESOLUTION
    periods -- one period per calendar month 0..horizon_months-1, each holding the value at h=month+1
    (its own period's END, the same convention bake_annual_periods uses at year boundaries,
    generalized to every month) -- the standardized periods shape, at the SAME resolution
    project_term_structure itself already computes internally.

    Replaces bake_annual_periods for insurance_annual/maintenance_annual/special_assessment
    (2026-09-30): bake_annual_periods threw away 11 of every 12 months project_term_structure had
    already computed, keeping only the year-end sample -- which is why a results plot built from
    those annual samples (10 points, connected by straight lines) looked noticeably coarser than
    model.png's own smooth 121-point curve, even though the underlying numbers matched exactly at
    the sampled points. Saving every month instead means model.png and any results plot reading these
    same periods are now LITERALLY drawing the same data at the same resolution -- the "should look
    nearly identical in form" the user rightly expected. known_value is always 0.0 (nothing here is
    ever already known)."""
    z_hi = norm.ppf(ci[1])
    periods = []
    for month in range(horizon_months):
        h = month + 1
        median = projection.loc[h, "median"]
        hi = projection.loc[h, "hi"]
        sigma = float(np.log(hi / median) / z_hi) if hi > median else 0.0
        periods.append(
            {
                "start_month": month,
                "end_month": month + 1,
                "known_value": 0.0,
                "projected_value": float(median),
                "growth_sigma": sigma,
            }
        )
    return periods


def bake_hoa_monthly_periods(
    annual_periods: list[dict[str, Any]], association_dues_fraction: float, horizon_months: int
) -> list[dict[str, Any]]:
    """Expands hoa_dues_annual's own ANNUAL periods (bake_annual_periods -- one combined
    dues+association draw per year) into GENUINELY MONTHLY periods with the January association-fee
    spike baked in DIRECTLY (2026-09-30, fixing a real gap the user caught: the earlier design saved
    only the annual periods plus a separate `association_dues_fraction`, so fit.json alone couldn't
    be used to recreate "the correct model, which includes the January annual association fee" --
    that reconstruction required extra fraction-math no reader could see just from the periods list).
    Now every month's own period already holds its own correct payment (including January's) --
    a plain lookup, exactly like every other entry, no separate fraction needed downstream at all."""
    recurring_fraction = 1 - association_dues_fraction
    periods = []
    for month in range(horizon_months):
        year_period = next(p for p in annual_periods if p["start_month"] <= month < p["end_month"])
        share = recurring_fraction / 12 + (association_dues_fraction if month % 12 == 0 else 0.0)
        periods.append(
            {
                "start_month": month,
                "end_month": month + 1,
                "known_value": year_period["known_value"] * share,
                "projected_value": year_period["projected_value"] * share,
                "growth_sigma": year_period["growth_sigma"],
            }
        )
    return periods


def build_hoa_dues_prior_model(deck: dict[str, Any] | None = None, horizon_months: int = 120) -> dict[str, Path]:
    """Builds carrying_costs.hoa_dues_annual's own KNOWN-VALUE+GROWTH projection.

    `initial_annual_dues` and `initial_annual_association_dues` are COMBINED into one starting value
    (`initial_combined`) that gets projected as a single draw from `latent`'s own fitted process --
    the user's own explicit request, so the association charge shares in the same growth and
    uncertainty as the rest, rather than being bolted on afterward as a frozen dollar amount. No
    extra_log_variance (both inputs are real known truth values) and NO floor/clamp: an earlier
    version of this function clipped the band at `initial_combined` to enforce "dues only go up,"
    which the user correctly flagged as producing a mixed/censored shape (a point mass at the floor
    plus a truncated lognormal above it), not a real lognormal one -- removed entirely.

    fit.json's saved `periods` are GENUINELY MONTHLY (bake_hoa_monthly_periods), each one already
    holding its own correct payment (the January ones already include the association share) -- fully
    self-sufficient, no separate fraction-math needed to reconstruct "the correct model" from fit.json
    alone (fixed 2026-09-30 -- `association_dues_fraction` is still echoed into fit.json for
    diagnostics, but nothing downstream needs it any more). model.png reuses the exact same
    plot_periods_stairstep market_value's own stair-step uses (HOA dues are billed monthly and held
    flat for 11 months then jump every January, structurally identical in shape to market_value's own
    reassessment stair-step, just monthly instead of multi-year).
    """
    deck = deck if deck is not None else _load_deck()
    config = get_hoa_dues_config(deck)
    latent_fit = get_growth_fit(config["latent"], deck)
    initial_combined = config["initial_annual_dues"] + config["initial_annual_association_dues"]
    association_dues_fraction = config["initial_annual_association_dues"] / initial_combined

    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])

    projection = project_term_structure(latent_fit, p0=initial_combined, horizon_months=horizon_months, ci=ci)
    annual_periods = bake_annual_periods(projection, ci, horizon_months)
    periods = bake_hoa_monthly_periods(annual_periods, association_dues_fraction, horizon_months)

    path_segment = carrying_cost_prior_path("hoa_dues_annual")
    fit_path = model_fit_path(path_segment)
    with fit_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "initial_annual_dues": config["initial_annual_dues"],
                "initial_annual_association_dues": config["initial_annual_association_dues"],
                "latent": config["latent"],
                "association_dues_fraction": association_dues_fraction,
                "periods": periods,
            },
            f,
            indent=2,
        )

    plot_path = model_plots_dir(path_segment) / "model.png"
    plot_periods_stairstep(
        periods,
        ci,
        _CARRYING_COST_PRIOR_OUTER_CI,
        horizon_months,
        title="HOA Dues (Monthly) — Uninformed Carrying-Cost Prior\nJanuary payments include the annual association charge",
        save_path=plot_path,
    )
    return {"fit_path": fit_path, "plot_path": plot_path}


def _build_guess_plus_growth_model(
    name: str, initial_guess: float, latent: str, deck: dict[str, Any], horizon_months: int, config_fields: dict[str, Any]
) -> dict[str, Path]:
    """Shared builder for the two GUESS+GROWTH entries (insurance_annual, maintenance_annual): a
    hand-elicited guess at today's value, with NO real truth data behind it, so its own point-in-time
    uncertainty is DERIVED from `latent`'s own fitted process (`extra_log_variance = sig^2 * tau / 2`
    -- the OU process's own MARGINAL/STATIONARY rate variance, the same quantity ou_fitting.
    project_monthly_rate's own docstring calls the process's long-run equilibrium spread) rather than
    a separately hand-picked cv. This is a CONSTANT term, not compounding with horizon -- uncertainty
    in the level NOW, not further inflated by extrapolating over a long horizon.

    config_fields are the caller's own resolved config values to echo into fit.json for diagnostics
    (e.g. {"initial_guess_mean": ...}) -- kept generic so both callers can pass their own field name.
    Saves just model.png, drawn DIRECTLY from the saved MONTHLY periods (bake_monthly_periods, full
    resolution -- not a separate project_term_structure recomputation) via plot_periods_projection, so
    model.png is guaranteed to show exactly what fit.json holds.
    """
    latent_fit = get_growth_fit(latent, deck)
    extra_log_variance = latent_fit["sig"] ** 2 * latent_fit["tau"] / 2

    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])

    projection = project_term_structure(
        latent_fit, p0=initial_guess, horizon_months=horizon_months, ci=ci, extra_log_variance=extra_log_variance
    )
    periods = bake_monthly_periods(projection, ci, horizon_months)

    path_segment = carrying_cost_prior_path(name)
    fit_path = model_fit_path(path_segment)
    with fit_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **config_fields,
                "latent": latent,
                "derived_extra_log_variance": extra_log_variance,
                "periods": periods,
            },
            f,
            indent=2,
        )

    plot_path = model_plots_dir(path_segment) / "model.png"
    plot_periods_projection(
        periods,
        ci,
        _CARRYING_COST_PRIOR_OUTER_CI,
        horizon_months,
        title=f"{name.replace('_', ' ').title()} — Uninformed Carrying-Cost Prior ({int(ci[0] * 100)}-{int(ci[1] * 100)}% Band)",
        save_path=plot_path,
        ylabel="USD/year",
    )
    return {"fit_path": fit_path, "plot_path": plot_path}


def build_insurance_prior_model(deck: dict[str, Any] | None = None, horizon_months: int = 120) -> dict[str, Path]:
    """Builds carrying_costs.insurance_annual's own GUESS+GROWTH projection (REDESIGNED 2026-09-30:
    moved from a known-value shape to this one -- a real current premium isn't actually known truth
    data the way hoa_dues_annual's dues are, it's an ESTIMATE, so it needs the same
    point-in-time-uncertainty treatment maintenance_annual already has -- see
    _build_guess_plus_growth_model for the shared mechanism)."""
    deck = deck if deck is not None else _load_deck()
    config = get_insurance_config(deck)
    return _build_guess_plus_growth_model(
        "insurance_annual",
        config["initial_guess_premium"],
        config["latent"],
        deck,
        horizon_months,
        {"initial_guess_premium": config["initial_guess_premium"]},
    )


def build_maintenance_prior_model(deck: dict[str, Any] | None = None, horizon_months: int = 120) -> dict[str, Path]:
    """Builds carrying_costs.maintenance_annual's own GUESS+GROWTH projection -- see
    _build_guess_plus_growth_model for the shared mechanism (insurance_annual uses the same one)."""
    deck = deck if deck is not None else _load_deck()
    config = get_maintenance_config(deck)
    return _build_guess_plus_growth_model(
        "maintenance_annual",
        config["initial_guess_mean"],
        config["latent"],
        deck,
        horizon_months,
        {"initial_guess_mean": config["initial_guess_mean"]},
    )


def build_special_assessment_prior_model(deck: dict[str, Any] | None = None, horizon_months: int = 120) -> dict[str, Path]:
    """Builds carrying_costs.special_assessment's FREQUENCY x SIZE model (redesigned 2026-11-22 from
    the old p x lognormal-severity expected-value projection -- see the deck comment).

    Unlike the other carrying costs, fit.json deliberately does NOT hold the standardized `periods`
    shape: a special assessment isn't a smooth cost with a confidence band, it's a count (Poisson)
    times a few discrete sizes, and it's excluded from every carrying-cost total. fit.json holds the
    validated deck config plus `size_growth_multiplier`: the growth latent's MEDIAN path from 1.0,
    one value per month h = 0..horizon_months (all 1.0 for `latent: none`). Only the median is used,
    not the latent's band: the size uncertainty is already carried by the discrete size_scenarios,
    and the latent's own spread over a few years is small next to the gap between a minor and a
    major assessment. carrying_costs.compute_special_assessment turns this into each scenario's own
    count and cost distribution.

    model.png shows P(N = k) against horizon length (1..horizon_months/12 years), via the SAME
    carrying_costs.special_assessment_count_probabilities the per-scenario model uses."""
    deck = deck if deck is not None else _load_deck()
    config = get_special_assessment_config(deck)

    if config["latent"] == SPECIAL_ASSESSMENT_NO_GROWTH:
        size_growth_multiplier = [1.0] * (horizon_months + 1)
    else:
        projection = project_term_structure(get_growth_fit(config["latent"], deck), p0=1.0, horizon_months=horizon_months)
        size_growth_multiplier = [float(m) for m in projection["median"]]

    path_segment = carrying_cost_prior_path("special_assessment")
    fit_path = model_fit_path(path_segment)
    with fit_path.open("w", encoding="utf-8") as f:
        json.dump({**config, "size_growth_multiplier": size_growth_multiplier}, f, indent=2)

    horizons = range(1, horizon_months // 12 + 1)
    count_table = pd.DataFrame(
        {
            years: special_assessment_count_probabilities(config["annual_rate"], years, config["k_max"])
            for years in horizons
        }
    ).T
    plot_path = model_plots_dir(path_segment) / "model.png"
    plot_special_assessment_count_by_horizon(
        count_table,
        config["k_max"],
        config["annual_rate"],
        title="Special Assessments — Number Over the Horizon, P(N = k)",
        save_path=plot_path,
    )
    return {"fit_path": fit_path, "plot_path": plot_path}


def bake_calendar_month_periods(
    latent_fit: dict[str, float],
    medians_by_month: dict[str, float],
    ci: tuple[float, float],
    horizon_months: int,
    closing_month: int,
    extra_log_variance: float = 0.0,
) -> list[dict[str, Any]]:
    """Bakes utilities' own 12-calendar-month costs into MONTHLY periods, one period per absolute
    month 0..horizon_months (each 1 month wide, `known_value` always 0.0) -- the standardized periods
    shape (see bake_annual_periods), at MONTHLY rather than annual granularity since utilities is the
    one entry with real seasonal (month-to-month) structure to preserve.

    CALENDAR ALIGNMENT (fixed 2026-11-22): absolute month h is calendar month
    MONTH_KEYS[(closing_month - 1 + h) % 12] -- month 0 is the CLOSING month (scenario.closing_month,
    TIMESTEP 0 for every month-indexed model), the same anchoring property tax already used (see
    months_to_first_reassessment). Before this fix, month 0 was always treated as January, so e.g. an
    October closing billed January's heating-season cost in October. medians_by_month must hold all
    12 MONTH_KEYS.

    Each absolute month is sampled from ITS calendar month's own independent OU trajectory (seeded at
    that month's year-1 cost) at h = the END of the absolute month's own scenario YEAR (year*12) --
    every month within a given year is evaluated at the SAME year-end horizon. extra_log_variance
    defaults to 0.0: each month's current value is treated as KNOWN (it's calibrated from a single
    real observed average bill), all uncertainty coming from the latent's own growth process.
    """
    z_hi = norm.ppf(ci[1])
    month_projections = {
        month: project_term_structure(
            latent_fit,
            p0=medians_by_month[month],
            horizon_months=horizon_months,
            ci=ci,
            extra_log_variance=extra_log_variance,
        )
        for month in MONTH_KEYS
    }
    periods = []
    for absolute_month in range(horizon_months):
        month_name = MONTH_KEYS[(closing_month - 1 + absolute_month) % 12]
        h = (absolute_month // 12 + 1) * 12
        projection = month_projections[month_name]
        median = projection.loc[h, "median"]
        hi = projection.loc[h, "hi"]
        sigma = float(np.log(hi / median) / z_hi) if hi > median else 0.0
        periods.append(
            {
                "start_month": absolute_month,
                "end_month": absolute_month + 1,
                "calendar_month": month_name,
                "known_value": 0.0,
                "projected_value": float(median),
                "growth_sigma": sigma,
            }
        )
    return periods


def load_temperature_seasonal_data(filename: str) -> dict[str, float]:
    """Loads data/manual/<filename> (columns "month","avg_temp_f" -- see that file's own README
    entry) into {month: avg_temp_f}. Raises ValueError unless the rows are exactly MONTH_KEYS, in
    order -- every month-keyed dict here relies on that order."""
    df = pd.read_csv(DATA_MANUAL_DIR / filename)
    temperatures = dict(zip(df["month"], df["avg_temp_f"].astype(float), strict=True))
    if tuple(temperatures) != MONTH_KEYS:
        raise ValueError(f"data/manual/{filename}: months must be exactly {MONTH_KEYS} in order, got {tuple(temperatures)}")
    return temperatures


# Non-leap days per calendar month. February uses 28: leap years add one day every 4 years, a ~0.1%
# effect on an annual bill, not worth modeling for a cost calibrated to an observed average.
DAYS_IN_MONTH = dict(zip(MONTH_KEYS, (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), strict=True))


def cap_occupied_days(occupied_days: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Caps each month's occupied days at the days in that month, so vacant days (days - occupied) can
    never go negative. Expected rented days + personal use CAN exceed the month in principle (the
    lognormal mean ignores occupancy's own rentable-days cap -- see
    input_deck.get_expected_occupied_days_by_month). Returns (capped days, one warning string per
    capped month), so the caller can print the warnings via reporting.py rather than silently
    clipping."""
    capped, warnings = {}, []
    for month in MONTH_KEYS:
        limit = DAYS_IN_MONTH[month]
        days = float(occupied_days[month])
        if days > limit:
            warnings.append(f"{month}: {days:.1f} occupied days exceeds the month's {limit} days -- capped at {limit}")
            days = float(limit)
        capped[month] = days
    return capped, warnings


def heating_degree_days_per_day(temperatures: dict[str, float], base_f: float, setback_f: float) -> dict[str, dict[str, float]]:
    """Heating degree-days PER DAY for each calendar month, from its MEAN temperature T:
    occupied = max(0, base - T); vacant = max(0, base - setback - T) (the thermostat is turned down by
    setback on vacant days, so heat only starts below a lower temperature).

    APPROXIMATION (flagged so it doesn't read as exact): real HDD are summed DAY BY DAY, and a month
    whose mean sits near the base still has cold days that need heat. Computing from the monthly mean
    UNDERCOUNTS such shoulder months. At Keystone every month's mean is below 65°F, so the effect is
    small (largest in summer); NOAA monthly HDD normals would remove it -- earmarked in the
    implementation plan."""
    return {
        month: {
            "occupied": max(0.0, base_f - temperature),
            "vacant": max(0.0, base_f - setback_f - temperature),
        }
        for month, temperature in temperatures.items()
    }


def heating_weights(occupied_days: dict[str, float], hdd_per_day: dict[str, dict[str, float]]) -> dict[str, float]:
    """Each month's heating weight w = occupied_days x occupied HDD/day + vacant_days x vacant HDD/day
    -- the month's total degree-days of heating actually delivered, occupied and vacant (setback)
    combined. Heating cost is proportional to w (cost per heating degree-day x w)."""
    return {
        month: days * hdd_per_day[month]["occupied"] + (DAYS_IN_MONTH[month] - days) * hdd_per_day[month]["vacant"]
        for month, days in occupied_days.items()
    }


def calibrate_utilities(
    config: dict[str, Any],
    heating: dict[str, Any],
    calibration_occupied_days: dict[str, float],
    hdd_per_day: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """CALIBRATION half of the three-category utilities model: sizes the three pools so that, under
    calibration_occupied_days (the occupancy the observed bill was measured under), one year adds up
    to exactly A = 12 x initial_annual_average_monthly_cost. Kept SEPARATE from apply_utilities so a
    different occupancy profile can be applied to the same calibration (e.g. the previous owner's real
    occupancy for calibration, this deck's for application).

    Steps: split A by component weight; carve each component's heating fraction (carrying_costs.
    resolve_heating_type) out into the HEATING pool H; split each component's non-heating remainder by
    its occupancy_fraction into the OCCUPANCY pool U and the FLAT pool F. Then rates:
        u = U / sum(occupied days)          (cost per occupied day)
        k = H / sum(heating weights)        (cost per heating degree-day, see heating_weights)
        flat per month = F / 12
    Raises ValueError if a pool has cost but nothing to spread it over (H > 0 with zero heating
    degree-days all year, or U > 0 with zero occupied days all year).

    Returns {"annual_observed_total", "flat_pool", "occupancy_pool", "heating_pool", "flat_monthly",
    "cost_per_occupied_day", "cost_per_heating_degree_day", "pools_by_component"}."""
    annual = 12 * config["initial_annual_average_monthly_cost"]
    pools_by_component = {}
    for name, component in config["components"].items():
        amount = component["weight"] * annual
        heat = heating["heating_fraction_by_component"][name] * amount
        rest = amount - heat
        pools_by_component[name] = {
            "heating": heat,
            "occupancy": rest * component["occupancy_fraction"],
            "flat": rest * (1 - component["occupancy_fraction"]),
        }
    heating_pool = sum(p["heating"] for p in pools_by_component.values())
    occupancy_pool = sum(p["occupancy"] for p in pools_by_component.values())
    flat_pool = sum(p["flat"] for p in pools_by_component.values())

    total_days = sum(calibration_occupied_days.values())
    total_weight = sum(heating_weights(calibration_occupied_days, hdd_per_day).values())
    if occupancy_pool > 0 and total_days == 0:
        raise ValueError("carrying_costs.utilities: occupancy-driven cost but zero occupied days all year to spread it over")
    if heating_pool > 0 and total_weight == 0:
        raise ValueError(
            "carrying_costs.utilities: heating cost but zero heating degree-days all year -- check hdd_base_temp_f "
            "against the temperature data"
        )
    return {
        "annual_observed_total": annual,
        "flat_pool": flat_pool,
        "occupancy_pool": occupancy_pool,
        "heating_pool": heating_pool,
        "flat_monthly": flat_pool / 12,
        "cost_per_occupied_day": occupancy_pool / total_days if total_days > 0 else 0.0,
        "cost_per_heating_degree_day": heating_pool / total_weight if total_weight > 0 else 0.0,
        "pools_by_component": pools_by_component,
    }


def apply_utilities(
    calibration: dict[str, Any],
    occupied_days: dict[str, float],
    hdd_per_day: dict[str, dict[str, float]],
    add_ons: list[dict[str, Any]],
) -> pd.DataFrame:
    """APPLICATION half of the three-category utilities model: one year of monthly costs (before
    growth) for a given occupancy profile, using calibrate_utilities' rates:
        cost_m = flat/12 + u x occupied_days_m + k x heating_weight_m + sum(add-ons)
    Returns a DataFrame indexed by MONTH_KEYS (calendar order) with columns flat, occupancy, heating,
    add_ons, total, plus the drivers occupied_days, hdd_occupied_per_day, heating_weight (kept for the
    plot and fit.json)."""
    weights = heating_weights(occupied_days, hdd_per_day)
    add_on_monthly = sum(a["amount_usd"] for a in add_ons)
    rows = {}
    for month in MONTH_KEYS:
        flat = calibration["flat_monthly"]
        occupancy = calibration["cost_per_occupied_day"] * occupied_days[month]
        heating = calibration["cost_per_heating_degree_day"] * weights[month]
        rows[month] = {
            "flat": flat,
            "occupancy": occupancy,
            "heating": heating,
            "add_ons": add_on_monthly,
            "total": flat + occupancy + heating + add_on_monthly,
            "occupied_days": occupied_days[month],
            "hdd_occupied_per_day": hdd_per_day[month]["occupied"],
            "heating_weight": weights[month],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def build_utilities_prior_model(deck: dict[str, Any] | None = None, horizon_months: int = 120) -> dict[str, Any]:
    """Builds carrying_costs.utilities' THREE-CATEGORY model (redesigned 2026-11-22, replacing the
    temperature-sine seasonal shape and its hand-picked 0.3 summer floor -- see the deck comment for the
    three categories). The old shape spread the bill by temperature alone; this one separates what
    actually drives each part of the bill (nothing, occupied days, or occupied days x cold), so a
    different occupancy -- or, later, an uncertain one -- moves the right part of the bill.

    1. Calibrate (calibrate_utilities) against calibration_occupied_days_by_month ("deck" = this deck's
       own expected occupancy, see input_deck.get_expected_occupied_days_by_month), capped at days in
       month (cap_occupied_days).
    2. SELF-CHECK: applying the calibration to its own occupancy, without add-ons, must reproduce
       12 x the observed average bill exactly. Raised as an explicit AssertionError (not `assert`, which
       `python -O` strips) -- it guards the calibration algebra, not user input.
    3. Apply (apply_utilities) to this deck's expected occupancy, with add-ons -> 12 calendar-month
       year-1 costs.
    4. Grow each calendar month via `latent` (bake_calendar_month_periods, anchored to
       scenario.closing_month).

    fit.json: the validated config, the resolved heating type, the calibration, the year-1 calendar-month
    table, any cap warnings, closing_month, and the MONTHLY `periods` carrying_costs._project_utilities_annual
    sums. Plots: model.png (the monthly periods, as before) and seasonal_shape.png (plotting.
    forecast_plots.plot_utilities_year1_breakdown: year-1 stacked categories over occupied days and HDD).

    Returns {"fit_path", "plot_path", "seasonal_shape_plot_path", "utilities_report"} -- the report is
    printed by reporting.print_carrying_cost_prior_build_report."""
    deck = deck if deck is not None else _load_deck()
    config = get_utilities_prior_config(deck)
    latent_fit = get_growth_fit(config["latent"], deck)
    closing_month = get_closing_date(deck)["closing_month"]

    heating = resolve_heating_type(config["heating_type"], config["heating_share"], config["components"])
    temperatures = load_temperature_seasonal_data(config["seasonal_data_source"])
    hdd_per_day = heating_degree_days_per_day(temperatures, config["hdd_base_temp_f"], config["vacant_setback_f"])

    deck_days, deck_warnings = cap_occupied_days(get_expected_occupied_days_by_month(deck))
    calibration_warnings: list[str]
    if config["calibration_occupied_days_by_month"] == UTILITIES_CALIBRATION_OCCUPANCY_FROM_DECK:
        calibration_days, calibration_warnings = deck_days, []
    else:
        calibration_days, calibration_warnings = cap_occupied_days(config["calibration_occupied_days_by_month"])

    calibration = calibrate_utilities(config, heating, calibration_days, hdd_per_day)
    check = apply_utilities(calibration, calibration_days, hdd_per_day, add_ons=[])
    check_total = float(check[["flat", "occupancy", "heating"]].to_numpy().sum())
    if not math.isclose(check_total, calibration["annual_observed_total"], rel_tol=1e-9):
        raise AssertionError(
            f"utilities calibration self-check failed: year-1 total excluding add-ons {check_total:.6f} != "
            f"12 x observed average {calibration['annual_observed_total']:.6f}"
        )

    year_1 = apply_utilities(calibration, deck_days, hdd_per_day, config["additional_flat_monthly"])
    monthly_costs = year_1["total"].to_dict()

    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])
    periods = bake_calendar_month_periods(latent_fit, monthly_costs, ci, horizon_months, closing_month)

    year_1_excluding_add_ons = float(year_1[["flat", "occupancy", "heating"]].to_numpy().sum())
    report = {
        "heating": heating,
        "calibration": calibration,
        "calibration_source": "deck occupancy"
        if config["calibration_occupied_days_by_month"] == UTILITIES_CALIBRATION_OCCUPANCY_FROM_DECK
        else "calibration_occupied_days_by_month",
        "year_1_total_excluding_add_ons": year_1_excluding_add_ons,
        "year_1_add_ons_total": float(year_1["add_ons"].sum()),
        "cap_warnings": [f"deck occupancy {w}" for w in deck_warnings]
        + [f"calibration occupancy {w}" for w in calibration_warnings],
    }

    path_segment = carrying_cost_prior_path("utilities")
    fit_path = model_fit_path(path_segment)
    with fit_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **config,
                "closing_month": closing_month,
                "heating": heating,
                "calibration": calibration,
                "calibration_occupied_days": calibration_days,
                "year_1_by_calendar_month": year_1.to_dict(orient="index"),
                "cap_warnings": report["cap_warnings"],
                "periods": periods,
            },
            f,
            indent=2,
        )

    plot_path = model_plots_dir(path_segment) / "model.png"
    plot_periods_projection(
        periods,
        ci,
        _CARRYING_COST_PRIOR_OUTER_CI,
        horizon_months,
        title=(
            f"Utilities (Monthly) — Uninformed Carrying-Cost Prior ({int(ci[0] * 100)}-{int(ci[1] * 100)}% Band)\n"
            f"Month 0 = closing month ({MONTH_KEYS[closing_month - 1].capitalize()})"
        ),
        save_path=plot_path,
        ylabel="USD/month",
    )

    seasonal_shape_plot_path = model_plots_dir(path_segment) / "seasonal_shape.png"
    plot_utilities_year1_breakdown(
        year_1,
        title=(
            f"Utilities — Year-1 Monthly Cost by Category (before growth)\n"
            f"Heating: {heating['name']}; observed average \\${config['initial_annual_average_monthly_cost']:,.0f}/month"
        ),
        save_path=seasonal_shape_plot_path,
    )
    return {
        "fit_path": fit_path,
        "plot_path": plot_path,
        "seasonal_shape_plot_path": seasonal_shape_plot_path,
        "utilities_report": report,
    }


def build_market_value_informed_model(deck: dict[str, Any] | None = None, horizon_months: int = 120) -> dict[str, Path]:
    """Builds carrying_costs.property_tax.market_value's own CHANGE-MAGNITUDE model (see
    build_market_value_change_magnitude for the actual computation and units -- ALREADY in USD, see
    that function's own docstring) and its sanity-check plot, PLUS (this is the "informed" half) a
    validation plot showing the fitted Denver-to-Keystone calibration line against the real
    observations that informed it, if load_market_value_history returns at least 2 records (the
    synthesized current-cycle one plus at least one real historical row from
    market_value.validation_data_source's own CSV).

    FITTING AND VALIDATION, IN BRIEF (see fit_market_value_calibration's own docstring for the full
    version): home_price's OU fit supplies the Denver-metro TREND and its propagated uncertainty from
    real Denver data; a separate least-squares LINE calibrates that trend into actual Keystone
    dollars, fit against however many of this property's own known real reassessment values exist.
    This is NOT diagnostic-only -- the fitted line is genuinely what the saved model uses to convert
    Denver units to USD for every FUTURE reassessment. It's still a bounded, low-risk use of the
    data: only 2 parameters (slope, intercept) are ever estimated, the OU model's own growth-rate/
    variance SHAPE is completely untouched, and market_value.initial_price -- 0% uncertain, the
    actual current answer -- is used to help fit the line but is NEVER itself re-derived from it (see
    build_market_value_change_magnitude's own docstring for the exact guarantee).

    The saved model.png draws each reassessment period as its OWN independent object (see
    plotting.forecast_plots.plot_periods_stairstep) -- a clean vertical break at each
    reassessment instead of a diagonal line connecting periods, since the value doesn't actually
    move between reassessments. Saves models/carrying_cost_priors/informed/market_value/fit.json +
    plots/model.png (+ plots/calibration_fit.png and plots/correction_diagnostic.png if history has
    at least 2 rows).

    correction_diagnostic.png (build_market_value_naive_reference +
    plotting.forecast_plots.plot_market_value_correction_diagnostic) is a MODEL-BUILD-ONLY diagnostic,
    not a scenario result: a naive, intercept-free (pure proportional) reference trajectory next to
    the actual saved model, both with the real known reassessment values overlaid, so the calibration
    line's own effect on the model's dynamics is visible directly over time, not just as the
    two-point scatter calibration_fit.png already shows.

    Returns {"fit_path", "plot_path", "calibration_plot_path"/"diagnostic_plot_path" (only if >= 2
    observations exist), "calibration_report" (only if >= 2 observations exist --
    fit_market_value_calibration's own slope/intercept/observations, for
    reporting.print_carrying_cost_prior_build_report)}.
    """
    deck = deck if deck is not None else _load_deck()
    change_magnitude = build_market_value_change_magnitude(deck, horizon_months)

    path_segment = carrying_cost_prior_path("market_value")
    fit_path = model_fit_path(path_segment)
    with fit_path.open("w", encoding="utf-8") as f:
        json.dump(change_magnitude, f, indent=2)

    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])
    plot_path = model_plots_dir(path_segment) / "model.png"
    plot_periods_stairstep(
        change_magnitude["periods"],
        ci,
        _CARRYING_COST_PRIOR_OUTER_CI,
        horizon_months=horizon_months,
        title=f"Market Value (Stair-Stepped) — Informed Carrying-Cost Prior ({int(ci[0] * 100)}-{int(ci[1] * 100)}% Band)",
        save_path=plot_path,
    )

    result: dict[str, Any] = {"fit_path": fit_path, "plot_path": plot_path}
    calibration = change_magnitude["calibration"]
    if len(calibration["observations"]) >= 2:
        calibration_plot_path = model_plots_dir(path_segment) / "calibration_fit.png"
        plot_market_value_calibration(
            calibration,
            title="Market Value — Denver-to-Keystone Calibration",
            save_path=calibration_plot_path,
        )
        result["calibration_plot_path"] = calibration_plot_path
        result["calibration_report"] = calibration

        history = load_market_value_history(deck)
        closing = get_closing_date(deck)
        closing_calendar_date = pd.Timestamp(year=closing["closing_year"], month=closing["closing_month"], day=1)
        naive_trajectory = build_market_value_naive_reference(deck, history, horizon_months)
        observations_with_timing = [
            annotate_observation_timing(record, closing_calendar_date, change_magnitude["reassessment_frequency_years"])
            for record in history
        ]
        diagnostic_plot_path = model_plots_dir(path_segment) / "correction_diagnostic.png"
        plot_market_value_correction_diagnostic(
            naive_trajectory,
            change_magnitude["periods"],
            observations_with_timing,
            ci,
            _CARRYING_COST_PRIOR_OUTER_CI,
            horizon_months=horizon_months,
            title="Market Value: Latent (home price) vs. Linearly Calibrated Model",
            save_path=diagnostic_plot_path,
        )
        result["diagnostic_plot_path"] = diagnostic_plot_path
    return result


def build_all_carrying_cost_prior_models(deck: dict[str, Any] | None = None, horizon_months: int = 120) -> dict[str, dict[str, Any]]:
    """Builds every carrying_costs prior: the five UNINFORMED entries (hoa_dues_annual/
    insurance_annual/maintenance_annual each now their own dedicated build_* function since their
    2026-09-29 redesign gave each a genuinely different shape -- see build_hoa_dues_prior_model/
    build_insurance_prior_model/build_maintenance_prior_model's own docstrings; special_assessment
    its own frequency x size model, see build_special_assessment_prior_model; utilities also
    uninformed, its own seasonal shape), and market_value's INFORMED build
    (see build_market_value_informed_model). See carrying_costs.INFORMED_CARRYING_COST_PRIORS for
    which is which. Returns {name: {"fit_path": ..., "plot_path": ..., ...}}."""
    deck = deck if deck is not None else _load_deck()
    return {
        "hoa_dues_annual": build_hoa_dues_prior_model(deck, horizon_months),
        "insurance_annual": build_insurance_prior_model(deck, horizon_months),
        "maintenance_annual": build_maintenance_prior_model(deck, horizon_months),
        "market_value": build_market_value_informed_model(deck, horizon_months),
        "special_assessment": build_special_assessment_prior_model(deck, horizon_months),
        "utilities": build_utilities_prior_model(deck, horizon_months),
    }
