"""Fits rental_operations' two priors -- occupancy (a RATE of nights available to rent) and ADR (USD
per rented night, in closing-date dollars) -- per calendar month, from the deck's `sources` (see that
field's deck comment for what the user enters).

THE FIT (user's spec, 2026-11-22):
1. Restate every source's ADR from its own `year`'s dollars to closing-date dollars with the
   `general` CPI raw data (forecast_models.general): factor = CPI at the closing month (or the latest
   month available, with a warning) / that year's average CPI.
2. Spread each ANNUAL source over the 12 months using the average seasonal pattern of the included
   MONTHLY sources (each monthly source's months divided by its own annual average, averaged across
   sources, rescaled so the year averages back to 1). So an annual source adds level information while
   the monthly sources supply the seasonal shape.
3. OCCUPANCY, per calendar month: the plain LINEAR mean and sample std of that month's occupancy across
   every included observation (zeros included), moment-matched to a lognormal: cv = std / mean,
   median = mean / sqrt(1 + cv^2). A month that is 0 in EVERY observation is a hard 0.
4. ADR, per calendar month: the same, over BOOKED observations only (a month with no bookings has no
   ADR).
5. A month with only ONE observation has no spread of its own: it borrows the POOLED cv (root-mean-
   square cv over the months with >= 2 observations), with a warning.
6. The fitted log-space sigma, sqrt(ln(1 + cv^2)), is treated as YEAR-TO-YEAR variation; the
   uncertainty in the fitted typical value itself is sigma / sqrt(n) and is treated as PERSISTENT
   (doesn't average out over the horizon). Caveat: a bias shared by every source (e.g. all
   projections optimistic) can't be detected by any averaging.

UPGRADE PATH (not built): a hurdle model -- P(no bookings in the month) + a lognormal for booked
months -- if tail probabilities (e.g. "chance April earns nothing") are ever needed. Moment matching
preserves each month's mean and variance only.
"""

from __future__ import annotations

import json
import math
import statistics
from typing import Any

from scipy.stats import norm

from condobuyuq2026.input_deck import (
    DAYS_IN_MONTH,
    MONTH_KEYS,
    get_closing_date,
    get_forecast_model_name,
    get_rental_operations_sources,
    get_report_quantiles,
)
from condobuyuq2026.input_deck import load_deck as _load_deck
from condobuyuq2026.utils.manual_utils import model_fit_path, model_plots_dir
from condobuyuq2026.raw_data import load_monthly_series


def cpi_restatement_factors(years: set[int], deck: dict[str, Any]) -> tuple[dict[int, float], list[str]]:
    """{year: CPI(closing month) / mean CPI over that year's available months} from the `general`
    forecast model's raw data. If the closing month isn't in the data yet, the latest available month
    is used instead (one warning). Raises if a year has no CPI data at all."""
    series = load_monthly_series(get_forecast_model_name("general", deck)).dropna()
    closing = get_closing_date(deck)
    closing_stamp = f"{closing['closing_year']}-{closing['closing_month']:02d}-01"
    warnings = []
    if closing_stamp in series.index:
        reference = float(series.loc[closing_stamp])
    else:
        latest = series.index[-1]
        reference = float(series.iloc[-1])
        warnings.append(
            f"CPI for the closing month ({closing_stamp[:7]}) isn't in the general CPI data yet -- ADR restated "
            f"to {latest:%Y-%m} dollars instead"
        )
    factors = {}
    for year in sorted(years):
        in_year = series[series.index.year == year]
        if in_year.empty:
            raise ValueError(f"rental_operations.sources: no general CPI data for year {year}, can't restate its ADR")
        factors[year] = reference / float(in_year.mean())
    return factors, warnings


def _available_nights(source: dict[str, Any]) -> dict[str, float]:
    """Nights the source's unit was available to rent per month: days in month minus the owner's nights
    when the source gives them (history), else the full month."""
    owner = source.get("owner_nights_by_month") or {}
    return {m: DAYS_IN_MONTH[m] - float(owner.get(m, 0.0)) for m in MONTH_KEYS}


def _annual_occupancy(months: dict[str, dict[str, float | None]], available: dict[str, float]) -> float:
    """Annual occupancy of a monthly source: total nights rented / total nights available. Matches an
    owner statement's own year total (e.g. 86 / 332) when owner nights are given."""
    return sum(float(months[m]["occupancy"] or 0.0) * available[m] for m in MONTH_KEYS) / sum(available.values())


def _annual_adr(months: dict[str, dict[str, float | None]], available: dict[str, float]) -> float:
    """Nights-weighted annual ADR of a monthly source (booked months only) = total revenue / total nights."""
    nights = {m: float(months[m]["occupancy"] or 0.0) * available[m] for m in MONTH_KEYS}
    booked = [m for m in MONTH_KEYS if months[m]["adr"] is not None]
    return sum(float(months[m]["adr"] or 0.0) * nights[m] for m in booked) / sum(nights[m] for m in booked)


def seasonal_patterns(monthly_sources: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Average seasonal pattern of the monthly sources (each already ADR-restated):
    "occupancy": {month: ratio to the annual occupancy}, rescaled so the days-weighted year averages 1;
    "adr": {month: ratio to the annual ADR, or None if no source had bookings that month}, rescaled so
    the year's nights-weighted average (using the occupancy pattern) is 1."""
    occupancy_ratios: dict[str, list[float]] = {m: [] for m in MONTH_KEYS}
    adr_ratios: dict[str, list[float]] = {m: [] for m in MONTH_KEYS}
    for source in monthly_sources:
        months = source["months"]
        available = _available_nights(source)
        annual_occupancy = _annual_occupancy(months, available)
        if annual_occupancy == 0:
            continue
        annual_adr = _annual_adr(months, available)
        for m in MONTH_KEYS:
            occupancy_ratios[m].append(float(months[m]["occupancy"] or 0.0) / annual_occupancy)
            if months[m]["adr"] is not None:
                adr_ratios[m].append(float(months[m]["adr"]) / annual_adr)

    occupancy_pattern = {m: statistics.fmean(occupancy_ratios[m]) for m in MONTH_KEYS}
    scale = sum(occupancy_pattern[m] * DAYS_IN_MONTH[m] for m in MONTH_KEYS) / sum(DAYS_IN_MONTH.values())
    occupancy_pattern = {m: value / scale for m, value in occupancy_pattern.items()}
    occupancy_result: dict[str, float | None] = dict(occupancy_pattern)

    adr_pattern: dict[str, float | None] = {m: statistics.fmean(adr_ratios[m]) if adr_ratios[m] else None for m in MONTH_KEYS}
    weights = {m: occupancy_pattern[m] * DAYS_IN_MONTH[m] for m in MONTH_KEYS if adr_pattern[m] is not None}
    adr_scale = sum(float(adr_pattern[m] or 0.0) * w for m, w in weights.items()) / sum(weights.values())
    adr_pattern = {m: (value / adr_scale if value is not None else None) for m, value in adr_pattern.items()}
    return {"occupancy": occupancy_result, "adr": adr_pattern}


def _moment_match(values: list[float]) -> dict[str, Any]:
    """Plain linear mean and sample std of `values` -> lognormal median/cv/sigma. n = 1 -> cv None
    (filled in later from the pooled cv)."""
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n >= 2 else None
    cv = std / mean if std is not None and mean > 0 else None
    return {"n": n, "mean": mean, "std": std, "cv": cv}


def _finish_lognormal(fit: dict[str, Any], cv: float) -> None:
    """Fills the lognormal fields for a moment-matched month with the given cv:
    median, sigma_log (year-to-year), level_sigma_log (persistent, sigma_log / sqrt(n)), and the
    PREDICTIVE lognormal used everywhere downstream: predictive_sigma = both parts combined, and
    predictive_median = mean / exp(predictive_sigma^2 / 2), so its mean is still the sources' plain
    average (the persistent uncertainty widens the spread around the same mean instead of raising it)."""
    sigma_log = math.sqrt(math.log1p(cv**2))
    level_sigma_log = sigma_log / math.sqrt(fit["n"])
    predictive_sigma = math.hypot(sigma_log, level_sigma_log)
    fit["cv"] = cv
    fit["median"] = fit["mean"] / math.sqrt(1 + cv**2)
    fit["sigma_log"] = sigma_log
    fit["level_sigma_log"] = level_sigma_log
    fit["predictive_sigma"] = predictive_sigma
    fit["predictive_median"] = fit["mean"] * math.exp(-(predictive_sigma**2) / 2)


def _fit_months(observations: dict[str, list[float]], quantity: str, warnings: list[str]) -> dict[str, dict[str, Any]]:
    """Per-month moment match (see module docstring steps 3-6). observations: {month: [values]}."""
    fits: dict[str, dict[str, Any]] = {}
    for m in MONTH_KEYS:
        values = observations[m]
        if not values:
            fits[m] = {"n": 0, "hard_zero": True, "mean": 0.0, "median": 0.0, "cv": 0.0}
            continue
        fit = _moment_match(values)
        fit["hard_zero"] = fit["mean"] == 0
        fits[m] = fit

    spread = [fits[m]["cv"] for m in MONTH_KEYS if not fits[m]["hard_zero"] and fits[m]["cv"] is not None]
    pooled_cv = math.sqrt(statistics.fmean(cv**2 for cv in spread)) if spread else None
    for m in MONTH_KEYS:
        fit = fits[m]
        if fit["hard_zero"]:
            fit.update({"median": 0.0, "cv": 0.0, "sigma_log": 0.0, "level_sigma_log": 0.0, "predictive_sigma": 0.0,
                        "predictive_median": 0.0, "pooled_cv_used": False})
            continue
        fit["pooled_cv_used"] = fit["cv"] is None
        if fit["pooled_cv_used"]:
            if pooled_cv is None:
                raise ValueError(
                    f"rental_operations {quantity}: no month has 2+ observations, so there's no spread to fit -- "
                    "include at least two sources"
                )
            warnings.append(f"{quantity} {m}: only 1 observation -- using the pooled cv {pooled_cv:.3f}")
        _finish_lognormal(fit, fit["cv"] if fit["cv"] is not None else float(pooled_cv or 0.0))
    return fits


def fit_rental_priors(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fits the occupancy and ADR priors from rental_operations.sources (see module docstring).

    Returns {"occupancy": {month: fit}, "adr": {month: fit}, "observations": [per included source:
    {"label", "kind", "year", "resolution", "cpi_factor", "months": {month: {"occupancy", "adr"}}}
    (annual sources already spread over the months)], "patterns": seasonal_patterns(...),
    "warnings": [...]}. Each month's fit has n, mean, std, cv, median, sigma_log (year-to-year),
    level_sigma_log (persistent), predictive_median/predictive_sigma (both combined, mean preserved),
    hard_zero, pooled_cv_used."""
    deck = deck if deck is not None else _load_deck()
    sources = [s for s in get_rental_operations_sources(deck) if s["include"]]
    factors, warnings = cpi_restatement_factors({s["year"] for s in sources}, deck)

    def restate(adr: float | None, year: int) -> float | None:
        return adr * factors[year] if adr is not None else None

    monthly = [
        {**s, "months": {m: {"occupancy": v["occupancy"], "adr": restate(v["adr"], s["year"])} for m, v in s["months"].items()}}
        for s in sources
        if s["resolution"] == "monthly"
    ]
    annual = [s for s in sources if s["resolution"] == "annual"]

    if monthly:
        patterns = seasonal_patterns(monthly)
    else:
        patterns = {"occupancy": dict.fromkeys(MONTH_KEYS, 1.0), "adr": dict.fromkeys(MONTH_KEYS, 1.0)}
        warnings.append("no monthly sources -- annual sources spread FLAT over the year (no seasonal pattern)")

    observations = [
        {"label": s["label"], "kind": s["kind"], "year": s["year"], "resolution": "monthly",
         "cpi_factor": factors[s["year"]], "months": s["months"]}
        for s in monthly
    ]
    for s in annual:
        months: dict[str, dict[str, float | None]] = {}
        for m in MONTH_KEYS:
            occupancy = float(s["occupancy"]) * float(patterns["occupancy"][m] or 0.0)
            if occupancy > 1:
                warnings.append(f"{s['label']} {m}: spread occupancy {occupancy:.3f} > 1 -- capped at 1")
                occupancy = 1.0
            # A month with occupancy > 0 always has an ADR pattern: some monthly source booked it.
            adr_ratio = patterns["adr"][m]
            adr = restate(float(s["adr"]) * adr_ratio, s["year"]) if occupancy > 0 and adr_ratio is not None else None
            months[m] = {"occupancy": occupancy, "adr": adr}
        observations.append(
            {"label": s["label"], "kind": s["kind"], "year": s["year"], "resolution": "annual",
             "cpi_factor": factors[s["year"]], "months": months}
        )

    occupancy_values = {m: [float(o["months"][m]["occupancy"] or 0.0) for o in observations] for m in MONTH_KEYS}
    adr_values = {m: [float(o["months"][m]["adr"]) for o in observations if o["months"][m]["adr"] is not None] for m in MONTH_KEYS}
    occupancy_fit = _fit_months(occupancy_values, "occupancy", warnings)
    adr_fit = _fit_months(adr_values, "adr", warnings)
    for m in MONTH_KEYS:
        if not occupancy_fit[m]["hard_zero"] and adr_fit[m]["hard_zero"]:
            raise ValueError(f"rental_operations {m}: occupancy is fitted > 0 but no source has an ADR for it")

    return {"occupancy": occupancy_fit, "adr": adr_fit, "observations": observations, "patterns": patterns, "warnings": warnings}


def history_occupied_days_by_month(
    sources: list[dict[str, Any]], occupancy_pattern: dict[str, float | None]
) -> dict[str, Any] | None:
    """The PREVIOUS owner's occupied days per calendar month (rented + owner nights), averaged over the
    included HISTORY sources that give owner_nights_by_month -- what utilities' calibration option
    `calibration_occupied_days_by_month: history` uses (the bills were measured under THAT occupancy,
    not yours). Monthly sources: rented = occupancy x (days - owner nights); annual sources: their
    occupancy spread by the seasonal pattern first. Returns {"days": {month: days}, "sources": [labels]}
    or None if no such source exists."""
    used = [s for s in sources if s["include"] and s["kind"] == "history" and s["owner_nights_by_month"]]
    if not used:
        return None
    totals = dict.fromkeys(MONTH_KEYS, 0.0)
    for source in used:
        owner = source["owner_nights_by_month"]
        for m in MONTH_KEYS:
            if source["resolution"] == "monthly":
                rate = float(source["months"][m]["occupancy"])
            else:
                rate = min(float(source["occupancy"]) * float(occupancy_pattern[m] or 0.0), 1.0)
            totals[m] += rate * (DAYS_IN_MONTH[m] - owner[m]) + owner[m]
    return {"days": {m: totals[m] / len(used) for m in MONTH_KEYS}, "sources": [s["label"] for s in used]}


# ---------------------------------------------------------------------------------------------------
# Build: fit.json + model-folder plots (section 7 plan, build step 2)
# ---------------------------------------------------------------------------------------------------

RENTAL_PRIOR_NAMES = ("occupancy", "adr")


def rental_prior_path(name: str) -> str:
    """models/rental_operations_priors/<name> (name: occupancy | adr). Shared by this build side and
    rental_operations.load_rental_priors (the read side)."""
    return f"rental_operations_priors/{name}"


def _month_quantile(month_fit: dict[str, Any], p: float) -> float:
    """Year-1 predictive quantile of one month: predictive_median x exp(z_p x predictive_sigma)
    (year-to-year and persistent parts combined; see _finish_lognormal)."""
    if month_fit["hard_zero"]:
        return 0.0
    return float(month_fit["predictive_median"] * math.exp(norm.ppf(p) * month_fit["predictive_sigma"]))


def _band_quantiles(ci: tuple[float, float], outer_ci: tuple[float, float]) -> dict[str, float]:
    return {"median": 0.5, "lo": ci[0], "hi": ci[1], "lo_outer": outer_ci[0], "hi_outer": outer_ci[1]}


def _monthly_bands(
    fits: dict[str, dict[str, Any]], quantiles: dict[str, float], cap: float | None = None
) -> dict[str, list[float]]:
    """{band: [12 monthly values]}; cap (e.g. occupancy rate 1.0) applied as min(q, cap) -- exact for
    quantiles of a capped variable."""
    bands = {}
    for band, p in quantiles.items():
        values = [_month_quantile(fits[m], p) for m in MONTH_KEYS]
        bands[band] = [min(v, cap) for v in values] if cap is not None else values
    return bands


def _weighted_annual_band(bands: dict[str, list[float]], weights: dict[str, float]) -> dict[str, float]:
    """Annual quantiles as the weighted average of the monthly quantiles. Exact if every month moves
    together (one shared yearly shock, as the model assumes for the year-to-year part); approximate
    for the persistent part, which is treated per month."""
    total = sum(weights.values())
    return {band: sum(values[i] * weights[m] for i, m in enumerate(MONTH_KEYS)) / total for band, values in bands.items()}


def build_rental_operations_prior_models(deck: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Fits (fit_rental_priors) and saves rental_operations' two priors:
    models/rental_operations_priors/{occupancy,adr}/fit.json + plots/model.png. fit.json holds the raw
    sources exactly as entered (included or not), the observations actually fitted (ADR CPI-restated,
    annual sources spread by season), the seasonal patterns, each month's fit (n, mean, std, cv, median,
    sigma_log = year-to-year, level_sigma_log = persistent, hard_zero, pooled_cv_used), year-1 bands,
    and warnings. Occupancy's fit.json also holds YOUR rentable and rented nights by month (they depend
    on personal_use, a deck constant, so still scenario-independent). Returns {name: {"fit_path",
    "plot_path", ...summary}} for reporting.print_rental_operations_prior_build_report."""
    # Imported here: rental_operations reads the fit.json this function writes.
    from condobuyuq2026.plotting.rental_plots import plot_rental_prior
    from condobuyuq2026.rental_operations import rentable_nights_by_month, rented_nights_by_month

    deck = deck if deck is not None else _load_deck()
    fit = fit_rental_priors(deck)
    all_sources = get_rental_operations_sources(deck)
    all_factors, _ = cpi_restatement_factors({s["year"] for s in all_sources}, deck)
    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])
    quantiles = _band_quantiles(ci, (0.01, 0.99))
    rentable = rentable_nights_by_month(deck)
    days = {m: float(d) for m, d in DAYS_IN_MONTH.items()}

    # ---- per-source annual values (as fitted: ADR restated) and monthly plot points ----
    observation_by_label = {o["label"]: o for o in fit["observations"]}
    occupancy_annual_sources, adr_annual_sources = [], []
    occupancy_points, adr_points = [], []
    for index, source in enumerate(all_sources):
        factor = all_factors[source["year"]]
        months: dict[str, dict[str, float | None]] | None
        if source["resolution"] == "monthly":
            months = {
                m: {"occupancy": v["occupancy"], "adr": v["adr"] * factor if v["adr"] is not None else None}
                for m, v in source["months"].items()
            }
            available = _available_nights(source)
            annual_occupancy, annual_adr = _annual_occupancy(months, available), _annual_adr(months, available)
        else:
            annual_occupancy, annual_adr = source["occupancy"], (source["adr"] or 0.0) * factor
            months = observation_by_label[source["label"]]["months"] if source["include"] else None
        common = {"label": source["label"], "index": index, "include": source["include"]}
        occupancy_annual_sources.append({**common, "value": annual_occupancy})
        adr_annual_sources.append({**common, "value": annual_adr})
        if months is not None:
            occupancy_points.append(
                {**common, "resolution": source["resolution"], "values": {m: months[m]["occupancy"] for m in MONTH_KEYS}}
            )
            adr_points.append(
                {**common, "resolution": source["resolution"], "values": {m: months[m]["adr"] for m in MONTH_KEYS}}
            )

    # ---- bands ----
    occupancy_bands = _monthly_bands(fit["occupancy"], quantiles, cap=1.0)
    adr_bands = _monthly_bands(fit["adr"], quantiles)
    occupancy_annual = _weighted_annual_band(occupancy_bands, days)
    booked_weights = {
        m: (fit["occupancy"][m]["median"] * days[m] if not fit["adr"][m]["hard_zero"] else 0.0) for m in MONTH_KEYS
    }
    adr_annual = _weighted_annual_band(adr_bands, booked_weights)
    nights_bands = {
        band: [min(values[i] * rentable[m], rentable[m]) for i, m in enumerate(MONTH_KEYS)]
        for band, values in occupancy_bands.items()
    }
    rented_median = rented_nights_by_month(deck, "median", fit)
    rented_mean = rented_nights_by_month(deck, "mean", fit)

    # ---- fit.json ----
    common_json = {
        "sources_as_entered": all_sources,
        "observations_fitted": fit["observations"],
        "warnings": fit["warnings"],
        "report_ci": list(ci),
        "method": (
            "Per calendar month: plain linear mean and sample std across every included observation "
            "(occupancy: zeros included; ADR: booked months only), moment-matched to a lognormal "
            "(cv = std/mean, median = mean/sqrt(1+cv^2)). sigma_log = sqrt(ln(1+cv^2)) is year-to-year "
            "variation; level_sigma_log = sigma_log/sqrt(n) is persistent. One observation -> pooled cv. "
            "predictive_sigma combines both; predictive_median = mean x exp(-predictive_sigma^2/2) keeps the "
            "mean equal to the sources' average. Bands are year-1 predictive."
        ),
    }
    occupancy_json = {
        **common_json,
        "quantity": "occupancy rate = nights rented / nights available to rent (0-1)",
        "seasonal_pattern": fit["patterns"]["occupancy"],
        "months": fit["occupancy"],
        "bands_by_month": occupancy_bands,
        "annual_band": occupancy_annual,
        "rentable_nights_by_month": rentable,
        "rented_nights_by_month": {"median": rented_median, "mean": rented_mean, "bands": nights_bands},
        "annual_rented_nights": {"median_sum": sum(rented_median.values()), "mean": sum(rented_mean.values())},
        "history_occupied_days_by_month": history_occupied_days_by_month(all_sources, fit["patterns"]["occupancy"]),
    }
    adr_json = {
        **common_json,
        "quantity": "ADR = gross rental revenue / nights rented, USD per night, before lodging tax and fees",
        "dollars": "closing-date dollars (each source restated from its own year with the general CPI)",
        "cpi_factors_by_year": {str(year): factor for year, factor in sorted(all_factors.items())},
        "seasonal_pattern": fit["patterns"]["adr"],
        "months": fit["adr"],
        "bands_by_month": adr_bands,
        "annual_band": adr_annual,
    }

    results: dict[str, dict[str, Any]] = {}
    note = "Warnings: " + "; ".join(fit["warnings"]) if fit["warnings"] else None
    ci_text = f"{ci[0]:.0%}-{ci[1]:.0%}"
    for name, payload in (("occupancy", occupancy_json), ("adr", adr_json)):
        fit_path = model_fit_path(rental_prior_path(name))
        with fit_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        plot_path = model_plots_dir(rental_prior_path(name)) / "model.png"
        if name == "occupancy":
            plot_rental_prior(
                title=f"Occupancy prior: fitted per month vs raw sources (band {ci_text}, year 1)",
                ylabel="Occupancy (share of nights available)",
                monthly_bands=occupancy_bands,
                points=occupancy_points,
                annual={"sources": occupancy_annual_sources, "fitted": occupancy_annual},
                annual_ylabel="Annual occupancy",
                value_format="{:.1%}",
                save_path=plot_path,
                nights={**nights_bands, "cap": [rentable[m] for m in MONTH_KEYS]},
                note=note,
            )
        else:
            plot_rental_prior(
                title=f"ADR prior: fitted per month vs raw sources (closing-date $, band {ci_text}, year 1)",
                ylabel="ADR (USD per rented night)",
                monthly_bands=adr_bands,
                points=adr_points,
                annual={"sources": adr_annual_sources, "fitted": adr_annual},
                annual_ylabel="Annual ADR (nights-weighted)",
                value_format="${:,.0f}",
                save_path=plot_path,
                note=note,
            )
        results[name] = {"fit_path": fit_path, "plot_path": plot_path, "annual_band": payload["annual_band"]}
    results["occupancy"]["annual_rented_nights"] = occupancy_json["annual_rented_nights"]
    results["occupancy"]["warnings"] = fit["warnings"]
    results["occupancy"]["source_count"] = sum(s["include"] for s in all_sources)
    return results
