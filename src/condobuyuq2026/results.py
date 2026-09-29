"""Saves per-scenario RESULT plots (results/<scenario_label>/carrying_costs/<name>.png) -- SCALED,
real-dollar visualizations of one specific scenario's own carrying-cost projections, as opposed to
models/carrying_cost_priors/<name>/plots/model.png (the UNSCALED change-magnitude sanity-check plot
every deck edit rebuilds, see manual_utils.build_all_carrying_cost_prior_models). Wired into
runner.run_scenarios, once per buy scenario.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

matplotlib.use("Agg")  # headless-safe: these scripts only ever save figures, never display them
import matplotlib.pyplot as plt

from condobuyuq2026.carrying_costs import (
    compute_total_carrying_costs_schedule,
    load_carrying_cost_prior_fit,
    load_market_value_change_magnitude,
    market_value_periods_to_tax_periods,
    resolve_period,
)
from condobuyuq2026.input_deck import get_report_quantiles, load_deck
from condobuyuq2026.paths import RESULTS_DIR
from condobuyuq2026.plotting.forecast_plots import plot_periods_stairstep


def scenario_results_dir(scenario: dict[str, Any]) -> Path:
    """results/buy_h<horizon_years>y_p<purchase_price>_<residency_state>/carrying_costs/ -- a
    readable folder name from the scenario's own SWEPT fields (the only ones that can distinguish
    two scenarios -- everything else is a fixed deck constant). Not a stable id across deck edits
    that change the sweep values themselves, but stable across repeated runs of the same deck."""
    label = f"buy_h{scenario['horizon_years']}y_p{scenario['purchase_price']}_{scenario['residency_state']}"
    path = RESULTS_DIR / label / "carrying_costs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _year_band(periods: list[dict[str, Any]], year: int, z_lo: float, z_hi: float) -> tuple[float, float, float]:
    """(median, lo, hi) for one calendar YEAR, summed over however many saved periods actually fall
    in it -- works identically whether an entry has ONE period per year (insurance_annual/
    maintenance_annual/special_assessment) or TWELVE (utilities' own monthly periods). Summing each
    period's own lo/hi (not just summing medians and reconstructing a band separately) implicitly
    assumes every period within the year moves together -- reasonable here since they all share the
    SAME underlying latent draw for that stretch, not independent months."""
    year_periods = [p for p in periods if (year - 1) * 12 <= p["start_month"] < year * 12]
    median = sum(p["known_value"] + p["projected_value"] for p in year_periods)
    lo = sum(p["known_value"] + p["projected_value"] * np.exp(z_lo * p["growth_sigma"]) for p in year_periods)
    hi = sum(p["known_value"] + p["projected_value"] * np.exp(z_hi * p["growth_sigma"]) for p in year_periods)
    return median, lo, hi


def _draw_final_distribution(ax, median: float, sigma: float, outer_ci: tuple[float, float], subtitle: str) -> None:
    """Draws ONE lognormal-ish (known=0 assumed -- caller already folded any known/deterministic part
    into `median`, since only the projected portion is genuinely random) distribution in STANDARD
    (unrotated) PDF orientation -- USD on the x-axis (real tick marks/labels), Density on a RIGHT-side
    y-axis (no ticks/labels, just the axis name) -- per the user's own explicit styling request,
    2026-09-30. Tails are drawn out to `outer_ci`'s own bounds (e.g. 1%/99%) and labeled, not just the
    inner confidence band. `subtitle` (e.g. "Year 10 Distribution") gets the distribution's own REAL
    mean/stdev (not the lognormal's median/sigma parameters) appended as a second title line."""
    z_lo_outer, z_hi_outer = norm.ppf(outer_ci)
    if sigma < 1e-9:
        ax.axvline(median, color="C1")
        ax.set_title(subtitle, fontsize=9)
    else:
        lo_outer = median * np.exp(z_lo_outer * sigma)
        hi_outer = median * np.exp(z_hi_outer * sigma)
        spread = hi_outer - lo_outer
        x = np.linspace(max(lo_outer - spread * 0.15, 1.0), hi_outer + spread * 0.15, 300)
        mu = np.log(median)
        pdf = np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))
        ax.plot(x, pdf, color="C1")
        ax.fill_between(x, 0, pdf, color="C1", alpha=0.2)
        ax.axvline(lo_outer, color="C1", linestyle="--", alpha=0.6, linewidth=1)
        ax.axvline(hi_outer, color="C1", linestyle="--", alpha=0.6, linewidth=1)
        y_label_pos = pdf.max() * 0.03
        ax.annotate(f"{outer_ci[0] * 100:.0f}%", (lo_outer, y_label_pos), ha="right", va="bottom", rotation=90, fontsize=8)
        ax.annotate(f"{outer_ci[1] * 100:.0f}%", (hi_outer, y_label_pos), ha="left", va="bottom", rotation=90, fontsize=8)
        mean = median * np.exp(sigma**2 / 2)
        std = mean * np.sqrt(np.exp(sigma**2) - 1)
        # escaped \$ -- a matched PAIR of unescaped $ in a matplotlib title triggers mathtext parsing
        # (rendering "$2,265, Std=$244" as garbled math instead of literal text) -- \$ keeps it literal
        ax.set_title(f"{subtitle}\nMean=\\${mean:,.0f}, Std=\\${std:,.0f}", fontsize=9)

    ax.set_xlabel("USD")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel("Density")
    ax.set_yticks([])


def _plot_annual_projection_with_final_distribution(
    periods: list[dict[str, Any]],
    horizon_months: int,
    ci: tuple[float, float],
    outer_ci: tuple[float, float],
    title: str,
    ylabel: str,
    save_path: Path,
    aggregate_final_year: bool,
) -> Path:
    """LEFT: the FULL saved MONTHLY periods, plotted DIRECTLY at their own native resolution -- not
    annual samples connected by straight lines (the user's own catch, 2026-09-30: coarse annual-only
    sampling made this look "nothing alike" model.png's own dense curve, even where the underlying
    numbers matched exactly at the sampled points). This is guaranteed to show exactly what fit.json
    holds, including utilities' own real seasonal oscillation.

    RIGHT: a distribution snapshot of the FINAL year. `aggregate_final_year` MUST match what this
    prior's own periods actually represent -- get it wrong and the right panel silently shows a wildly
    wrong number (a real bug caught 2026-09-30: utilities' own genuine monthly BILLS need SUMMING
    (_year_band) to get "how much do I pay this year," but insurance_annual/maintenance_annual/
    special_assessment's periods each already hold a CONTINUOUS ANNUAL ESTIMATE sampled monthly --
    summing 12 of those over-counts by ~12x, since it's the same annual figure re-evaluated 12 times,
    not 12 independent months' worth of cost). True (utilities/hoa-shaped monthly bills): sum the
    final year's own periods via _year_band. False (a continuously-evolving annual estimate): just
    look up the single period covering the final month, the same query
    carrying_costs._project_carrying_cost_prior itself uses."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    z_lo, z_hi = norm.ppf(ci)
    z_hi_outer = norm.ppf(outer_ci[1])

    months, medians, los, his = [], [], [], []
    for period in periods:
        if period["start_month"] >= horizon_months:
            break
        known, projected, sigma = period["known_value"], period["projected_value"], period["growth_sigma"]
        months.append(period["start_month"])
        medians.append(known + projected)
        los.append(known + projected * np.exp(z_lo * sigma))
        his.append(known + projected * np.exp(z_hi * sigma))

    fig, (ax_line, ax_dist) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    ax_line.plot(months, medians, color="C1", label="Projected Median")
    ax_line.fill_between(months, los, his, color="C1", alpha=0.2, label="Confidence Band")
    ax_line.set_title(title)
    ax_line.set_xlabel("Months Ahead")
    ax_line.set_ylabel(ylabel)
    ax_line.legend(fontsize="small")

    final_year = horizon_months // 12
    if aggregate_final_year:
        year_median, _, year_hi_outer = _year_band(periods, final_year, norm.ppf(outer_ci[0]), z_hi_outer)
    else:
        final_period = resolve_period(periods, final_year * 12 - 1)
        known, projected, sigma = final_period["known_value"], final_period["projected_value"], final_period["growth_sigma"]
        year_median = known + projected
        year_hi_outer = known + projected * np.exp(z_hi_outer * sigma)
    year_sigma = float(np.log(year_hi_outer / year_median) / z_hi_outer) if year_hi_outer > year_median else 0.0
    _draw_final_distribution(ax_dist, year_median, year_sigma, outer_ci, subtitle=f"Year {final_year} Distribution")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def _plot_period_distributions(
    periods: list[dict[str, Any]], horizon_years: int, ci: tuple[float, float], title: str, save_path: Path
) -> Path:
    """Composite plot: one lognormal PDF curve (or a vertical marker line, for a period with zero
    growth_sigma -- a known constant) per DISTINCT saved market_value-shaped period ({"start_month",
    "end_month", "known_value", "projected_value", "growth_sigma"} -- see utils.prior_utils.
    build_market_value_change_magnitude) touching this scenario's horizon, overlaid on a shared x-axis
    so how the distribution shifts/widens is visible at a glance. Works equally for market_value's own
    periods (USD) or property_tax's rescaled ones (carrying_costs.market_value_periods_to_tax_periods
    -- USD/year), since both share the identical {known_value, projected_value, growth_sigma}
    structure and only differ by a scalar rate.

    Grouped by the underlying saved period, not by calendar year: consecutive years that resolve to
    the IDENTICAL period (carrying_costs.resolve_period, same check
    project_market_value_schedule itself uses) share ONE legend entry/curve labeled with the year
    range they cover (e.g. "year 2-3"), instead of drawing redundant overlapping curves per year.

    Period 0 (the value in effect from closing until the first reassessment) is ALWAYS given its own
    entry when it's short enough that no year's own 12-month sample point (year*12 months) ever lands
    inside it -- e.g. closing 3 months before a reassessment means year 1 itself is split across TWO
    real known values (the pre- and post-reassessment ones), and sampling only at month 12 would
    otherwise silently only ever show the second one, hiding real months of ownership at a different
    (and just as real) assessed value.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    z_hi = norm.ppf(ci[1])

    def _entry(label_prefix: str, period: dict[str, Any]) -> tuple[str, float, float, float, float]:
        """(label, median, effective_sigma, lo, hi) -- effective_sigma is the lognormal-around-median
        sigma reconstructed from the period's REAL lo/hi (known + projected*exp(z*sigma)), the same
        way this plot's own PDF curve always has: it differs from the period's raw growth_sigma
        whenever known_value != 0 (the usual case), since only the projected portion is lognormally
        distributed, not the period's full median."""
        known, projected, sigma = period["known_value"], period["projected_value"], period["growth_sigma"]
        median = known + projected
        if sigma < 1e-9 or projected == 0:
            return (f"{label_prefix} (Known, ${median:,.0f})", median, 0.0, median, median)
        z_lo = norm.ppf(ci[0])
        lo = known + projected * np.exp(z_lo * sigma)
        hi = known + projected * np.exp(z_hi * sigma)
        effective_sigma = float(np.log(hi / median) / z_hi) if hi > median else 0.0
        return (label_prefix, median, effective_sigma, lo, hi)

    year_periods = [resolve_period(periods, year * 12) for year in range(1, horizon_years + 1)]

    entries: list[tuple[str, float, float, float, float]] = []
    first_period = periods[0]
    if first_period is not year_periods[0]:
        entries.append(_entry("At Closing", first_period))

    start_idx = 0
    for i in range(1, len(year_periods) + 1):
        if i < len(year_periods) and year_periods[i] is year_periods[start_idx]:
            continue
        period = year_periods[start_idx]
        first_year, last_year = start_idx + 1, i
        year_label = f"Year {first_year}" if first_year == last_year else f"Year {first_year}-{last_year}"
        entries.append(_entry(year_label, period))
        start_idx = i

    lo_extent = min(lo for _, _, _, lo, _ in entries)
    hi_extent = max(hi for _, _, _, _, hi in entries)
    spread = max(hi_extent - lo_extent, entries[0][1] * 0.05)
    x = np.linspace(max(lo_extent - spread * 0.5, 1.0), hi_extent + spread * 0.5, 400)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, median, sigma, _, _ in entries:
        if sigma < 1e-9:
            ax.axvline(median, linestyle="--", label=label)
            continue
        mu = np.log(median)
        pdf = np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))
        ax.plot(x, pdf, label=label)

    ax.set_title(title)
    ax.set_xlabel("USD")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def _plot_total_carrying_costs(schedule: pd.DataFrame, ci: tuple[float, float], title: str, save_path: Path) -> Path:
    """Simple monthly median+confidence-band plot of
    carrying_costs.compute_total_carrying_costs_schedule's own combined total -- saved in the
    PARENT results/<scenario>/ folder, not the carrying_costs/ subfolder (the user's own explicit
    call: every per-variable plot lives in the subfolder, this one total summary lives one level up)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(schedule.index, schedule["median"], color="C1", label="Projected Median")
    ax.fill_between(
        schedule.index,
        schedule["lo"],
        schedule["hi"],
        color="C1",
        alpha=0.2,
        label=f"{ci[0] * 100:.0f}-{ci[1] * 100:.0f}% Band",
    )
    ax.set_title(title)
    ax.set_xlabel("Months from Closing")
    ax.set_ylabel("USD/month")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def save_carrying_cost_result_plots(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Path]:
    """Saves this scenario's own SCALED carrying-cost plots into results/<scenario>/carrying_costs/
    (with ONE exception -- the total, see below). Every plot reads DIRECTLY from each prior's own
    saved `fit.json` (carrying_costs.load_carrying_cost_prior_fit), at that prior's own FULL saved
    resolution -- never a separate re-projection -- so every plot here is guaranteed to show exactly
    what's in fit.json, and to look like (a truncated slice of) that prior's own model.png (fixed
    2026-09-30: an earlier version sampled only 10 annual points for insurance/maintenance/
    special_assessment/utilities, which looked visibly coarser than model.png's own dense curve even
    though the underlying numbers matched -- see utils.prior_utils.bake_monthly_periods).

    insurance/maintenance/special_assessment_expected/utilities each get
    _plot_annual_projection_with_final_distribution (the full monthly curve on the left, a
    final-year distribution snapshot on the right -- every plot here shows a REAL distribution, not
    just a median, the user's own catch, 2026-09-30). hoa_dues gets `plotting.forecast_plots.
    plot_periods_stairstep` (the SAME stair-step drawer market_value's own model.png uses) -- HOA
    dues are billed monthly and held flat for 11 months then jump every January (the association
    charge), structurally identical in shape to market_value's own reassessment stair-step, so no
    dedicated function is needed any more. property_tax keeps its own per-year distribution plot
    (_plot_period_distributions), grouped by market_value's own underlying saved period (rescaled to
    tax dollars via carrying_costs.market_value_periods_to_tax_periods), not by calendar year.

    No market_value plot of its own is saved here (2026-09-29) -- market_value is an internal input to
    property_tax, not itself a carrying cost paid; its own unscaled sanity-check plots live under
    models/carrying_cost_priors/informed/market_value/plots/ instead (utils.prior_utils.
    build_market_value_informed_model).

    ONE EXCEPTION to the "carrying_costs/ subfolder" rule: `total_carrying_costs.png` (see
    carrying_costs.compute_total_carrying_costs_schedule -- the combined MONTHLY total across
    property_tax/hoa_dues/insurance/maintenance/utilities, EXCLUDING special_assessment) is saved in
    the PARENT results/<scenario>/ folder directly, per the user's own explicit request -- it's the
    scenario's own headline carrying-cost number, not one variable among the rest.

    Returns {name: path} for every plot saved.
    """
    deck = deck if deck is not None else load_deck()
    out_dir = scenario_results_dir(scenario)
    horizon_years = scenario["horizon_years"]
    horizon_months = horizon_years * 12
    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])
    outer_ci = (0.01, 0.99)

    paths: dict[str, Path] = {}
    for result_name, prior_name, title, ylabel, aggregate_final_year in (
        ("insurance", "insurance_annual", "Insurance — Scenario Projection", "USD/year", False),
        ("maintenance", "maintenance_annual", "Maintenance — Scenario Projection", "USD/year", False),
        (
            "special_assessment_expected",
            "special_assessment",
            "Special Assessment (Expected) — Scenario Projection",
            "USD/year",
            False,
        ),
        ("utilities", "utilities", "Utilities — Scenario Projection", "USD/month", True),
    ):
        periods = load_carrying_cost_prior_fit(prior_name)["periods"]
        paths[result_name] = _plot_annual_projection_with_final_distribution(
            periods,
            horizon_months,
            ci,
            outer_ci,
            title,
            ylabel,
            save_path=out_dir / f"{result_name}.png",
            aggregate_final_year=aggregate_final_year,
        )

    hoa_fit = load_carrying_cost_prior_fit("hoa_dues_annual")
    paths["hoa_dues"] = plot_periods_stairstep(
        hoa_fit["periods"],
        ci,
        outer_ci,
        horizon_months,
        title="HOA Dues (Monthly) — Scenario Projection\nJanuary payments include the annual association charge",
        save_path=out_dir / "hoa_dues.png",
    )

    change_magnitude = load_market_value_change_magnitude()
    tax_periods = market_value_periods_to_tax_periods(change_magnitude["periods"], deck)
    paths["property_tax"] = _plot_period_distributions(
        tax_periods,
        horizon_years,
        ci,
        title="Property Tax — Distribution by Year",
        save_path=out_dir / "property_tax.png",
    )

    total_schedule = compute_total_carrying_costs_schedule(scenario, deck)
    paths["total_carrying_costs"] = _plot_total_carrying_costs(
        total_schedule,
        ci,
        title="Total Carrying Costs — Scenario Projection\n(excludes special assessment -- see its own deferred rare-event treatment)",
        save_path=out_dir.parent / "total_carrying_costs.png",
    )
    return paths
