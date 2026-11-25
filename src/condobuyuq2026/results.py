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
    compute_special_assessment,
    compute_total_carrying_costs_schedule,
    load_carrying_cost_prior_fit,
    load_market_value_change_magnitude,
    market_value_periods_to_tax_periods,
    resolve_period,
)
from condobuyuq2026.input_deck import get_report_quantiles, load_deck
from condobuyuq2026.paths import RESULTS_DIR
from condobuyuq2026.plotting.forecast_plots import (
    plot_periods_stairstep,
    plot_special_assessment_cost,
)


def scenario_results_dir(scenario: dict[str, Any]) -> Path:
    """results/buy_h<horizon_years>y_p<purchase_price>_<residency_state>/carrying_costs/ -- a
    readable folder name from the scenario's own SWEPT fields (the only ones that can distinguish
    two scenarios -- everything else is a fixed deck constant). Not a stable id across deck edits
    that change the sweep values themselves, but stable across repeated runs of the same deck."""
    label = f"buy_h{scenario['horizon_years']}y_p{scenario['purchase_price']}_{scenario['residency_state']}"
    path = RESULTS_DIR / label / "carrying_costs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reference_results_dir(reference_id: int, reference_case: dict[str, Any]) -> Path:
    """results/rent_ref<reference_id>_h<horizon_years>y_<residency_state>/ -- one folder per reference
    case (scenarios.get_scenario_set), the rent-branch counterpart of scenario_results_dir. The id is
    first-seen order, so it's stable across repeated runs of the same deck only."""
    label = f"rent_ref{reference_id}_h{reference_case['horizon_years']}y_{reference_case['residency_state']}"
    path = RESULTS_DIR / label
    path.mkdir(parents=True, exist_ok=True)
    return path


def _year_band(periods: list[dict[str, Any]], year: int, z_lo: float, z_hi: float) -> tuple[float, float, float]:
    """(median, lo, hi) for one calendar YEAR, summed over however many saved periods actually fall
    in it -- works identically whether an entry has ONE period per year (insurance_annual/
    maintenance_annual) or TWELVE (utilities' own monthly periods). Summing each
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
    (_year_band) to get "how much do I pay this year," but insurance_annual/maintenance_annual's
    periods each already hold a CONTINUOUS ANNUAL ESTIMATE sampled monthly --
    summing 12 of those over-counts by ~12x, since it's the same annual figure re-evaluated 12 times,
    not 12 independent months' worth of cost). True (utilities/hoa-shaped monthly bills): sum the
    final year's own periods via _year_band. False (a continuously-evolving annual estimate): just
    look up the single period covering the final month, the same query
    carrying_costs._project_carrying_cost_prior itself uses."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    z_lo, z_hi = norm.ppf(ci)
    z_hi_outer = norm.ppf(outer_ci[1])

    z_lo_outer = norm.ppf(outer_ci[0])
    months, medians, los, his, los_outer, his_outer = [], [], [], [], [], []
    for period in periods:
        if period["start_month"] >= horizon_months:
            break
        known, projected, sigma = period["known_value"], period["projected_value"], period["growth_sigma"]
        months.append(period["start_month"])
        medians.append(known + projected)
        los.append(known + projected * np.exp(z_lo * sigma))
        his.append(known + projected * np.exp(z_hi * sigma))
        los_outer.append(known + projected * np.exp(z_lo_outer * sigma))
        his_outer.append(known + projected * np.exp(z_hi_outer * sigma))

    fig, (ax_line, ax_dist) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    ax_line.plot(months, medians, color="C1", label="Projected Median")
    ax_line.fill_between(months, los, his, color="C1", alpha=0.2, label=f"{ci[0] * 100:.0f}-{ci[1] * 100:.0f}% Band")
    outer_label = f"{outer_ci[0] * 100:.0f}-{outer_ci[1] * 100:.0f}% Band"
    ax_line.plot(months, los_outer, linestyle="--", color="C1", alpha=0.6, linewidth=1, label=outer_label)
    ax_line.plot(months, his_outer, linestyle="--", color="C1", alpha=0.6, linewidth=1)
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


def _plot_total_carrying_costs(
    schedule: pd.DataFrame, ci: tuple[float, float], outer_ci: tuple[float, float], title: str, save_path: Path
) -> Path:
    """Monthly STAIR-STEP plot of carrying_costs.compute_total_carrying_costs_schedule's combined
    total: each month drawn as its own flat box (confidence band) with a flat median bar, like
    hoa_dues.png -- redesigned 2026-11-22 at the user's request. The old continuous line joined
    consecutive months with diagonals, so the one-month January HOA association charge read as a sharp
    spike in a continuous (daily-looking) cost, when every value here is one month's total. Schedule
    row m (1..horizon_months) is scenario month m - 1, drawn over [m - 1, m].

    Saved in the PARENT results/<scenario>/ folder, not the carrying_costs/ subfolder (the user's own
    explicit call: every per-variable plot lives in the subfolder, this one total summary lives one
    level up)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    starts = schedule.index.to_numpy() - 1
    ends = schedule.index.to_numpy()
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (start, end, lo, hi) in enumerate(zip(starts, ends, schedule["lo"], schedule["hi"], strict=True)):
        ax.fill_between(
            [start, end], [lo, lo], [hi, hi], color="C1", alpha=0.2,
            label=f"{ci[0] * 100:.0f}-{ci[1] * 100:.0f}% Band" if i == 0 else None,
        )
    ax.hlines(schedule["median"], starts, ends, color="C1", linewidth=2, label="Projected Median")
    outer_label = f"{outer_ci[0] * 100:.0f}-{outer_ci[1] * 100:.0f}% Band"
    # "_"-prefixed labels are matplotlib's convention for "leave out of the legend" (one entry per band).
    for column, label in (("lo_outer", outer_label), ("hi_outer", "_hi_outer")):
        ax.hlines(schedule[column], starts, ends, colors="C1", alpha=0.6, linewidth=1, linestyles="--", label=label)
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

    special_assessment gets ONE plot of its own, special_assessment_cost.png: the exact discrete
    total-cost distribution over this scenario's horizon (carrying_costs.compute_special_assessment),
    subtitled as excluded from the totals; skipped entirely when annual_rate is 0. The P(N = k) count
    plot was REMOVED from results (2026-11-22, user decision): it was just one horizon's column of the
    model folder's own special_assessment/plots/model.png.

    insurance/maintenance/utilities each get
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

    special_assessment = (scenario.get("carrying_costs") or {}).get("special_assessment") or compute_special_assessment(
        scenario, deck
    )
    if special_assessment["enabled"]:  # annual_rate: 0 is the deck's off switch -- nothing to plot
        horizon_label = f"{horizon_years}-Year Horizon (λ = {special_assessment['annual_rate']:g}/yr)"
        paths["special_assessment_cost"] = plot_special_assessment_cost(
            special_assessment["outcomes"],
            special_assessment["k_max"],
            title=f"Special Assessments — Total Cost Distribution, {horizon_label}, timing: {special_assessment['timing']}",
            save_path=out_dir / "special_assessment_cost.png",
        )

    total_schedule = compute_total_carrying_costs_schedule(scenario, deck, outer_ci)
    paths["total_carrying_costs"] = _plot_total_carrying_costs(
        total_schedule,
        ci,
        outer_ci,
        title="Total Carrying Costs — Scenario Projection\n(excludes special assessments -- see their own tail-risk plots)",
        save_path=out_dir.parent / "total_carrying_costs.png",
    )
    return paths


# ---------------------------------------------------------------------------------------------------
# rental_operations result plots (section 7 plan, build step 5)
# ---------------------------------------------------------------------------------------------------


def _draw_monthly_stairstep(
    ax, monthly: pd.DataFrame, prefix: str, ci: tuple[float, float], outer_ci: tuple[float, float]
) -> None:
    """Draws one monthly quantity from rental_operations.compute_rental_operations' "monthly" DataFrame
    (columns <prefix>_median/_lo/_hi/_lo_outer/_hi_outer, index = scenario month h) as a stair-step: each
    month its own flat box (main band), flat median bar and dashed 1-99% bars, over [h, h + 1] -- the
    same style as total_carrying_costs.png. Months with no value (NaN: e.g. no bookings, so no ADR) are
    left blank."""
    data = monthly[[f"{prefix}_{band}" for band in ("median", "lo", "hi", "lo_outer", "hi_outer")]].dropna()
    starts = data.index.to_numpy()
    ends = starts + 1
    for i, (start, end, lo, hi) in enumerate(zip(starts, ends, data[f"{prefix}_lo"], data[f"{prefix}_hi"], strict=True)):
        ax.fill_between(
            [start, end], [lo, lo], [hi, hi], color="C1", alpha=0.2,
            label=f"{ci[0] * 100:.0f}-{ci[1] * 100:.0f}% Band" if i == 0 else None,
        )
    ax.hlines(data[f"{prefix}_median"], starts, ends, color="C1", linewidth=2, label="Projected Median")
    outer_label = f"{outer_ci[0] * 100:.0f}-{outer_ci[1] * 100:.0f}% Band"
    for column, label in ((f"{prefix}_lo_outer", outer_label), (f"{prefix}_hi_outer", "_hi_outer")):
        ax.hlines(data[column], starts, ends, colors="C1", alpha=0.6, linewidth=1, linestyles="--", label=label)
    ax.set_xlabel("Months from Closing")
    ax.grid(alpha=0.3)


def _plot_rental_monthly(
    monthly: pd.DataFrame,
    prefix: str,
    ci: tuple[float, float],
    outer_ci: tuple[float, float],
    title: str,
    ylabel: str,
    save_path: Path,
    cap: list[float] | None = None,
    final_year: dict[str, Any] | None = None,
    zoom_months: int = 12,
) -> Path:
    """Monthly stair-step of one rental quantity over the whole horizon (top), plus a zoom on the first
    `zoom_months` months with calendar-month labels (bottom) -- the full horizon is too dense to read
    month by month. Optional `cap` (nights you leave available, per scenario month) is drawn as a dotted
    step line. Optional `final_year` ({"median", "hi_outer", "year", optional "what" for its subtitle})
    adds a top-right panel with that year's TOTAL distribution (lognormal, as computed by
    compute_rental_operations / compute_personal_use). Also used for personal_use's lodging cost."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14 if final_year is not None else 12, 9))
    grid = fig.add_gridspec(2, 2 if final_year is not None else 1, width_ratios=[3, 1] if final_year is not None else [1])
    ax = fig.add_subplot(grid[0, 0])
    ax_zoom = fig.add_subplot(grid[1, :])
    zoom = monthly.loc[monthly.index < zoom_months]
    for axis, data in ((ax, monthly), (ax_zoom, zoom)):
        _draw_monthly_stairstep(axis, data, prefix, ci, outer_ci)
        if cap is not None:
            edges = np.append(data.index.to_numpy(), data.index[-1] + 1)
            axis.stairs(cap[: len(data)], edges, baseline=None, color="k", linewidth=1, linestyle=":",
                        label="Cap: nights you leave available")
        axis.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax_zoom.set_title(f"First {zoom_months} months (month 0 = closing month)", fontsize=10)
    ax_zoom.set_xticks(zoom.index.to_numpy() + 0.5,
                       [f"{h}\n{m.capitalize()}" for h, m in zip(zoom.index, zoom["calendar_month"], strict=True)])
    if final_year is not None:
        ax_dist = fig.add_subplot(grid[0, 1])
        median, hi_outer = final_year["median"], final_year["hi_outer"]
        z_hi_outer = norm.ppf(outer_ci[1])
        sigma = float(np.log(hi_outer / median) / z_hi_outer) if hi_outer > median > 0 else 0.0
        what = final_year.get("what", "Total Gross Revenue")
        _draw_final_distribution(ax_dist, median, sigma, outer_ci, subtitle=f"Year {final_year['year']} {what}")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def _plot_net_rental_income(
    annual: pd.DataFrame, fees: dict[str, float], ci: tuple[float, float], outer_ci: tuple[float, float], save_path: Path
) -> Path:
    """Per scenario year: EXPECTED gross revenue as a stacked bar split into net rental income (bottom) and
    each deduction (management fee, channel commissions); net's own main band as a solid whisker and its
    1-99% band as a dashed whisker. Title and caption state every deduction explicitly."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    years = annual.index.to_numpy()
    net = annual["net_mean"].to_numpy()
    mgmt = annual["mgmt_fee_mean"].to_numpy()
    commissions = annual["channel_commissions_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5.5))
    width = 0.6
    ax.bar(years, net, width, color="C0", label="Net rental income (expected)")
    ax.bar(years, mgmt, width, bottom=net, color="C1", alpha=0.6,
           label=f"Management fee ({fees['mgmt_fee_fraction']:.1%} of gross after commissions)")
    if commissions.any():
        ax.bar(years, commissions, width, bottom=net + mgmt, color="C7", alpha=0.6,
               label=f"Channel commissions ({fees['channel_commission_fraction']:.1%} of gross)")
    ax.errorbar(years, annual["net_median"], yerr=[annual["net_median"] - annual["net_lo"], annual["net_hi"] - annual["net_median"]],
                fmt="o", color="k", markersize=4, capsize=4, linewidth=1.2,
                label=f"Net: median and {ci[0] * 100:.0f}-{ci[1] * 100:.0f}% band")
    outer = ax.errorbar(years + 0.18, annual["net_median"],
                        yerr=[annual["net_median"] - annual["net_lo_outer"], annual["net_hi_outer"] - annual["net_median"]],
                        fmt="none", ecolor="k", alpha=0.5, capsize=3, linewidth=1,
                        label=f"Net: {outer_ci[0] * 100:.0f}-{outer_ci[1] * 100:.0f}% band")
    outer[-1][0].set_linestyle("--")
    ax.set_xticks(years)
    ax.set_xlim(years[0] - 0.8, years[-1] + 0.8)
    ax.set_xlabel("Scenario year")
    ax.set_ylabel("USD/year")
    ax.set_title(
        "Net Rental Income = Gross Rental Revenue − Channel Commissions − Management Fee\n"
        f"(bar height = expected gross; net = gross × (1 − {fees['channel_commission_fraction']:.3g}) × "
        f"(1 − {fees['mgmt_fee_fraction']:.3g}) = {fees['keep_fraction']:.1%} of gross)"
    )
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize="small", loc="upper left")
    fig.text(
        0.01, 0.005,
        "Lodging tax is NOT deducted: it is charged to guests on top of the nightly rate and remitted by the "
        "manager/platform.\nCarrying costs and income taxes are NOT subtracted here (see total_carrying_costs.png "
        "and the taxes stage).",
        fontsize=8, ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def save_rental_operations_result_plots(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Path]:
    """Saves this scenario's rental plots (section 7):
      results/<scenario>/rental_operations/occupancy.png      rented nights per month (with your cap)
      results/<scenario>/rental_operations/adr.png            ADR per month, grown by the adr latent
      results/<scenario>/rental_operations/gross_revenue.png  gross per month + final-year total distribution
      results/<scenario>/net_rental_income.png                per year: gross split into net + each deduction
    All read scenario["rental_operations"] (rental_operations.compute_rental_operations), computed here if
    the runner hasn't stored it yet."""
    from condobuyuq2026.rental_operations import compute_rental_operations, rentable_nights_by_month

    deck = deck if deck is not None else load_deck()
    rental = scenario.get("rental_operations") or compute_rental_operations(scenario, deck)
    monthly, annual = rental["monthly"], rental["annual"]
    ci, outer_ci = rental["ci"], rental["outer_ci"]
    scenario_dir = scenario_results_dir(scenario).parent
    out_dir = scenario_dir / "rental_operations"
    rentable = rentable_nights_by_month(deck)
    final = annual.iloc[-1]

    return {
        "occupancy": _plot_rental_monthly(
            monthly, "nights", ci, outer_ci,
            title="Nights Rented per Month — Scenario Projection\n(fitted occupancy × nights you leave available, capped)",
            ylabel="Nights", save_path=out_dir / "occupancy.png",
            cap=[rentable[m] for m in monthly["calendar_month"]],
        ),
        "adr": _plot_rental_monthly(
            monthly, "adr", ci, outer_ci,
            title="ADR per Month — Scenario Projection\n(fitted ADR, closing-date dollars, grown by the adr latent; "
            "blank = no bookings expected)",
            ylabel="USD per rented night", save_path=out_dir / "adr.png",
        ),
        "gross_revenue": _plot_rental_monthly(
            monthly, "gross", ci, outer_ci,
            title="Gross Rental Revenue per Month — Scenario Projection\n(nights × ADR, before lodging tax and fees)",
            ylabel="USD/month", save_path=out_dir / "gross_revenue.png",
            final_year={"median": float(final["gross_median"]), "hi_outer": float(final["gross_hi_outer"]),
                        "year": int(annual.index[-1])},
        ),
        "net_rental_income": _plot_net_rental_income(
            annual, rental["fees"], ci, outer_ci, save_path=scenario_dir / "net_rental_income.png"
        ),
    }


def save_personal_use_result_plots(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Path]:
    """Saves this scenario's personal_use plot (section 8):
      results/<scenario>/personal_use/alternate_lodging_cost.png  alternate lodging per month (charged as
                                                                  entered) + final-year total distribution
    Reads scenario["personal_use"] (personal_use.compute_personal_use), computed here if the runner hasn't
    stored it yet."""
    from condobuyuq2026.personal_use import compute_personal_use

    deck = deck if deck is not None else load_deck()
    personal_use = scenario.get("personal_use") or compute_personal_use(scenario, deck)
    monthly, annual = personal_use["monthly"], personal_use["annual"]
    final = annual.iloc[-1]
    latent = personal_use["inputs"]["alternate_lodging_growth_latent"]
    out_dir = scenario_results_dir(scenario).parent / "personal_use"
    final_year = None
    if final["lodging_median"] > 0:
        final_year = {"median": float(final["lodging_median"]), "hi_outer": float(final["lodging_hi_outer"]),
                      "year": int(annual.index[-1]), "what": "Total Alternate Lodging"}
    return {
        "alternate_lodging_cost": _plot_rental_monthly(
            monthly, "lodging", personal_use["ci"], personal_use["outer_ci"],
            title="Alternate Lodging Cost per Month — Scenario Projection (buy branch, personal cost)\n"
            f"(your entered amount, charged in full every month — no proration — grown by the {latent} latent)",
            ylabel="USD/month", save_path=out_dir / "alternate_lodging_cost.png", final_year=final_year,
        ),
    }


def save_rent_reference_result_plots(
    reference_id: int, reference_case: dict[str, Any], rent_reference: dict[str, Any]
) -> dict[str, Path]:
    """Saves this reference case's rent_reference plot (section 10):
      results/rent_ref<id>_h<h>y_<residency>/rent_reference/alt_lease_cost.png  the rent branch's lease per
          month (charged as entered, stepped up once a year at each closing anniversary) + final-year total
    `rent_reference` is rent_reference.compute_rent_reference's result."""
    monthly, annual = rent_reference["monthly"], rent_reference["annual"]
    final = annual.iloc[-1]
    latent = rent_reference["inputs"]["lease_growth_latent"]
    out_dir = reference_results_dir(reference_id, reference_case) / "rent_reference"
    final_year = None
    if final["lease_median"] > 0 and final["lease_hi_outer"] > final["lease_median"]:
        final_year = {"median": float(final["lease_median"]), "hi_outer": float(final["lease_hi_outer"]),
                      "year": int(annual.index[-1]), "what": "Total Lease Cost"}
    return {
        "alt_lease_cost": _plot_rental_monthly(
            monthly, "lease", rent_reference["ci"], rent_reference["outer_ci"],
            title="Lease Cost per Month — Rent Branch (you don't buy)\n"
            f"(your entered all-in amount, charged in full every month; steps up once a year at each closing "
            f"anniversary, by the {latent} latent)",
            ylabel="USD/month", save_path=out_dir / "alt_lease_cost.png", final_year=final_year,
        ),
    }


def save_tax_result_plots(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Path]:
    """Saves this scenario's taxes plot (section 9):
      results/<scenario>/taxes/tax_effects_by_year.png  the condo's tax effect each year + at the sale,
                                                        split federal / property state / residence state
    Each bar is the household's tax WITH the condo minus WITHOUT it (taxes.py): above zero the condo
    adds tax, below zero it saves tax. Reads scenario["taxes"] (taxes.compute_owner_taxes), computed
    here if the runner hasn't stored it yet."""
    from condobuyuq2026.taxes import compute_owner_taxes

    deck = deck if deck is not None else load_deck()
    taxes = scenario.get("taxes") or compute_owner_taxes(scenario, deck)
    schedule, sale, jurisdiction, rules = taxes["annual_schedule"], taxes["sale"], taxes["jurisdiction"], taxes["rules"]
    lines = [
        ("federal", "Federal", "C0"),
        ("property_state", jurisdiction.property_state.name, "C2"),
        ("residence_state", f"{jurisdiction.residence_state.name} (residence)", "C4"),
    ]
    lines = [
        line
        for line in lines
        if line[0] != "residence_state"
        or jurisdiction.residence_state.has_income_tax
        and not jurisdiction.resident_of_property_state
    ]
    labels = [str(int(year)) for year in schedule["tax_year"]] + ["Sale"]
    values = {key: [*schedule[key].tolist(), sale[key]] for key, _, _ in lines}
    x = np.arange(len(labels))

    fig, (ax, ax_rent) = plt.subplots(2, 1, figsize=(11, 8.5), height_ratios=[3, 2], sharex=True)
    pos_base, neg_base = np.zeros(len(x)), np.zeros(len(x))
    for key, label, color in lines:
        v = np.asarray(values[key], dtype=float)
        base = np.where(v >= 0, pos_base, neg_base)
        ax.bar(x, v, bottom=base, width=0.7, color=color, edgecolor="white", linewidth=1.5, label=label)
        pos_base += np.where(v >= 0, v, 0)
        neg_base += np.where(v < 0, v, 0)
    net = pos_base + neg_base
    ax.scatter(x, net, marker="_", s=400, color="k", linewidths=2, zorder=3, label="Net effect")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axvline(len(x) - 1.5, color="gray", linewidth=0.8, linestyle=":")
    span = max(abs(float(net.max())), abs(float(net.min())), 1.0)
    ax.annotate(
        f"${net[-1]:,.0f}",
        (x[-1], net[-1]),
        xytext=(0, 8 if net[-1] >= 0 else -14),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
    ax.set_ylim(min(float(neg_base.min()), 0) - 0.12 * span, max(float(pos_base.max()), 0) + 0.12 * span)
    ax.set_ylabel("USD (above 0 = the condo adds tax; below 0 = it saves tax)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    passive = "" if rules["passive_status"] == "n/a" else f", losses {rules['passive_status']}"
    ax.set_title(
        f"Condo Tax Effect by Year — {jurisdiction.label}; residence {jurisdiction.residence_state.name}\n"
        f"total ${taxes['total_tax']:,.0f} (annual ${schedule['tax'].sum():,.0f} + sale ${sale['tax']:,.0f}); "
        f"use class {rules['tax_use_class']}, depreciation {rules['depreciation_years']:g} yr{passive}",
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=9)

    # Lower panel: where the rent went -- gross rent vs. the part that became taxable, and the deductions
    # carried forward (280A, lost at sale) or losses suspended (passive, released at sale).
    years = np.arange(len(schedule))
    ax_rent.bar(years - 0.18, schedule["gross_rent"], width=0.36, color="C7", edgecolor="white", linewidth=1.5,
                label="Expected gross rent")
    ax_rent.bar(years + 0.18, schedule["taxable_rental"], width=0.36, color="C1", edgecolor="white", linewidth=1.5,
                label="Taxable rental income (in AGI)")
    carried = schedule["carry_280a_balance"] + schedule["carry_280a_depr_balance"]
    if carried.abs().sum() > 0:
        ax_rent.plot(years, carried, color="C3", marker="o", linewidth=2, label="280A deductions carried forward (lost at sale)")
    if schedule["suspended_passive_balance"].abs().sum() > 0:
        ax_rent.plot(years, schedule["suspended_passive_balance"], color="C9", marker="s", linewidth=2,
                     label="Suspended passive losses (released at sale)")
    ax_rent.axhline(0, color="k", linewidth=0.8)
    ax_rent.axvline(len(x) - 1.5, color="gray", linewidth=0.8, linestyle=":")
    ax_rent.set_xticks(x, labels)
    ax_rent.set_xlabel("Tax year")
    ax_rent.set_ylabel("USD per year")
    ax_rent.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax_rent.grid(axis="y", alpha=0.3)
    ax_rent.set_axisbelow(True)
    ax_rent.set_title("Where the rent went", fontsize=10)
    ax_rent.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_dir = scenario_results_dir(scenario).parent / "taxes"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "tax_effects_by_year.png"
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return {"tax_effects_by_year": save_path}
