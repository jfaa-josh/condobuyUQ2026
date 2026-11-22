"""Plotting helpers for a fitted forecast_models process -- a visual sanity check on a term
structure (median + confidence band) projected from a fitted model, against the historical data it
was fit from, or on the rate's own path with no price level involved.

Every plot here draws two bands: the "main" one (projection's own "lo"/"hi", shaded -- whatever CI
the caller chose, typically run.report_quantiles' outer two) and a wider reference band (dashed,
"outer_projection", by convention the model's own (0.01, 0.99) band -- see each build script) so a
1-99% range is always visible for scale, regardless of what the shaded band's own CI happens to be.
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


def _plot_outer_band(ax, x, outer_projection: pd.DataFrame) -> None:
    """Draws outer_projection's lo/hi as dashed reference lines -- the wide (0.01, 0.99) band every
    plot in this module shows alongside its own (typically narrower) shaded confidence band."""
    ax.plot(x, outer_projection["lo"], linestyle="--", color="C1", alpha=0.6, linewidth=1, label="1-99% Band")
    ax.plot(x, outer_projection["hi"], linestyle="--", color="C1", alpha=0.6, linewidth=1)


def plot_term_structure(
    history: pd.Series, projection: pd.DataFrame, outer_projection: pd.DataFrame, title: str, save_path: Path
) -> Path:
    """Plots history (a monthly-indexed Series, e.g. from manual_utils.to_monthly_series) as a
    scatter, and projection (an h_months-indexed DataFrame with "median"/"lo"/"hi" columns, e.g.
    from ou_fitting.project_term_structure) as a median line with a shaded confidence band,
    starting from history's last observed month, plus outer_projection (same shape, wider CI --
    see module docstring) as a dashed reference band. Saves to save_path (creating parent
    directories as needed) and returns the path written.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    projection_dates = pd.date_range(history.index[-1], periods=len(projection), freq="MS")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(history.index, history.values, s=8, label="History")
    ax.plot(projection_dates, projection["median"], color="C1", label="Projected Median")
    ax.fill_between(
        projection_dates, projection["lo"], projection["hi"], color="C1", alpha=0.2, label="Confidence Band"
    )
    _plot_outer_band(ax, projection_dates, outer_projection)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_derived_latent_model(
    component_projections: dict[str, pd.DataFrame],
    component_weights: dict[str, float],
    combined_projection: pd.DataFrame,
    combined_outer_projection: pd.DataFrame,
    title: str,
    save_path: Path,
) -> Path:
    """Plots each named component's own REAL-UNIT monthly-rate projection (e.g. from
    ou_fitting.project_monthly_rate -- the same curve as that component's own monthly_rate.png,
    UNWEIGHTED) as a median line + shaded band, at an alpha proportional to
    component_weights[label] -- not the curve's VALUE scaled by weight, just its visual
    prominence: a 0-weight component doesn't show up at all, a high-weight one is clearly visible,
    and each stays directly comparable to its own monthly_rate.png. Superimposes the combined
    derived-latent model (see manual_utils.build_derived_latent_model) as a fully-opaque thick
    black line + band, drawn on top so it's always the most readable, plus
    combined_outer_projection as a dashed reference band (see module docstring). No scatter/
    history -- like plot_monthly_rate, a pure model-shape view. Components won't necessarily "line
    up" with each other or the combined line (they're different real quantities, only linearly
    blended for comparison) -- that's expected, not a bug.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, proj in component_projections.items():
        weight = component_weights[label]
        if abs(weight) < 1e-9:
            continue  # a true zero-weight component contributes nothing -- don't even legend it
        alpha = min(abs(weight), 1.0)
        display_label = label.replace("_", " ").title()
        ax.plot(proj.index, proj["median"], linewidth=1.5, alpha=alpha, label=f"{display_label} (Weight {weight:.2g})")
        ax.fill_between(proj.index, proj["lo"], proj["hi"], alpha=alpha * 0.25)
    ax.plot(
        combined_projection.index,
        combined_projection["median"],
        color="black",
        linewidth=3,
        alpha=1.0,
        label="Combined",
        zorder=10,
    )
    ax.fill_between(
        combined_projection.index,
        combined_projection["lo"],
        combined_projection["hi"],
        color="black",
        alpha=0.15,
        zorder=9,
    )
    _plot_outer_band(ax, combined_outer_projection.index, combined_outer_projection)
    ax.set_title(title)
    ax.set_xlabel("Months Ahead")
    ax.set_ylabel("Monthly Price Increase (Fraction)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_price_level_projection(
    projection: pd.DataFrame, outer_projection: pd.DataFrame, title: str, save_path: Path, ylabel: str = "USD"
) -> Path:
    """Plots a projected price LEVEL (e.g. from ou_fitting.project_term_structure) as a median line
    with a shaded confidence band, x-axis in months-ahead, plus outer_projection as a dashed
    reference band (see module docstring). No history/scatter -- unlike plot_term_structure, this
    is for a hand-elicited level prior (carrying_costs.py's hoa_dues_annual/insurance_annual/
    maintenance_annual/utilities/special_assessment) with no real observed data series to plot
    against, only the projection itself."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(projection.index, projection["median"], color="C1", label="Projected Median")
    ax.fill_between(
        projection.index, projection["lo"], projection["hi"], color="C1", alpha=0.2, label="Confidence Band"
    )
    _plot_outer_band(ax, projection.index, outer_projection)
    ax.set_title(title)
    ax.set_xlabel("Months Ahead")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def _draw_periods_stairstep(
    ax,
    periods: list[dict[str, Any]],
    ci: tuple[float, float],
    outer_ci: tuple[float, float],
    horizon_months: int,
) -> None:
    """Draws any prior's own saved {start_month, end_month, known_value, projected_value,
    growth_sigma} periods onto an existing Axes -- already in ACTUAL USD (no scale factor to apply
    here) -- each period drawn as its OWN independent set of plot calls (a 2-point horizontal
    line/fill per period, never one array spanning several periods) so consecutive periods show a
    clean vertical break instead of a continuous line's usual diagonal connecting the last point of
    one period to the first of the next -- misleading whenever a period's own value genuinely holds
    flat and then JUMPS (market_value's real reassessment cycle, hoa_dues_annual's own combined
    dues+association total, held flat for a year then jumping) rather than drifting continuously.

    A period's own median/lo/hi at a given z-score: median = known_value + projected_value, lo/hi =
    known_value + projected_value*exp(z*sigma) -- see utils.prior_utils.bake_annual_periods for the
    derivation. Factored out of plot_periods_stairstep so plot_market_value_correction_diagnostic can
    draw the same stair-step on one of its own subplot Axes, and so hoa_dues_annual's own monthly
    periods (utils.prior_utils.bake_hoa_monthly_periods) can reuse this exact same drawing.
    """
    z_lo, z_hi = norm.ppf(ci)
    z_lo_outer, z_hi_outer = norm.ppf(outer_ci)

    for i, period in enumerate(periods):
        start = period["start_month"]
        if start >= horizon_months:
            break
        end = horizon_months if period["end_month"] is None else min(period["end_month"], horizon_months)
        known, projected, sigma = period["known_value"], period["projected_value"], period["growth_sigma"]
        x = [start, end]
        median = known + projected
        lo = known + projected * np.exp(z_lo * sigma)
        hi = known + projected * np.exp(z_hi * sigma)
        lo_outer = known + projected * np.exp(z_lo_outer * sigma)
        hi_outer = known + projected * np.exp(z_hi_outer * sigma)
        ax.fill_between(x, [lo, lo], [hi, hi], color="C1", alpha=0.2, label="Confidence Band" if i == 0 else None)
        ax.plot(x, [median, median], color="C1", linewidth=2, label="Projected Median" if i == 0 else None)
        ax.plot(
            x, [lo_outer, lo_outer], linestyle="--", color="C1", alpha=0.6, linewidth=1, label="1-99% Band" if i == 0 else None
        )
        ax.plot(x, [hi_outer, hi_outer], linestyle="--", color="C1", alpha=0.6, linewidth=1)


def plot_periods_stairstep(
    periods: list[dict[str, Any]],
    ci: tuple[float, float],
    outer_ci: tuple[float, float],
    horizon_months: int,
    title: str,
    save_path: Path,
) -> Path:
    """Plots a prior's own saved stair-stepped periods (market_value's real reassessment cycle, or
    hoa_dues_annual's own combined-dues-plus-association-charge total, held flat within each period
    then jumping) -- see _draw_periods_stairstep for the actual drawing. Generic over any periods
    list in that shape, not market_value-specific despite the name of the concept it came from."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    _draw_periods_stairstep(ax, periods, ci, outer_ci, horizon_months)
    ax.set_title(title)
    ax.set_xlabel("Months from Closing")
    ax.set_ylabel("USD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_market_value_calibration(calibration: dict[str, Any], title: str, save_path: Path) -> Path:
    """Plots the Denver-to-Keystone calibration (utils.prior_utils.fit_market_value_calibration) as
    a scatter of the real (Denver window average, Keystone actual value) observations that informed
    it, with the fitted line (slope*x + intercept) drawn across their range -- the natural way to
    show a linear fit, generalizing cleanly whether there are 2 observations (the line passes through
    both exactly) or many (a genuine least-squares fit, points won't all land on the line)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    observations = calibration["observations"]
    denver_values = [o["denver_window_avg"] for o in observations]
    keystone_values = [o["keystone_actual"] for o in observations]
    labels = [str(o["effective_year"]) for o in observations]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(denver_values, keystone_values, color="C0", zorder=3, label="Actuals")
    for x, y, label in zip(denver_values, keystone_values, labels, strict=True):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6))

    x_line = np.linspace(min(denver_values) * 0.98, max(denver_values) * 1.02, 50)
    y_line = calibration["slope"] * x_line + calibration["intercept"]
    ax.plot(
        x_line,
        y_line,
        color="C3",
        linestyle="--",
        label=f"Fitted Line (slope={calibration['slope']:,.1f}, intercept={calibration['intercept']:,.0f})",
    )

    ax.set_title(title)
    ax.set_xlabel("Denver Home Price Index (Window Average)")
    ax.set_ylabel("Keystone Assessed Value (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_market_value_correction_diagnostic(
    naive_trajectory: pd.DataFrame,
    periods: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    ci: tuple[float, float],
    outer_ci: tuple[float, float],
    horizon_months: int,
    title: str,
    save_path: Path,
) -> Path:
    """MODEL-DIAGNOSTIC ONLY (models/carrying_cost_priors/informed/market_value/plots/ -- never a
    scenario's own results/ plot): two panels side by side, sharing a y-axis, so the effect of
    fit_market_value_calibration's own slope+intercept line is visible directly in dollar terms and
    over time, not just as a scatter of the two fitting points (see plot_market_value_calibration).

    Left: the NAIVE reference -- home_price's raw OU fit projected as one continuous trajectory,
    scaled by a PLAIN proportional factor (no intercept) so it passes through initial_price exactly
    at closing -- i.e. what the model would say if fit_market_value_calibration's intercept were
    forced to 0 (a pure unit conversion). Right: the ACTUAL saved, linearly-calibrated stair-step
    model (same drawing as plot_periods_stairstep). Both panels overlay the SAME real known
    reassessment values (utils.prior_utils.load_market_value_history, annotated with
    utils.prior_utils.annotate_observation_timing) as a scatter, at each observation's own averaging
    window MIDPOINT -- so it's visible at a glance whether the naive, intercept-free reference
    actually passes near the real data, or whether (as here) it systematically misses it, which is
    the whole reason fit_market_value_calibration fits a line instead of a plain ratio. Not a claim
    the naive panel is "more correct" -- it's a reference baseline for how much the fitted intercept
    is doing to the model's own dynamics.

    Each Actual also gets a thin red dashed "Averaging Window" line spanning window_start_month to
    window_end_month (the real dates the Denver average was actually taken over -- e.g. Jan to the
    following Jun) at that Actual's own y-value, on BOTH panels. The right panel additionally draws a
    thin black dashed "Assessment Period" line spanning period_start_month to period_end_month -- the
    reassessment_frequency_years-long stretch that assessed value actually HELD for real tax purposes
    (distinct from, and always AFTER, the averaging window that produced it).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    naive_projection = naive_trajectory[naive_trajectory.index >= 0]
    naive_history = naive_trajectory[naive_trajectory.index < 0]

    fig, (ax_naive, ax_actual) = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

    ax_naive.plot(naive_projection.index, naive_projection["median"], color="C1", label="Projected Median")
    ax_naive.fill_between(
        naive_projection.index, naive_projection["lo"], naive_projection["hi"], color="C1", alpha=0.2, label="Confidence Band"
    )
    ax_naive.plot(naive_projection.index, naive_projection["lo_outer"], linestyle="--", color="C1", alpha=0.6, linewidth=1, label="1-99% Band")
    ax_naive.plot(naive_projection.index, naive_projection["hi_outer"], linestyle="--", color="C1", alpha=0.6, linewidth=1)
    if len(naive_history) > 0:
        ax_naive.plot(naive_history.index, naive_history["median"], color="C0", alpha=0.5, label="Raw Denver History")
    ax_naive.set_title("Latent Home Price Model\nNote: Actuals are 18 mo av lagging 6 mos")

    _draw_periods_stairstep(ax_actual, periods, ci, outer_ci, horizon_months)
    ax_actual.set_title("Calibrated Model:\nLatent model with linear calibration and averaging to 2 year reporting period")

    for ax in (ax_naive, ax_actual):
        months = [o["month_since_closing"] for o in observations]
        values = [o["assessed_value"] for o in observations]
        ax.scatter(months, values, color="C3", zorder=5, s=40, label="Actuals")
        for month, value, obs in zip(months, values, observations, strict=True):
            ax.annotate(str(obs["effective_year"]), (month, value), textcoords="offset points", xytext=(6, 6))
        for i, obs in enumerate(observations):
            ax.plot(
                [obs["window_start_month"], obs["window_end_month"]],
                [obs["assessed_value"], obs["assessed_value"]],
                color="red",
                linestyle="--",
                linewidth=1,
                label="Averaging Window" if i == 0 else None,
            )
        ax.set_xlabel("Months from Closing")

    for i, obs in enumerate(observations):
        ax_actual.plot(
            [obs["period_start_month"], obs["period_end_month"]],
            [obs["assessed_value"], obs["assessed_value"]],
            color="black",
            linestyle="--",
            linewidth=1,
            label="Assessment Period" if i == 0 else None,
        )

    ax_naive.legend(fontsize="small")
    ax_actual.legend(fontsize="small")
    ax_naive.set_ylabel("USD")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_periods_projection(
    periods: list[dict[str, Any]],
    ci: tuple[float, float],
    outer_ci: tuple[float, float],
    horizon_months: int,
    title: str,
    save_path: Path,
    ylabel: str = "USD",
) -> Path:
    """Plots a prior's own saved MONTHLY periods (see utils.prior_utils.bake_monthly_periods/
    bake_calendar_month_periods) as a CONTINUOUS median+confidence-band line, one point per saved
    period -- DIRECTLY from the saved data, never a fresh ou_fitting.project_term_structure
    recomputation, so this is guaranteed to show exactly what fit.json holds. For an entry with real
    seasonal structure (utilities), this shows the genuine month-to-month oscillation; for a smoothly
    -evolving continuous estimate (insurance_annual/maintenance_annual/special_assessment), it looks
    like a normal drifting-and-widening band, just sampled at full monthly resolution instead of only
    at year marks."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    z_lo, z_hi = norm.ppf(ci)
    z_lo_outer, z_hi_outer = norm.ppf(outer_ci)

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

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(months, medians, color="C1", label="Projected Median")
    ax.fill_between(months, los, his, color="C1", alpha=0.2, label="Confidence Band")
    ax.plot(months, los_outer, linestyle="--", color="C1", alpha=0.6, linewidth=1, label="1-99% Band")
    ax.plot(months, his_outer, linestyle="--", color="C1", alpha=0.6, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Months Ahead")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_utilities_year1_breakdown(year_1: pd.DataFrame, title: str, save_path: Path) -> Path:
    """Two-panel view of utilities' three-category model (utils.prior_utils.build_utilities_prior_model),
    year 1, calendar order Jan..Dec, before growth.
    TOP: stacked monthly cost bars -- flat / occupancy-driven / heating / add-ons -- so it's visible
    which category drives each month's bill.
    BOTTOM: the two drivers -- occupied days (left axis) and occupied heating degree-days per day
    (right axis). Kept in their own panel rather than a secondary axis on the cost chart (user
    decision 2026-11-22): days and degree-days are different units, and stacking both onto the cost
    chart's spare axis made it unreadable.
    year_1: DataFrame indexed by month key with columns flat, occupancy, heating, add_ons,
    occupied_days, hdd_occupied_per_day (see utils.prior_utils.apply_utilities)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [month.capitalize() for month in year_1.index]
    x = np.arange(len(labels))

    fig, (ax_cost, ax_driver) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    bottom = np.zeros(len(labels))
    for column, label, color in (
        ("flat", "Flat", "C7"),
        ("occupancy", "Occupancy-driven", "C0"),
        ("heating", "Heating", "C3"),
        ("add_ons", "Add-ons (new costs)", "C2"),
    ):
        values = year_1[column].to_numpy(dtype=float)
        if column == "add_ons" and not values.any():
            continue
        ax_cost.bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    for xi, total in zip(x, bottom, strict=True):
        ax_cost.annotate(f"\\${total:,.0f}", (xi, total), ha="center", va="bottom", fontsize=8)
    ax_cost.set_title(title)
    ax_cost.set_ylabel("USD/month")
    ax_cost.set_ylim(0, bottom.max() * 1.12)
    ax_cost.legend(fontsize="small", loc="upper center", ncol=4)

    days_line = ax_driver.plot(x, year_1["occupied_days"], marker="o", color="C0", label="Occupied days")
    ax_driver.set_ylabel("Occupied days")
    ax_driver.set_ylim(0, 31)
    ax_hdd = ax_driver.twinx()
    hdd_line = ax_hdd.plot(
        x, year_1["hdd_occupied_per_day"], marker="s", color="C3", linestyle="--", label="Heating degree-days/day (occupied)"
    )
    ax_hdd.set_ylabel("HDD per day")
    ax_hdd.set_ylim(0, max(1.0, float(year_1["hdd_occupied_per_day"].max()) * 1.15))
    lines = days_line + hdd_line
    ax_driver.legend(lines, [line.get_label() for line in lines], fontsize="small", loc="upper center", ncol=2)
    ax_driver.set_xticks(x)
    ax_driver.set_xticklabels(labels)
    ax_driver.set_xlabel("Calendar month (year 1)")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_monthly_rate(projection: pd.DataFrame, outer_projection: pd.DataFrame, title: str, save_path: Path) -> Path:
    """Plots a projected monthly fractional rate (e.g. from ou_fitting.project_monthly_rate) as a
    median line with a shaded confidence band, x-axis in months-ahead, plus outer_projection as a
    dashed reference band (see module docstring). No history/p0 to anchor it to -- a rate doesn't
    need a starting price level, unlike plot_term_structure."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(projection.index, projection["median"], color="C1", label="Projected Median")
    ax.fill_between(
        projection.index, projection["lo"], projection["hi"], color="C1", alpha=0.2, label="Confidence Band"
    )
    _plot_outer_band(ax, projection.index, outer_projection)
    ax.set_title(title)
    ax.set_xlabel("Months Ahead")
    ax.set_ylabel("Monthly Price Increase (Fraction)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# Shown under every special-assessment plot, so a reader never mistakes these for part of the
# carrying-cost totals (see carrying_costs.compute_special_assessment for why they're kept apart).
SPECIAL_ASSESSMENT_PLOT_SUBTITLE = "Excluded from total carrying costs — applied as tail-risk overlay on ΔW in final results."


def _count_label(k: int, k_max: int) -> str:
    """"2" for an ordinary count, "3+" for the k_max entry, which also holds the folded-in chance of
    more than k_max assessments (see carrying_costs.special_assessment_count_probabilities)."""
    return f"{k}+" if k == k_max else str(k)


def plot_special_assessment_count_by_horizon(
    count_table: pd.DataFrame, k_max: int, annual_rate: float, title: str, save_path: Path
) -> Path:
    """Build-time model.png for carrying_costs.special_assessment: P(N = k) against horizon length,
    one line per k. The model itself doesn't depend on any one scenario (only annual_rate and k_max
    from the deck), so this shows every horizon at once; each scenario's own single-horizon view is
    saved by results.py. count_table: index = horizon years, columns = k (0..k_max), values = P(N=k)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    for k in count_table.columns:
        ax.plot(count_table.index, count_table[k], marker="o", label=f"N = {_count_label(int(k), k_max)}")
    ax.set_title(f"{title}\n{SPECIAL_ASSESSMENT_PLOT_SUBTITLE}", fontsize=11)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_xticks(list(count_table.index))
    ax.legend(title=f"Assessments (λ = {annual_rate:g}/yr)", fontsize="small")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_special_assessment_count(count_probabilities: pd.Series, k_max: int, title: str, save_path: Path) -> Path:
    """One scenario's P(N = k) as a bar chart, k = 0..k_max, each bar labeled with its percentage.
    The last bar is "k_max+" (see _count_label)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [_count_label(int(k), k_max) for k in count_probabilities.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, count_probabilities.to_numpy(), color="C1", alpha=0.8)
    for bar, p in zip(bars, count_probabilities.to_numpy(), strict=True):
        ax.annotate(f"{p * 100:.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{title}\n{SPECIAL_ASSESSMENT_PLOT_SUBTITLE}", fontsize=10)
    ax.set_xlabel("Number of special assessments over the horizon (N)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, min(1.0, count_probabilities.max() * 1.15 + 0.02))
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_special_assessment_cost(outcomes: pd.DataFrame, k_max: int, title: str, save_path: Path) -> Path:
    """One scenario's total special-assessment cost distribution: a stem at each possible total cost
    (USD, nominal at payment time), height = its probability, colored by how many assessments (k)
    produced it and labeled with its size combination and percentage. A stem plot, not a smooth
    density, because the distribution genuinely IS a short list of exact points (see
    carrying_costs.compute_special_assessment). outcomes: DataFrame with k, combination,
    total_cost_usd, probability."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, group in outcomes.groupby("k"):
        color = f"C{int(k)}"
        ax.vlines(group["total_cost_usd"], 0, group["probability"], color=color, linewidth=2)
        ax.scatter(group["total_cost_usd"], group["probability"], color=color, zorder=3, label=f"N = {_count_label(int(k), k_max)}")
        for _, row in group.iterrows():
            # escaped \$ -- an unescaped pair of $ would trigger matplotlib's mathtext parsing
            ax.annotate(
                f"{row['combination']}\n\\${row['total_cost_usd']:,.0f} · {row['probability'] * 100:.2f}%",
                (row["total_cost_usd"], row["probability"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
            )
    ax.set_title(f"{title}\n{SPECIAL_ASSESSMENT_PLOT_SUBTITLE}", fontsize=10)
    ax.set_xlabel("Total special-assessment cost over the horizon (USD, nominal at payment time)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, min(1.0, outcomes["probability"].max() * 1.2 + 0.03))
    ax.legend(title="Assessments", fontsize="small")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
