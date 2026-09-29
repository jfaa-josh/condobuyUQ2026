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


def plot_utilities_seasonal_shape(
    temperatures: dict[str, float],
    seasonal_fit: dict[str, Any],
    monthly_costs: dict[str, float],
    title: str,
    save_path: Path,
) -> Path:
    """Two-panel plot: LEFT shows the raw monthly temperature data (scatter) against its own fitted
    sinusoid (utils.prior_utils.fit_temperature_seasonal_shape -- drawn as a genuinely SMOOTH curve
    from the fit's own a/b/c coefficients on a fine grid, not a 12-point connect-the-dots line), in
    the data's own units (°F) -- the "derivation" step. RIGHT shows the resulting seasonal
    utility-cost shape (utils.prior_utils.seasonal_cost_weights applied to utilities.
    initial_annual_average_monthly_cost) at INITIAL levels (i.e. before any growth is applied), USD/
    month -- so the temperature-to-cost inversion (cold month = high cost) is visible directly, not
    just the end-result numbers."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    months = list(temperatures.keys())
    x = np.arange(len(months))
    x_smooth = np.linspace(0, len(months), 200)
    fitted_smooth = seasonal_fit["a"] + seasonal_fit["b"] * np.cos(2 * np.pi * x_smooth / 12) + seasonal_fit[
        "c"
    ] * np.sin(2 * np.pi * x_smooth / 12)

    fig, (ax_temp, ax_cost) = plt.subplots(1, 2, figsize=(14, 5))
    ax_temp.scatter(x, [temperatures[m] for m in months], color="C0", zorder=3, label="Observed average")
    ax_temp.plot(x_smooth, fitted_smooth, color="C3", linestyle="--", label="Sine fit")
    ax_temp.set_xticks(x)
    ax_temp.set_xticklabels(months)
    ax_temp.set_ylabel("°F")
    ax_temp.set_title("Temperature Data and Sine Fit")
    ax_temp.legend()

    ax_cost.bar(x, [monthly_costs[m] for m in months], color="C1")
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(months)
    ax_cost.set_ylabel("USD/month")
    ax_cost.set_title("Seasonal Cost Shape (Initial Levels)")

    fig.suptitle(title)
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
