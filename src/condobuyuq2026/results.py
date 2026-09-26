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

from condobuyuq2026.carrying_costs import load_market_value_change_magnitude, project_market_value_schedule
from condobuyuq2026.input_deck import get_market_value_config, get_report_quantiles, load_deck
from condobuyuq2026.paths import RESULTS_DIR
from condobuyuq2026.plotting.forecast_plots import plot_market_value_periods

_RESULT_PLOT_OUTER_CI = (0.01, 0.99)  # wide reference band, same convention as the model-build plots


def scenario_results_dir(scenario: dict[str, Any]) -> Path:
    """results/buy_h<horizon_years>y_p<purchase_price>_<residency_state>/carrying_costs/ -- a
    readable folder name from the scenario's own SWEPT fields (the only ones that can distinguish
    two scenarios -- everything else is a fixed deck constant). Not a stable id across deck edits
    that change the sweep values themselves, but stable across repeated runs of the same deck."""
    label = f"buy_h{scenario['horizon_years']}y_p{scenario['purchase_price']}_{scenario['residency_state']}"
    path = RESULTS_DIR / label / "carrying_costs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plot_annual_median_line(series: pd.Series, title: str, ylabel: str, save_path: Path) -> Path:
    """Simple median-only annual line/marker plot -- for the carrying-cost lines that are only ever
    propagated as a median (see carrying_costs.py's own median-only-by-convention rule)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(series.index, series.values, color="C0", marker="o")
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(series.index))
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def _plot_annual_band(df: pd.DataFrame, title: str, ylabel: str, save_path: Path) -> Path:
    """Median/lo/hi band plot, annual x-axis -- for property_tax, the one line among the six
    compute_carrying_costs_schedule columns that keeps a real distribution (see
    carrying_costs.compute_property_tax_schedule)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df["median"], color="C1", label="median")
    ax.fill_between(df.index, df["lo"], df["hi"], color="C1", alpha=0.2, label="confidence band")
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(df.index))
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def _plot_market_value_year_distributions(
    market_value_schedule: pd.DataFrame, ci: tuple[float, float], title: str, save_path: Path
) -> Path:
    """Composite plot: one lognormal PDF curve per year of market_value_schedule (median/lo/hi, see
    carrying_costs.project_market_value_schedule), overlaid on a shared x-axis so how the
    distribution shifts/widens year to year is visible at a glance -- reconstructs each year's own
    log-space sigma from its median/hi the same way carrying_costs.py itself does internally
    (log(hi/median)/z_hi). A year with zero uncertainty (the pre-first-reassessment period, a known
    constant) is drawn as a vertical marker line instead of a degenerate zero-width density curve.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    z_hi = norm.ppf(ci[1])

    sigmas = {}
    for year in market_value_schedule.index:
        median = market_value_schedule.loc[year, "median"]
        hi = market_value_schedule.loc[year, "hi"]
        sigmas[year] = float(np.log(hi / median) / z_hi) if hi > median else 0.0

    lo_extent = market_value_schedule["lo"].min()
    hi_extent = market_value_schedule["hi"].max()
    spread = max(hi_extent - lo_extent, market_value_schedule["median"].iloc[0] * 0.05)
    x = np.linspace(max(lo_extent - spread * 0.5, 1.0), hi_extent + spread * 0.5, 400)

    fig, ax = plt.subplots(figsize=(10, 5))
    for year in market_value_schedule.index:
        median = market_value_schedule.loc[year, "median"]
        sigma = sigmas[year]
        if sigma < 1e-9:
            ax.axvline(median, linestyle="--", label=f"year {year} (known)")
            continue
        mu = np.log(median)
        pdf = np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))
        ax.plot(x, pdf, label=f"year {year}")

    ax.set_title(title)
    ax.set_xlabel("USD")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def save_carrying_cost_result_plots(
    scenario: dict[str, Any], carrying_costs: dict[str, Any], deck: dict[str, Any] | None = None
) -> dict[str, Path]:
    """Saves this scenario's own SCALED carrying-cost plots into results/<scenario>/carrying_costs/
    -- one median-line plot per LEVEL+GROWTH entry (hoa_dues, insurance, maintenance,
    special_assessment_expected, utilities), a median/band plot for property_tax (the one real
    distribution among the six annual_schedule columns), market_value's own stair-stepped plot
    (reusing plot_market_value_periods against the SAVED change-magnitude periods, scaled by the
    deck's live initial_price and truncated to this scenario's own horizon -- no re-fitting, per
    this module's own "change magnitude" split), and a composite per-year distribution plot (see
    _plot_market_value_year_distributions). carrying_costs as returned by
    carrying_costs.compute_carrying_costs for this scenario.

    Returns {name: path} for every plot saved.
    """
    deck = deck if deck is not None else load_deck()
    out_dir = scenario_results_dir(scenario)
    schedule = carrying_costs["annual_schedule"]
    property_tax_schedule = carrying_costs["property_tax_schedule"]
    report_quantiles = get_report_quantiles(deck)
    ci = (report_quantiles[0], report_quantiles[-1])

    paths: dict[str, Path] = {}
    for name, ylabel in (
        ("hoa_dues", "USD/year"),
        ("insurance", "USD/year"),
        ("maintenance", "USD/year"),
        ("special_assessment_expected", "USD/year (expected)"),
        ("utilities", "USD/year"),
    ):
        paths[name] = _plot_annual_median_line(
            schedule[name], title=f"{name} — scenario projection", ylabel=ylabel, save_path=out_dir / f"{name}.png"
        )

    tax_band = property_tax_schedule[["tax_median", "tax_lo", "tax_hi"]].rename(
        columns={"tax_median": "median", "tax_lo": "lo", "tax_hi": "hi"}
    )
    paths["property_tax"] = _plot_annual_band(
        tax_band, title="property_tax — scenario projection", ylabel="USD/year", save_path=out_dir / "property_tax.png"
    )

    change_magnitude = load_market_value_change_magnitude()
    initial_price = get_market_value_config(deck)["initial_price"]
    horizon_months = scenario["horizon_years"] * 12
    paths["market_value"] = plot_market_value_periods(
        change_magnitude["periods"],
        initial_price,
        ci,
        _RESULT_PLOT_OUTER_CI,
        horizon_months=horizon_months,
        title="market_value — scenario projection",
        save_path=out_dir / "market_value.png",
    )

    market_value_schedule = project_market_value_schedule(scenario, deck)
    paths["market_value_year_distributions"] = _plot_market_value_year_distributions(
        market_value_schedule,
        ci,
        title="market_value — distribution by year",
        save_path=out_dir / "market_value_year_distributions.png",
    )
    return paths
