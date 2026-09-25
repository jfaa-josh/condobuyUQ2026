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

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless-safe: these scripts only ever save figures, never display them
import matplotlib.pyplot as plt


def _plot_outer_band(ax, x, outer_projection: pd.DataFrame) -> None:
    """Draws outer_projection's lo/hi as dashed reference lines -- the wide (0.01, 0.99) band every
    plot in this module shows alongside its own (typically narrower) shaded confidence band."""
    ax.plot(x, outer_projection["lo"], linestyle="--", color="C1", alpha=0.6, linewidth=1, label="1-99% band")
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
    ax.scatter(history.index, history.values, s=8, label="history")
    ax.plot(projection_dates, projection["median"], color="C1", label="projected median")
    ax.fill_between(
        projection_dates, projection["lo"], projection["hi"], color="C1", alpha=0.2, label="confidence band"
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
        ax.plot(proj.index, proj["median"], linewidth=1.5, alpha=alpha, label=f"{label} (weight {weight:.2g})")
        ax.fill_between(proj.index, proj["lo"], proj["hi"], alpha=alpha * 0.25)
    ax.plot(
        combined_projection.index,
        combined_projection["median"],
        color="black",
        linewidth=3,
        alpha=1.0,
        label="combined",
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
    ax.set_xlabel("Months ahead")
    ax.set_ylabel("Monthly price increase (fraction)")
    ax.legend()
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
    ax.plot(projection.index, projection["median"], color="C1", label="projected median")
    ax.fill_between(
        projection.index, projection["lo"], projection["hi"], color="C1", alpha=0.2, label="confidence band"
    )
    _plot_outer_band(ax, projection.index, outer_projection)
    ax.set_title(title)
    ax.set_xlabel("Months ahead")
    ax.set_ylabel("Monthly price increase (fraction)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
