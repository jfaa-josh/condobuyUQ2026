"""Plotting helpers for a fitted forecast_models process -- a visual sanity check on a term
structure (median + confidence band) projected from a fitted model, against the historical data it
was fit from, or on the rate's own path with no price level involved.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless-safe: these scripts only ever save figures, never display them
import matplotlib.pyplot as plt


def plot_term_structure(
    history: pd.Series, projection: pd.DataFrame, title: str, save_path: Path
) -> Path:
    """Plots history (a monthly-indexed Series, e.g. from manual_utils.to_monthly_series) as a
    scatter, and projection (an h_months-indexed DataFrame with "median"/"lo"/"hi" columns, e.g.
    from ou_fitting.project_term_structure) as a median line with a shaded confidence band,
    starting from history's last observed month. Saves to save_path (creating parent directories as
    needed) and returns the path written.
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
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_monthly_rate(projection: pd.DataFrame, title: str, save_path: Path) -> Path:
    """Plots a projected monthly fractional rate (e.g. from ou_fitting.project_monthly_rate) as a
    median line with a shaded confidence band, x-axis in months-ahead. No history/p0 to anchor
    it to -- a rate doesn't need a starting price level, unlike plot_term_structure."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(projection.index, projection["median"], color="C1", label="projected median")
    ax.fill_between(
        projection.index, projection["lo"], projection["hi"], color="C1", alpha=0.2, label="confidence band"
    )
    ax.set_title(title)
    ax.set_xlabel("Months ahead")
    ax.set_ylabel("Monthly price increase (fraction)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
