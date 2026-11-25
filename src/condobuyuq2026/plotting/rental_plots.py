"""Model-folder check plots for rental_operations' two priors (occupancy, ADR): the fitted per-month
prior against every source's RAW values, so the fit can be eyeballed against the data it came from
(user request, 2026-11-22). Scenario-independent (no horizon, no growth): results/ holds the per-
scenario plots.

Conventions match plotting.forecast_plots: shaded band = run.report_quantiles' outer pair, dashed C1
lines = run.outer_band, median in C1. Sources take fixed colors in deck order (C0, C2, C3, ...; C1 is
reserved for the fit), so a source keeps its color across both plots. Excluded sources are hollow gray.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from condobuyuq2026.input_deck import MONTH_KEYS, format_band, get_outer_band

_SOURCE_COLORS = ("C0", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9")
_MONTH_LABELS = [m.capitalize() for m in MONTH_KEYS]


def _source_color(index: int) -> str:
    return _SOURCE_COLORS[index % len(_SOURCE_COLORS)]


def _draw_monthly_fit(ax, bands: dict[str, list[float]]) -> None:
    """Fitted median (step) + shaded main band + dashed outer band (run.outer_band) over the 12 calendar months."""
    edges = np.arange(13) - 0.5  # each month spans its full tick slot, Jan and Dec included
    ax.stairs(bands["hi"], edges, baseline=bands["lo"], fill=True, color="C1", alpha=0.2, label="Fitted band")
    ax.stairs(bands["lo_outer"], edges, baseline=None, linestyle="--", color="C1", alpha=0.6, linewidth=1, label=f"{format_band(*get_outer_band())} Band")
    ax.stairs(bands["hi_outer"], edges, baseline=None, linestyle="--", color="C1", alpha=0.6, linewidth=1)
    ax.stairs(bands["median"], edges, baseline=None, color="C1", linewidth=2, label="Fitted median")
    ax.set_xticks(np.arange(12), _MONTH_LABELS)
    ax.set_xlim(-0.5, 11.5)
    ax.grid(alpha=0.3)


def _draw_monthly_points(ax, points: list[dict[str, Any]]) -> None:
    """points: [{"label", "index" (deck order), "include", "resolution", "values": {month: value|None}}].
    Monthly sources: filled circles; annual sources (spread with the seasonal pattern): crosses;
    excluded sources: hollow gray. Small horizontal offsets keep sources from overlapping."""
    count = max(len(points), 1)
    for k, point in enumerate(points):
        offset = (k - (count - 1) / 2) * 0.12
        x = [i + offset for i, m in enumerate(MONTH_KEYS) if point["values"][m] is not None]
        y = [point["values"][m] for m in MONTH_KEYS if point["values"][m] is not None]
        suffix = " (annual, spread by season)" if point["resolution"] == "annual" else ""
        if point["include"]:
            marker = "x" if point["resolution"] == "annual" else "o"
            ax.scatter(x, y, s=40, marker=marker, color=_source_color(point["index"]), zorder=3,
                       label=f"{point['label']}{suffix}")
        else:
            ax.scatter(x, y, s=40, marker="o", facecolors="none", edgecolors="gray", zorder=3,
                       label=f"{point['label']} (excluded){suffix}")


def _draw_annual(ax, annual: dict[str, Any], value_format: str) -> None:
    """Each source's annual value (dots, deck colors) against the fitted annual median + bands."""
    sources = annual["sources"]
    x = np.arange(len(sources))
    for i, source in enumerate(sources):
        if source["include"]:
            ax.scatter([i], [source["value"]], s=60, color=_source_color(source["index"]), zorder=3)
        else:
            ax.scatter([i], [source["value"]], s=60, facecolors="none", edgecolors="gray", zorder=3)
        ax.annotate(value_format.format(source["value"]), (i, source["value"]), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=9)
    fitted = annual["fitted"]
    ax.axhspan(fitted["lo"], fitted["hi"], color="C1", alpha=0.2, label="Fitted band")
    ax.axhline(fitted["lo_outer"], linestyle="--", color="C1", alpha=0.6, linewidth=1, label=f"{format_band(*get_outer_band())} Band")
    ax.axhline(fitted["hi_outer"], linestyle="--", color="C1", alpha=0.6, linewidth=1)
    ax.axhline(fitted["median"], color="C1", linewidth=2, label=f"Fitted median {value_format.format(fitted['median'])}")
    ax.set_xticks(x, [s["label"] + ("" if s["include"] else "\n(excluded)") for s in sources], fontsize=9)
    ax.set_xlim(-0.5, len(sources) - 0.5)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9)


def plot_rental_prior(
    title: str,
    ylabel: str,
    monthly_bands: dict[str, list[float]],
    points: list[dict[str, Any]],
    annual: dict[str, Any],
    annual_ylabel: str,
    value_format: str,
    save_path: Path,
    nights: dict[str, Any] | None = None,
    note: str | None = None,
) -> Path:
    """Two panels (three for occupancy, when `nights` is given):
    1. by calendar month: every source's raw value vs the fitted median and bands;
    2. annual: each source's annual value vs the fitted annual median and bands;
    3. (occupancy) YOUR rented nights by month: fitted median + bands, with the cap (nights you leave
       available = days - your personal use) marked.
    nights: {"median", "lo", "hi", "lo_outer", "hi_outer", "cap"} lists over the 12 months."""
    rows = 3 if nights is not None else 2
    fig, axes = plt.subplots(rows, 1, figsize=(12, 4.6 * rows), gridspec_kw={"height_ratios": [1.3] + [1] * (rows - 1)})

    ax = axes[0]
    _draw_monthly_fit(ax, monthly_bands)
    _draw_monthly_points(ax, points)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)  # headroom for the legend
    ax.legend(loc="upper center", fontsize=8, ncol=3)

    _draw_annual(axes[1], annual, value_format)
    axes[1].set_ylabel(annual_ylabel)
    axes[1].set_title("Annual: each source vs the fitted year", fontsize=11)

    if nights is not None:
        ax = axes[2]
        _draw_monthly_fit(ax, nights)
        ax.stairs(nights["cap"], np.arange(13) - 0.5, baseline=None, color="k", linewidth=1.2, linestyle=":",
                  label="Cap: nights you leave available")
        ax.set_ylim(0, max(nights["cap"]) * 1.3)  # headroom so the legend clears the cap line
        ax.set_ylabel("Nights rented")
        ax.set_title("Your rented nights per month (occupancy x nights you leave available, capped)", fontsize=11)
        ax.legend(loc="upper center", fontsize=8, ncol=4)

    if note:
        fig.text(0.01, 0.005, note, fontsize=8, va="bottom", ha="left", wrap=True)
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
