"""Run-level outputs (deck section 13, `run`), written once after every scenario has run:

- the SUMMARY across scenarios (summary_table -> results/summary.csv; results.save_run_summary_plot ->
  results/summary.png): every buy scenario side by side, P(buy beats rent) with and without the
  special-assessment overlay, ΔW's band, expected W_buy / W_rent, who holds the difference fund at the
  sale, and the biggest source of ΔW's uncertainty;
- STALE RESULT FOLDERS (find_stale_results_folders): folders under results/ this run did NOT write (scenarios
  since removed from the sweeps). Reported only, never deleted (user, 2026-11-25). Automated results-storage
  management may come later (see the deck's run header);
- the RUN MANIFEST (write_run_manifest -> results/run_info.json): what produced the results now in results/ --
  when the run happened, the code's git commit (and whether the working tree had uncommitted changes), the
  deck's SHA-256 fingerprint, the run settings used, how current each forecast model's data is, every
  warning and note raised, and the folders written / found stale. The GUI will read it once it's wired.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from condobuyuq2026.freshness import FORECAST_MODEL_BUILD_SCRIPTS, expected_latest_month, latest_data_month
from condobuyuq2026.input_deck import (
    DECK_PATH,
    get_build_horizon_months,
    get_outer_band,
    get_report_quantiles,
    get_solver_settings,
    get_warning_thresholds,
)
from condobuyuq2026.paths import REPO_ROOT, RESULTS_DIR
from condobuyuq2026.results import reference_results_dir, scenario_results_dir

RUN_ROOT_FILES = ("summary.csv", "summary.png", "run_info.json")
SWEPT_FIELDS = ("alternative", "horizon_years", "residency_state", "purchase_price", "family_loan_repaid_at_sale")


def scenario_folder(scenario: dict[str, Any]) -> Path:
    """results/<scenario>/ (the parent of scenario_results_dir's carrying_costs/ subfolder)."""
    return scenario_results_dir(scenario).parent


def summary_table(run: dict[str, Any]) -> pd.DataFrame:
    """One row per buy scenario (runner.run_scenarios' result): the swept values, its results folder and
    reference case, and the ΔW headline numbers (see the module docstring)."""
    rows = []
    for scenario in run["scenarios"]:
        dw = scenario["delta_w"]
        lo_label, hi_label = (f"p{round(q * 100):02d}" for q in dw["ci"])
        breakdown = sorted(dw["variance_breakdown"].items(), key=lambda kv: -kv[1])
        overlay = dw["overlay"]
        rows.append({
            "folder": scenario_folder(scenario).name,
            **{field: scenario[field] for field in SWEPT_FIELDS},
            "reference_id": scenario["reference_id"],
            "p_buy_beats_rent": dw["p_buy_beats_rent"],
            "p_buy_beats_rent_with_special_assessments": overlay["p_buy_beats_rent"] if overlay["enabled"] else None,
            "delta_w_mean": dw["mean"],
            "delta_w_median": dw["bands"]["median"],
            f"delta_w_{lo_label}": dw["bands"]["lo"],
            f"delta_w_{hi_label}": dw["bands"]["hi"],
            "delta_w_sd": dw["sd"],
            "w_buy_expected": dw["components"]["w_buy"],
            "w_rent_expected": dw["components"]["w_rent"],
            "fund_holder_at_sale": {1: "rent", -1: "buy", 0: "none"}[dw["fund"]["final_holder"]],
            "top_variance_source": breakdown[0][0] if breakdown else None,
            "top_variance_share": breakdown[0][1] if breakdown else None,
        })
    return pd.DataFrame(rows)


def written_folders(run: dict[str, Any]) -> list[Path]:
    """Every results/ folder this run wrote: each buy scenario's and each reference case's."""
    folders = {scenario_folder(s) for s in run["scenarios"]}
    folders |= {reference_results_dir(rid, case) for rid, case in run["reference_cases"].items()}
    return sorted(folders)


def find_stale_results_folders(written: list[Path]) -> list[Path]:
    """Folders directly under results/ this run did NOT write. Reported, never deleted."""
    if not RESULTS_DIR.exists():
        return []
    keep = {p.resolve() for p in written}
    return sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir() and p.resolve() not in keep)


def collect_warnings(run: dict[str, Any], stale_forecasts: list[str], rental_prior_results: dict[str, Any],
                     carrying_cost_prior_results: dict[str, Any], stale_folders: list[Path],
                     deck: dict[str, Any]) -> list[dict[str, str]]:
    """Every warning / note the run raised, as {"where", "message"} (the same ones the CLI prints)."""
    warnings: list[dict[str, str]] = []
    if stale_forecasts:
        warnings.append({"where": "forecast_models", "message": f"stale, not refreshed: {', '.join(stale_forecasts)}"})
    for message in rental_prior_results.get("occupancy", {}).get("warnings", []):
        warnings.append({"where": "rental_operations priors", "message": message})
    for message in carrying_cost_prior_results.get("utilities", {}).get("cap_warnings", []):
        warnings.append({"where": "utilities prior", "message": message})
    threshold = get_warning_thresholds(deck)["special_assessment_tail"]
    for scenario in run["scenarios"]:
        where = scenario_folder(scenario).name
        special = scenario["carrying_costs"]["special_assessment"]
        if special["enabled"] and special["folded_tail_probability"] > threshold:
            warnings.append({"where": where, "message": (
                f"special_assessment P(more than k_max={special['k_max']})={special['folded_tail_probability']:.1%} "
                f"> {threshold:.1%}: raise carrying_costs.special_assessment.k_max")})
        warnings += [{"where": where, "message": flag} for flag in scenario["taxes"]["flags"]]
        warnings += [{"where": where, "message": flag} for flag in scenario["delta_w"]["flags"]]
    if stale_folders:
        warnings.append({"where": "results/", "message": (
            f"{len(stale_folders)} folder(s) not written by this run (old scenarios?): "
            + ", ".join(p.name for p in stale_folders))})
    return warnings


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def write_run_manifest(run: dict[str, Any], deck: dict[str, Any], started: datetime, warnings: list[dict[str, str]],
                       written: list[Path], stale_folders: list[Path], outputs: dict[str, Path]) -> Path:
    """Writes results/run_info.json (see the module docstring) and returns its path."""
    finished = datetime.now().astimezone()
    status = _git("status", "--porcelain")
    forecast_data = {}
    for name in FORECAST_MODEL_BUILD_SCRIPTS:
        model_name = deck["forecast_models"][name]["model_name"]
        latest, expected = latest_data_month(model_name), expected_latest_month(name)
        forecast_data[name] = {
            "model_name": model_name,
            "latest_data_month": f"{latest[0]}-{latest[1]:02d}",
            "expected_by_now": f"{expected[0]}-{expected[1]:02d}",
            "stale": latest < expected,
        }
    manifest = {
        "run_started": started.isoformat(timespec="seconds"),
        "run_finished": finished.isoformat(timespec="seconds"),
        "duration_seconds": round((finished - started).total_seconds(), 1),
        "code": {"git_commit": _git("rev-parse", "HEAD"),
                 "uncommitted_changes": None if status is None else bool(status)},
        "deck": {"path": DECK_PATH.relative_to(REPO_ROOT).as_posix(),
                 "sha256": hashlib.sha256(DECK_PATH.read_bytes()).hexdigest()},
        "run_settings": {
            "report_quantiles": get_report_quantiles(deck),
            "outer_band": list(get_outer_band(deck)),
            "solver": get_solver_settings(deck),
            "warning_thresholds": get_warning_thresholds(deck),
            "build_horizon_months": get_build_horizon_months(deck),
        },
        "forecast_data": forecast_data,
        "scenario_count": len(run["scenarios"]),
        "reference_case_count": len(run["reference_cases"]),
        "outputs": {key: path.relative_to(REPO_ROOT).as_posix() for key, path in outputs.items()},
        "folders_written": [p.relative_to(REPO_ROOT).as_posix() for p in written],
        "stale_folders": [p.relative_to(REPO_ROOT).as_posix() for p in stale_folders],
        "warnings": warnings,
    }
    path = RESULTS_DIR / "run_info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
