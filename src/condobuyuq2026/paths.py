"""Filesystem path conventions shared between the core package (e.g. input_deck.py, reading a
fitted model's saved output) and the manual, one-off scripts in run/manual/ (via
utils/manual_utils.py, which writes that output). The one place both sides get these from, so the
convention can't drift apart between the write side and the read side.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
MODELS_DIR = PACKAGE_DIR / "models"
RESULTS_DIR = REPO_ROOT / "results"


def model_dir(model_name: str) -> Path:
    """models/<model_name> -- a given model's build outputs (plots, fit.json, ...)."""
    return MODELS_DIR / model_name


def model_fit_path(model_name: str) -> Path:
    """models/<model_name>/fit.json -- see utils.ou_fitting.save_fit/load_fit."""
    return model_dir(model_name) / "fit.json"
