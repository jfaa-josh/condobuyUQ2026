"""Loads, validates, and normalizes input_deck.yaml into Python objects for the rest of the
package to consume.

Only barely started — the deck's structure is still being iterated on with the user (see
.claude/implementation-plan.md). So far this loads the raw YAML, can unwrap a `kind: constant`
entry to its plain value (see get_report_quantiles), and can resolve a fitted forecast_models
entry (general, home_price, market -- none of these have a `kind`, since they're fitted rather
than hand-elicited, so they just name a model) to the fit it names, with some basic sanity checks
(see get_fitted_forecast_model). Once the deck stabilizes, this module should be its single
validation "catch-all": everything else in this package should be able to assume a deck that's
come through here is well-formed, without re-checking it itself. That means:

- Parsing the YAML into typed Python objects, one shape per `kind` (constant, sweep, prior,
  process), rather than leaving callers to work with raw dicts.
- Resolving every `latent:` / `growth.latent:` reference against `forecast_models` and
  `derived_latents`, and erroring clearly on anything that doesn't resolve.
- Checking `derived_latents.*.weights` satisfy a^2 + b^2 + c^2 + d^2 <= 1.
- Checking bounded-fraction/count constraints (e.g. occupancy's day-count against the actual
  number of days in that calendar month and year, net of personal_use_days_by_month).
- Checking every field marked `# REQUIRED` in the deck's comments actually has a value, and
  every categorical field holds one of its documented OPTIONS.
- Catching conflicting settings between variables, not just invalid values within one — this is
  the reason this module exists as a single catch-all rather than validating each field in
  isolation wherever it's used.

As functionality gets added here, expand this docstring into real module documentation instead
of a list of intentions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from condobuyuq2026.paths import REPO_ROOT, model_fit_path
from condobuyuq2026.utils.ou_fitting import load_fit

DECK_PATH = REPO_ROOT / "input_deck.yaml"
_REQUIRED_FIT_KEYS = {"alpha", "beta", "i_inf", "tau", "sig", "i0", "phi_monthly"}


def load_deck(path: Path = DECK_PATH) -> dict[str, Any]:
    """Loads input_deck.yaml as a plain nested dict -- no validation or type normalization yet
    (see module docstring)."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _constant(entry: dict[str, Any]) -> Any:
    """Unwraps a `kind: constant` deck entry to its plain value. Other kinds (sweep, prior,
    process) aren't handled yet -- raises clearly rather than silently returning something wrong."""
    if entry["kind"] != "constant":
        raise NotImplementedError(f"input_deck.py only understands kind: constant so far, got {entry['kind']!r}")
    return entry["value"]


def get_report_quantiles(deck: dict[str, Any] | None = None) -> list[float]:
    """Returns run.report_quantiles (e.g. [0.1, 0.5, 0.9]) -- the probability quantiles the model
    should report in its outputs."""
    deck = deck if deck is not None else load_deck()
    return _constant(deck["run"]["report_quantiles"])


def _load_checked_fit(model_name: str) -> dict[str, float]:
    """Loads models/<model_name>/fit.json (see utils.ou_fitting.save_fit) and checks it's
    well-formed enough to use as a process: the file exists and isn't empty, it has every key
    fit_annual_ou produces, and phi_monthly is a valid AR(1) persistence (0 < phi < 1 -- outside
    that range the fit isn't a proper mean-reverting process; see utils.ou_fitting.fit_annual_ou).
    """
    path = model_fit_path(model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"forecast_models references model {model_name!r}, but {path} doesn't exist -- run its "
            "build script (e.g. run/manual/build_general_price_model.py, "
            "run/manual/build_co_home_price_model.py, or run/manual/build_market_model.py) first."
        )
    if path.stat().st_size == 0:
        raise ValueError(f"Fit file for model {model_name!r} at {path} is empty.")

    fit = load_fit(path)
    missing = _REQUIRED_FIT_KEYS - fit.keys()
    if missing:
        raise ValueError(f"Fit file for model {model_name!r} at {path} is missing keys: {sorted(missing)}.")
    if not (0 < fit["phi_monthly"] < 1):
        raise ValueError(
            f"Fit for model {model_name!r} has phi_monthly={fit['phi_monthly']!r}, outside (0, 1) -- "
            "not a valid mean-reverting process."
        )
    return fit


def get_fitted_forecast_model(name: str, deck: dict[str, Any] | None = None) -> dict[str, float]:
    """Resolves forecast_models.<name> -- an entry that just names a model fit (model_name) rather
    than holding hand-elicited distribution parameters inline (no `kind` at all -- see the deck's
    schema reference) -- to its checked, loaded fit (see _load_checked_fit). Callers use the fit
    as-is (e.g. via utils.ou_fitting.project_term_structure/project_monthly_rate, or
    fit["phi_monthly"] directly) -- no conversion into some other shape needed.
    """
    deck = deck if deck is not None else load_deck()
    return _load_checked_fit(deck["forecast_models"][name]["model_name"])


def get_forecast_model_general(deck: dict[str, Any] | None = None) -> dict[str, float]:
    """Resolves forecast_models.general -- see get_fitted_forecast_model. Fit built by
    run/manual/build_general_price_model.py."""
    return get_fitted_forecast_model("general", deck)


def get_forecast_model_home_price(deck: dict[str, Any] | None = None) -> dict[str, float]:
    """Resolves forecast_models.home_price -- see get_fitted_forecast_model. Fit built by
    run/manual/build_co_home_price_model.py."""
    return get_fitted_forecast_model("home_price", deck)


def get_forecast_model_market(deck: dict[str, Any] | None = None) -> dict[str, float]:
    """Resolves forecast_models.market -- see get_fitted_forecast_model. Fit built by
    run/manual/build_market_model.py."""
    return get_fitted_forecast_model("market", deck)
