"""Loads, validates, and normalizes input_deck.yaml into Python objects for the rest of the
package to consume.

Only barely started — the deck's structure is still being iterated on with the user (see
.claude/implementation-plan.md). So far this loads the raw YAML, can unwrap a `kind: constant`
entry to its plain value (see get_report_quantiles), can resolve a fitted forecast_models entry
(general, home_price, market -- none of these have a `kind`, since they're fitted rather than
hand-elicited, so they just name a model) to the fit it names, with some basic sanity checks (see
get_fitted_forecast_model), and can resolve `derived_latents.*.weights` for every entry, checked
for a^2+b^2+c^2 <= 1 (see get_derived_latents). Once the deck stabilizes, this module should be its
single validation "catch-all": everything else in this package should be able to assume a deck
that's come through here is well-formed, without re-checking it itself. That means:

- Parsing the YAML into typed Python objects, one shape per `kind` (constant, sweep, prior,
  process), rather than leaving callers to work with raw dicts.
- Resolving every `latent:` / `growth.latent:` reference against `forecast_models` and
  `derived_latents`, and erroring clearly on anything that doesn't resolve.
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
_OU_FIT_KEYS = frozenset({"alpha", "beta", "i_inf", "tau", "sig", "i0", "phi_monthly"})
DERIVED_LATENT_COMPONENTS = ("market", "home_price", "general")


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


def _sweep(entry: dict[str, Any]) -> list[Any]:
    """Unwraps a `kind: sweep` deck entry to its list of values -- the sweep-side counterpart to
    _constant."""
    if entry["kind"] != "sweep":
        raise NotImplementedError(f"_sweep only understands kind: sweep, got {entry['kind']!r}")
    return entry["values"]


def get_scenarios(deck: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Expands `scenario`'s sweeps -- alternatives x horizon_years x residency_state x
    purchase_price, see the deck's "SWEEPS ARE DELIBERATELY MINIMAL" note -- into the full
    cartesian product of scenarios to run. Each returned dict also carries scenario's two fixed
    (`kind: constant`) fields, reference and property_state, alongside that combination's swept
    values, e.g.:

        {"reference": "rent", "alternative": "buy", "horizon_years": 1, "residency_state": "FL",
         "property_state": "CO", "purchase_price": 375000}

    Note scenario.purchase_price is this deck's actual input for purchase price --
    acquisition.purchase_price is a discoverability stub only (see the deck's "SEE scenario.X"
    note); a scenario dict's "purchase_price" key is what acquisition.compute_acquisition_costs
    and any other per-scenario computation should read.
    """
    deck = deck if deck is not None else load_deck()
    scenario = deck["scenario"]
    reference = _constant(scenario["reference"])
    property_state = _constant(scenario["property_state"])

    scenarios = []
    for alternative in _sweep(scenario["alternatives"]):
        for horizon_years in _sweep(scenario["horizon_years"]):
            for residency_state in _sweep(scenario["residency_state"]):
                for purchase_price in _sweep(scenario["purchase_price"]):
                    scenarios.append(
                        {
                            "reference": reference,
                            "alternative": alternative,
                            "horizon_years": horizon_years,
                            "residency_state": residency_state,
                            "property_state": property_state,
                            "purchase_price": purchase_price,
                        }
                    )
    return scenarios


def get_acquisition_inputs(deck: dict[str, Any] | None = None) -> dict[str, float]:
    """Unwraps `acquisition`'s own `kind: constant` fields to plain values -- everything except
    purchase_price, which is a discoverability stub only (the real value comes from a scenario's
    own "purchase_price", see get_scenarios). Returns:
    {"closing_costs_fixed_fees", "closing_costs_percent_of_price", "land_fraction",
    "personal_cash_at_closing", "family_loan", "initial_furnishing"}, straight off
    acquisition.purchase_closing_costs.fixed_fees/.percent_of_price, acquisition.land_fraction,
    acquisition.personal_cash_at_closing, acquisition.family_loan, acquisition.initial_furnishing
    respectively. total_cash_outlay (personal_cash_at_closing + family_loan) is DERIVED, not read
    here -- see acquisition.compute_acquisition_costs.
    """
    deck = deck if deck is not None else load_deck()
    acquisition = deck["acquisition"]
    closing_costs = acquisition["purchase_closing_costs"]
    return {
        "closing_costs_fixed_fees": _constant(closing_costs["fixed_fees"]),
        "closing_costs_percent_of_price": _constant(closing_costs["percent_of_price"]),
        "land_fraction": _constant(acquisition["land_fraction"]),
        "personal_cash_at_closing": _constant(acquisition["personal_cash_at_closing"]),
        "family_loan": _constant(acquisition["family_loan"]),
        "initial_furnishing": _constant(acquisition["initial_furnishing"]),
    }


def get_financing_inputs(deck: dict[str, Any] | None = None) -> dict[str, float | int]:
    """Unwraps `financing`'s own `kind: constant` fields to plain values. Returns
    {"mortgage_rate_primary_residence", "mortgage_rate_second_home", "mortgage_rate_investment",
    "non_warrantable_premium", "loan_term_years"}, straight off financing.mortgage_rate's three
    sub-fields, financing.non_warrantable_premium, financing.loan_term_years respectively.
    loan_occupancy_class itself is DERIVED, not read here -- see
    classification.get_loan_occupancy_class."""
    deck = deck if deck is not None else load_deck()
    financing = deck["financing"]
    mortgage_rate = financing["mortgage_rate"]
    return {
        "mortgage_rate_primary_residence": _constant(mortgage_rate["primary_residence"]),
        "mortgage_rate_second_home": _constant(mortgage_rate["second_home"]),
        "mortgage_rate_investment": _constant(mortgage_rate["investment"]),
        "non_warrantable_premium": _constant(financing["non_warrantable_premium"]),
        "loan_term_years": _constant(financing["loan_term_years"]),
    }


def get_rental_operations_inputs(deck: dict[str, Any] | None = None) -> dict[str, str]:
    """Unwraps `rental_operations`'s own classification-relevant `kind: constant` field. Returns
    {"managed_by"} (OPTIONS: self | company), straight off rental_operations.managed_by -- see
    that field's own deck comment and classification.get_loan_occupancy_class, which is the only
    thing that reads it."""
    deck = deck if deck is not None else load_deck()
    return {"managed_by": _constant(deck["rental_operations"]["managed_by"])}


def get_annual_rented_days(deck: dict[str, Any] | None = None) -> float:
    """Sums rental_operations.occupancy's monthly medians (`kind: prior`, `resolution: monthly` --
    not a `kind: constant`, so _constant doesn't apply) into an expected annual rented-days figure.
    Used only for occupancy/tax classification (see classification.get_tax_use_class), never for
    revenue itself (that still uses the full occupancy distribution, median AND cv, elsewhere)."""
    deck = deck if deck is not None else load_deck()
    months = deck["rental_operations"]["occupancy"]["months"]
    return sum(month["median"] for month in months.values())


def get_annual_personal_days(deck: dict[str, Any] | None = None) -> float:
    """Sums personal_use.personal_use_days_by_month (`kind: constant`, a plain {month: days} dict)
    into an annual personal-use-days figure. Used only for occupancy/tax classification (see
    classification.get_tax_use_class)."""
    deck = deck if deck is not None else load_deck()
    days_by_month = _constant(deck["personal_use"]["personal_use_days_by_month"])
    return sum(days_by_month.values())


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
    missing = _OU_FIT_KEYS - fit.keys()
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


def get_derived_latents(deck: dict[str, Any] | None = None) -> dict[str, dict[str, float]]:
    """Returns derived_latents.*.weights for every entry (construction_cost, insurance,
    maintenance, adr, utilities), keyed by entry name -> {"market": a, "home_price": b, "general":
    c} (see DERIVED_LATENT_COMPONENTS). Checks a^2+b^2+c^2 <= 1 for each -- the shortfall is that
    entry's own idiosyncratic share (see the deck's derived_latents section comment and
    .claude/implementation-plan.md's "Derived latents" design note); raises clearly if violated,
    since a violation would make the idiosyncratic weight imaginary.
    """
    deck = deck if deck is not None else load_deck()
    result: dict[str, dict[str, float]] = {}
    for name, entry in deck["derived_latents"].items():
        weights = {key: entry["weights"][key] for key in DERIVED_LATENT_COMPONENTS}
        sum_sq = sum(w**2 for w in weights.values())
        if sum_sq > 1:
            raise ValueError(
                f"derived_latents.{name}.weights has a^2+b^2+c^2={sum_sq!r} > 1 -- "
                "idiosyncratic share would be negative."
            )
        result[name] = weights
    return result
