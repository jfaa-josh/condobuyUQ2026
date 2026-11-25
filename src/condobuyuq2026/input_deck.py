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
from typing import Any, cast

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


def _expect_kind(entry: dict[str, Any], expected: str, where: str) -> None:
    """Validates a COMPOUND deck entry's own `kind` label (market_value, hoa_dues_annual, etc. --
    a whole multi-field sub-object, not a single scalar leaf the way _constant/_sweep's own `kind`
    is) actually matches the shape its own getter is about to read. Before 2026-09-29 these
    block-level `kind` tags were pure, UNCHECKED documentation -- and wrong at least once
    (market_value was labeled `kind: constant` despite being a multi-field stair-step config, not a
    single value, which is exactly the kind of mistake this catches going forward instead of
    silently reading a mislabeled block)."""
    actual = entry.get("kind")
    if actual != expected:
        raise ValueError(f"{where}: expected kind: {expected!r}, got {actual!r} -- deck entry shape looks wrong")


def get_scenarios(deck: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Expands `scenario`'s sweeps -- alternatives x horizon_years x residency_state x
    purchase_price x family_loan_repaid_at_sale (bool), see the deck's "SWEEPS ARE DELIBERATELY
    MINIMAL" note -- into the full
    cartesian product of scenarios to run. Each returned dict also carries scenario's fixed
    (`kind: constant`) fields (reference, property_state, closing_month, closing_year) alongside
    that combination's swept values, e.g.:

        {"reference": "rent", "alternative": "buy", "horizon_years": 1, "residency_state": "FL",
         "property_state": "CO", "purchase_price": 375000, "family_loan_repaid_at_sale": True,
         "closing_month": 10, "closing_year": 2026}

    closing_month/closing_year are TIMESTEP 0 for month-indexed models (see
    carrying_costs.project_market_value_schedule) -- not swept, a single fixed reference point for
    every scenario.

    Note scenario.purchase_price is this deck's actual input for purchase price --
    acquisition.purchase_price is a discoverability stub only (see the deck's "SEE scenario.X"
    note); a scenario dict's "purchase_price" key is what acquisition.compute_acquisition_costs
    and any other per-scenario computation should read.
    """
    deck = deck if deck is not None else load_deck()
    scenario = deck["scenario"]
    reference = _constant(scenario["reference"])
    property_state = _constant(scenario["property_state"])
    closing_month = _constant(scenario["closing_month"])
    closing_year = _constant(scenario["closing_year"])
    loan_repaid_values = _sweep(scenario["family_loan_repaid_at_sale"])
    if not loan_repaid_values or any(not isinstance(v, bool) for v in loan_repaid_values):
        raise ValueError(f"scenario.family_loan_repaid_at_sale must list true and/or false, got {loan_repaid_values!r}")

    scenarios = []
    for alternative in _sweep(scenario["alternatives"]):
        for horizon_years in _sweep(scenario["horizon_years"]):
            for residency_state in _sweep(scenario["residency_state"]):
                for purchase_price in _sweep(scenario["purchase_price"]):
                    for family_loan_repaid_at_sale in loan_repaid_values:
                        scenarios.append(
                            {
                                "reference": reference,
                                "alternative": alternative,
                                "horizon_years": horizon_years,
                                "residency_state": residency_state,
                                "property_state": property_state,
                                "purchase_price": purchase_price,
                                "family_loan_repaid_at_sale": family_loan_repaid_at_sale,
                                "closing_month": closing_month,
                                "closing_year": closing_year,
                            }
                        )
    return scenarios


def get_closing_date(deck: dict[str, Any] | None = None) -> dict[str, int]:
    """Returns scenario.closing_month/closing_year (see get_scenarios' own docstring -- TIMESTEP 0
    for every month-indexed model in this deck, e.g. carrying_costs.project_market_value_schedule).
    Returns {"closing_month", "closing_year"}."""
    deck = deck if deck is not None else load_deck()
    scenario = deck["scenario"]
    return {
        "closing_month": _constant(scenario["closing_month"]),
        "closing_year": _constant(scenario["closing_year"]),
    }


def get_acquisition_inputs(deck: dict[str, Any] | None = None) -> dict[str, float]:
    """Unwraps `acquisition`'s own `kind: constant` fields to plain values -- everything except
    purchase_price, which is a discoverability stub only (the real value comes from a scenario's
    own "purchase_price", see get_scenarios). Returns:
    {"closing_costs_fixed_fees", "closing_costs_percent_of_price", "building_fraction",
    "personal_cash_at_closing", "family_loan", "initial_furnishing"}, straight off
    acquisition.purchase_closing_costs.fixed_fees/.percent_of_price, acquisition.percent_improved
    (the listing sheet's "% Improved", 0-100, returned as a 0-1 building_fraction -- the depreciable
    share), acquisition.personal_cash_at_closing, acquisition.family_loan, acquisition.initial_furnishing
    respectively. total_cash_outlay (personal_cash_at_closing + family_loan) is DERIVED, not read
    here -- see acquisition.compute_acquisition_costs.
    """
    deck = deck if deck is not None else load_deck()
    acquisition = deck["acquisition"]
    closing_costs = acquisition["purchase_closing_costs"]
    return {
        "closing_costs_fixed_fees": _constant(closing_costs["fixed_fees"]),
        "closing_costs_percent_of_price": _constant(closing_costs["percent_of_price"]),
        "building_fraction": _fraction(
            _constant(acquisition["percent_improved"]), "acquisition.percent_improved", upper=100
        )
        / 100,
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


RENTAL_SOURCE_KINDS = ("history", "estimate")
RENTAL_SOURCE_RESOLUTIONS = ("annual", "monthly")


def _fraction(value: Any, where: str, *, upper: float = 1.0, upper_inclusive: bool = True) -> float:
    """Validates a fraction in [0, upper] (or [0, upper) if not upper_inclusive) and returns it as a float."""
    if not _is_number(value):
        raise ValueError(f"{where}: expected a number, got {value!r}")
    ok = 0 <= value <= upper if upper_inclusive else 0 <= value < upper
    if not ok:
        bracket = "]" if upper_inclusive else ")"
        raise ValueError(f"{where}: must be in [0, {upper}{bracket}, got {value}")
    return float(value)


def get_rental_operations_inputs(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Unwraps and validates `rental_operations`'s `kind: constant` fields (everything except
    `sources`, see get_rental_operations_sources). Returns:
      managed_by: self | company (read by classification.get_loan_occupancy_class);
      occupancy_adr_correlation: in [-1, 1];
      adr_growth_latent: name of the derived latent that grows ADR;
      channel_commission_fraction, mgmt_fee_fraction: in [0, 1). Net rental income =
        gross x (1 - channel_commission_fraction) x (1 - mgmt_fee_fraction)."""
    deck = deck if deck is not None else load_deck()
    rental_operations = deck["rental_operations"]
    where = "rental_operations"
    managed_by = _constant(rental_operations["managed_by"])
    if managed_by not in ("self", "company"):
        raise ValueError(f"{where}.managed_by: must be self or company, got {managed_by!r}")
    correlation = _constant(rental_operations["occupancy_adr_correlation"])
    if not _is_number(correlation) or not -1 <= correlation <= 1:
        raise ValueError(f"{where}.occupancy_adr_correlation: must be a number in [-1, 1], got {correlation!r}")
    return {
        "managed_by": managed_by,
        "occupancy_adr_correlation": float(correlation),
        "adr_growth_latent": str(_constant(rental_operations["adr_growth_latent"])),
        "channel_commission_fraction": _fraction(
            _constant(rental_operations["channel_commission_fraction"]),
            f"{where}.channel_commission_fraction",
            upper_inclusive=False,
        ),
        "mgmt_fee_fraction": _fraction(
            _constant(rental_operations["mgmt_fee_fraction"]), f"{where}.mgmt_fee_fraction", upper_inclusive=False
        ),
    }


def _rental_observation(raw: Any, where: str) -> dict[str, float | None]:
    """Validates one {occupancy, adr} pair: occupancy in [0, 1]; adr > 0 when occupancy > 0 (no
    bookings -> no ADR, so adr must be null/omitted when occupancy is 0)."""
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected {{occupancy, adr}}, got {raw!r}")
    occupancy = _fraction(raw.get("occupancy"), f"{where}.occupancy")
    adr = raw.get("adr")
    if occupancy == 0:
        if adr is not None:
            raise ValueError(f"{where}: occupancy is 0 (no bookings), so adr must be null, got {adr!r}")
        return {"occupancy": 0.0, "adr": None}
    if not _is_number(adr) or float(cast(float, adr)) <= 0:
        raise ValueError(f"{where}.adr: occupancy is {occupancy}, so adr must be a number > 0, got {adr!r}")
    return {"occupancy": occupancy, "adr": float(cast(float, adr))}


def get_rental_operations_sources(deck: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Validates and normalizes `rental_operations.sources` (see that field's deck comment for what
    each field means). Returns EVERY source, included or not (the fit plots show excluded ones too),
    each as {"label", "include", "kind", "year", "resolution", "occupancy"/"adr" (annual only),
    "months" (monthly only: {month: {"occupancy", "adr" (None if no bookings)}}),
    "owner_nights_by_month" (history only, None if not given)}. Raises if no source is included."""
    deck = deck if deck is not None else load_deck()
    raw_sources = deck["rental_operations"]["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("rental_operations.sources: must be a non-empty list (see its deck comment)")

    sources = []
    for index, raw in enumerate(raw_sources):
        label = raw.get("label") or f"source {index + 1}"
        where = f"rental_operations.sources[{index}] ({label})"
        include = raw.get("include", True)
        if not isinstance(include, bool):
            raise ValueError(f"{where}.include: must be true or false, got {include!r}")
        kind = raw.get("kind")
        if kind not in RENTAL_SOURCE_KINDS:
            raise ValueError(f"{where}.kind: must be one of {RENTAL_SOURCE_KINDS}, got {kind!r}")
        year = raw.get("year")
        if not isinstance(year, int) or isinstance(year, bool):
            raise ValueError(f"{where}.year: must be a whole year (e.g. 2025), got {year!r}")
        resolution = raw.get("resolution")
        if resolution not in RENTAL_SOURCE_RESOLUTIONS:
            raise ValueError(f"{where}.resolution: must be one of {RENTAL_SOURCE_RESOLUTIONS}, got {resolution!r}")

        source: dict[str, Any] = {"label": label, "include": include, "kind": kind, "year": year, "resolution": resolution}
        if resolution == "annual":
            source.update(_rental_observation(raw, where))
        else:
            months = raw.get("months")
            if not isinstance(months, dict) or set(months) != set(MONTH_KEYS):
                raise ValueError(f"{where}.months: must list all 12 months ({', '.join(MONTH_KEYS)})")
            source["months"] = {month: _rental_observation(months[month], f"{where}.months.{month}") for month in MONTH_KEYS}

        owner_nights = raw.get("owner_nights_by_month")
        if owner_nights is not None:
            if kind != "history":
                raise ValueError(f"{where}.owner_nights_by_month: only meaningful for kind: history")
            if not isinstance(owner_nights, dict) or set(owner_nights) != set(MONTH_KEYS):
                raise ValueError(f"{where}.owner_nights_by_month: must list all 12 months")
            for month in MONTH_KEYS:
                nights = owner_nights[month]
                if not _is_number(nights) or not 0 <= nights <= DAYS_IN_MONTH[month]:
                    raise ValueError(f"{where}.owner_nights_by_month.{month}: must be 0..{DAYS_IN_MONTH[month]}, got {nights!r}")
            owner_nights = {month: float(owner_nights[month]) for month in MONTH_KEYS}
        source["owner_nights_by_month"] = owner_nights
        sources.append(source)

    if not any(source["include"] for source in sources):
        raise ValueError("rental_operations.sources: at least one source must have include: true")
    return sources


def get_personal_use_days_by_month(deck: dict[str, Any] | None = None) -> dict[str, float]:
    """personal_use.personal_use_days_by_month as {month: days}, each checked to fit in its month
    (no leap-day February)."""
    deck = deck if deck is not None else load_deck()
    days_by_month = _constant(deck["personal_use"]["personal_use_days_by_month"])
    result = {}
    for month in MONTH_KEYS:
        days = days_by_month[month]
        if not _is_number(days) or not 0 <= days <= DAYS_IN_MONTH[month]:
            raise ValueError(f"personal_use.personal_use_days_by_month.{month}: must be 0..{DAYS_IN_MONTH[month]}, got {days!r}")
        result[month] = float(days)
    return result


def get_annual_personal_days(deck: dict[str, Any] | None = None) -> float:
    """Sums personal_use.personal_use_days_by_month (`kind: constant`, a plain {month: days} dict)
    into an annual personal-use-days figure. Used only for occupancy/tax classification (see
    classification.get_tax_use_class)."""
    return sum(get_personal_use_days_by_month(deck).values())


def get_personal_use_inputs(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Unwraps and validates section 8 (`personal_use`). Returns:
      personal_use_days_by_month: {month: days} (see get_personal_use_days_by_month);
      alternate_lodging_cost_by_month: {month: USD >= 0}, today's (closing-date) dollars, charged IN
        FULL every month as entered -- no proration by personal days or bookings (user decision,
        2026-11-23: the user works out what they'd really pay and enters that);
      alternate_lodging_growth_latent: name of the derived latent that grows it (resolved later by
        get_growth_fit)."""
    deck = deck if deck is not None else load_deck()
    personal_use = deck["personal_use"]
    return {
        "personal_use_days_by_month": get_personal_use_days_by_month(deck),
        "alternate_lodging_cost_by_month": _monthly_costs(
            personal_use["alternate_lodging_cost_by_month"], "personal_use.alternate_lodging_cost_by_month"
        ),
        "alternate_lodging_growth_latent": str(_constant(personal_use["alternate_lodging_growth_latent"])),
    }


def get_rent_reference_inputs(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Unwraps and validates section 10 (`rent_reference`). Returns:
      lease_cost_by_month: {month: USD >= 0}, today's (closing-date) dollars, all-in, charged IN FULL every
        month as entered (any lease arrangement -- 12-month, month to month, seasonal -- is expressed by the
        user's own month values);
      lease_growth_latent: name of the latent that grows it, ONCE A YEAR at each closing anniversary
        (resolved later by get_growth_fit)."""
    deck = deck if deck is not None else load_deck()
    rent_reference = deck["rent_reference"]
    return {
        "lease_cost_by_month": _monthly_costs(rent_reference["lease_cost_by_month"], "rent_reference.lease_cost_by_month"),
        "lease_growth_latent": str(_constant(rent_reference["lease_growth_latent"])),
    }


def _monthly_costs(entry: Any, where: str) -> dict[str, float]:
    """A `kind: constant` {month: USD/month} map: all 12 months, each a number >= 0."""
    costs = _constant(entry)
    if not isinstance(costs, dict) or set(costs) != set(MONTH_KEYS):
        raise ValueError(f"{where}: must list all 12 months ({', '.join(MONTH_KEYS)})")
    for month in MONTH_KEYS:
        if not _is_number(costs[month]) or costs[month] < 0:
            raise ValueError(f"{where}.{month}: must be a number >= 0 (USD/month), got {costs[month]!r}")
    return {month: float(costs[month]) for month in MONTH_KEYS}


def get_property_tax_inputs(deck: dict[str, Any] | None = None) -> dict[str, dict[str, float]]:
    """Unwraps `carrying_costs.property_tax.mill_levy` -- the parcel's own levies, one per component of
    the property state's property-tax model (CO: local, school), each a plain fraction >= 0. The
    mechanic itself (assessment rates, value reductions, classification) comes from the tax
    jurisdiction's state layer (jurisdictions.TaxJurisdiction.property_tax_model), which also checks the
    component names match. Returns {"mill_levies": {component: fraction}}. market_value has its own
    dedicated shape -- see get_market_value_config."""
    deck = deck if deck is not None else load_deck()
    raw = deck["carrying_costs"]["property_tax"]["mill_levy"]
    mill_levies = {}
    for name, entry in raw.items():
        value = _constant(entry)
        if not _is_number(value) or value < 0:
            raise ValueError(f"carrying_costs.property_tax.mill_levy.{name} must be a number >= 0, got {value!r}")
        mill_levies[str(name)] = float(value)
    return {"mill_levies": mill_levies}


def get_hoa_dues_config(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Unwraps carrying_costs.hoa_dues_annual's KNOWN-VALUE+GROWTH shape (REDESIGNED 2026-09-30 --
    see that field's own deck comment for the full reasoning). Two separate REAL truth inputs --
    `initial_annual_dues` (the regular monthly-billed dues, annualized) and
    `initial_annual_association_dues` (a separate once-a-year special charge) -- are COMBINED into
    ONE value (utils.prior_utils.build_hoa_dues_prior_model) that gets grown/uncertain TOGETHER via
    `latent` (construction_cost)'s own fitted OU process: the user's own explicit call, so the
    association charge shares in the same growth and uncertainty as the rest instead of being frozen
    at today's dollar amount forever. No hand-elicited cv (0% uncertain at h=0, same role as
    market_value.initial_price) and NO floor/clamp of any kind (an earlier version of this design
    clipped the whole band at the initial value to enforce "dues only go up" -- the user correctly
    called this out as producing a mixed/censored shape, not a real lognormal one; removed entirely --
    a real lognormal, driven purely by construction_cost's own fitted drift, is trusted as-is).
    Returns {"initial_annual_dues", "initial_annual_association_dues", "latent"}."""
    deck = deck if deck is not None else load_deck()
    entry = deck["carrying_costs"]["hoa_dues_annual"]
    _expect_kind(entry, "known_plus_growth", "carrying_costs.hoa_dues_annual")
    return {
        "initial_annual_dues": entry["initial_annual_dues"],
        "initial_annual_association_dues": entry["initial_annual_association_dues"],
        "latent": entry["growth"]["latent"],
    }


def get_insurance_config(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Unwraps carrying_costs.insurance_annual's GUESS+GROWTH shape (REDESIGNED 2026-09-30: moved
    from a KNOWN-VALUE shape to the same shape maintenance_annual uses, per the user's own
    correction -- a real current insurance premium isn't actually known truth data the way
    hoa_dues_annual's dues are; it's an ESTIMATE, so there should be real point-in-time uncertainty
    about it too). `initial_guess_premium` is a genuine hand-elicited guess; its own point-uncertainty
    is DERIVED from `latent`'s own fitted process rather than a separately hand-picked cv -- see
    get_maintenance_config's own docstring for the exact reasoning, shared identically here (both
    built by utils.prior_utils's shared _build_guess_plus_growth_model). Returns
    {"initial_guess_premium", "latent"}."""
    deck = deck if deck is not None else load_deck()
    entry = deck["carrying_costs"]["insurance_annual"]
    _expect_kind(entry, "guess_plus_growth", "carrying_costs.insurance_annual")
    return {"initial_guess_premium": entry["initial_guess_premium"], "latent": entry["growth"]["latent"]}


def get_maintenance_config(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Unwraps carrying_costs.maintenance_annual's GUESS+GROWTH shape: there's no real truth data for
    TODAY's maintenance cost -- `initial_guess_mean` is a genuine hand-elicited guess, so there IS
    real point-in-time uncertainty about it. Rather than a separately hand-picked cv, that
    point-uncertainty is DERIVED from `latent`'s own fitted process (see utils.prior_utils.
    _build_guess_plus_growth_model for the exact formula: the OU process's own marginal/stationary
    rate variance, sig^2*tau/2, applied as a CONSTANT extra_log_variance -- i.e. "how uncertain is a
    plausible guess about the level right now," not "how much does this compound over a 10-year
    horizon," which is what a naively-large cv would otherwise smuggle in). Returns
    {"initial_guess_mean", "latent"}."""
    deck = deck if deck is not None else load_deck()
    entry = deck["carrying_costs"]["maintenance_annual"]
    _expect_kind(entry, "guess_plus_growth", "carrying_costs.maintenance_annual")
    return {"initial_guess_mean": entry["initial_guess_mean"], "latent": entry["growth"]["latent"]}


def get_market_value_config(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Unwraps carrying_costs.property_tax.market_value's OWN shape (see that field's own deck
    comment for why it's not the generic LEVEL+GROWTH pattern -- a closing-date-anchored,
    averaged-window stair-step instead, and for the "reassessment means the EFFECTIVE DATE, not the
    averaging window" clarification). Returns {"initial_price", "reassessment_frequency_years",
    "first_reassessment_year" (an int -- 1 = the January immediately after closing_year, see that
    field's own deck comment), "latent", "validation_data_source" (a filename under data/manual/, or
    None if the field is absent/blank -- see that field's own deck comment; used by
    utils.prior_utils.load_market_value_history)}. Used by carrying_costs.project_market_value_schedule
    and utils.prior_utils.build_market_value_change_magnitude."""
    deck = deck if deck is not None else load_deck()
    entry = deck["carrying_costs"]["property_tax"]["market_value"]
    _expect_kind(entry, "market_value_schedule", "carrying_costs.property_tax.market_value")
    return {
        "initial_price": entry["initial_price"],
        "reassessment_frequency_years": entry["reassessment_frequency_years"],
        "first_reassessment_year": entry["first_reassessment_year"],
        "latent": entry["latent"],
        "validation_data_source": entry.get("validation_data_source") or None,
    }


SPECIAL_ASSESSMENT_TIMINGS = ("uniform", "worst_case_year1")
SPECIAL_ASSESSMENT_NO_GROWTH = "none"


def get_special_assessment_config(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Unwraps and VALIDATES carrying_costs.special_assessment's frequency_severity shape
    (redesigned 2026-11-22 from the old p x lognormal-severity mixture -- see the deck comment).
    Validated here, not later, because every bad value below would otherwise produce a silently
    wrong probability table rather than a crash: probabilities that don't sum to 1 make the whole
    cost distribution sum to something other than 100%, and a negative rate or amount has no
    meaning. Same "raise on an invalid deck state rather than guess" convention as
    get_derived_latents.

    Returns {"annual_rate", "size_scenarios" (list of {"label", "amount_today_usd", "probability"}),
    "latent" (a forecast_models/derived_latents key, or "none" for no size growth), "timing",
    "k_max"}."""
    deck = deck if deck is not None else load_deck()
    where = "carrying_costs.special_assessment"
    entry = deck["carrying_costs"]["special_assessment"]
    _expect_kind(entry, "frequency_severity", where)

    annual_rate = entry["annual_rate"]
    if isinstance(annual_rate, bool) or not isinstance(annual_rate, int | float) or not annual_rate >= 0:
        raise ValueError(f"{where}.annual_rate must be a number >= 0, got {annual_rate!r}")

    scenarios = entry["size_scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"{where}.size_scenarios must be a non-empty list")
    labels = [s.get("label") for s in scenarios]
    if any(not isinstance(label, str) or not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError(f"{where}.size_scenarios: every label must be a non-empty, unique string, got {labels!r}")
    for s in scenarios:
        for key in ("amount_today_usd", "probability"):
            value = s.get(key)
            if isinstance(value, bool) or not isinstance(value, int | float) or not value >= 0:
                raise ValueError(f"{where}.size_scenarios[{s['label']}].{key} must be a number >= 0, got {value!r}")
    probability_total = sum(s["probability"] for s in scenarios)
    if abs(probability_total - 1.0) > 1e-6:
        raise ValueError(
            f"{where}.size_scenarios probabilities must sum to 1 (they're conditional on an assessment "
            f"happening), got {probability_total}"
        )

    latent = entry["growth"]["latent"]
    known_latents = set(deck["forecast_models"]) | set(deck["derived_latents"]) | {SPECIAL_ASSESSMENT_NO_GROWTH}
    if latent not in known_latents:
        raise ValueError(f"{where}.growth.latent must be one of {sorted(known_latents)}, got {latent!r}")

    timing = entry["timing"]
    if timing not in SPECIAL_ASSESSMENT_TIMINGS:
        raise ValueError(f"{where}.timing must be one of {SPECIAL_ASSESSMENT_TIMINGS}, got {timing!r}")

    k_max = entry["k_max"]
    if isinstance(k_max, bool) or not isinstance(k_max, int) or k_max < 1:
        raise ValueError(f"{where}.k_max must be an integer >= 1, got {k_max!r}")

    return {
        "annual_rate": float(annual_rate),
        "size_scenarios": [
            {"label": s["label"], "amount_today_usd": float(s["amount_today_usd"]), "probability": float(s["probability"])}
            for s in scenarios
        ],
        "latent": latent,
        "timing": timing,
        "k_max": k_max,
    }


MONTH_KEYS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
# Non-leap days per calendar month. February uses 28: leap years add one day every 4 years, a ~0.1%
# effect on annual figures, not worth modeling.
DAYS_IN_MONTH = dict(zip(MONTH_KEYS, (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), strict=True))
UTILITY_COMPONENTS = ("electric", "gas", "water", "sewer", "trash", "internet")
UTILITIES_CALIBRATION_OCCUPANCY_FROM_DECK = "deck"
UTILITIES_CALIBRATION_OCCUPANCY_FROM_HISTORY = "history"


def _is_number(value: Any) -> bool:
    """True for a real int/float (YAML numbers), False for bools (which are ints in Python) and
    anything else -- so `True` can never sneak through as a 1."""
    return not isinstance(value, bool) and isinstance(value, int | float)


def get_utilities_prior_config(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Unwraps and VALIDATES carrying_costs.utilities' THREE-CATEGORY shape (redesigned 2026-11-22,
    replacing the temperature-sine seasonal shape -- see the deck comment and
    utils.prior_utils.build_utilities_prior_model). Validated here because every bad value below
    would otherwise yield a silently wrong calibration rather than a crash (e.g. weights not summing
    to 1 would make year 1 no longer match the observed bill). heating_type itself is checked by
    carrying_costs.resolve_heating_type, next to the HEATING_TYPES table it indexes (that table
    can't live here: carrying_costs imports this module, not the other way around).

    Returns {"initial_annual_average_monthly_cost", "calibration_occupied_days_by_month" ("deck",
    "history" or a {month: days} dict), "components" ({name: {"weight", "occupancy_fraction"}}, every name in
    UTILITY_COMPONENTS, missing ones filled with zeros), "heating_type", "heating_share",
    "occupied_setpoint_f", "vacant_setpoint_f", "daily_temperature_std_f", "additional_flat_monthly" (list of {"label",
    "amount_usd"}), "seasonal_data_source", "latent"}."""
    deck = deck if deck is not None else load_deck()
    where = "carrying_costs.utilities"
    entry = deck["carrying_costs"]["utilities"]
    _expect_kind(entry, "three_category", where)

    average = entry["initial_annual_average_monthly_cost"]
    if not _is_number(average) or not average > 0:
        raise ValueError(f"{where}.initial_annual_average_monthly_cost must be a number > 0, got {average!r}")

    calibration = entry["calibration_occupied_days_by_month"]
    if calibration not in (UTILITIES_CALIBRATION_OCCUPANCY_FROM_DECK, UTILITIES_CALIBRATION_OCCUPANCY_FROM_HISTORY):
        if not isinstance(calibration, dict) or set(calibration) != set(MONTH_KEYS):
            raise ValueError(
                f"{where}.calibration_occupied_days_by_month must be 'deck', 'history' or a map of all 12 months "
                f"({', '.join(MONTH_KEYS)}), got {calibration!r}"
            )
        bad = {m: d for m, d in calibration.items() if not _is_number(d) or d < 0}
        if bad:
            raise ValueError(f"{where}.calibration_occupied_days_by_month: days must be numbers >= 0, got {bad}")
        calibration = {month: float(calibration[month]) for month in MONTH_KEYS}

    raw_components = entry["components"]
    unknown = set(raw_components) - set(UTILITY_COMPONENTS)
    if unknown:
        raise ValueError(f"{where}.components: unknown component(s) {sorted(unknown)}; allowed: {UTILITY_COMPONENTS}")
    components = {}
    for name in UTILITY_COMPONENTS:
        component = raw_components.get(name, {"weight": 0.0, "occupancy_fraction": 0.0})
        weight, fraction = component.get("weight"), component.get("occupancy_fraction")
        if not _is_number(weight) or weight < 0:
            raise ValueError(f"{where}.components.{name}.weight must be a number >= 0, got {weight!r}")
        if not _is_number(fraction) or not 0 <= fraction <= 1:
            raise ValueError(f"{where}.components.{name}.occupancy_fraction must be between 0 and 1, got {fraction!r}")
        components[name] = {"weight": float(weight), "occupancy_fraction": float(fraction)}
    weight_total = sum(c["weight"] for c in components.values())
    if abs(weight_total - 1.0) > 1e-6:
        raise ValueError(f"{where}.components weights must sum to 1, got {weight_total}")

    heating_share = entry["heating_share"]
    if not _is_number(heating_share) or not 0 <= heating_share < 1:
        raise ValueError(f"{where}.heating_share must be >= 0 and < 1, got {heating_share!r}")
    occupied_setpoint = entry["occupied_setpoint_f"]
    if not _is_number(occupied_setpoint):
        raise ValueError(f"{where}.occupied_setpoint_f must be a number, got {occupied_setpoint!r}")
    vacant_setpoint = entry["vacant_setpoint_f"]
    if not _is_number(vacant_setpoint) or vacant_setpoint > occupied_setpoint:
        raise ValueError(
            f"{where}.vacant_setpoint_f must be a number <= occupied_setpoint_f ({occupied_setpoint!r}), got {vacant_setpoint!r}"
        )
    temperature_std = entry["daily_temperature_std_f"]
    if not _is_number(temperature_std) or not temperature_std > 0:
        raise ValueError(f"{where}.daily_temperature_std_f must be a number > 0, got {temperature_std!r}")
    heating_type = entry["heating_type"]
    if not isinstance(heating_type, int) or isinstance(heating_type, bool):
        raise ValueError(f"{where}.heating_type must be an integer code, got {heating_type!r}")

    add_ons = entry.get("additional_flat_monthly") or []
    if not isinstance(add_ons, list):
        raise ValueError(f"{where}.additional_flat_monthly must be a list of {{label, amount_usd}}, got {add_ons!r}")
    labels = [a.get("label") for a in add_ons]
    if any(not isinstance(label, str) or not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError(f"{where}.additional_flat_monthly: labels must be non-empty and unique, got {labels!r}")
    for a in add_ons:
        if not _is_number(a.get("amount_usd")) or a["amount_usd"] < 0:
            raise ValueError(f"{where}.additional_flat_monthly[{a['label']}].amount_usd must be >= 0, got {a.get('amount_usd')!r}")

    latent = entry["growth"]["latent"]
    known_latents = set(deck["forecast_models"]) | set(deck["derived_latents"])
    if latent not in known_latents:
        raise ValueError(f"{where}.growth.latent must be one of {sorted(known_latents)}, got {latent!r}")

    return {
        "initial_annual_average_monthly_cost": float(average),
        "calibration_occupied_days_by_month": calibration,
        "components": components,
        "heating_type": heating_type,
        "heating_share": float(heating_share),
        "occupied_setpoint_f": float(occupied_setpoint),
        "vacant_setpoint_f": float(vacant_setpoint),
        "daily_temperature_std_f": float(temperature_std),
        "additional_flat_monthly": [{"label": a["label"], "amount_usd": float(a["amount_usd"])} for a in add_ons],
        "seasonal_data_source": entry["seasonal_data_source"],
        "latent": latent,
    }


TAX_FILING_STATUSES = ("mfj",)


def get_tax_jurisdiction_selection(deck: dict[str, Any] | None = None) -> dict[str, str | None]:
    """Unwraps `taxes.jurisdiction` -> {"state", "county", "municipality"} (county/municipality may be
    None = that layer off). Checks the shape and that state matches scenario.property_state; whether
    the layers exist on disk is checked when jurisdictions.load_tax_jurisdiction loads them."""
    deck = deck if deck is not None else load_deck()
    raw = _constant(deck["taxes"]["jurisdiction"])
    if not isinstance(raw, dict) or not raw.get("state"):
        raise ValueError(f"taxes.jurisdiction must be {{state, county, municipality}} with a state, got {raw!r}")
    unknown = set(raw) - {"state", "county", "municipality"}
    if unknown:
        raise ValueError(f"taxes.jurisdiction has unknown key(s) {sorted(unknown)}")
    selection = {level: (str(raw[level]) if raw.get(level) else None) for level in ("state", "county", "municipality")}
    if selection["municipality"] and not selection["county"]:
        raise ValueError("taxes.jurisdiction: a municipality needs its county selected too")
    property_state = _constant(deck["scenario"]["property_state"])
    if selection["state"] != property_state:
        raise ValueError(
            f"taxes.jurisdiction.state={selection['state']!r} must match scenario.property_state={property_state!r}"
        )
    return selection


def get_taxes_inputs(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Unwraps `taxes`' household facts (the law lives in data/tax_jurisdictions/, selected by
    get_tax_jurisdiction_selection). Returns {"jurisdiction" (see get_tax_jurisdiction_selection),
    "filing_status", "household_agi", "other_itemized_deductions", "other_salt_paid" (USD/year, >= 0),
    "average_guest_stay_nights" (> 0), "owner_materially_participates" (bool)}."""
    deck = deck if deck is not None else load_deck()
    taxes = deck["taxes"]
    filing_status = _constant(taxes["filing_status"])
    if filing_status not in TAX_FILING_STATUSES:
        raise ValueError(f"taxes.filing_status must be one of {TAX_FILING_STATUSES}, got {filing_status!r}")
    dollars = {}
    for name in ("household_agi", "other_itemized_deductions", "other_salt_paid"):
        value = _constant(taxes[name])
        if not _is_number(value) or value < 0:
            raise ValueError(f"taxes.{name} must be a number >= 0 (USD/year), got {value!r}")
        dollars[name] = float(value)
    stay = _constant(taxes["average_guest_stay_nights"])
    if not _is_number(stay) or stay <= 0:
        raise ValueError(f"taxes.average_guest_stay_nights must be a number > 0, got {stay!r}")
    participates = _constant(taxes["owner_materially_participates"])
    if not isinstance(participates, bool):
        raise ValueError(f"taxes.owner_materially_participates must be true or false, got {participates!r}")
    return {
        "jurisdiction": get_tax_jurisdiction_selection(deck),
        "filing_status": filing_status,
        **dollars,
        "average_guest_stay_nights": float(stay),
        "owner_materially_participates": participates,
    }


def get_exit_inputs(deck: dict[str, Any] | None = None) -> dict[str, float]:
    """Unwraps `exit.selling_costs`' own `kind: constant` fields. Returns
    {"selling_costs_fixed_fees", "selling_costs_percent_of_price"}. exit_year/sale_price are
    DERIVED, not read here -- see carrying_costs.project_sale_price."""
    deck = deck if deck is not None else load_deck()
    selling_costs = deck["exit"]["selling_costs"]
    return {
        "selling_costs_fixed_fees": _constant(selling_costs["fixed_fees"]),
        "selling_costs_percent_of_price": _constant(selling_costs["percent_of_price"]),
    }


CAPITAL_MARKETS_ACCOUNTS = ("market", "cash")


def get_capital_markets_inputs(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Unwraps and validates section 11 (`capital_markets`). Returns:
      initial_investment: acquisition.personal_cash_at_closing (USD; the stub in capital_markets only
        points there -- never family_loan or total_cash_outlay, see the deck comment);
      allocation: {"market", "cash"} fractions >= 0 summing to 1;
      cash_yield: annual interest rate on the cash account (>= 0);
      dividend_paying_fraction (0-1) and dividend_yield (>= 0): effective dividend yield on the market
        account = paying_fraction x yield."""
    deck = deck if deck is not None else load_deck()
    section = deck["capital_markets"]
    if _constant(section["initial_investment"]) is not None:
        raise ValueError(
            "capital_markets.initial_investment must stay null: the amount is acquisition.personal_cash_at_closing"
        )
    investment = get_acquisition_inputs(deck)["personal_cash_at_closing"]
    if not _is_number(investment) or investment < 0:
        raise ValueError(f"acquisition.personal_cash_at_closing must be a number >= 0, got {investment!r}")

    allocation = _constant(section["allocation"])
    if not isinstance(allocation, dict) or set(allocation) != set(CAPITAL_MARKETS_ACCOUNTS):
        raise ValueError(f"capital_markets.allocation must list exactly {CAPITAL_MARKETS_ACCOUNTS}")
    for account in CAPITAL_MARKETS_ACCOUNTS:
        if not _is_number(allocation[account]) or allocation[account] < 0:
            raise ValueError(f"capital_markets.allocation.{account} must be a number >= 0, got {allocation[account]!r}")
    if abs(sum(allocation.values()) - 1.0) > 1e-9:
        raise ValueError(f"capital_markets.allocation must add up to 1, got {sum(allocation.values()):g}")

    cash_yield = _constant(section["cash_yield"])
    paying = _constant(section["dividends"]["paying_fraction"])
    dividend_yield = _constant(section["dividends"]["yield"])
    if not _is_number(cash_yield) or cash_yield < 0:
        raise ValueError(f"capital_markets.cash_yield must be a number >= 0, got {cash_yield!r}")
    if not _is_number(paying) or not 0 <= paying <= 1:
        raise ValueError(f"capital_markets.dividends.paying_fraction must be 0..1, got {paying!r}")
    if not _is_number(dividend_yield) or dividend_yield < 0:
        raise ValueError(f"capital_markets.dividends.yield must be a number >= 0, got {dividend_yield!r}")
    return {
        "initial_investment": float(investment),
        "allocation": {account: float(allocation[account]) for account in CAPITAL_MARKETS_ACCOUNTS},
        "cash_yield": float(cash_yield),
        "dividend_paying_fraction": float(paying),
        "dividend_yield": float(dividend_yield),
    }


def get_report_quantiles(deck: dict[str, Any] | None = None) -> list[float]:
    """Returns run.report_quantiles (e.g. [0.1, 0.5, 0.9]), validated: at least 2 probabilities, each in
    (0, 1), strictly increasing. Only the FIRST and LAST are used (every reported band); the median is
    always reported separately. (May become a two-value report_band later -- see the deck note.)"""
    deck = deck if deck is not None else load_deck()
    quantiles = _constant(deck["run"]["report_quantiles"])
    if (not isinstance(quantiles, list) or len(quantiles) < 2 or not all(_is_number(q) and 0 < q < 1 for q in quantiles)
            or any(b <= a for a, b in zip(quantiles, quantiles[1:], strict=False))):
        raise ValueError(f"run.report_quantiles must be >= 2 strictly increasing probabilities in (0, 1), got {quantiles!r}")
    return [float(q) for q in quantiles]


def get_report_band(deck: dict[str, Any] | None = None) -> tuple[float, float]:
    """(first, last) of run.report_quantiles: the reported uncertainty band."""
    quantiles = get_report_quantiles(deck)
    return quantiles[0], quantiles[-1]


def get_outer_band(deck: dict[str, Any] | None = None) -> tuple[float, float]:
    """run.outer_band (lo, hi): the wide reference band on every plot and ΔW's outer band. Validated: both in
    (0, 1), lo below report_quantiles' first value, hi above its last."""
    deck = deck if deck is not None else load_deck()
    band = deck["run"]["outer_band"]
    lo, hi = _constant(band["lo"]), _constant(band["hi"])
    report_lo, report_hi = get_report_band(deck)
    if not (_is_number(lo) and _is_number(hi) and 0 < lo < report_lo and report_hi < hi < 1):
        raise ValueError(
            f"run.outer_band must satisfy 0 < lo < {report_lo:g} (report_quantiles' first) and "
            f"{report_hi:g} (its last) < hi < 1, got lo={lo!r}, hi={hi!r}"
        )
    return float(lo), float(hi)


def format_band(lo: float, hi: float) -> str:
    """'1-99%' style label for a probability band (plot legends and titles)."""
    return f"{lo * 100:g}-{hi * 100:g}%"


def get_solver_settings(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """run.solver, validated. Returns {"exit_price_integration_points" (int >= 5), "delta_w_moment_grouping_months"
    (int >= 1), "cornish_fisher_z_range" (float > check_z), "cornish_fisher_grid_points" (int >= 101),
    "cornish_fisher_check_z" (float, 0 < check_z < z_range)}. See the deck's comments for what each does."""
    deck = deck if deck is not None else load_deck()
    solver = deck["run"]["solver"]
    cf = solver["cornish_fisher"]
    points = _constant(solver["exit_price_integration_points"])
    grouping = _constant(solver["delta_w_moment_grouping_months"])
    z_range, grid, check_z = _constant(cf["z_range"]), _constant(cf["grid_points"]), _constant(cf["check_z"])
    if not (isinstance(points, int) and not isinstance(points, bool) and points >= 5):
        raise ValueError(f"run.solver.exit_price_integration_points must be a whole number >= 5, got {points!r}")
    if not (isinstance(grouping, int) and not isinstance(grouping, bool) and grouping >= 1):
        raise ValueError(f"run.solver.delta_w_moment_grouping_months must be a whole number >= 1, got {grouping!r}")
    if not (isinstance(grid, int) and not isinstance(grid, bool) and grid >= 101):
        raise ValueError(f"run.solver.cornish_fisher.grid_points must be a whole number >= 101, got {grid!r}")
    if not (_is_number(z_range) and _is_number(check_z) and 0 < check_z < z_range):
        raise ValueError(
            f"run.solver.cornish_fisher needs 0 < check_z < z_range, got check_z={check_z!r}, z_range={z_range!r}"
        )
    return {
        "exit_price_integration_points": points,
        "delta_w_moment_grouping_months": grouping,
        "cornish_fisher_z_range": float(z_range),
        "cornish_fisher_grid_points": grid,
        "cornish_fisher_check_z": float(check_z),
    }


def get_warning_thresholds(deck: dict[str, Any] | None = None) -> dict[str, float]:
    """run.warning_thresholds, validated: {"special_assessment_tail": probability in (0, 1)}."""
    deck = deck if deck is not None else load_deck()
    tail = _constant(deck["run"]["warning_thresholds"]["special_assessment_tail"])
    if not (_is_number(tail) and 0 < tail < 1):
        raise ValueError(f"run.warning_thresholds.special_assessment_tail must be a probability in (0, 1), got {tail!r}")
    return {"special_assessment_tail": float(tail)}


def get_build_horizon_months(deck: dict[str, Any] | None = None) -> int:
    """How far (months) the every-run model builds must project: the longest scenario.horizon_years in the
    sweep x 12. Replaces the old hardcoded 120."""
    deck = deck if deck is not None else load_deck()
    horizons = _sweep(deck["scenario"]["horizon_years"])
    if not horizons or not all(isinstance(h, int) and not isinstance(h, bool) and h >= 1 for h in horizons):
        raise ValueError(f"scenario.horizon_years must list whole numbers >= 1, got {horizons!r}")
    return 12 * max(horizons)


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


def get_forecast_model_name(name: str, deck: dict[str, Any] | None = None) -> str:
    """Returns forecast_models.<name>.model_name (e.g. "fred_denver_homes_1987_1_2026_8" for
    "home_price") -- the raw string identifying which model this founding process resolves to, for
    a caller that needs to locate that model's raw historical data (data/raw/<model_name>.csv, see
    manual_utils.raw_csv_path) rather than its fitted OU parameters (see get_fitted_forecast_model
    for that). name must be one of DERIVED_LATENT_COMPONENTS -- a derived_latents entry has no raw
    data of its own (it's a synthetic weighted combination, not fetched from anywhere)."""
    deck = deck if deck is not None else load_deck()
    return deck["forecast_models"][name]["model_name"]


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


def get_derived_latent_fit(name: str, deck: dict[str, Any] | None = None) -> dict[str, float]:
    """Resolves derived_latents.<name>'s own BUILT fit (models/derived_variables/<name>/fit.json --
    see manual_utils.build_derived_latent_model) to its checked, loaded fit (see
    _load_checked_fit -- the same validation get_fitted_forecast_model uses for forecast_models, at
    a different path). name must be an actual derived_latents entry (construction_cost/insurance/
    maintenance/adr/utilities) -- for a `growth.latent` that might instead name one of the three
    FOUNDING processes (general/home_price/market, e.g. carrying_costs.property_tax.market_value's
    own growth), use get_growth_fit instead, which dispatches to whichever this actually is."""
    return _load_checked_fit(f"derived_variables/{name}")


def get_growth_fit(name: str, deck: dict[str, Any] | None = None) -> dict[str, float]:
    """Resolves a `growth.latent` reference to its fitted OU model, whichever kind of process it
    names: one of the three FOUNDING processes (DERIVED_LATENT_COMPONENTS: general/home_price/
    market -- e.g. carrying_costs.property_tax.market_value's own growth, via
    get_fitted_forecast_model) or a derived_latents entry (construction_cost/insurance/maintenance/
    adr/utilities -- e.g. hoa_dues_annual's, via get_derived_latent_fit). Prefer this over calling
    either directly when the caller doesn't already know which kind a given `growth.latent` name is
    -- every utils.prior_utils.build_* carrying-cost-prior function does."""
    deck = deck if deck is not None else load_deck()
    if name in DERIVED_LATENT_COMPONENTS:
        return get_fitted_forecast_model(name, deck)
    return get_derived_latent_fit(name, deck)
