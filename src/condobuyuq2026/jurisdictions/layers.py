"""Jurisdiction layer classes. Each data file under data/tax_jurisdictions/ names its class with a
`model:` key (see LAYER_MODELS at the bottom); the class turns that file's statutory values into
behavior. The base class carries what every level can have (guest taxes, notes, sources, a hook to
adjust the property-tax model); FederalLayer and StateLayer add the income-tax math for their level.

Adding a jurisdiction usually needs only a new data file using an existing model (`state`, `county`,
`municipality`). A new class is needed only when a place's rules BEHAVE differently (e.g. a state with
its own property-tax mechanic -- see ColoradoState).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from condobuyuq2026.jurisdictions.property_tax import PROPERTY_TAX_MODELS, PropertyTaxModel, value_for_year


def bracket_tax(income: float, brackets: list[list[float]]) -> float:
    """Tax on `income` under a progressive [[floor, rate], ...] table (floors ascending, first floor 0)."""
    if income <= 0:
        return 0.0
    tax = 0.0
    for i, (floor, rate) in enumerate(brackets):
        top = brackets[i + 1][0] if i + 1 < len(brackets) else math.inf
        if income <= floor:
            break
        tax += (min(income, top) - floor) * rate
    return tax


def _stacked_tax(start: float, amount: float, brackets: list[list[float]]) -> float:
    """Tax on `amount` of income stacked on top of `start` (how capital gains are taxed)."""
    return bracket_tax(start + amount, brackets) - bracket_tax(start, brackets)


@dataclass
class JurisdictionLayer:
    """One data file: `key` is its path id ("federal", "CO", "CO/summit", "CO/summit/keystone")."""

    key: str
    path: Path
    data: dict[str, Any] = field(repr=False)

    @property
    def name(self) -> str:
        return str(self.data["name"])

    @property
    def level(self) -> str:
        return str(self.data["level"])

    @property
    def as_of(self) -> str:
        return str(self.data["as_of"])

    @property
    def sources(self) -> dict[str, str]:
        return dict(self.data.get("sources") or {})

    @property
    def notes(self) -> list[str]:
        return list(self.data.get("notes") or [])

    def guest_tax_rates(self) -> dict[str, float]:
        """Taxes this layer charges STR guests (information only; guest-paid pass-throughs)."""
        return {name: float(rate) for name, rate in (self.data.get("guest_taxes") or {}).items()}

    def adjust_property_tax_model(self, model: PropertyTaxModel) -> PropertyTaxModel:
        """Hook for a county/municipality whose levies behave differently; the default changes nothing."""
        return model


class FederalLayer(JurisdictionLayer):
    """Federal income tax: ordinary brackets, capital-gain stacking, NIIT, SALT cap, and the federal
    rules taxes.py and classification.py apply (280A day counts, depreciation, passive activity)."""

    @property
    def standard_deduction(self) -> float:
        return float(self.data["standard_deduction"])

    def income_tax(self, taxable_income: float, ltcg: float = 0.0, unrecaptured_1250: float = 0.0) -> float:
        """Regular federal tax (Schedule D worksheet, simplified): ordinary income at the brackets;
        unrecaptured 1250 gain at the ordinary rates capped at its max rate; long-term gain stacked on
        top at the 0/15/20% brackets; never more than the all-ordinary tax."""
        ordinary_brackets = self.data["ordinary_brackets"]
        ti = max(0.0, taxable_income)
        gain = min(max(0.0, ltcg), ti)
        recapture = min(max(0.0, unrecaptured_1250), ti - gain)
        ordinary = ti - gain - recapture
        tax_ordinary = bracket_tax(ordinary, ordinary_brackets)
        tax_recapture = min(
            _stacked_tax(ordinary, recapture, ordinary_brackets), self.data["unrecaptured_1250_max_rate"] * recapture
        )
        tax_gain = _stacked_tax(ordinary + recapture, gain, self.data["long_term_capital_gain_brackets"])
        return min(bracket_tax(ti, ordinary_brackets), tax_ordinary + tax_recapture + tax_gain)

    def niit(self, net_investment_income: float, magi: float) -> float:
        rule = self.data["niit"]
        return rule["rate"] * max(0.0, min(net_investment_income, magi - rule["magi_threshold"]))

    def salt_cap(self, tax_year: int, magi: float) -> float:
        rule = self.data["salt_cap"]
        caps = {int(year): float(cap) for year, cap in rule["cap_by_year"].items()}
        if tax_year > max(caps):
            return float(rule["after_last_year"])
        year = max(min(tax_year, max(caps)), min(caps))
        threshold = float(rule["phasedown_threshold_by_year"][year])
        reduced = caps[year] - rule["phasedown_rate"] * max(0.0, magi - threshold)
        return max(float(rule["floor"]), reduced)

    def depreciation_years(self, average_guest_stay_nights: float) -> float:
        rule = self.data["depreciation"]
        transient = average_guest_stay_nights <= rule["transient_max_average_stay_nights"]
        return float(rule["nonresidential_years"] if transient else rule["residential_rental_years"])

    def passive_status(self, average_guest_stay_nights: float, owner_materially_participates: bool) -> str:
        """'nonpassive' | 'passive' (not a rental activity, no material participation) |
        'passive_rental_activity' (a rental activity: passive, with the special allowance)."""
        rule = self.data["passive_activity"]
        if average_guest_stay_nights <= rule["not_rental_activity_max_average_stay_nights"]:
            return "nonpassive" if owner_materially_participates else "passive"
        return "passive_rental_activity"

    def rental_special_allowance(self, magi: float) -> float:
        rule = self.data["passive_activity"]
        phaseout = rule["allowance_phaseout_rate"] * max(0.0, magi - rule["allowance_phaseout_start_magi"])
        return max(0.0, rule["rental_special_allowance"] - phaseout)

    @property
    def vacation_home(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.data["vacation_home"].items()}

    @property
    def qualified_residence_interest_classes(self) -> list[str]:
        return list(self.data["qualified_residence_interest_classes"])

    @property
    def section_121(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.data["section_121"].items()}


class StateLayer(JurisdictionLayer):
    """A state: income tax by bracket table per year (or none), and optionally a property-tax model."""

    @property
    def has_income_tax(self) -> bool:
        return bool(self.data.get("income_tax"))

    def income_tax(self, taxable_income: float, tax_year: int) -> float:
        """State tax on `taxable_income` (the state's own base -- for CO, federal taxable income)."""
        if not self.has_income_tax:
            return 0.0
        by_year = {int(year): table for year, table in self.data["income_tax"]["brackets_by_year"].items()}
        years = sorted(by_year)
        chosen = max([year for year in years if year <= tax_year], default=years[0])
        return bracket_tax(taxable_income, by_year[chosen])

    @property
    def statutory_residency_days(self) -> float | None:
        income_tax = self.data.get("income_tax") or {}
        days = income_tax.get("statutory_residency_days")
        return float(days) if days is not None else None

    def property_tax_model(self) -> PropertyTaxModel:
        config = self.data.get("property_tax")
        if not config:
            raise NotImplementedError(
                f"{self.name} has no property-tax model in {self.path} -- it can only be a residence state, not "
                "taxes.jurisdiction.state."
            )
        return PROPERTY_TAX_MODELS[config["model"]](config)

    def describe_income_tax(self, tax_year: int) -> str:
        if not self.has_income_tax:
            return "no personal income tax"
        by_year = {int(year): table for year, table in self.data["income_tax"]["brackets_by_year"].items()}
        chosen = max([year for year in by_year if year <= tax_year], default=min(by_year))
        table = by_year[chosen]
        if len(table) == 1:
            return f"flat {table[0][1]:.2%}"
        return "graduated " + "/".join(f"{rate:.1%}" for _, rate in table)


class ColoradoState(StateLayer):
    """Colorado: StateLayer's bracket tables, and nonresidents apportioned by Colorado-source income
    (DR 0104PN, see TaxJurisdiction.property_state_tax). Its property_tax block names the
    colorado_residential model. A home for Colorado-only behavior; none beyond the data is needed yet."""


class CountyLayer(JurisdictionLayer):
    """A county: guest taxes, notes, and the property-tax adjustment hook (none needed for Summit)."""


class MunicipalityLayer(JurisdictionLayer):
    """A municipality: guest taxes, notes, and the property-tax adjustment hook (none needed for Keystone)."""


LAYER_MODELS: dict[str, type[JurisdictionLayer]] = {
    "federal_us": FederalLayer,
    "state": StateLayer,
    "colorado": ColoradoState,
    "county": CountyLayer,
    "municipality": MunicipalityLayer,
}

__all__ = [
    "LAYER_MODELS",
    "ColoradoState",
    "CountyLayer",
    "FederalLayer",
    "JurisdictionLayer",
    "MunicipalityLayer",
    "StateLayer",
    "bracket_tax",
    "value_for_year",
]
