"""Property-tax models, one class per state's mechanic. A state layer (see layers.StateLayer.
property_tax_model) builds its own model from its data file's `property_tax` block; county and
municipality layers can then adjust it (layers.JurisdictionLayer.adjust_property_tax_model), which is
the hook for a locality whose levies behave differently. The MILL LEVIES themselves are never stored
here or in the data files: they're property-specific (set per tax area) and come from
input_deck.yaml's carrying_costs.property_tax.mill_levy, keyed by the model's own `components`.

Every model is MONOTONE in market value, so a band on market value (median/lo/hi) maps straight
through to a band on tax -- see carrying_costs.compute_property_tax_schedule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def value_for_year(by_year: dict[int, float], tax_year: int) -> float:
    """Looks up a {year: value} schedule: the entry for the latest listed year <= tax_year (years after
    the last listed one keep the last value; a year before the first listed one uses the first)."""
    years = sorted(int(year) for year in by_year)
    chosen = years[0]
    for year in years:
        if year <= tax_year:
            chosen = year
    return float(by_year[chosen])


class PropertyTaxModel(ABC):
    """Converts the assessor's market (actual) value into assessed value and tax, per component. Built
    from the state data file's `property_tax` block (`config`)."""

    components: tuple[str, ...]

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.components = tuple(config["components"])

    @abstractmethod
    def assessed_values(self, market_value: np.ndarray | float, tax_year: int) -> dict[str, np.ndarray]:
        """Each component's assessed value for this market value and tax year."""

    def assessed_value(self, market_value: np.ndarray | float, tax_year: int) -> np.ndarray:
        """Total assessed value across components (each component may use a different base)."""
        return sum(
            self.assessed_values(market_value, tax_year).values(), np.zeros_like(np.asarray(market_value, dtype=float))
        )

    def tax(self, market_value: np.ndarray | float, mill_levies: dict[str, float], tax_year: int) -> np.ndarray:
        """Tax = sum over components of assessed value x that component's mill levy (a plain fraction)."""
        missing = set(self.components) - set(mill_levies)
        if missing:
            raise ValueError(f"mill_levy is missing component(s) {sorted(missing)} for this property-tax model")
        assessed = self.assessed_values(market_value, tax_year)
        return sum(
            (assessed[name] * mill_levies[name] for name in self.components),
            np.zeros_like(np.asarray(market_value, dtype=float)),
        )

    def describe(self, tax_year: int) -> str:
        """One-line summary for the run report."""
        return type(self).__name__


class ColoradoResidentialPropertyTax(PropertyTaxModel):
    """Colorado's residential two-step mechanic (HB24B-1001): for each component (local, school),
    assessed value = (actual value - reduction) x assessment rate, where the reduction is the lesser of
    fraction x value and cap (local levies only in current law)."""

    def __init__(self, config: dict[str, Any]) -> None:
        if config.get("classification") != "residential":
            raise NotImplementedError(
                f"Colorado property-tax classification {config.get('classification')!r} isn't implemented -- only "
                "'residential' exists in current law for this property (see data/tax_jurisdictions/CO/state.yaml)."
            )
        super().__init__(config)

    def _component(self, name: str, market_value: np.ndarray, tax_year: int) -> np.ndarray:
        component = self.config["components"][name]
        rate = value_for_year(component["assessment_rate_by_year"], tax_year)
        reduction_rule = component.get("actual_value_reduction")
        reduction = (
            np.minimum(reduction_rule["fraction"] * market_value, reduction_rule["cap"]) if reduction_rule else 0.0
        )
        return np.maximum(market_value - reduction, 0.0) * rate

    def assessed_values(self, market_value: np.ndarray | float, tax_year: int) -> dict[str, np.ndarray]:
        value = np.asarray(market_value, dtype=float)
        return {name: self._component(name, value, tax_year) for name in self.components}

    def describe(self, tax_year: int) -> str:
        parts = []
        for name in self.components:
            component = self.config["components"][name]
            rate = value_for_year(component["assessment_rate_by_year"], tax_year)
            rule = component.get("actual_value_reduction")
            reduction = f" after -min({rule['fraction']:.0%} of value, ${rule['cap']:,.0f})" if rule else ""
            parts.append(f"{name} {rate:.2%}{reduction}")
        return f"Colorado residential ({tax_year}): " + "; ".join(parts)


PROPERTY_TAX_MODELS: dict[str, type[PropertyTaxModel]] = {"colorado_residential": ColoradoResidentialPropertyTax}
