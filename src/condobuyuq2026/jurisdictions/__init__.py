"""Hierarchical tax jurisdictions: federal -> state -> county -> municipality, each layer its own data
file under data/tax_jurisdictions/ (see that folder's README for the layout and how to add a place).

input_deck.yaml's taxes.jurisdiction turns layers on: {state: CO, county: summit, municipality: keystone}
loads federal.yaml + CO/state.yaml + CO/summit/county.yaml + CO/summit/keystone/municipality.yaml;
county/municipality null leaves that layer off (e.g. Denver would be county: denver, municipality: null).
The scenario's residency_state loads its own <STATE>/state.yaml as the RESIDENCE state (resident income
tax; FL has none). load_tax_jurisdiction composes the selected layers into one TaxJurisdiction, the
single object taxes.py, classification.py and carrying_costs.py ask for tax rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import yaml

from condobuyuq2026.jurisdictions.layers import (
    LAYER_MODELS,
    CountyLayer,
    FederalLayer,
    JurisdictionLayer,
    MunicipalityLayer,
    StateLayer,
)
from condobuyuq2026.jurisdictions.property_tax import PropertyTaxModel
from condobuyuq2026.paths import TAX_JURISDICTIONS_DIR

LAYER_FILES = {
    "federal": "federal.yaml",
    "state": "state.yaml",
    "county": "county.yaml",
    "municipality": "municipality.yaml",
}


def _load_layer(key: str, level: str) -> JurisdictionLayer:
    """Loads data/tax_jurisdictions/<key>/<level>.yaml (federal.yaml at the root) into its model class."""
    path = (
        TAX_JURISDICTIONS_DIR / LAYER_FILES[level]
        if level == "federal"
        else TAX_JURISDICTIONS_DIR / key / LAYER_FILES[level]
    )
    if not path.exists():
        raise ValueError(
            f"no {level} tax data for {key!r}: expected {path} (see {TAX_JURISDICTIONS_DIR / 'README.md'})"
        )
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    for required in ("name", "level", "model", "as_of"):
        if required not in data:
            raise ValueError(f"{path}: missing required key {required!r}")
    if data["level"] != level:
        raise ValueError(f"{path}: level is {data['level']!r}, expected {level!r}")
    model = LAYER_MODELS.get(data["model"])
    if model is None:
        raise ValueError(f"{path}: unknown model {data['model']!r} (known: {sorted(LAYER_MODELS)})")
    return model(key=key, path=path, data=data)


def _load_as[LayerT: JurisdictionLayer](key: str, level: str, cls: type[LayerT]) -> LayerT:
    """_load_layer, checked to be the class this level needs (e.g. a state file must be a StateLayer)."""
    layer = _load_layer(key, level)
    if not isinstance(layer, cls):
        raise ValueError(f"{layer.path}: model {layer.data['model']!r} isn't a {cls.__name__}")
    return layer


def available_jurisdictions() -> dict[str, dict[str, list[str]]]:
    """The selectable tree on disk: {state: {county: [municipality, ...]}}."""
    tree: dict[str, dict[str, list[str]]] = {}
    for state_file in sorted(TAX_JURISDICTIONS_DIR.glob("*/state.yaml")):
        state_dir = state_file.parent
        tree[state_dir.name] = {
            county_file.parent.name: sorted(m.parent.name for m in county_file.parent.glob("*/municipality.yaml"))
            for county_file in sorted(state_dir.glob("*/county.yaml"))
        }
    return tree


@dataclass(frozen=True)
class TaxJurisdiction:
    """The composed layers for one run: always federal + the property's state; county/municipality if
    selected; plus the taxpayer's residence state (the same object as property_state when they match)."""

    federal: FederalLayer
    property_state: StateLayer
    county: CountyLayer | None
    municipality: MunicipalityLayer | None
    residence_state: StateLayer

    @property
    def local_layers(self) -> list[JurisdictionLayer]:
        return [layer for layer in (self.county, self.municipality) if layer is not None]

    @property
    def property_layers(self) -> list[JurisdictionLayer]:
        """The layers where the property sits, outermost first (state, county, municipality)."""
        return [self.property_state, *self.local_layers]

    @property
    def label(self) -> str:
        return " > ".join(layer.name for layer in self.property_layers)

    @property
    def as_of(self) -> str:
        return min(layer.as_of for layer in (self.federal, *self.property_layers, self.residence_state))

    @property
    def resident_of_property_state(self) -> bool:
        return self.residence_state.key == self.property_state.key

    def guest_tax_rates(self) -> list[tuple[str, str, float]]:
        """(layer name, tax name, rate) for every tax charged to STR guests, outermost layer first."""
        return [
            (layer.name, name, rate) for layer in self.property_layers for name, rate in layer.guest_tax_rates().items()
        ]

    def guest_tax_total(self) -> float:
        return sum(rate for _, _, rate in self.guest_tax_rates())

    def notes(self) -> list[str]:
        return [note for layer in self.property_layers for note in layer.notes]

    def property_tax_model(self) -> PropertyTaxModel:
        """The property state's model, then each local layer's adjustment (outermost first)."""
        model = self.property_state.property_tax_model()
        for layer in self.local_layers:
            model = layer.adjust_property_tax_model(model)
        return model

    def state_income_taxes(
        self, taxable_income: float, agi: float, property_source_income: float, tax_year: int
    ) -> dict[str, float]:
        """State income tax on a household's whole return. The property state taxes a resident on
        everything; a nonresident pays the resident-basis tax x property-source income / AGI (CO's
        DR 0104PN). A different residence state taxes everything, with a credit for the property state's
        tax (never below zero). Returns {"property_state", "residence_state"}."""
        full = self.property_state.income_tax(taxable_income, tax_year)
        if self.resident_of_property_state:
            return {"property_state": full, "residence_state": 0.0}
        method = (self.property_state.data.get("income_tax") or {}).get("nonresident_method")
        if self.property_state.has_income_tax and method != "apportion_by_source_income":
            raise NotImplementedError(f"{self.property_state.name}: nonresident method {method!r} isn't implemented")
        share = min(1.0, max(0.0, property_source_income / agi)) if agi > 0 else 0.0
        property_state_tax = full * share
        residence_tax = max(0.0, self.residence_state.income_tax(taxable_income, tax_year) - property_state_tax)
        return {"property_state": property_state_tax, "residence_state": residence_tax}


@cache
def _compose(state: str, county: str | None, municipality: str | None, residency_state: str) -> TaxJurisdiction:
    if municipality is not None and county is None:
        raise ValueError("taxes.jurisdiction: a municipality needs its county selected too")
    property_state = _load_as(state, "state", StateLayer)
    return TaxJurisdiction(
        federal=_load_as("federal", "federal", FederalLayer),
        property_state=property_state,
        county=_load_as(f"{state}/{county}", "county", CountyLayer) if county else None,
        municipality=_load_as(f"{state}/{county}/{municipality}", "municipality", MunicipalityLayer)
        if municipality
        else None,
        residence_state=property_state if residency_state == state else _load_as(residency_state, "state", StateLayer),
    )


def load_tax_jurisdiction(selection: dict[str, Any], residency_state: str | None = None) -> TaxJurisdiction:
    """Composes the layers named by `selection` ({"state", "county", "municipality"}; county/municipality
    may be None) for a taxpayer resident in `residency_state` (defaults to the property's state, which is
    all property tax needs). Cached: each distinct selection is read from disk once per run."""
    state = selection["state"]
    return _compose(state, selection.get("county"), selection.get("municipality"), residency_state or state)


def load_federal_layer() -> FederalLayer:
    """Just the federal layer (always on), for callers that only need federal rules."""
    return _load_as("federal", "federal", FederalLayer)


__all__ = ["TaxJurisdiction", "available_jurisdictions", "load_federal_layer", "load_tax_jurisdiction"]
