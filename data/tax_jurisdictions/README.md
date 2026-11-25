# data/tax_jurisdictions/

The tax LAW the model uses, one sourced data file per jurisdiction layer. Tracked in git and meant to
be updated periodically (bump each file's `as_of` when you do). Household facts never go here; they
belong in `input_deck.yaml`'s `taxes` section.

## Layout

```
federal.yaml                              always on
<STATE>/state.yaml                        e.g. CO/state.yaml, FL/state.yaml
<STATE>/<county>/county.yaml              e.g. CO/summit/county.yaml
<STATE>/<county>/<municipality>/municipality.yaml   e.g. CO/summit/keystone/municipality.yaml
```

`input_deck.yaml` picks the layers for the PROPERTY:

```yaml
taxes:
  jurisdiction:
    value: {state: CO, county: summit, municipality: keystone}
```

`county: null` or `municipality: null` leaves that layer off: an unincorporated area, or a city that is
its own county. Denver, for example, would be `{state: CO, county: denver, municipality: null}` once
`CO/denver/county.yaml` exists. `state` must match `scenario.property_state`. The taxpayer's residence
state (`scenario.residency_state`) is loaded from `<STATE>/state.yaml` too, for resident income tax
(FL: none).

Available now: federal; CO (property state or residence); FL (residence only, no property-tax model);
CO/summit; CO/summit/keystone.

## How the layers are used (src/condobuyuq2026/jurisdictions/)

Each file's `model:` key names the Python class that gives it behavior (`jurisdictions/layers.py`,
`LAYER_MODELS`):

| model          | class               | what it provides |
|----------------|---------------------|------------------|
| `federal_us`   | `FederalLayer`      | brackets, standard deduction, capital-gain stacking, NIIT, SALT cap, section 121, 280A day counts, depreciation periods, passive-activity rules |
| `state`        | `StateLayer`        | income tax by bracket table per year (or none), statutory residency days, optional property-tax model |
| `colorado`     | `ColoradoState`     | `StateLayer` + nonresident apportionment by Colorado-source income; `property_tax` block → `ColoradoResidentialPropertyTax` |
| `county`       | `CountyLayer`       | guest taxes, notes, property-tax adjustment hook |
| `municipality` | `MunicipalityLayer` | guest taxes, notes, property-tax adjustment hook |

`load_tax_jurisdiction` composes the selected files into one `TaxJurisdiction`:
- **The property state** supplies the property-tax mechanic (`property_tax.model`). Each county or
  municipality layer can then adjust it via `adjust_property_tax_model`; that's the hook for a place
  whose levies behave differently.
- **Mill levies are NOT here.** They're set per tax area, which can differ within one town, so they're
  entered per property in `carrying_costs.property_tax.mill_levy`, keyed by the model's components
  (CO: `local`, `school`).
- **Guest taxes** (`guest_taxes`) are summed across layers for the report. They are information only:
  guest-paid pass-throughs, never owner income.
- **Income tax** is computed on the household's whole return with and without the condo (`taxes.py`).
  The property state taxes a resident on everything and a nonresident on its share of income
  (`nonresident_method`). A different residence state taxes everything, with a credit for the property
  state's tax.

## Adding a place

- **A town or county with ordinary rules:** create its folder and file with `model: county` or
  `model: municipality`, then fill in `name`, `level`, `as_of`, `local_income_tax`, `guest_taxes`,
  `notes` and `sources`. No code change.
- **A new state:** `model: state` with `income_tax.brackets_by_year`. If it can be a property state,
  it also needs a `property_tax` block and, if its mechanic differs from Colorado's, a new
  `PropertyTaxModel` subclass in `jurisdictions/property_tax.py`.
- **The GUI** (`gui/input_deck.html`) mirrors this tree for its combo boxes (`TAX_JURISDICTIONS`).
  Add the new entry there too until the GUI is wired to the backend.

## Conventions

- Rates are plain fractions (0.044 = 4.4%); money in USD; bracket tables are `[[floor, rate], ...]`
  with floors ascending from 0.
- `*_by_year` maps: a year after the last listed one uses the last value.
- Every value's source goes under `sources:` (keyed by field). If something can't be sourced, say so in
  a comment rather than guessing.
