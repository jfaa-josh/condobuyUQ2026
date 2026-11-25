"""Computes the incremental tax effects of the buy-vs-rent choice: ONLY taxes caused by the real-estate
decision itself (rental income, depreciation, mortgage interest / property tax deductions, the sale).

HOW: each year, the household's whole federal + state income tax is computed twice from
taxes.household_agi and the household's other deductions -- once WITHOUT the condo and once WITH it --
and the difference is the condo's effect (see _household_tax). So bracket jumps, capital-gain
stacking (0/15/20%), the 3.8% NIIT, itemizing vs. the standard deduction, and the SALT cap all come out
of the tables rather than a hand-set marginal rate.

THE LAW comes from the selected tax jurisdiction (jurisdictions.load_tax_jurisdiction: federal + the
property's state + county + municipality data files under data/tax_jurisdictions/, plus the residence
state). This module holds no statutory numbers of its own.

PER-YEAR PROPERTY ITEMS (compute_owner_tax_schedule), by tax_use_class (read from
scenario["classification"], never recomputed; held constant across years):
- r = rented / (rented + personal) days splits shared costs (vacant days don't count, IRS method).
- rental: depreciation r x full; net = rent - r x (interest + property tax + shared) - fees - depr.
  A loss is passive unless the federal passive rules say otherwise (average_guest_stay_nights,
  owner_materially_participates): suspended and carried to the sale, released in full then. The
  personal share of mortgage interest is NOT deductible (not a qualified residence); the personal share
  of property tax is (SALT).
- mixed_use (a residence under 280A): deductions capped at rent in the 280A order (interest/tax +
  fees, then operating costs, then depreciation), the excess carried forward and lost at sale. The
  personal shares of interest and property tax are itemized deductions.
- personal_use (rented <= 14 days): rent excluded; interest and property tax are itemized deductions.
Depreciable basis = (purchase price + closing costs) x building_fraction (acquisition.percent_improved),
over the federal period for the average stay (39 yr if transient, else 27.5).

SALE (compute_owner_sale_tax): recapture (unrecaptured 1250 gain), section 121 exclusion (primary
residence held >= the minimum years), long-term gain, and released passive losses, taxed as an increment
on top of the final year's return.

gross_rent: each year's EXPECTED (mean) gross rental revenue from scenario["rental_operations"],
before lodging tax (guest-paid) and fees. Taxes are POINT estimates on these means; propagating revenue
uncertainty through the nonlinear tax rules is a later (ΔW-stage) concern.

Tax year for scenario year n = closing_year + n - 1.

THE RENT BRANCH (compute_renter_taxes): renting itself has no tax effect (rent isn't deductible), but the
invested portfolio (capital_markets, section 11) does. investment_income_tax / investment_sale_tax tax its
interest (ordinary), qualified dividends and sale gain (long-term rates + NIIT) with the same
with-minus-without engine. Investment income is NOT property-source income: a nonresident of the property
state pays that state nothing on it; the residence state taxes it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from condobuyuq2026.carrying_costs import compute_carrying_costs_schedule, project_condo_value_schedule
from condobuyuq2026.input_deck import (
    get_acquisition_inputs,
    get_closing_date,
    get_exit_inputs,
    get_rental_operations_inputs,
    get_taxes_inputs,
    load_deck,
)
from condobuyuq2026.jurisdictions import TaxJurisdiction, load_tax_jurisdiction
from condobuyuq2026.rental_operations import compute_rental_operations

NO_PROPERTY: dict[str, float] = {
    "rental_income": 0.0,
    "itemized_interest": 0.0,
    "personal_property_tax": 0.0,
    "ltcg": 0.0,
    "unrecaptured_1250": 0.0,
    # The rent branch's portfolio (section 11); never property-source income.
    "interest_income": 0.0,
    "qualified_dividends": 0.0,
    "investment_ltcg": 0.0,
}


def get_scenario_tax_jurisdiction(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> TaxJurisdiction:
    """The composed tax layers for this scenario: the deck's taxes.jurisdiction + the scenario's
    residency_state as the residence state."""
    deck = deck if deck is not None else load_deck()
    return load_tax_jurisdiction(get_taxes_inputs(deck)["jurisdiction"], scenario["residency_state"])


def gross_rent_by_year(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> pd.Series:
    """Expected (mean) gross rental revenue per scenario year, pre-lodging-tax, from
    scenario["rental_operations"] (computed here if runner hasn't stored it yet)."""
    rental_operations = scenario.get("rental_operations") or compute_rental_operations(scenario, deck)
    return rental_operations["annual"]["gross_mean"]


def estimate_sale_proceeds(
    scenario: dict[str, Any], condo_value_schedule: pd.Series, deck: dict[str, Any] | None = None
) -> dict[str, float]:
    """REAL: sale_price = condo_value_schedule's last year; selling_costs_total = exit.
    selling_costs.fixed_fees + percent_of_price * sale_price (the deck's own composite-cost
    pattern); net_sale_proceeds = sale_price - selling_costs_total."""
    deck = deck if deck is not None else load_deck()
    exit_inputs = get_exit_inputs(deck)
    sale_price = condo_value_schedule.iloc[-1]
    selling_costs_total = (
        exit_inputs["selling_costs_fixed_fees"] + exit_inputs["selling_costs_percent_of_price"] * sale_price
    )
    return {
        "sale_price": sale_price,
        "selling_costs_total": selling_costs_total,
        "net_sale_proceeds": sale_price - selling_costs_total,
    }


def _annual_mortgage_interest(scenario: dict[str, Any], horizon_years: int) -> list[float]:
    """Resamples scenario["financing"]["amortization_schedule"] (monthly, h_months=0..
    horizon_years*12) into one interest total per year -- year n covers h_months
    (n-1)*12+1 .. n*12 (h=0 is loan origination, no interest yet)."""
    interest = scenario["financing"]["amortization_schedule"]["interest"]
    return [interest.loc[(year - 1) * 12 + 1 : year * 12].sum() for year in range(1, horizon_years + 1)]


def _household_tax(
    jurisdiction: TaxJurisdiction, taxes_inputs: dict[str, Any], tax_year: int, items: dict[str, float]
) -> dict[str, float]:
    """The household's whole-return income tax for one year with the given property items (NO_PROPERTY
    for the baseline). items: rental_income (ordinary, in AGI; negative only for an allowed loss),
    itemized_interest (deductible mortgage interest), personal_property_tax (SALT-eligible),
    ltcg (long-term gain), unrecaptured_1250 (recaptured depreciation); and the portfolio's
    interest_income (ordinary), qualified_dividends and investment_ltcg (both at long-term rates), which
    are investment income for NIIT but NOT property-source income for the state apportionment. Returns
    {"federal", "property_state", "residence_state", "total", "itemizes", "taxable_income", "agi"}."""
    federal = jurisdiction.federal
    property_income = items["rental_income"] + items["ltcg"] + items["unrecaptured_1250"]
    dividends_and_gains = items.get("qualified_dividends", 0.0) + items.get("investment_ltcg", 0.0)
    investment_income = items.get("interest_income", 0.0) + dividends_and_gains
    agi = taxes_inputs["household_agi"] + property_income + investment_income
    salt = min(federal.salt_cap(tax_year, agi), taxes_inputs["other_salt_paid"] + items["personal_property_tax"])
    itemized = taxes_inputs["other_itemized_deductions"] + items["itemized_interest"] + salt
    deduction = max(federal.standard_deduction, itemized)
    taxable_income = max(0.0, agi - deduction)
    federal_tax = federal.income_tax(
        taxable_income, items["ltcg"] + dividends_and_gains, items["unrecaptured_1250"]
    ) + federal.niit(max(0.0, property_income) + max(0.0, investment_income), agi)
    state = jurisdiction.state_income_taxes(taxable_income, agi, property_income, tax_year)
    return {
        "federal": federal_tax,
        "property_state": state["property_state"],
        "residence_state": state["residence_state"],
        "total": federal_tax + state["property_state"] + state["residence_state"],
        "itemizes": float(itemized > federal.standard_deduction),
        "taxable_income": taxable_income,
        "agi": agi,
    }


def _tax_effect(
    jurisdiction: TaxJurisdiction,
    taxes_inputs: dict[str, Any],
    tax_year: int,
    with_items: dict[str, float],
    without_items: dict[str, float] = NO_PROPERTY,
) -> dict[str, float]:
    """with - without, per tax line; plus whether the household itemizes in the WITH case."""
    with_tax = _household_tax(jurisdiction, taxes_inputs, tax_year, with_items)
    without_tax = _household_tax(jurisdiction, taxes_inputs, tax_year, without_items)
    effect = {
        line: with_tax[line] - without_tax[line] for line in ("federal", "property_state", "residence_state", "total")
    }
    effect["itemizes"] = with_tax["itemizes"]
    return effect


def tax_rules_applied(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """The derived rules this scenario's taxes use (for the schedule and the report): use class,
    depreciation period, passive status, and the shares/basis they act on."""
    deck = deck if deck is not None else load_deck()
    taxes_inputs = get_taxes_inputs(deck)
    federal = get_scenario_tax_jurisdiction(scenario, deck).federal
    classification = scenario["classification"]
    rented, personal = classification["annual_rented_days"], classification["annual_personal_days"]
    used = rented + personal
    tax_use_class = classification["tax_use_class"]
    stay = taxes_inputs["average_guest_stay_nights"]
    cost_basis = scenario["acquisition"]["purchase_price"] + scenario["acquisition"]["closing_costs_total"]
    depreciation_years = federal.depreciation_years(stay)
    building_fraction = get_acquisition_inputs(deck)["building_fraction"]
    return {
        "tax_use_class": tax_use_class,
        "rental_share": rented / used if used else 0.0,
        "average_guest_stay_nights": stay,
        "depreciation_years": depreciation_years,
        "annual_depreciation_full": cost_basis * building_fraction / depreciation_years,
        "building_fraction": building_fraction,
        "cost_basis": cost_basis,
        "passive_status": (
            federal.passive_status(stay, taxes_inputs["owner_materially_participates"])
            if tax_use_class == "rental"
            else "n/a"
        ),
        "personal_interest_deductible": tax_use_class in federal.qualified_residence_interest_classes,
    }


def compute_owner_tax_schedule(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> pd.DataFrame:
    """Per-year OWN-tenure tax effects for scenario["horizon_years"] (see the module docstring for the
    rules by use class). Threads depreciation taken, suspended passive losses, and 280A carryforwards
    through the loop as CUMULATIVE columns, so the last row is the ending state
    compute_owner_sale_tax needs.

    Returns a DataFrame indexed by year (1..horizon_years) with columns: tax_year, use_class,
    mortgage_interest, property_tax, gross_rent, shared_expenses, direct_rental_expenses,
    depr_this_year, taxable_rental (the rental income that lands in AGI; negative = an allowed loss),
    itemized_interest, personal_property_tax, itemizes (0/1, with the condo), federal, property_state,
    residence_state, tax (their sum = the condo's effect that year), cumulative_depr_taken,
    suspended_passive_balance, carry_280a_balance, carry_280a_depr_balance.
    """
    deck = deck if deck is not None else load_deck()
    taxes_inputs = get_taxes_inputs(deck)
    jurisdiction = get_scenario_tax_jurisdiction(scenario, deck)
    federal = jurisdiction.federal
    rules = tax_rules_applied(scenario, deck)
    rental_operations_inputs = get_rental_operations_inputs(deck)
    closing_year = get_closing_date(deck)["closing_year"]

    horizon_years = scenario["horizon_years"]
    tax_use_class = rules["tax_use_class"]
    r = rules["rental_share"]
    annual_depr_full = rules["annual_depreciation_full"]

    # carrying_costs.py already computed this (runner.run_scenarios stores it before taxes runs).
    carrying_costs = scenario.get("carrying_costs")
    carrying_costs_schedule = (
        carrying_costs["annual_schedule"] if carrying_costs else compute_carrying_costs_schedule(scenario, deck)
    )
    property_tax_by_year = carrying_costs_schedule["property_tax"]
    # Special assessments are deliberately NOT here (2026-11-22): they're excluded from every
    # carrying-cost total and applied later as a tail-risk overlay on ΔW, using PRE-TAX amounts.
    # OPEN QUESTION, to revisit when finalizing that model: the IRS generally ADDS an assessment for
    # improvements (roof, siding) to the property's cost basis (reducing the taxable gain at sale)
    # rather than deducting it as a yearly expense -- see .claude/implementation-plan.md's "Final
    # reporting: special-assessment overlay".
    shared_expenses_by_year = carrying_costs_schedule[["hoa_dues", "insurance", "maintenance", "utilities"]].sum(axis=1)

    mortgage_interest_by_year = _annual_mortgage_interest(scenario, horizon_years)
    gross_rent_schedule = gross_rent_by_year(scenario, deck)
    total_fee_fraction = 1 - (1 - rental_operations_inputs["channel_commission_fraction"]) * (
        1 - rental_operations_inputs["mgmt_fee_fraction"]
    )

    depr_taken, suspended_passive, carry_280a, carry_280a_depr = 0.0, 0.0, 0.0, 0.0
    rows = []
    for year in range(1, horizon_years + 1):
        tax_year = closing_year + year - 1
        mortgage_interest = mortgage_interest_by_year[year - 1]
        property_tax = float(property_tax_by_year.loc[year])
        shared_expenses = float(shared_expenses_by_year.loc[year])
        gross_rent = float(gross_rent_schedule.loc[year])
        direct_rental_expenses = total_fee_fraction * gross_rent  # channel commissions + management fee
        interest_tax = mortgage_interest + property_tax
        rental_income, depr_this_year = 0.0, 0.0
        personal_share = 1.0 if tax_use_class == "personal_use" else 1 - r

        if tax_use_class == "rental":
            depr_this_year = annual_depr_full * r
            depr_taken += depr_this_year
            net = gross_rent - (r * (interest_tax + shared_expenses) + direct_rental_expenses + depr_this_year)
            if net >= 0:
                used_suspended = min(net, suspended_passive)
                suspended_passive -= used_suspended
                rental_income = net - used_suspended
            else:
                loss = -net
                status = rules["passive_status"]
                if status == "nonpassive":
                    allowed = loss
                elif status == "passive_rental_activity":
                    allowed = min(loss, federal.rental_special_allowance(taxes_inputs["household_agi"]))
                else:
                    allowed = 0.0
                suspended_passive += loss - allowed
                rental_income = -allowed

        elif tax_use_class == "mixed_use":
            # Section 280A ordering: interest/tax + direct costs, then operating costs, then depreciation.
            tier1 = r * interest_tax + direct_rental_expenses
            room = max(0.0, gross_rent - tier1)
            tier2 = r * shared_expenses + carry_280a
            tier2_deducted = min(tier2, room)
            room -= tier2_deducted
            depr_available = annual_depr_full * r + carry_280a_depr
            depr_this_year = min(depr_available, room)
            carry_280a = tier2 - tier2_deducted
            carry_280a_depr = depr_available - depr_this_year
            depr_taken += depr_this_year
            rental_income = max(0.0, gross_rent - tier1 - tier2_deducted - depr_this_year)

        # "personal_use": rent excluded, nothing deducted against it -- only the itemized items below.

        items = {
            **NO_PROPERTY,
            "rental_income": rental_income,
            "itemized_interest": personal_share * mortgage_interest if rules["personal_interest_deductible"] else 0.0,
            "personal_property_tax": personal_share * property_tax,
        }
        effect = _tax_effect(jurisdiction, taxes_inputs, tax_year, items)
        rows.append(
            {
                "tax_year": tax_year,
                "use_class": tax_use_class,
                "mortgage_interest": mortgage_interest,
                "property_tax": property_tax,
                "gross_rent": gross_rent,
                "shared_expenses": shared_expenses,
                "direct_rental_expenses": direct_rental_expenses,
                "depr_this_year": depr_this_year,
                "taxable_rental": rental_income,
                "itemized_interest": items["itemized_interest"],
                "personal_property_tax": items["personal_property_tax"],
                "itemizes": effect["itemizes"],
                "federal": effect["federal"],
                "property_state": effect["property_state"],
                "residence_state": effect["residence_state"],
                "tax": effect["total"],
                "cumulative_depr_taken": depr_taken,
                "suspended_passive_balance": suspended_passive,
                "carry_280a_balance": carry_280a,
                "carry_280a_depr_balance": carry_280a_depr,
            }
        )

    return pd.DataFrame(rows, index=pd.Index(range(1, horizon_years + 1), name="year"))


def compute_owner_sale_tax(
    scenario: dict[str, Any], annual_schedule: pd.DataFrame, deck: dict[str, Any] | None = None
) -> dict[str, float]:
    """The sale's tax effect, as an increment on top of the final year's return: depreciation recapture
    (unrecaptured 1250 gain), the section 121 exclusion (only if loan_occupancy_class ==
    "primary_residence" AND held >= the federal minimum years), the remaining long-term gain, and the
    release of suspended passive losses (an ordinary loss); 280A carryforwards are lost. Reads the
    ending state off annual_schedule's LAST row."""
    deck = deck if deck is not None else load_deck()
    taxes_inputs = get_taxes_inputs(deck)
    jurisdiction = get_scenario_tax_jurisdiction(scenario, deck)
    section_121 = jurisdiction.federal.section_121

    cost_basis = scenario["acquisition"]["purchase_price"] + scenario["acquisition"]["closing_costs_total"]
    ending = annual_schedule.iloc[-1]
    depr_taken = float(ending["cumulative_depr_taken"])

    condo_value_schedule = project_condo_value_schedule(scenario, deck)
    proceeds = estimate_sale_proceeds(scenario, condo_value_schedule, deck)

    adj_basis = cost_basis - depr_taken
    gain = proceeds["net_sale_proceeds"] - adj_basis
    recapture = max(0.0, min(gain, depr_taken))
    rest = max(0.0, gain - recapture)

    is_primary_residence = scenario["classification"]["loan_occupancy_class"] == "primary_residence"
    held_long_enough = scenario["horizon_years"] >= section_121["min_ownership_and_use_years"]
    excluded_121 = min(rest, section_121["exclusion"]) if (is_primary_residence and held_long_enough) else 0.0
    ltcg = rest - excluded_121
    released_passive = float(ending["suspended_passive_balance"])

    final_year_items = {
        **NO_PROPERTY,
        "rental_income": float(ending["taxable_rental"]),
        "itemized_interest": float(ending["itemized_interest"]),
        "personal_property_tax": float(ending["personal_property_tax"]),
    }
    with_sale_items = {
        **final_year_items,
        "rental_income": final_year_items["rental_income"] - released_passive,
        "ltcg": ltcg,
        "unrecaptured_1250": recapture,
    }
    effect = _tax_effect(jurisdiction, taxes_inputs, int(ending["tax_year"]), with_sale_items, final_year_items)

    return {
        "sale_price": proceeds["sale_price"],
        "selling_costs_total": proceeds["selling_costs_total"],
        "net_sale_proceeds": proceeds["net_sale_proceeds"],
        "gain": gain,
        "recapture": recapture,
        "excluded_121": excluded_121,
        "ltcg": ltcg,
        "released_passive": released_passive,
        "lost_280a_carry": float(ending["carry_280a_balance"] + ending["carry_280a_depr_balance"]),
        "federal": effect["federal"],
        "property_state": effect["property_state"],
        "residence_state": effect["residence_state"],
        "tax": effect["total"],
    }


def compute_owner_taxes(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Top-level entry point for a BUY scenario (tenure=OWN). Requires scenario["acquisition"],
    scenario["classification"], scenario["financing"] to already be set (see runner.run_scenarios'
    ordering). Returns {"annual_schedule" (see compute_owner_tax_schedule), "sale" (see
    compute_owner_sale_tax), "total_tax" (annual "tax" summed + the sale's), "rules" (see
    tax_rules_applied), "jurisdiction" (the composed TaxJurisdiction), "flags" (e.g. the property
    state's statutory-residency warning; informational, doesn't change any number)}."""
    deck = deck if deck is not None else load_deck()
    annual_schedule = compute_owner_tax_schedule(scenario, deck)
    sale = compute_owner_sale_tax(scenario, annual_schedule, deck)
    jurisdiction = get_scenario_tax_jurisdiction(scenario, deck)

    flags = []
    residency_days = jurisdiction.property_state.statutory_residency_days
    personal_days = scenario["classification"]["annual_personal_days"]
    if not jurisdiction.resident_of_property_state and residency_days is not None and personal_days >= residency_days:
        flags.append(
            f"WARNING: annual_personal_days={personal_days:g} >= {residency_days:g} -- "
            f"{scenario['residency_state']} residency claim at risk ({jurisdiction.property_state.name} statutory-resident test)."
        )

    return {
        "annual_schedule": annual_schedule,
        "sale": sale,
        "total_tax": annual_schedule["tax"].sum() + sale["tax"],
        "rules": tax_rules_applied(scenario, deck),
        "jurisdiction": jurisdiction,
        "flags": flags,
    }


def investment_income_tax(
    jurisdiction: TaxJurisdiction, taxes_inputs: dict[str, Any], tax_year: int, interest: float, dividends: float
) -> dict[str, float]:
    """One year's tax on the portfolio's income (with minus without): interest_tax (the interest alone,
    ordinary), dividend_tax (what the qualified dividends add on top), plus the federal / property_state /
    residence_state / total split of the two together."""
    interest_items = {**NO_PROPERTY, "interest_income": interest}
    interest_only = _tax_effect(jurisdiction, taxes_inputs, tax_year, interest_items)
    both = _tax_effect(jurisdiction, taxes_inputs, tax_year, {**interest_items, "qualified_dividends": dividends})
    return {
        "interest_tax": interest_only["total"],
        "dividend_tax": both["total"] - interest_only["total"],
        **{line: both[line] for line in ("federal", "property_state", "residence_state", "total")},
    }


def investment_sale_tax(
    jurisdiction: TaxJurisdiction,
    taxes_inputs: dict[str, Any],
    tax_year: int,
    gain: float,
    final_interest: float,
    final_dividends: float,
) -> dict[str, float]:
    """The portfolio sale's tax, as an increment on top of the final year's return (which already holds
    that year's interest and dividends): a long-term gain. A loss (gain <= 0) is taxed $0 here; its
    $3,000/yr deduction and carryforward aren't modeled."""
    base = {**NO_PROPERTY, "interest_income": final_interest, "qualified_dividends": final_dividends}
    effect = _tax_effect(jurisdiction, taxes_inputs, tax_year, {**base, "investment_ltcg": max(0.0, gain)}, base)
    return {line: effect[line] for line in ("federal", "property_state", "residence_state", "total")}


def compute_renter_taxes(capital_markets: dict[str, Any]) -> dict[str, Any]:
    """Top-level entry point for a reference case (tenure=RENT). Renting itself has no tax effect (rent
    isn't deductible); the taxes are the portfolio's, already computed by
    capital_markets.compute_capital_markets with investment_income_tax / investment_sale_tax. This
    repackages them in compute_owner_taxes' shape: annual_schedule (tax_year, interest_tax, dividend_tax,
    federal, property_state, residence_state, tax), sale, total_tax."""
    annual = capital_markets["annual"]
    columns = ["tax_year", "interest_tax", "dividend_tax", "federal", "property_state", "residence_state", "tax"]
    sale = capital_markets["sale"]
    return {
        "annual_schedule": annual[columns],
        "sale": sale,
        "total_tax": float(annual["tax"].sum()) + sale["tax"],
        "flags": [],
        "use_class": "TENANT",
    }
