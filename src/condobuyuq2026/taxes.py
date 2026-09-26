"""Computes the incremental tax effects of the buy-vs-rent choice -- ONLY taxes caused by the
real-estate decision itself (mortgage interest, rental income, depreciation, capital gains at
sale). Work income, and the general household income tax you'd owe on it regardless of this
decision, are OUT OF SCOPE -- see input_deck.yaml's taxes section disclaimer.

Adapted from a reference implementation the user supplied (a `TaxModel` class with mutable
year-to-year state: depreciation taken, suspended passive losses, section 280A carryforwards) into
this project's functional style (see acquisition.py/financing.py): one function computes the full
year-by-year schedule in one call, threading that state through an internal loop -- the same
pattern as financing.amortize_loan, with the state itself kept as CUMULATIVE columns in the
returned DataFrame (so the last row IS the "ending state" needed for the sale calculation, exactly
like amortize_loan's "balance" column) rather than a separate return value.

SCENARIOS (residency x tenure) -- not a new axis, exactly scenario["residency_state"] (CO|FL) x
whether this is a buy scenario (alternative="buy" -> OWN, see compute_owner_taxes) or its paired
reference case (-> RENT, see compute_renter_taxes -- always a real, confirmed ZERO result, not a
placeholder: renting has no real-estate tax effect at all):
    CO / RENT -> no real-estate tax effect
    CO / OWN  -> primary residence (loan_occupancy_class == "primary_residence"): full section 121
                 exclusion at sale
    FL / RENT -> no real-estate tax effect
    FL / OWN  -> second home/investment, CO nonresident: no section 121 exclusion

tax_use_class (personal_use/mixed_use/rental) is NEVER recomputed here -- it's read straight from
scenario["classification"] (classification.py already applies the exact same day-count rule this
module's reference implementation used internally). Same for loan_occupancy_class, cost_basis
(acquisition), mortgage_interest (financing's own amortization_schedule, resampled to annual), and
property_tax/hoa_dues/insurance/maintenance/special_assessment/utilities (carrying_costs.py's own
per-year schedule -- see carrying_costs.compute_carrying_costs_schedule; property_tax uses
Colorado's REAL two-step assessment/mill-levy mechanic, and the other five are each projected from
their own pre-built carrying_cost_prior model, both real, not placeholders, as of 2026-09-26).

===========================================================================================
*** ONE REMAINING PLACEHOLDER INPUT -- READ THIS BEFORE TRUSTING gross_rent-DERIVED FIGURES ***
===========================================================================================
A real rental-revenue model (rental_operations.adr x occupancy, with actual growth propagation)
DOESN'T EXIST YET. `gross_rent` (placeholder_gross_rent) is a flat, UN-GROWN estimate taken
directly off the deck's own ADR/occupancy medians (ignoring cv and growth entirely), applied
identically to every year of the horizon -- see PLACEHOLDER_FIELDS, echoed back in every
compute_owner_taxes result and printed loudly by reporting.print_taxes_report. `direct_rental_
expenses` is REAL logic despite depending on this placeholder: mgmt_fee_fraction * gross_rent won't
change once gross_rent itself is real -- only the number it's fed will.

(As of 2026-09-25 this section also covered property_tax/shared_expenses -- both are now REAL,
built via carrying_costs.py; see .claude/implementation-plan.md's "Backend build progress" for
that history and the still-open TODO on gross_rent.)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from condobuyuq2026.carrying_costs import compute_carrying_costs_schedule, project_condo_value_schedule
from condobuyuq2026.input_deck import (
    get_acquisition_inputs,
    get_exit_inputs,
    get_rental_operations_inputs,
    get_rental_operations_monthly_medians,
    get_taxes_inputs,
    load_deck,
)

# Statutory day-count threshold this module itself owns (a WARNING only, doesn't affect any tax
# math below) -- kept as a module constant rather than a deck field, the same precedent
# classification.py already set for its own day-count thresholds (14 days, 10% of rented days).
CO_STATUTORY_RESIDENCY_DAYS = 183

PLACEHOLDER_FIELDS = {
    "gross_rent": (
        "flat, un-grown ADR x occupancy month-by-month sum from deck medians only -- needs the "
        "real rental revenue model (growth, cv, actual per-year variation)."
    ),
}


def placeholder_gross_rent(deck: dict[str, Any] | None = None) -> float:
    """PLACEHOLDER (see module docstring) -- ADR x occupancy, multiplied MONTH BY MONTH (not
    annual-total x annual-total, so the seasonal correlation between high-ADR and high-occupancy
    months, e.g. ski season, is at least preserved) then summed, using only each month's MEDIAN
    (ignoring cv and growth entirely). Replace with a real revenue model's output once it exists."""
    deck = deck if deck is not None else load_deck()
    medians = get_rental_operations_monthly_medians(deck)
    return sum(medians["adr"][month] * medians["occupancy"][month] for month in medians["adr"])


def estimate_sale_proceeds(
    scenario: dict[str, Any], condo_value_schedule: pd.Series, deck: dict[str, Any] | None = None
) -> dict[str, float]:
    """REAL: sale_price = condo_value_schedule's last year; selling_costs_total = exit.
    selling_costs.fixed_fees + percent_of_price * sale_price (the deck's own composite-cost
    pattern); net_sale_proceeds = sale_price - selling_costs_total."""
    deck = deck if deck is not None else load_deck()
    exit_inputs = get_exit_inputs(deck)
    sale_price = condo_value_schedule.iloc[-1]
    selling_costs_total = exit_inputs["selling_costs_fixed_fees"] + exit_inputs["selling_costs_percent_of_price"] * sale_price
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


def compute_owner_tax_schedule(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> pd.DataFrame:
    """Computes the OWN-tenure per-year tax effects for scenario["horizon_years"], mirroring the
    reference TaxModel.year()'s §280A/passive-loss logic exactly, but functionally: threads
    depreciation taken, suspended passive losses, and §280A carryforwards through the loop as
    CUMULATIVE columns (so the last row is the "ending state" compute_owner_sale_tax needs).

    tax_use_class is read from scenario["classification"] (never recomputed -- see module
    docstring) and held CONSTANT across every year, matching classification.py's own annual
    (not year-varying) day-count assumption. gross_rent is a PLACEHOLDER (see module docstring)
    held flat across every year for that reason; mortgage_interest, property_tax, and
    shared_expenses are all real and DO vary year to year (the amortization schedule, the condo's
    own projected appreciation, and carrying_costs.py's own per-year schedule, respectively).

    Returns a DataFrame indexed by year (1..horizon_years) with columns: use_class, mortgage_
    interest, property_tax, gross_rent, shared_expenses, direct_rental_expenses, taxable_rental,
    fed, co, sch_a_benefit, tax, depr_this_year, cumulative_depr_taken, suspended_passive_balance,
    carry_280a_balance, carry_280a_depr_balance.
    """
    deck = deck if deck is not None else load_deck()
    taxes_inputs = get_taxes_inputs(deck)
    rental_operations_inputs = get_rental_operations_inputs(deck)

    horizon_years = scenario["horizon_years"]
    classification = scenario["classification"]
    tax_use_class = classification["tax_use_class"]
    rental_days = classification["annual_rented_days"]
    personal_days = classification["annual_personal_days"]
    used_days = rental_days + personal_days
    r = rental_days / used_days if used_days else 0.0

    purchase_price = scenario["acquisition"]["purchase_price"]
    cost_basis = purchase_price + scenario["acquisition"]["closing_costs_total"]
    building_fraction = 1 - get_acquisition_inputs(deck)["land_fraction"]
    annual_depr_full = cost_basis * building_fraction / taxes_inputs["depreciation_years_building"]

    fed_marginal_rate = taxes_inputs["federal_marginal_rate"]
    niit_rate = taxes_inputs["niit_rate"] if taxes_inputs["niit_applies"] else 0.0
    co_income_tax_rate = taxes_inputs["state_tax_profiles"]["CO"]["income_tax_rate"]

    # carrying_costs.py already computed this (runner.run_scenarios stores it before taxes runs) --
    # reuse it instead of recomputing, same pattern as acquisition/financing's own reuse.
    carrying_costs = scenario.get("carrying_costs")
    carrying_costs_schedule = carrying_costs["annual_schedule"] if carrying_costs else compute_carrying_costs_schedule(scenario, deck)
    property_tax_by_year = carrying_costs_schedule["property_tax"]
    shared_expenses_by_year = carrying_costs_schedule[
        ["hoa_dues", "insurance", "maintenance", "special_assessment_expected", "utilities"]
    ].sum(axis=1)

    mortgage_interest_by_year = _annual_mortgage_interest(scenario, horizon_years)
    gross_rent = placeholder_gross_rent(deck)  # PLACEHOLDER -- flat every year, see module docstring
    direct_rental_expenses = rental_operations_inputs["mgmt_fee_fraction"] * gross_rent  # REAL formula

    depr_taken, suspended_passive, carry_280a, carry_280a_depr = 0.0, 0.0, 0.0, 0.0
    rows = []
    for year in range(1, horizon_years + 1):
        mortgage_interest = mortgage_interest_by_year[year - 1]
        property_tax = property_tax_by_year.loc[year]  # REAL -- Colorado's own assessment/mill-levy mechanic
        shared_expenses = shared_expenses_by_year.loc[year]  # REAL -- carrying_costs.py's own per-year schedule
        interest_tax = mortgage_interest + property_tax
        taxable, personal_it, depr_this_year = 0.0, interest_tax, 0.0

        if tax_use_class == "rental":
            depr_this_year = annual_depr_full * r
            depr_taken += depr_this_year
            net = gross_rent - (r * (interest_tax + shared_expenses) + direct_rental_expenses + depr_this_year)
            personal_it = (1 - r) * interest_tax
            if net >= 0:
                used_suspended = min(net, suspended_passive)
                suspended_passive -= used_suspended
                taxable = net - used_suspended
            else:
                suspended_passive += -net

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
            taxable = max(0.0, gross_rent - tier1 - tier2_deducted - depr_this_year)
            personal_it = (1 - r) * interest_tax

        # "personal_use": rent excluded, everything personal -- taxable/personal_it stay as above.

        fed = taxable * (fed_marginal_rate + niit_rate)
        co = taxable * co_income_tax_rate  # CO taxes CO-source rental income regardless of residency
        sch_a_benefit = -(personal_it * fed_marginal_rate) if taxes_inputs["itemizes_deductions"] else 0.0

        rows.append(
            {
                "use_class": tax_use_class,
                "mortgage_interest": mortgage_interest,
                "property_tax": property_tax,
                "gross_rent": gross_rent,
                "shared_expenses": shared_expenses,
                "direct_rental_expenses": direct_rental_expenses,
                "taxable_rental": taxable,
                "fed": fed,
                "co": co,
                "sch_a_benefit": sch_a_benefit,
                "tax": fed + co + sch_a_benefit,
                "depr_this_year": depr_this_year,
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
    """Computes the sale-year tax effect: depreciation recapture, section 121 exclusion (only if
    loan_occupancy_class == "primary_residence" AND held >= taxes.sec121_min_years), remaining
    long-term capital gain, and the release of any suspended passive losses / lost section 280A
    carryforwards -- reading the ending state straight off annual_schedule's LAST row (see
    compute_owner_tax_schedule's own docstring)."""
    deck = deck if deck is not None else load_deck()
    taxes_inputs = get_taxes_inputs(deck)

    cost_basis = scenario["acquisition"]["purchase_price"] + scenario["acquisition"]["closing_costs_total"]
    ending = annual_schedule.iloc[-1]
    depr_taken = ending["cumulative_depr_taken"]

    condo_value_schedule = project_condo_value_schedule(scenario, deck)
    proceeds = estimate_sale_proceeds(scenario, condo_value_schedule, deck)

    adj_basis = cost_basis - depr_taken
    gain = proceeds["net_sale_proceeds"] - adj_basis
    recapture = max(0.0, min(gain, depr_taken))
    rest = max(0.0, gain - recapture)

    is_primary_residence = scenario["classification"]["loan_occupancy_class"] == "primary_residence"
    held_long_enough = scenario["horizon_years"] >= taxes_inputs["sec121_min_years"]
    excluded_121 = taxes_inputs["sec121_exclusion"] if (is_primary_residence and held_long_enough) else 0.0
    excluded_121 = min(rest, excluded_121)
    ltcg = rest - excluded_121
    taxable_gain = recapture + ltcg

    fed_marginal_rate = taxes_inputs["federal_marginal_rate"]
    niit_rate = taxes_inputs["niit_rate"] if taxes_inputs["niit_applies"] else 0.0
    co_income_tax_rate = taxes_inputs["state_tax_profiles"]["CO"]["income_tax_rate"]

    fed = (
        recapture * min(fed_marginal_rate, taxes_inputs["recapture_rate"])
        + ltcg * taxes_inputs["ltcg_rate"]
        + taxable_gain * niit_rate
    )
    co = taxable_gain * co_income_tax_rate

    released_passive = ending["suspended_passive_balance"]
    release_benefit = -released_passive * (fed_marginal_rate + niit_rate + co_income_tax_rate)
    lost_280a_carry = ending["carry_280a_balance"] + ending["carry_280a_depr_balance"]

    return {
        "sale_price": proceeds["sale_price"],
        "selling_costs_total": proceeds["selling_costs_total"],
        "net_sale_proceeds": proceeds["net_sale_proceeds"],
        "gain": gain,
        "recapture": recapture,
        "excluded_121": excluded_121,
        "ltcg": ltcg,
        "fed": fed,
        "co": co,
        "released_passive": released_passive,
        "release_benefit": release_benefit,
        "lost_280a_carry": lost_280a_carry,
        "tax": fed + co + release_benefit,
    }


def compute_owner_taxes(scenario: dict[str, Any], deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """Top-level entry point for a BUY scenario (tenure=OWN). Requires scenario["acquisition"],
    scenario["classification"], scenario["financing"] to already be set (see runner.run_scenarios'
    ordering). Returns {"annual_schedule" (DataFrame, see compute_owner_tax_schedule), "sale"
    (dict, see compute_owner_sale_tax), "total_tax" (annual_schedule's "tax" column summed + sale's
    "tax"), "flags" (CO statutory-residency warning if triggered -- see
    CO_STATUTORY_RESIDENCY_DAYS; informational only, doesn't affect any number here),
    "placeholder_fields" (PLACEHOLDER_FIELDS' keys, echoed back so a caller/print never mistakes
    this for a fully-real result)."""
    deck = deck if deck is not None else load_deck()
    annual_schedule = compute_owner_tax_schedule(scenario, deck)
    sale = compute_owner_sale_tax(scenario, annual_schedule, deck)

    flags = []
    if scenario["residency_state"] == "FL":
        personal_days = scenario["classification"]["annual_personal_days"]
        if personal_days >= CO_STATUTORY_RESIDENCY_DAYS:
            flags.append(
                f"WARNING: annual_personal_days={personal_days} >= {CO_STATUTORY_RESIDENCY_DAYS} -- "
                "FL residency claim at risk (CO statutory-resident day test)."
            )

    return {
        "annual_schedule": annual_schedule,
        "sale": sale,
        "total_tax": annual_schedule["tax"].sum() + sale["tax"],
        "flags": flags,
        "placeholder_fields": list(PLACEHOLDER_FIELDS.keys()),
    }


def compute_renter_taxes(reference_case: dict[str, Any] | None = None) -> dict[str, Any]:
    """Top-level entry point for a reference case (tenure=RENT) -- a REAL, confirmed result, not a
    placeholder: renting has NO real-estate tax effect at all (rent isn't deductible, there's no
    property to depreciate or sell). Ignores reference_case's contents entirely; it's accepted only
    so callers can pass one through uniformly alongside compute_owner_taxes."""
    return {"annual_schedule": None, "sale": {"tax": 0.0}, "total_tax": 0.0, "flags": [], "use_class": "TENANT"}
