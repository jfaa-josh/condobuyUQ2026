"""ΔW stage: W_buy - W_rent at the horizon, per buy scenario vs its reference (rent) case, as a
distribution -- the model's headline: mean, sd, report bands, P(buy beats rent) = P(ΔW > 0), a variance
breakdown by source, and the special-assessment overlay. Closed form (no Monte Carlo inside the model).
Plan: .claude/implementation-plan.md, "ΔW stage plan".

    ΔW = net equity at the sale (exit.py)  -  the rent branch's portfolio after tax (capital_markets.py)
         -  F_H, the DIFFERENCE FUND after tax

BUDGET MATCHING (user decision, 2026-11-25): both households have the same income, and each invests what
it doesn't spend. The shared savings cancel out of ΔW exactly, so only the monthly spending gap matters:
    D_h = buy spending - rent spending
        = mortgage P&I + carrying costs + alternate lodging + the condo's income-tax effect
          + initial furnishing (h = 0) - net rental income - lease.
Special assessments are NOT in D (a separate overlay, below). The fund F is SIGNED: D_h > 0 (buying costs
more that month) moves F toward the rent household, which invests the difference; D_h < 0 moves it toward
the buy household. It holds the same portfolio as section 11 (capital_markets.allocation, cash_yield,
dividends), buy and hold.

TAXES ON THE FUND (user, 2026-11-25: "ok as long as we aren't ignoring any tax implications"). One fund at
one tax rate would be wrong: the money sits on a real household's return. On the EXPECTED path (taxes are
point estimates on expected values, like everywhere in the model):
  - HOLDER: the money belongs to whichever household is ahead. Rent-held money is just more of the section 11
    portfolio: its interest / dividends / gains are taxed ON TOP of the portfolio's on the renter's return
    (one stacked engine call, taxes.tax_effect). Buy-held money's income goes on the OWNER's return with the
    condo (taxes.compute_owner_tax_schedule's investment_income_by_year): brackets, NIIT, the SALT phase-down,
    the rental special allowance's MAGI phase-out, the CO apportionment (investment income isn't CO-source).
    In the sale year it stacks with the condo sale too (owner_sale_tax_function on that return).
  - DRAW-DOWNS: when the holder's lead shrinks it sells (cash and market in proportion; market lots FIFO).
    The realized gain is taxed that year, short-term (ordinary) for lots held <= 12 months, long-term otherwise.
    When the lead flips, the old holder sells everything.
  - Each year's tax is paid out of the fund at the next month's start (a refund comes back in).
  - HORIZON: what's left is sold, taxed with that household's other sale-year items. The sale tax's effective
    rate t_F is applied linearly to the random market value (as section 11 does).
  - NOT captured (stated assumption, user: "don't add savings for now"): the shared savings are assumed taxed
    alike in both branches. A capital LOSS on the fund is taxed $0 (no $3,000/yr deduction or carryforward),
    as in section 11. Reinvested dividends take their lot's purchase date (holding period).
The deterministic pieces come from a month-by-month EXPECTED-VALUE ledger (_FundLedger); its market/cash
split of each month's flow then carries the random flows.

TERMS (step 2): every random piece is a signed lognormal term c x exp(mu + l . Z), Z independent standard
normals, plus constants:
  - FACTORS (step 3): each founding process's (market, home_price, general) monthly log-growth increments;
    each derived latent's own monthly increments (its idiosyncratic share); item noise (revenue's occupancy /
    ADR year-to-year and level shocks; each carrying cost's level noise). A derived latent X = a x market +
    b x home_price + c x general + own share loads its month-t increment, of size sqrt(V_X(t) - V_X(t-1)),
    on those. ASSUMPTION: the founding processes are uncorrelated (flagged to revisit; see the plan). Every
    latent's growth log-variance V_X(h) and median come from its fitted OU term structure, exactly what the
    stages use, so Cov(log G_X(i), log G_X(j)) = V_X(min(i, j)) (the independent-increments approximation
    used throughout) and the same increments make cross-latent covariance consistent.
  - Revenue, lease, alternate lodging, carrying costs (known part + lognormal part), mortgage, the condo's
    yearly tax effect, furnishing: flows into the fund at month h, each split into a cash part (grows at
    cash_yield) and a market part (grows with market G_H / G_h and reinvested dividends, x (1 - t_F)).
  - Sale price: net equity ~ a x P + b (exit.py's "linear": the band secant slope, mean exact).
  - The rent portfolio: section 11's market lots (capital_markets' "terms").

THE DISTRIBUTION (step 4): exact mean and variance over all terms (E[exp] algebra). Skewness and excess
kurtosis from exact 3rd/4th moments of the terms AGGREGATED by (source, part, block of
run.solver.delta_w_moment_grouping_months months; default 12 = by year) -- each group's months are merged into
one lognormal with the mean-weighted loading (flagged approximation; ~200 terms keep the 4th moment's O(n^4) cost
low). Cornish-Fisher's grid and monotonicity check come from run.solver.cornish_fisher. Quantiles and P(ΔW > 0) by CORNISH-FISHER from those four moments
(flagged approximation; a warning is recorded if its expansion isn't monotone over the reported range).
VARIANCE BREAKDOWN: first-order (delta-method) shares by factor group, scaled to the exact variance.
SPECIAL-ASSESSMENT OVERLAY (plan's "Final reporting"): ΔW_total = ΔW - A, A each outcome's total
(carrying_costs.compute_special_assessment's exact outcomes table) compounded to the horizon at EXPECTED
market growth with dividends (the cash's opportunity cost; PRE-TAX, open question kept), from the payment
month per `timing`. N is assumed independent of ΔW: P(ΔW_total > 0) = sum_j p_j x P(ΔW > A_j).

APPROXIMATIONS to read the results with: taxes as point estimates; the exit sale tax linearized around the
band secant inside ΔW (exit's own bands are exact); the fund's cash/market split and holder taken from the
expected path; independent-increments covariance; aggregated higher moments; Cornish-Fisher quantiles.
Utilities-occupancy covariance still ignored (existing earmark). Market fit understates stock risk
(capital_markets' KNOWN LIMITATION). Flagged for later: a Monte Carlo cross-check; founding correlations.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from condobuyuq2026.carrying_costs import (
    load_carrying_cost_prior_fit,
    load_market_value_change_magnitude,
    market_value_periods_to_tax_periods,
    resolve_period,
)
from condobuyuq2026.input_deck import (
    DERIVED_LATENT_COMPONENTS,
    MONTH_KEYS,
    get_capital_markets_inputs,
    get_closing_date,
    get_derived_latents,
    get_growth_fit,
    get_outer_band,
    get_personal_use_inputs,
    get_rent_reference_inputs,
    get_solver_settings,
    get_report_quantiles,
    get_taxes_inputs,
    load_deck,
)
from condobuyuq2026.rent_reference import LEASE_RENEWAL_MONTHS
from condobuyuq2026.taxes import (
    INVESTMENT_ITEMS,
    NO_PROPERTY,
    compute_owner_tax_schedule,
    get_scenario_tax_jurisdiction,
    owner_sale_tax_function,
    tax_effect,
)
from condobuyuq2026.utils.ou_fitting import project_term_structure

MARKET_LATENT = "market"
RENT, BUY = 1, -1  # fund holder: +1 = the rent household, -1 = the buy household
LONG_TERM_HOLDING_MONTHS = 12  # matches federal.yaml property_sale (the same IRS holding-period rule)
BAND_NAMES = ("median", "lo", "hi", "lo_outer", "hi_outer")

# Carrying costs: (label, saved prior name, monthly scale). property_tax/insurance/maintenance are saved as
# annual-cadence amounts (/12 per month); hoa_dues/utilities are monthly bills (same as
# carrying_costs.compute_total_carrying_costs_schedule).
CARRYING_ITEMS = (
    ("property tax", "market_value", 1 / 12),
    ("HOA dues", "hoa_dues_annual", 1.0),
    ("insurance", "insurance_annual", 1 / 12),
    ("maintenance", "maintenance_annual", 1 / 12),
    ("utilities", "utilities", 1.0),
)


# ---------------------------------------------------------------------------------------------------
# Factors: the shared standard normals every term loads on
# ---------------------------------------------------------------------------------------------------


class _Factors:
    """Column layout of the factor loadings. Columns 0..K0-1: monthly increments t = 1..horizon_months of
    each founding process, then of each derived latent's own share. Item factors (K1 of them) are
    registered by name as terms need them. growth(X)[h] is the loading vector of log G_X(h) - median."""

    def __init__(self, horizon_months: int, deck: dict[str, Any]) -> None:
        self.horizon_months = horizon_months
        self.deck = deck
        report_quantiles = get_report_quantiles(deck)
        self.ci = (report_quantiles[0], report_quantiles[-1])
        self.weights = get_derived_latents(deck)
        self.blocks = [*DERIVED_LATENT_COMPONENTS, *self.weights]
        self.block_index = {name: i for i, name in enumerate(self.blocks)}
        self.k0 = len(self.blocks) * horizon_months
        self.group = [
            name if name in DERIVED_LATENT_COMPONENTS else f"{name} (own share)"
            for name in self.blocks
            for _ in range(horizon_months)
        ]
        self.items: dict[str, int] = {}
        self.item_group: list[str] = []
        self._growth: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def item(self, name: str, group: str) -> int:
        if name not in self.items:
            self.items[name] = len(self.items)
            self.item_group.append(group)
        return self.items[name]

    def latent(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(mu, var, loadings) for latent `name`: median log growth and its variance at h = 0..H, and the
        (H+1) x K0 loading matrix of log G(h) - mu(h)."""
        if name not in self._growth:
            hm = self.horizon_months
            projection = project_term_structure(get_growth_fit(name, self.deck), 1.0, hm, ci=self.ci)
            median = projection["median"].to_numpy()[: hm + 1]
            mu = np.log(median)
            var = ((np.log(projection["hi"].to_numpy()[: hm + 1] / median) / norm.ppf(self.ci[1])) ** 2)
            step = np.sqrt(np.maximum(np.diff(var), 0.0))  # increment size for t = 1..H
            increments = np.zeros((hm + 1, self.k0))
            months = np.arange(1, hm + 1)
            if name in DERIVED_LATENT_COMPONENTS:
                base = self.block_index[name] * hm
                increments[months, base + months - 1] = step
            else:
                weights = self.weights[name]
                for founding in DERIVED_LATENT_COMPONENTS:
                    base = self.block_index[founding] * hm
                    increments[months, base + months - 1] = weights[founding] * step
                own = math.sqrt(max(0.0, 1 - sum(w**2 for w in weights.values())))
                base = self.block_index[name] * hm
                increments[months, base + months - 1] = own * step
            self._growth[name] = (mu, var, np.cumsum(increments, axis=0))
        return self._growth[name]


class _Terms:
    """Signed lognormal terms coef x exp(mu + l0 . Z0 + l1 . Z1) plus constants, each tagged with its
    source (for the variance and component breakdowns), its fund part (cash / market / none) and the month its
    cash flow happens (for the higher-moment grouping)."""

    def __init__(self, factors: _Factors) -> None:
        self.factors = factors
        self.coef: list[float] = []
        self.mu: list[float] = []
        self.l0: list[np.ndarray] = []
        self.l1: list[dict[int, float]] = []
        self.source: list[str] = []
        self.part: list[str] = []
        self.month: list[int] = []
        self.constants: dict[str, float] = defaultdict(float)

    def add(self, coef: float, mu: float, l0: np.ndarray | None, l1: dict[int, float] | None, source: str,
            part: str, month: int) -> None:
        if coef == 0:
            return
        if (l0 is None or not np.any(l0)) and not any(l1 or {}):
            self.constants[source] += coef * math.exp(mu)
            return
        self.coef.append(coef)
        self.mu.append(mu)
        self.l0.append(l0 if l0 is not None else np.zeros(self.factors.k0))
        self.l1.append(dict(l1 or {}))
        self.source.append(source)
        self.part.append(part)
        self.month.append(month)

    def matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(coef x exp(mu) as weights at the median, the n x K loading matrix, the column groups)."""
        n, k1 = len(self.coef), len(self.factors.items)
        loadings = np.zeros((n, self.factors.k0 + k1))
        if n:
            loadings[:, : self.factors.k0] = np.vstack(self.l0)
        for i, row in enumerate(self.l1):
            for column, value in row.items():
                loadings[i, self.factors.k0 + column] += value
        groups = np.array([*self.factors.group, *self.factors.item_group])
        return np.asarray(self.coef) * np.exp(np.asarray(self.mu)), loadings, groups


# ---------------------------------------------------------------------------------------------------
# Step 1: the monthly spending gap D_h as terms
# ---------------------------------------------------------------------------------------------------


def _carrying_item_loading(factors: _Factors, latent: str, h: int, total_var: float, item: str
                           ) -> tuple[np.ndarray | None, dict[int, float]]:
    """A saved carrying-cost period's lognormal part has log-variance growth_sigma^2. It loads on its latent's
    growth up to month h; any extra variance (the item's own level uncertainty, e.g. insurance's
    derived_extra_log_variance) goes on ONE persistent item factor; if the period's variance is smaller than
    the latent's (property tax: the assessor's averaged, earlier window), the latent loading is scaled down."""
    if total_var <= 0:
        return None, {}
    _, var, loadings = factors.latent(latent)
    h = min(h, factors.horizon_months)
    base_var = var[h]
    if base_var <= 0:
        return None, {factors.item(f"{item} level", f"{item} (own level)"): math.sqrt(total_var)}
    if total_var >= base_var:
        extra = total_var - base_var
        l1 = {factors.item(f"{item} level", f"{item} (own level)"): math.sqrt(extra)} if extra > 1e-12 else {}
        return loadings[h], l1
    return loadings[h] * math.sqrt(total_var / base_var), {}


def _flow_terms(scenario: dict[str, Any], factors: _Factors,
                deck: dict[str, Any]) -> tuple[list[tuple[int, float, float, np.ndarray | None, dict[int, float], str]],
                                               np.ndarray]:
    """Every piece of D_h = buy spending - rent spending as (h, coef, mu, l0, l1, source), h = 0..H (H = the
    horizon instant), plus E[D_h] by month."""
    hm = factors.horizon_months
    closing_month = get_closing_date(deck)["closing_month"]
    calendar = [MONTH_KEYS[(closing_month - 1 + h) % 12] for h in range(hm)]
    flows: list[tuple[int, float, float, np.ndarray | None, dict[int, float], str]] = []

    # Mortgage P&I: payment k at the end of month k (= start of month k), k = 1..H.
    schedule = scenario["financing"]["amortization_schedule"]
    for k in range(1, hm + 1):
        if k in schedule.index and schedule.loc[k, "payment"] > 0:
            flows.append((k, float(schedule.loc[k, "payment"]), 0.0, None, {}, "mortgage"))

    # Carrying costs: known part + lognormal part per month (the saved periods, as the stages use them).
    for label, name, scale in CARRYING_ITEMS:
        if name == "market_value":
            config = load_market_value_change_magnitude()
            periods = market_value_periods_to_tax_periods(config["periods"], deck)
        else:
            config = load_carrying_cost_prior_fit(name)
            periods = config["periods"]
        for h in range(hm):
            period = resolve_period(periods, h)
            known, projected, sigma = (period["known_value"] * scale, period["projected_value"] * scale,
                                       period["growth_sigma"])
            if known:
                flows.append((h, known, 0.0, None, {}, label))
            if projected:
                at = period["start_month"] if name == "market_value" else h
                l0, l1 = _carrying_item_loading(factors, config["latent"], at, sigma**2, label)
                flows.append((h, projected, 0.0, l0, l1, label))

    # The condo's income-tax effect, paid at each scenario year's end (base schedule, without the fund).
    for year, tax in scenario["taxes"]["annual_schedule"]["tax"].items():
        if tax:
            flows.append((int(year) * 12, float(tax), 0.0, None, {}, "income tax (condo)"))

    furnishing = float(scenario["acquisition"].get("initial_furnishing") or 0.0)
    if furnishing:
        flows.append((0, furnishing, 0.0, None, {}, "furnishing"))

    # Alternate lodging (moves every month) and the lease (resets each closing anniversary); same math as
    # grown_cost.project_grown_monthly_cost.
    personal = get_personal_use_inputs(deck)
    lease = get_rent_reference_inputs(deck)
    for costs, latent, renew, sign, source in (
        (personal["alternate_lodging_cost_by_month"], personal["alternate_lodging_growth_latent"], 1, 1.0,
         "alternate lodging"),
        (lease["lease_cost_by_month"], lease["lease_growth_latent"], LEASE_RENEWAL_MONTHS, -1.0, "lease"),
    ):
        mu, _, loadings = factors.latent(latent)
        for h in range(hm):
            if costs[calendar[h]] > 0:
                reset = (h // renew) * renew
                flows.append((h, sign * costs[calendar[h]], float(mu[reset]), loadings[reset], {}, source))

    # Net rental income (reduces the buy household's spending): rental_operations' own log structure.
    terms = scenario["rental_operations"]["terms"]
    _, _, revenue_growth = factors.latent(terms["growth_latent"])
    rho = terms["rho"]
    occ_level = factors.item("occupancy level", "revenue: occupancy")
    adr_level = factors.item("ADR level", "revenue: ADR")
    for month in np.flatnonzero(terms["active"]):
        h = int(month)
        year = h // 12 + 1
        occ_yy = factors.item(f"occupancy year {year}", "revenue: occupancy")
        adr_yy = factors.item(f"ADR year {year}", "revenue: ADR")
        a, b, c, d = terms["occ_yy"][h], terms["occ_level"][h], terms["adr_yy"][h], terms["adr_level"][h]
        l1 = {occ_level: b, adr_level: d, adr_yy: c * math.sqrt(max(0.0, 1 - rho**2))}
        l1[occ_yy] = a + c * rho  # occupancy year-to-year shock + the ADR shock's correlated part
        flows.append((h, -terms["keep_fraction"], float(terms["mu"][h]), revenue_growth[h], l1, "net rental income"))

    expected = np.zeros(hm + 1)
    for at, coef, log_median, l0, l1, _ in flows:
        var = (float(l0 @ l0) if l0 is not None else 0.0) + sum(v**2 for v in l1.values())
        expected[at] += coef * math.exp(log_median + var / 2)
    return flows, expected


# ---------------------------------------------------------------------------------------------------
# Step 1: the difference fund's expected-value ledger (holder, draw-downs, taxes)
# ---------------------------------------------------------------------------------------------------


class _FundLedger:
    """The signed fund on the EXPECTED path: one account (cash + FIFO market lots [bought month, value,
    basis]) held by whichever household is ahead. Records each month's market share of the flow, and every
    income item and tax by holder and year (see the module docstring's TAXES)."""

    def __init__(self, horizon_months: int, market_share: float, cash_rate: float, dividend_rate: float,
                 price_growth: np.ndarray) -> None:
        self.hm = horizon_months
        self.market_share = market_share
        self.cash_rate = cash_rate
        self.dividend_rate = dividend_rate
        self.price_growth = price_growth
        self.holder = 0
        self.cash = 0.0
        self.lots: list[list[float]] = []
        self.income: dict[int, dict[int, dict[str, float]]] = {
            side: defaultdict(lambda: dict.fromkeys(INVESTMENT_ITEMS, 0.0)) for side in (RENT, BUY)
        }
        self.liquidation = dict.fromkeys(INVESTMENT_ITEMS, 0.0)

    def balance(self) -> float:
        return self.cash + sum(lot[1] for lot in self.lots)

    def market(self) -> float:
        return sum(lot[1] for lot in self.lots)

    def basis(self) -> float:
        return sum(lot[2] for lot in self.lots)

    def _year(self, h: int) -> int:
        return min(h // 12 + 1, self.hm // 12)

    def _realize(self, h: int, gain: float, held_months: int, liquidation: bool) -> None:
        key = "investment_ltcg" if held_months > LONG_TERM_HOLDING_MONTHS else "interest_income"  # ST = ordinary
        if liquidation:
            self.liquidation[key] += gain
        else:
            self.income[self.holder][self._year(h)][key] += gain

    def _deposit(self, h: int, amount: float, side: int) -> float:
        self.holder = side
        market_in = self.market_share * amount
        self.cash += amount - market_in
        if market_in > 0:
            self.lots.append([h, market_in, market_in])
        return market_in

    def _withdraw(self, h: int, amount: float, liquidation: bool = False) -> float:
        """Sells `amount` (<= balance) from cash and market in proportion; market lots FIFO. Returns the market
        dollars sold."""
        total = self.balance()
        if total <= 0:
            return 0.0
        fraction = min(1.0, amount / total)
        self.cash -= self.cash * fraction
        to_sell = self.market() * fraction
        sold = 0.0
        while to_sell > 1e-9 and self.lots:
            lot = self.lots[0]
            take = min(to_sell, lot[1])
            basis_out = lot[2] * take / lot[1] if lot[1] > 0 else 0.0
            self._realize(h, take - basis_out, h - int(lot[0]), liquidation)
            lot[1] -= take
            lot[2] -= basis_out
            to_sell -= take
            sold += take
            if lot[1] <= 1e-9:
                self.lots.pop(0)
        if fraction >= 1.0:
            self.cash, self.lots = 0.0, []
        return sold

    def flow(self, h: int, amount: float) -> float:
        """Signed flow into F at the start of month h (+ toward the rent household). Returns the signed
        market dollars it moves."""
        if abs(amount) < 1e-9:
            return 0.0
        side = RENT if amount > 0 else BUY
        if self.holder in (0, side):
            return side * self._deposit(h, abs(amount), side)
        balance = self.balance()
        if abs(amount) < balance:
            return side * self._withdraw(h, abs(amount))
        sold = self._withdraw(h, balance)
        self.holder = 0
        market = side * sold
        leftover = abs(amount) - balance
        if leftover > 1e-9:
            market += side * self._deposit(h, leftover, side)
        return market

    def grow(self, h: int) -> None:
        """Month h's interest and dividends (reinvested into the same lot; basis up by the dividend)."""
        if self.holder == 0:
            return
        income = self.income[self.holder][self._year(h)]
        interest = self.cash * self.cash_rate
        self.cash += interest
        income["interest_income"] += interest
        for lot in self.lots:
            lot[1] *= self.price_growth[h] * (1 + self.dividend_rate)
            dividend = lot[1] * self.dividend_rate / (1 + self.dividend_rate)
            lot[2] += dividend
            income["qualified_dividends"] += dividend


def _run_ledger(scenario: dict[str, Any], reference_result: dict[str, Any], expected_flows: np.ndarray,
                factors: _Factors, deck: dict[str, Any]) -> dict[str, Any]:
    """Runs the fund on the expected path with yearly taxes by holder (see _FundLedger)."""
    hm = factors.horizon_months
    horizon_years = hm // 12
    inputs = get_capital_markets_inputs(deck)
    taxes_inputs = get_taxes_inputs(deck)
    closing_year = get_closing_date(deck)["closing_year"]
    mu_m, var_m, _ = factors.latent(MARKET_LATENT)
    price_growth = np.exp(np.diff(mu_m) + np.diff(var_m) / 2)
    ledger = _FundLedger(hm, inputs["allocation"]["market"], inputs["cash_yield"] / 12,
                         inputs["dividend_paying_fraction"] * inputs["dividend_yield"] / 12, price_growth)

    # The rent household's return: the section 11 portfolio's items each year (and its sale in the last year).
    jurisdiction = get_scenario_tax_jurisdiction(scenario, deck)
    portfolio = reference_result["capital_markets"]
    portfolio_gain = max(0.0, float(portfolio["sale"]["gain"]))
    base_owner = scenario["taxes"]["annual_schedule"]
    base_owner_sale = float(scenario["taxes"]["sale"]["tax"])
    expected_price = float(scenario["taxes"]["sale"]["sale_price"])

    def rent_tax(year: int, fund_items: dict[str, float]) -> float:
        row = portfolio["annual"].loc[year]
        base = {**NO_PROPERTY, "interest_income": float(row["interest"]), "qualified_dividends": float(row["dividends"]),
                "investment_ltcg": portfolio_gain if year == horizon_years else 0.0}
        with_fund = {**base, **{key: base[key] + fund_items[key] for key in INVESTMENT_ITEMS}}
        return tax_effect(jurisdiction, taxes_inputs, closing_year + year - 1, with_fund, base)["total"]

    def buy_tax(year: int, final_items: dict[str, float]) -> float:
        by_year = {y: dict(items) for y, items in ledger.income[BUY].items() if y < year}
        by_year[year] = final_items
        schedule = compute_owner_tax_schedule(scenario, deck, investment_income_by_year=by_year)
        tax = float(schedule.loc[year, "tax"] - base_owner.loc[year, "tax"])
        if year == horizon_years:
            tax += owner_sale_tax_function(scenario, schedule, deck)(expected_price)["tax"] - base_owner_sale
        return tax

    def has_income(items: dict[str, float]) -> bool:
        return any(abs(v) > 1e-9 for v in items.values())

    def year_tax(side: int, year: int, items: dict[str, float]) -> float:
        if side == RENT:
            return rent_tax(year, items) if has_income(items) else 0.0
        # The buy side's earlier fund income can still move this year's tax (e.g. suspended passive losses
        # carried forward), so it's recomputed whenever the buy household has EVER held the fund.
        ever = has_income(items) or any(has_income(v) for y, v in ledger.income[BUY].items() if y < year)
        return buy_tax(year, items) if ever else 0.0

    market_flow = np.zeros(hm + 1)
    tax_events: list[tuple[int, float, float]] = []  # (h, signed amount, signed market part)
    taxes_by_year: dict[int, dict[int, float]] = {RENT: {}, BUY: {}}
    rows = []
    pending: list[float] = []
    for h in range(hm):
        for amount in pending:
            tax_events.append((h, amount, ledger.flow(h, amount)))
        pending = []
        market_flow[h] = ledger.flow(h, float(expected_flows[h]))
        ledger.grow(h)
        if (h + 1) % 12 == 0 and (h + 1) < hm:
            year = (h + 1) // 12
            for side in (RENT, BUY):
                tax = year_tax(side, year, ledger.income[side][year])
                taxes_by_year[side][year] = tax
                if tax:
                    pending.append(-side * tax)
            rows.append(_ledger_row(ledger, year, expected_flows, taxes_by_year))

    # The horizon instant: last flows (the final mortgage payment and the last year's condo tax), then the sale.
    for amount in pending:
        tax_events.append((hm, amount, ledger.flow(hm, amount)))
    market_flow[hm] = ledger.flow(hm, float(expected_flows[hm]))
    final_holder = ledger.holder
    final = {"cash": ledger.cash, "market": ledger.market(), "basis": ledger.basis()}
    if final_holder:
        ledger._withdraw(hm, ledger.balance(), liquidation=True)

    income_tax_final, liquidation_tax = {}, 0.0
    for side in (RENT, BUY):
        income = ledger.income[side][horizon_years]
        income_tax_final[side] = year_tax(side, horizon_years, income)
        if side == final_holder:
            with_sale = {key: income[key] + ledger.liquidation[key] for key in INVESTMENT_ITEMS}
            liquidation_tax = year_tax(side, horizon_years, with_sale) - income_tax_final[side]
        taxes_by_year[side][horizon_years] = income_tax_final[side] + (liquidation_tax if side == final_holder else 0)
    gain = final["market"] - final["basis"]  # the holder's (unsigned) expected gain
    sale_rate = liquidation_tax / gain if final_holder and abs(gain) > 1.0 else 0.0
    constant_sale_tax = 0.0 if sale_rate or not final_holder else liquidation_tax
    signed_cash = final_holder * final["cash"]
    signed_market = final_holder * final["market"]
    signed_basis = final_holder * final["basis"]
    after_tax = (signed_cash + signed_market - sale_rate * (signed_market - signed_basis)
                 - final_holder * constant_sale_tax - sum(side * tax for side, tax in income_tax_final.items()))
    rows.append({
        "year": horizon_years, "holder": final_holder, "flows": float(expected_flows[(horizon_years - 1) * 12: hm + 1].sum()),
        "rent_tax": taxes_by_year[RENT][horizon_years], "buy_tax": taxes_by_year[BUY][horizon_years],
        "balance_signed": after_tax, "basis_signed": signed_basis,
        **{f"rent_{k}": ledger.income[RENT][horizon_years][k] for k in INVESTMENT_ITEMS},
        **{f"buy_{k}": ledger.income[BUY][horizon_years][k] for k in INVESTMENT_ITEMS},
    })
    for key in INVESTMENT_ITEMS:
        rows[-1][f"{'rent' if final_holder == RENT else 'buy'}_{key}"] += ledger.liquidation[key] if final_holder else 0.0

    return {
        "market_flow": market_flow,
        "tax_events": tax_events,
        "final_holder": final_holder,
        "sale_rate": sale_rate,
        "signed_basis": signed_basis,
        "constant_sale_tax": constant_sale_tax,
        "income_tax_final": income_tax_final,
        "liquidation_tax": liquidation_tax,
        "after_tax_expected": after_tax,
        "annual": pd.DataFrame(rows).set_index("year"),
        "inputs": inputs,
    }


def _ledger_row(ledger: _FundLedger, year: int, expected_flows: np.ndarray,
                taxes_by_year: dict[int, dict[int, float]]) -> dict[str, Any]:
    return {
        "year": year, "holder": ledger.holder, "flows": float(expected_flows[(year - 1) * 12: year * 12].sum()),
        "rent_tax": taxes_by_year[RENT].get(year, 0.0), "buy_tax": taxes_by_year[BUY].get(year, 0.0),
        "balance_signed": ledger.holder * ledger.balance(), "basis_signed": ledger.holder * ledger.basis(),
        **{f"rent_{k}": ledger.income[RENT][year][k] for k in INVESTMENT_ITEMS},
        **{f"buy_{k}": ledger.income[BUY][year][k] for k in INVESTMENT_ITEMS},
    }


# ---------------------------------------------------------------------------------------------------
# Step 4: moments, Cornish-Fisher, breakdown
# ---------------------------------------------------------------------------------------------------


def _mean_var(weights: np.ndarray, loadings: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Exact mean and variance of sum_i weights_i x exp(l_i . Z) (weights at the median, Z standard normal):
    E = sum w_i e^(s_i/2), Var = m' (exp(L L') - 1) m with m the terms' means. Returns (mean, variance, m)."""
    s = np.einsum("ij,ij->i", loadings, loadings)
    means = weights * np.exp(s / 2)
    cov = loadings @ loadings.T
    variance = float(means @ np.expm1(cov) @ means)
    return float(means.sum()), max(variance, 0.0), means


def _aggregate(means: np.ndarray, loadings: np.ndarray, keys: list[tuple[Any, ...]]) -> tuple[np.ndarray, np.ndarray]:
    """Merges terms by key into one lognormal each: weight = the group's mean, loading = mean-weighted."""
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        groups[(*key, means[i] > 0)].append(i)
    out_means, out_loadings = [], []
    for index in groups.values():
        w = means[index]
        total = float(w.sum())
        if abs(total) < 1e-9:
            continue
        out_means.append(total)
        out_loadings.append((w @ loadings[index]) / total)
    return np.asarray(out_means), np.vstack(out_loadings) if out_loadings else np.zeros((0, loadings.shape[1]))


def _shape(means: np.ndarray, loadings: np.ndarray) -> tuple[float, float]:
    """Skewness and excess kurtosis of T = sum_g W_g exp(l_g.Z - |l_g|^2/2) from its exact raw moments:
    E[T^2] = W'FW, E[T^3] = tr((F diag W)^3), E[T^4] = sum_i W_i tr((F diag(W o F_i))^3), F = exp(l l')."""
    if means.size == 0:
        return 0.0, 0.0
    scale = float(np.abs(means).max())
    w = means / scale
    f = np.exp(loadings @ loadings.T)
    m1 = float(w.sum())
    e2 = float(w @ f @ w)
    p = f * w[None, :]
    p2 = p @ p
    e3 = float(np.einsum("ij,ji->", p2, p))
    e4 = 0.0
    for i in range(w.size):
        q = f * (w * f[i])[None, :]
        e4 += w[i] * float(np.einsum("ij,ji->", q @ q, q))
    k2 = e2 - m1**2
    if k2 <= 0:
        return 0.0, 0.0
    k3 = e3 - 3 * e2 * m1 + 2 * m1**3
    mu4 = e4 - 4 * e3 * m1 + 6 * e2 * m1**2 - 3 * m1**4
    return k3 / k2**1.5, mu4 / k2**2 - 3


class CornishFisher:
    """x(z) = mean + sd x [z + (z^2-1) g1/6 + (z^3-3z) g2/24 - (2z^3-5z) g1^2/36] on a z grid (z_range,
    grid_points: run.solver.cornish_fisher); the CDF inverts it. If the expansion isn't increasing over
    |z| <= check_z it's flagged (`monotone` False); it's always made non-decreasing over the whole grid."""

    def __init__(self, mean: float, sd: float, skew: float, excess_kurtosis: float, z_range: float = 6.0,
                 grid_points: int = 4801, check_z: float = 3.5) -> None:
        self.z_range, self.check_z = z_range, check_z
        self.z = np.linspace(-z_range, z_range, grid_points)
        z, g1, g2 = self.z, skew, excess_kurtosis
        x = mean + sd * (z + (z**2 - 1) * g1 / 6 + (z**3 - 3 * z) * g2 / 24 - (2 * z**3 - 5 * z) * g1**2 / 36)
        inner = np.abs(z) <= check_z  # where reported numbers come from
        self.monotone = bool(np.all(np.diff(x[inner]) > 0)) if sd > 0 else True
        self.x = np.maximum.accumulate(x)
        self.mean, self.sd = mean, sd

    def quantile(self, p: float) -> float:
        return float(np.interp(norm.ppf(p), self.z, self.x))

    def cdf(self, x: np.ndarray | float) -> np.ndarray:
        if self.sd <= 0:
            return np.where(np.asarray(x) >= self.mean, 1.0, 0.0)
        return norm.cdf(np.interp(x, self.x, self.z, left=-self.z_range, right=self.z_range))


# ---------------------------------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------------------------------


def compute_delta_w(scenario: dict[str, Any], reference_result: dict[str, Any],
                    deck: dict[str, Any] | None = None) -> dict[str, Any]:
    """ΔW for one buy scenario vs its reference case (see the module docstring). Needs scenario["financing"],
    ["carrying_costs"], ["rental_operations"], ["taxes"], ["exit"], and reference_result["capital_markets"].
    Returns:
      "mean", "sd", "skewness", "excess_kurtosis", "bands" ({median, lo, hi, lo_outer, hi_outer}),
      "p_buy_beats_rent", "cornish_fisher" (the CornishFisher object, for plots), "cf_monotone";
      "components": expected W_buy (net_equity, buy_held_fund) and W_rent (portfolio, rent_held_fund);
      "fund": the ledger (annual DataFrame: holder +1 rent / -1 buy, flows, each side's income items and taxes,
        signed balance and basis), final_holder, sale_rate, after_tax_expected, expected_flows (by month);
      "flows_by_source": expected D by source over the horizon;
      "variance_breakdown": {group: share of variance};
      "overlay": the special-assessment overlay (enabled, p_buy_beats_rent, mean, by_count table,
        breakeven_assessment, compound_factor, outcomes) -- see the module docstring;
      "flags": approximation warnings; "ci", "outer_ci"."""
    deck = deck if deck is not None else load_deck()
    hm = int(scenario["horizon_years"]) * 12
    factors = _Factors(hm, deck)
    solver = get_solver_settings(deck)
    ci, outer_ci = factors.ci, get_outer_band(deck)
    quantiles = {"median": 0.5, "lo": ci[0], "hi": ci[1], "lo_outer": outer_ci[0], "hi_outer": outer_ci[1]}
    flags: list[str] = []

    # Step 1: the spending gap and the fund on the expected path.
    flows, expected_flows = _flow_terms(scenario, factors, deck)
    ledger = _run_ledger(scenario, reference_result, expected_flows, factors, deck)
    inputs = ledger["inputs"]
    market_share = inputs["allocation"]["market"]
    cash_rate = inputs["cash_yield"] / 12
    dividend_rate = inputs["dividend_paying_fraction"] * inputs["dividend_yield"] / 12
    mu_m, _, market_loadings = factors.latent(MARKET_LATENT)
    keep_after_sale = 1 - ledger["sale_rate"]

    # Step 2: the terms. ΔW = equity - portfolio - F.
    terms = _Terms(factors)

    def add_fund(h: int, coef: float, mu: float, l0: np.ndarray | None, l1: dict[int, float], source: str,
                 phi: float) -> None:
        terms.add(-coef * (1 - phi) * (1 + cash_rate) ** (hm - h), mu, l0, l1, source, "fund cash", h)
        market_l0 = market_loadings[hm] - market_loadings[h] + (l0 if l0 is not None else 0.0)
        terms.add(-coef * phi * (1 + dividend_rate) ** (hm - h) * keep_after_sale,
                  mu + mu_m[hm] - mu_m[h], market_l0, l1, source, "fund market", h)

    phi_by_month = np.where(np.abs(expected_flows) > 1e-9,
                            ledger["market_flow"] / np.where(np.abs(expected_flows) > 1e-9, expected_flows, 1.0),
                            market_share)
    for h, coef, mu, l0, l1, source in flows:
        add_fund(h, coef, mu, l0, l1, source, float(phi_by_month[h]))
    for h, amount, market in ledger["tax_events"]:
        add_fund(h, amount, 0.0, None, {}, "fund taxes", market / amount if amount else 0.0)
    # At the sale: F_after = cash + (1 - t_F) x market + t_F x basis - the final year's taxes (signed).
    final_adjustment = (ledger["sale_rate"] * ledger["signed_basis"]
                        - ledger["final_holder"] * ledger["constant_sale_tax"]
                        - sum(side * tax for side, tax in ledger["income_tax_final"].items()))
    terms.constants["fund taxes"] -= final_adjustment

    portfolio = reference_result["capital_markets"]["terms"]
    for start, log_factor in zip(portfolio["start_month"], portfolio["log_mean_factor"], strict=True):
        terms.add(-(1 - portfolio["sale_rate"]), float(log_factor), market_loadings[hm] - market_loadings[int(start)],
                  {}, "rent portfolio", "portfolio", hm)
    terms.constants["rent portfolio"] -= portfolio["cash"] + portfolio["sale_rate"] * portfolio["basis"]

    exit_result = scenario["exit"]
    price, linear = exit_result["sale_price"], exit_result["linear"]
    _, hp_var, hp_loadings = factors.latent("home_price")
    hp_scale = price["sigma"] / math.sqrt(hp_var[hm]) if hp_var[hm] > 0 else 0.0
    terms.add(linear["a"], price["mu"], hp_loadings[hm] * hp_scale, {}, "sale price", "equity", hm)
    terms.constants["sale price"] += linear["b"]

    # Step 4: moments.
    weights, loadings, groups = terms.matrix()
    constant = float(sum(terms.constants.values()))
    random_mean, variance, term_means = _mean_var(weights, loadings)
    mean, sd = constant + random_mean, math.sqrt(variance)
    grouping = solver["delta_w_moment_grouping_months"]
    keys = [(source, part, month // grouping) for source, part, month in zip(terms.source, terms.part, terms.month,
                                                                             strict=True)]
    agg_means, agg_loadings = _aggregate(term_means, loadings, keys)
    skew, kurt = _shape(agg_means, agg_loadings)
    cf = CornishFisher(mean, sd, skew, kurt, solver["cornish_fisher_z_range"], solver["cornish_fisher_grid_points"],
                       solver["cornish_fisher_check_z"])
    if not cf.monotone:
        flags.append(f"Cornish-Fisher expansion not monotone over z in [-{cf.check_z:g}, {cf.check_z:g}] "
                     "(run.solver.cornish_fisher.check_z): tail quantiles are unreliable.")
    bands = {band: cf.quantile(p) for band, p in quantiles.items()}
    p_buy = float(1 - cf.cdf(0.0))

    # Variance breakdown: first-order shares by factor group.
    gradient = term_means @ loadings
    contributions = pd.Series(gradient**2).groupby(groups).sum()
    contributions = contributions[contributions > 0].sort_values(ascending=False)
    breakdown = (contributions / contributions.sum()).to_dict() if contributions.sum() > 0 else {}

    # Components (expected): W_buy = equity + buy-held fund; W_rent = portfolio + rent-held fund.
    by_source = pd.Series(term_means).groupby(np.asarray(terms.source)).sum()
    for source, value in terms.constants.items():
        by_source[source] = by_source.get(source, 0.0) + value
    fund_after_tax = -float(sum(v for s, v in by_source.items() if s not in ("sale price", "rent portfolio")))
    components = {
        "net_equity": float(by_source.get("sale price", 0.0)),
        "buy_held_fund": max(0.0, -fund_after_tax),
        "portfolio": -float(by_source.get("rent portfolio", 0.0)),
        "rent_held_fund": max(0.0, fund_after_tax),
        "fund_after_tax_signed": fund_after_tax,
        "fund_after_tax_ledger": ledger["after_tax_expected"],
    }
    components["w_buy"] = components["net_equity"] + components["buy_held_fund"]
    components["w_rent"] = components["portfolio"] + components["rent_held_fund"]
    if abs(fund_after_tax - ledger["after_tax_expected"]) > 1e-6 * max(1.0, abs(fund_after_tax)) + 1.0:
        flags.append(f"fund check: terms ${fund_after_tax:,.0f} vs ledger ${ledger['after_tax_expected']:,.0f}")

    flows_by_source: dict[str, float] = defaultdict(float)
    for _h, coef, mu, l0, l1, source in flows:
        var = (float(l0 @ l0) if l0 is not None else 0.0) + sum(v**2 for v in l1.values())
        flows_by_source[source] += coef * math.exp(mu + var / 2)

    overlay = _special_assessment_overlay(scenario, cf, mean, hm, mu_m, factors, dividend_rate)

    return {
        "mean": mean, "sd": sd, "skewness": skew, "excess_kurtosis": kurt, "bands": bands,
        "p_buy_beats_rent": p_buy, "cornish_fisher": cf, "cf_monotone": cf.monotone,
        "components": components,
        "fund": {**{k: ledger[k] for k in ("annual", "final_holder", "sale_rate", "after_tax_expected",
                                            "liquidation_tax")}, "expected_flows": expected_flows},
        "flows_by_source": dict(flows_by_source),
        "variance_breakdown": breakdown,
        "overlay": overlay,
        "term_count": {"exact": int(weights.size), "aggregated": int(agg_means.size)},
        "solver": solver, "outer_band": outer_ci,
        "flags": flags, "ci": ci, "outer_ci": outer_ci,
    }


def _special_assessment_overlay(scenario: dict[str, Any], cf: CornishFisher, mean: float, hm: int,
                                mu_m: np.ndarray, factors: _Factors, dividend_rate: float) -> dict[str, Any]:
    """See the module docstring's SPECIAL-ASSESSMENT OVERLAY."""
    special = scenario["carrying_costs"]["special_assessment"]
    if not special["enabled"]:
        return {"enabled": False}
    _, var_m, _ = factors.latent(MARKET_LATENT)
    months = np.arange(hm)
    growth = np.exp(mu_m[hm] - mu_m[months] + (var_m[hm] - var_m[months]) / 2) * (1 + dividend_rate) ** (hm - months)
    compound = float(growth[0]) if special["timing"] == "worst_case_year1" else float(growth.mean())
    outcomes = special["outcomes"].copy()
    outcomes["at_horizon_usd"] = outcomes["total_cost_usd"] * compound
    outcomes["p_buy_beats_rent"] = 1 - cf.cdf(outcomes["at_horizon_usd"].to_numpy())

    def mixture_median(rows: pd.DataFrame) -> float:
        p = rows["probability"].to_numpy() / rows["probability"].sum()
        shifts = rows["at_horizon_usd"].to_numpy()
        lo, hi = cf.x[0] - shifts.max(), cf.x[-1]
        for _ in range(80):
            mid = (lo + hi) / 2
            if float(p @ cf.cdf(mid + shifts)) < 0.5:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    by_count = []
    for k, rows in outcomes.groupby("k"):
        p_k = float(rows["probability"].sum())
        by_count.append({
            "k": int(k), "probability": p_k,
            "p_buy_beats_rent": float((rows["probability"] * rows["p_buy_beats_rent"]).sum() / p_k) if p_k else math.nan,
            "median": mixture_median(rows) if p_k else math.nan,
        })
    median = cf.quantile(0.5)
    return {
        "enabled": True,
        "compound_factor": compound,
        "outcomes": outcomes,
        "p_buy_beats_rent": float((outcomes["probability"] * outcomes["p_buy_beats_rent"]).sum()),
        "mean": mean - float((outcomes["probability"] * outcomes["at_horizon_usd"]).sum()),
        "median": mixture_median(outcomes),
        "by_count": pd.DataFrame(by_count).set_index("k"),
        "breakeven_assessment": median if median > 0 else None,
        "folded_tail_probability": special["folded_tail_probability"],
    }


def delta_w_cdf(delta_w: dict[str, Any], x: np.ndarray, with_overlay: bool = False) -> np.ndarray:
    """P(ΔW <= x) (or ΔW_total with the special-assessment overlay) -- for plots and later stages."""
    cf: CornishFisher = delta_w["cornish_fisher"]
    overlay = delta_w["overlay"]
    if not with_overlay or not overlay["enabled"]:
        return cf.cdf(x)
    outcomes = overlay["outcomes"]
    return sum(p * cf.cdf(np.asarray(x) + a) for p, a in zip(outcomes["probability"], outcomes["at_horizon_usd"],
                                                              strict=True))
