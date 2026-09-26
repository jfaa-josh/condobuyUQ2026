"""Manual script: build forecast_models.general (CPI inflation) from real BLS data.

Pulls monthly CPI-U (all items, U.S. city average, not seasonally adjusted) from the BLS Public
Data API, saves it to data/raw/, then fits a continuous-time OU (mean-reverting) model to CPI's
annual log growth rate and saves it to models/<MODEL_NAME>/fit.json for reuse (see
condobuyuq2026.utils.ou_fitting.fit_annual_ou for the fitting method, and
.claude/implementation-plan.md's Earmarks for how this maps onto forecast_models.general's
mean/sd/monthly_change/phi). Also saves two sanity-check plots: the projected price-level term
structure (needs a starting price, p0) and the projected monthly growth-rate path (doesn't -- see
ou_fitting.project_monthly_rate for why these are different shapes).

Run with: uv run python run/manual/build_general_price_model.py

NOTE: this makes real calls against the live BLS API and counts against your daily quota (25
requests/day unregistered — no API key needed for a pull this size) — run it deliberately, not
as part of routine testing.
"""

from datetime import datetime

from condobuyuq2026.input_deck import get_report_quantiles
from condobuyuq2026.plotting.forecast_plots import plot_monthly_rate, plot_term_structure
from condobuyuq2026.raw_data import load_raw_csv, raw_csv_path, save_raw_csv, to_monthly_series
from condobuyuq2026.utils.manual_utils import fetch_bls_series, filter_monthly, model_fit_path, model_plots_dir
from condobuyuq2026.utils.ou_fitting import fit_annual_ou, project_monthly_rate, project_term_structure, save_fit

# =============================================================================
# MANUAL INPUTS
# =============================================================================
BLS_SERIES_ID = "CUUR0000SA0"      # CPI-U, all items, U.S. city average, not seasonally adjusted
START_YEAR = 1984                  # one year before the intended 1985 fit window -- the first
                                    # annual (12-month) change needs a prior year of data, see
                                    # ou_fitting.fit_annual_ou
END_YEAR = datetime.now().year
MODEL_NAME = "bls_cpi_all_nsa_1984_1_2026_8"   # saved to data/raw/<MODEL_NAME>.csv and models/<MODEL_NAME>/
PROJECTION_HORIZON_MONTHS = 120    # 10 years -- matches a plausible scenario.horizon_years max
OUTER_CI = (0.01, 0.99)            # wide reference band every plot shows (dashed), regardless of
                                    # report_quantiles -- see plotting.forecast_plots module docstring
# =============================================================================


if __name__ == "__main__":
    data_points = fetch_bls_series(BLS_SERIES_ID, START_YEAR, END_YEAR)
    monthly_points = filter_monthly(data_points)
    out_path = save_raw_csv(monthly_points, MODEL_NAME)
    print(f"Saved {len(monthly_points)} monthly observations to {out_path}")

    monthly_points = load_raw_csv(MODEL_NAME)
    print(f"Loaded {len(monthly_points)} monthly observations from {raw_csv_path(MODEL_NAME)}")

    cpi = to_monthly_series(monthly_points).dropna()
    fit = fit_annual_ou(cpi)
    print(
        f"Fitted OU model: i_inf={fit['i_inf']:.4f}/yr, tau={fit['tau']:.1f}mo, "
        f"phi_monthly={fit['phi_monthly']:.4f}, sig={fit['sig']:.4f}/yr, i0={fit['i0']:.4f}/yr"
    )

    fit_path = save_fit(fit, model_fit_path(MODEL_NAME))
    print(f"Saved fitted model to {fit_path}")

    # run.report_quantiles is [lo, ..., hi] (e.g. [0.1, 0.5, 0.9]) -- the outer two are the band
    # plotted below; the middle one(s) are implicit in the median line already plotted separately.
    report_quantiles = get_report_quantiles()
    projection_ci = (report_quantiles[0], report_quantiles[-1])

    term_structure_path = model_plots_dir(MODEL_NAME) / "term_structure.png"
    plot_term_structure(
        cpi,
        project_term_structure(fit, p0=cpi.iloc[-1], horizon_months=PROJECTION_HORIZON_MONTHS, ci=projection_ci),
        project_term_structure(fit, p0=cpi.iloc[-1], horizon_months=PROJECTION_HORIZON_MONTHS, ci=OUTER_CI),
        title=f"{MODEL_NAME} — fitted price-level projection ({int(projection_ci[0] * 100)}-{int(projection_ci[1] * 100)}% band)",
        save_path=term_structure_path,
    )
    print(f"Saved term-structure plot to {term_structure_path}")

    monthly_rate_path = model_plots_dir(MODEL_NAME) / "monthly_rate.png"
    plot_monthly_rate(
        project_monthly_rate(fit, horizon_months=PROJECTION_HORIZON_MONTHS, ci=projection_ci),
        project_monthly_rate(fit, horizon_months=PROJECTION_HORIZON_MONTHS, ci=OUTER_CI),
        title=f"{MODEL_NAME} — fitted monthly growth-rate projection ({int(projection_ci[0] * 100)}-{int(projection_ci[1] * 100)}% band)",
        save_path=monthly_rate_path,
    )
    print(f"Saved monthly-rate plot to {monthly_rate_path}")
