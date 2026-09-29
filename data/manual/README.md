# data/manual/

Your own hand-collected data (monthly rented days, ADR, utilities, etc.) — not something an API
can re-fetch if it's lost, so this folder is tracked in git, unlike `data/raw/` (scraped/external
data, gitignored, and cheap to regenerate by re-running a scraper).

- `166_Argentine_Ct_Unit1433_keystone_mkt_value.csv`: the county assessor's own ACTUAL past
  reassessment values for this specific property (effective_year, the averaging window it was based
  on, and the dollar amount). Named directly by `input_deck.yaml`'s own
  `carrying_costs.property_tax.market_value.validation_data_source` field -- not hardcoded in
  `utils.prior_utils.py`, so renaming this file (or pointing a different deck at a different one)
  only ever needs a deck edit, not a code change. Combined with a Denver-metro `home_price` index
  average over each record's own averaging window into a LINEAR FIT (slope + intercept, least
  squares) that calibrates Denver's own index units into actual Keystone dollars for the forward
  model (`utils.prior_utils.fit_market_value_calibration`) -- see that function's docstring for why
  this is a bounded, low-risk use of real data (2 parameters, however many points exist), not
  overfitting.

  Currently one row (2023-2024 cycle, $523,800). **Do NOT add a row for the CURRENT cycle** (the one
  `market_value.initial_price` reflects) -- `utils.prior_utils.load_market_value_history`
  synthesizes that one automatically from `initial_price`/`first_reassessment_year`/
  `reassessment_frequency_years` directly, specifically so it can never drift out of sync with
  `initial_price` the way a hand-entered duplicate row could. Add a row here only for a cycle that's
  fully in the PAST relative to the current one.

- `keystone_co_monthly_temperature_averages.csv`: 12 rows (month, avg_temp_f) -- each month's average
  of that month's own average daily high and average daily low, for Keystone, CO (source: Weather
  Spark's climate normals, https://weatherspark.com/y/3543/, fetched 2026-09-29 -- a representative
  external reference, not a live/scriptable API, so this is a one-time manual pull like the assessor
  data above, not something `raw_data.py`'s own fetch scripts re-derive). Named by
  `carrying_costs.utilities.seasonal_data_source`. Used by `utils.prior_utils.
  fit_temperature_seasonal_shape` to fit a 3-parameter sinusoid (`a + b*cos(2*pi*m/12) +
  c*sin(2*pi*m/12)`) to the annual temperature cycle, which `build_utilities_prior_model` then
  inverts (cold month = high utility cost) and rescales so utilities.
  `initial_annual_average_monthly_cost` is the resulting seasonal shape's own annual average -- see
  that function's docstring for the exact floor/rescale formula.
