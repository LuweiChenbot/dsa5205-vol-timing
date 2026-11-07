# Models and validation: implementation contract

Requested work: implement ridge models and walk-forward validation on a separate
branch, isolate the contribution of term-structure features, and provide usable
prediction files to Member 4. Preserve the existing data and feature stages.

## Frozen experiment

- Primary target: daily SPY minus BIL total return, from close(t+1) to close(t+2).
  Decisions occur after the volatility close on t; execution is next close.
- Daily rolling refits on exactly 756 fully realised training labels. At decision
  T, a training label must end on or before T. Use the full panel's trading
  calendar, including January 2026, for dates and targets.
- Fit the existing PastScaler on each training window only, including its 1%/99%
  clipping; never scale on validation or evaluation rows. Use ridge with an
  unpenalised intercept and alpha in [0.1, 1, 10, 100, 1000].
- Compare four nested sets: base (5), vix_level (9: base plus vix_log, vix_z,
  vix_chg5, vrp), vix (11: adds ts_ratio and ts_backw), full (14: interactions).
  Also forecast the mean of the same 756 training labels as a benchmark.
- Tune alpha separately per set using pooled daily walk-forward MSE over
  2017–2018, only on labels realised by 2018-12-31. Log missing warm-up and
  boundary-purged dates explicitly; never silently shorten training windows.
  Break exact MSE ties in favour of larger alpha. Freeze these alphas before
  generating 2019–2025 forecasts. No test-period optimisation of any settings.
- Full ridge is the predeclared primary research model; all models are exported
  and scored. Do not declare a trading winner from evaluation-period results.
  In addition to independently tuned models, report the vix_level versus vix
  comparison at a shared alpha (selected using vix_level validation MSE), so the
  term-structure comparison also holds regularisation fixed.
- Later refits may use earlier evaluation labels once realised. This is an
  adaptive rolling evaluation with frozen design, not one static fitted model.
- Report daily and yearly forecast errors, OOS R-squared against the rolling
  mean, and results separately on decline days. Report matched differences for
  term structure and interactions. Event-label diagnostics are secondary: the
  one-day return forecast is not a ten-day recovery probability.

## Delivery and verification

- One offline command builds all outputs from committed processed data. Save
  configuration, feature lists, input/source hashes, package versions, alpha
  selection, coverage, coefficients, and metrics for reproducibility.
- Member 4 gets long and wide evaluation predictions with explicit decision,
  execution, and label-end dates, decimal-return units, model IDs, train-window
  audit fields, and a validated Python loader. Future outcomes live in a separate
  file, not in the trading signal exports. Include validation predictions for
  designing downstream allocation rules without using evaluation outcomes.
- Fail early on invalid dates, duplicate/missing/nonfinite inputs, misaligned
  targets, insufficient validation history, and incomplete evaluation coverage.
- Test label maturity, future-data independence, selection-period boundaries,
  ridge numerics, comparison design, error metrics, and the export/loader contract.
  Run existing stage tests and a full real-data run; review before completion.
- Record the existing research-history limitation: earlier feature work examined
  2017–2025 outcomes (docs/report_features_sections.md). The new pipeline must be
  mechanically causal, but 2019–2025 is not a pristine, untouched research holdout.
- Do not implement portfolio sizing, costs, leverage, or strategy selection here;
  those belong to Member 4. Do not imply forecast performance proves profitability.
