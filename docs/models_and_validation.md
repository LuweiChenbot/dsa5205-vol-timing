# Models, validation and Member 4 handoff

Run from the repository root after installing `requirements.txt`:

```bash
python -m src.models.pipeline
python -m unittest tests.test_models -v
```

The pipeline is offline: it uses the committed panel, features and events, with
Parquet preferred over CSV, and does not download or rewrite those inputs. It
writes signals to `data/processed/models/` and analysis to `reports/models/`.
Settings are frozen in `src/models/config.py`; the CLI deliberately offers no
test-period tuning options. `VT_WORK_DIR` redirects input and output directories
as in the data stage. See `models_validation_spec.md` for the experiment contract.

## Estimator and comparisons

The target is `fwd_spy_xs`: **SPY total return minus BIL total return from
close(t+1) to close(t+2)**. Values are decimal returns: `0.001` is 0.1 percentage
points, or 10 basis points, of expected excess return. This is regression on all
daily observations, not a classifier trained only on decline episodes.

Ridge minimises `sum((y - intercept - X beta)^2) + alpha * sum(beta^2)`.
The intercept is unpenalised; the target is not scaled. The fixed alpha grid is
`[0.1, 1, 10, 100, 1000]`, under this sum-of-squares convention (not mean loss).
The implementation uses scikit-learn Ridge with its deterministic Cholesky
solver. Each training window fits the existing `PastScaler`: continuous features
are clipped at training 1%/99% quantiles, then all columns are standardised using
training means and population standard deviations. Binary and interaction
columns are not clipped. Constant columns use scale 1. No full-sample scaler is
fitted. Scaling all 14 columns once per window gives exactly the same columnwise
transform as fitting each subset separately.

| Model ID | Inputs | Purpose |
|---|---|---|
| `historical_mean` | Mean of the same 756 matured labels | Forecast benchmark |
| `ridge_base` | 5 price/decline features | Price-information baseline |
| `ridge_vix_level` | Base + `vix_log`, `vix_z`, `vix_chg5`, `vrp` (9) | VIX-level information, without VIX3M |
| `ridge_vix` | VIX level + `ts_ratio`, `ts_backw` (11) | Incremental term structure |
| `ridge_full` | VIX + 3 existing interaction features (14) | Predeclared primary research model |
| `ridge_vix_matched` | Same 11 as `ridge_vix`; alpha from `ridge_vix_level` | Term structure at identical regularisation |

`vix_level` is a model-stage subset; it does not modify Member 2's saved feature
sets. The column named `vrp` is an implied-minus-realised **volatility gap proxy**,
not a direct measurement of the variance risk premium.

Compare `ridge_vix_level` with `ridge_vix` for the independently tuned feature-set
comparison, and with `ridge_vix_matched` to hold alpha fixed as well. The only
added inputs in the latter are the two term-structure features. Adding features
can change other fitted coefficients: this is a predictive ablation, not causal
identification. If both models choose the same alpha, the two comparisons are
identical; this is expected. `ridge_full` versus `ridge_vix` tests interactions,
with their penalties selected independently on validation data.

## Walk-forward protocol

1. On decision date T, use exactly the most recent **756 fully realised labels**.
   Features dated s enter training only when their outcome date `s+2` is at or
   before T. Today's close is usable because the decision is after 16:15 ET.
   For h=1, the latest training decision is T−2 trading sessions. The row for T−1
   remains unavailable even though its features are already known.
2. Fit the training-only scaler and ridge, then predict T. Refit daily; neither
   random K-fold CV nor full-sample fitting is used. Calendar positions come from
   the full SPY panel, not calendar-day or generic weekday arithmetic.
3. Validate on decisions in **2017–2018**, retaining only outcomes completed by
   **2018-12-31**. Pool squared errors across eligible dates, choose one alpha for
   each set, and prefer larger alpha only for exact MSE ties. The historical mean
   uses the identical training window and prediction dates.
4. Freeze alphas and all other settings, then forecast **2019–2025**. Later daily
   refits can use earlier evaluation labels once they mature; evaluation outcomes
   never choose the alpha, feature list, window length or primary model.

In this snapshot, 2017-01-03 has only 755 matured training labels and is recorded
as `insufficient_history`. Validation starts on 2017-01-04. Decisions on
2018-12-28 and 2018-12-31 have outcomes in 2019 and are recorded as
`outcome_after_design_end`. This leaves **499 validation dates**. The evaluation
contains **1,760 decisions per model**. The final decision, 2025-12-31, executes
2026-01-02 and earns its return through 2026-01-05.

The output is a daily rolling evaluation with frozen design, rather than one
static model fitted in 2018. Validation forecast files use alphas chosen using
the entire validation period. They are suitable for design work, but their
selected-model performance is not an unbiased second evaluation.

**Research-history limitation:** Member 2's report discloses earlier parameter
exploration and rolling-ridge probes involving 2017–2025 outcomes
(`docs/report_features_sections.md`, “Design log”). The new implementation
enforces information timing, but cannot make 2019–2025 a pristine research
holdout retroactively. Preserve this disclosure in the final report. Do not
retune settings after seeing the generated evaluation results.

## Output files

| File | Contents |
|---|---|
| `data/processed/models/predictions.csv` | Long evaluation signals, six rows per decision |
| `data/processed/models/predictions_wide.csv` | One decision per row, six prediction columns plus dates and decline flag |
| `data/processed/models/validation_predictions.csv` | Selected configurations on eligible design/validation dates |
| `data/processed/models/validation_predictions_wide.csv` | Wide version of the above |
| `data/processed/models/outcomes.csv` | Realised SPY, BIL and excess returns, separately from signals |
| `data/processed/models/validation_outcomes.csv` | Corresponding design-period outcomes |
| `data/processed/models/manifest.json` | Settings, feature lists, selections, versions, SHA-256 input/source/output hashes |
| `reports/models/alpha_selection.csv` | Every candidate's validation MSE/MAE, counts and selection flag |
| `reports/models/validation_candidates.csv` | All validation grid predictions and rolling-mean forecasts |
| `reports/models/coverage.csv` | Every requested date, period, available history and status |
| `reports/models/forecast_metrics.csv` | Pooled and decision-year metrics, all/decline/non-decline |
| `reports/models/feature_comparisons.csv` | Paired MSE/MAE differences between the nested sets |
| `reports/models/event_summary.csv` | Event counts and resolved-event ranking AUC |
| `reports/models/event_predictions.csv` | Event-day forecasts joined to future barrier labels; analysis only |
| `reports/models/coefficients.csv` | Daily evaluation intercepts and standardised-feature coefficients; blank for unused features |
| `reports/models/summary.md` | Generated report tables and limitations |
| `reports/models/validation_mse.png`, `yearly_oos_r2.png` | Validation curve and annual forecast comparison |

The long signal schema includes `decision_date`, `execution_date`,
`label_end_date`, `model_id`, `feature_set`, `horizon`, `alpha`, `prediction`,
`decline`, `n_train`, `train_start_date`, `train_end_date`, and
`train_last_label_end_date`. Training start/end refer to **training decision
dates**; the last-label field records when the latest training outcome became
available. `alpha` is blank for the historical mean. Date columns are exchange
session dates, not UTC timestamps. `decline` is known at the decision time.

Outputs are staged before publication and the manifest is replaced last. Keep
the signal files and manifest together: the Python loader checks their hash,
in addition to schema, finite predictions, frozen alphas, matched model dates,
and timing. Without a manifest, a standalone file receives structural validation
only. No saved fitted model is needed to replay these historical forecasts; the
daily fits are reproducible from the pinned snapshot and settings. Small floating
point differences can occur across numerical-library versions or platforms.

## Member 4: loading and aligning the signals

```python
from src.models.handoff import load_signal, load_predictions, wide_predictions

signal = load_signal("ridge_full")   # default: evaluation file, indexed by decision date
# signal[["prediction", "execution_date", "label_end_date", "decline"]]

all_models = wide_predictions(load_predictions())
# Compare ridge_vix_level, ridge_vix, and ridge_vix_matched with the same allocation rule.

design = load_signal(
    "ridge_full", "data/processed/models/validation_predictions.csv"
)
```

Predictions are neither probabilities nor weights. Member 4 must define an
allocation rule on design data and freeze it, together with constraints, costs,
rebalance and initial-position conventions, before the final comparison. Apply
the same rule and execution convention across models to isolate signal quality.
Positive excess forecasts favour SPY relative to BIL; that sign alone does not
establish an optimal allocation or profitability after costs.

For an explicit date join, after creating a decision-indexed Series `weights`:

```python
from src.data.build import load_panel

panel = load_panel()  # retain the January 2026 buffer
trades = signal[["execution_date", "label_end_date"]].join(weights.rename("spy_weight"))
trades = trades.merge(
    panel[["spy_tr_cc", "bil_tr_cc"]], left_on="label_end_date",
    right_index=True, how="left", validate="one_to_one",
)
assert trades[["spy_weight", "spy_tr_cc", "bil_tr_cc"]].notna().all().all()
trades["gross_return"] = (
    trades.spy_weight * trades.spy_tr_cc + (1 - trades.spy_weight) * trades.bil_tr_cc
)
```

This shows **gross return alignment only**. Charge transaction costs on execution
dates under the chosen backtest accounting, and retain each model's first entry
and final position convention. If using the existing `src.metrics.apply_weights`,
pass weights indexed by **decision date** without another shift; that helper
already applies the next-close lag. Its simple turnover-based cost calculation
does not charge the initial move from an unspecified starting allocation and
does not model drifted asset weights. Member 4 should explicitly handle those
conventions rather than interpreting its default output as a complete cost model.
Use identical actual holding dates for SPY/BIL benchmarks and model comparisons.

## Interpreting the diagnostics

- `mse`, `rmse`, `mae` describe return forecast errors; they are not P&L metrics.
- `oos_r2 = 1 - sum(model_errors²) / sum(rolling_mean_errors²)`, calculated on
  identical dates. Negative values mean worse than the historical-mean forecast.
  The benchmark has R²=0. This differs from scikit-learn's default `score()`.
- `direction_accuracy` compares the sign of excess prediction and excess return;
  zero is treated as its own sign. Yearly rows group by decision year, including
  any trailing next-year realised returns for those decisions.
- Paired `mse_difference` and `mse_change_pct` subtract the smaller model's error
  from the larger model's error; negative favours the larger set. These tables
  are descriptive and contain no significance claim or confidence interval.
- Event AUC compares +1 recoveries with −1 continuations; the 0/unresolved class
  remains in the event export and is counted, but is excluded from AUC. AUC is
  undefined (blank) if only one resolved class is present. The daily forecast
  targets a different horizon and excess return, while event labels refer to
  ten-day SPY barrier paths. Treat this as a secondary ranking diagnostic.
- Ridge coefficients do not come with ordinary OLS standard errors or p-values.
  A separate predeclared small OLS/HAC analysis would be a different study.
- Alpha selection is restricted to the declared grid. Choosing an endpoint does
  not establish a global optimum; the grid must not be expanded in response to
  the evaluation results.

The implementation deliberately covers the primary one-day ridge experiment.
Multi-horizon models, classifiers, portfolio sizing and trading performance are
not inferred from these results. A negative result is a valid project finding.

## Checks and references

Run the existing stage checks and the new model tests:

```bash
python -m tests.test_pipeline
python -m tests.test_features
python -m tests.test_events
python -m tests.test_targets
python -m unittest tests.test_models -v
```

The model tests cover future-data shocks, exact training maturity, warm-up and
split-boundary exclusions, fixed-penalty feature comparisons, analytic ridge
agreement, invalid inputs, metrics, export reload, hash tampering and the
Member 4 execution alignment. A full snapshot run additionally verifies coverage.

- Ridge objective and API: https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.Ridge.html
- Time-ordered validation and gaps: https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Local data timing and total-return conventions: `DATA.md`.
- Existing research-history disclosure: `docs/report_features_sections.md`.

The implementation uses explicit label-end dates rather than blindly applying a
fixed `TimeSeriesSplit.gap`: a close realised on T is available to an after-close
decision on T, while validation selection additionally excludes outcomes after
the design cutoff.
