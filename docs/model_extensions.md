# Exploratory modelling extensions

These additions address the gap between a one-day return forecast and the original
ten-day recovery/continuation question. They were specified **after** the original
evaluation results had been seen. They are exploratory, even though penalties are
chosen using design-period data only. Keep the original one-day results and the
research-history disclosure in the final report.

Run from the repository root with the existing pinned requirements:

```bash
python -m src.models.extensions.pipeline
python -m unittest tests.test_model_extensions -v
```

No downloads or new packages are required. The build reads the committed panel,
features and event trigger flags. It does not rewrite the original events,
features, targets, one-day forecasts or performance tables. Shared ridge code now
supports a configurable horizon; its default remains one day. Regenerating the
original pipeline updates source provenance while reproducing its numerical
outputs unchanged.

The detailed contract is `docs/model_extensions_spec.md`. Build selections and
coverage are in `reports/models/extensions/summary.md`. These extensions generate
signals and training diagnostics only, **not evaluation-performance metrics,
significance tests, allocation rules or backtests**.

The subsequently requested frozen evaluation is a separate command:
`python -m src.analysis.recovery_evaluation`. It scores the saved 2019–2025
probabilities without invoking fitting or selection. See
`reports/recovery_evaluation/report.md` and `docs/recovery_evaluation_protocol.md`.
It does not change this extension's settings, predictions or build artifacts.

Additional frozen diagnostics run with
`python -m src.analysis.recovery_diagnostics`. See
`reports/recovery_diagnostics/report.md` for class-specific discrimination,
probability distributions, confusion matrices, calibration curves, validation
ranking sensitivity and saved coefficient contrasts. No models or calibrators
are fitted, and no settings or forecasts are changed. Omitting an episode from
candidate scoring is a sensitivity check, not refitted cross-validation.

## Direct recovery classifier

Each decline-trigger day receives the existing ten-session barrier label, using
entry at the next session close and SPY total-return prices. The barriers are
plus/minus `0.75 * dec_vol * sqrt(10)`, fixed using decision-day information:

- `+1`: the upper barrier is reached before the lower barrier (recovery).
- `-1`: the lower barrier is reached first (continuation).
- `0`: neither barrier is reached within ten holding sessions (unresolved).

The new dense labels include **every trigger day**. The original non-overlapping
event file remains intact. At most one close-based barrier can be reached on a
given day because the widths are positive. Intraday barrier hits are outside the
existing close-based definition.

The model is L2-regularised multinomial logistic regression, `solver='lbfgs'`,
`max_iter=2000`, `tol=1e-8`. A convergence warning fails the build rather than
silently exporting an unconverged model. There are three deliberately small
feature sets:

| Model ID | Features |
|---|---|
| `class_prior` | Episode-weighted training class frequencies, with one pseudocount per class |
| `logit_price` | `dec_dd`, `dec_mom` |
| `logit_vix_level` | Price + `vix_log` |
| `logit_term_structure` | VIX level + `ts_ratio` |
| `logit_term_structure_matched` | Same term-structure inputs, but C fixed to the VIX-level model's C |

The term-structure model is the designated direct-recovery model. No claim that it
is a test-period winner is made. On decline-only rows, `dec_cond` is always 1 and
`x_cond_ts` is a shifted copy of `ts_ratio`; neither is included. This comparison
adds one term-structure feature at a time instead of fitting the original 14
predictors to a small conditional sample.

Training uses an **expanding history of matured decline labels** from the saved
2014 sample. A label dated s becomes available at close(s+11), even if a barrier
was reached earlier. This deliberately conservative rule applies equally to all
three classes. All feature clipping/scaling uses only those training rows.

Episodes are grouped with the existing causal start rule: the first trigger
starts a group, and triggers in the next ten calendar trading sessions share
that group. Groups use full-panel trigger history, including warm-up dates. In
each refit, a group represented among matured training rows receives total sample
weight 1; its individual rows receive `1 / group_size`. Counts and weights are
recomputed from the matured rows only, so later triggers cannot change an earlier
fit. The scaler is the existing unweighted, columnwise PastScaler fitted on that
same training sample; episode weights apply to the likelihood objective.

This weighting limits concentration in long episodes. **It does not remove all
dependence between overlapping target windows**, including windows in neighbouring
groups, and it does not create more independent market episodes. Probability
estimates reflect an episode-weighted fitting objective, not a guarantee of
calibration under the frequency distribution of all daily triggers.

The C grid is `[0.01, 0.1, 1, 10]`; smaller C means stronger regularisation. Each
feature set selects C by episode-weighted three-class log loss on 2017–2018
validation decisions whose entire label is complete by 2018-12-31. Exact ties
prefer smaller C. The matched term-structure model takes the VIX-level model's C.
Selected settings stay fixed during subsequent expanding refits. These validation
predictions are design/tuning data, not an unbiased additional test sample.

If fewer than 12 training episodes are represented or any class is absent, every
classifier exports the smoothed training prior and `fit_status='prior_fallback'`.
With no history the prior is uniform. The prior itself has `fit_status='prior'`.
This is explicit fallback behaviour, not a claim that a logistic model was fitted.

In the design sample, 108 trigger-day labels have complete horizons by the cutoff;
there are 34 original non-overlapping starts with complete horizons. Overlap is
why these counts must not be treated as 108 independent training events.

## Ten-day ridge

This uses the existing compounded SPY-minus-BIL target over close(s+1) through
close(s+11). It retains exactly 756 matured training observations and the existing
base, VIX-level, term-structure and interaction feature sets, with a rolling-mean
benchmark and matched-alpha term-structure model. The penalty grid is
`[0.1, 1, 10, 100, 1000, 10000, 100000]`, selected by design-period MSE.

The ten-day forecast is a **cumulative excess-return estimate**. It is not a daily
return, a recovery probability, or a direct model of which barrier is hit first.
Consecutive ten-day targets overlap. Their errors and resulting holding periods
must not be treated as independent daily observations or daily portfolio returns.

The full-window requirement means ten early validation dates lack enough matured
training observations, and eleven late validation outcomes cross the design
boundary. There are 481 usable validation dates. The committed panel ends on
2026-01-15 and lacks the complete ten-day horizon for the 2025-12-31 decision.
The pipeline therefore exports 1,759 evaluation decisions through 2025-12-30 and
records the final decision as `incomplete_outcome_calendar` in coverage. It does
not guess a missing calendar date, fill an outcome, shorten the horizon, or
download revised prices. Classifier coverage is checked separately on trigger
dates; a missing non-trigger forecast is expected, not an error.

## One-day penalty sensitivity

The larger alpha grid is also applied to the **one-day design sample only**.
Candidate predictions, selected settings and coverage are saved under
`reports/models/extensions/ridge_h1_design_sensitivity/`. No new one-day
evaluation forecasts are generated from these penalties, and the original
one-day choices remain unchanged. Endpoint selections remain endpoint selections,
not proof of a global optimum; the program never expands grids automatically.

## Member 4 handoff

```python
from src.models.extensions.handoff import load_recovery_signal, load_ten_day_signal

recovery = load_recovery_signal("logit_term_structure")
# Index: decision_date, on decline-trigger dates only.
# p_recovery, p_continuation, p_unresolved sum to one.

ten_day = load_ten_day_signal("ridge_full")
# prediction: decimal ten-day SPY-minus-BIL cumulative total return.

recovery_design = load_recovery_signal(
    "logit_term_structure",
    "data/processed/models/extensions/recovery/validation_predictions.csv",
)
```

The default folders are `data/processed/models/extensions/recovery/` and
`data/processed/models/extensions/ridge_h10/`. Each contains:

- `predictions.csv` and `predictions_wide.csv` for evaluation-period decisions;
- `validation_predictions.csv` and its wide version for design-period decisions;
- `outcomes.csv` and `validation_outcomes.csv`, separate from trading signals;
- `manifest.json` with settings, sources, inputs, output hashes and version data.

Keep each manifest with its files. Loaders validate dates, schemas, matched models,
penalties, probabilities and checksums when the adjacent manifest is available.
Without a manifest only structural checks are possible. Unused coefficients and
prior penalties are blank; no-history fallback training dates are blank as well.

Recovery signals include the three probabilities, feature set, frozen C, fit
status, represented episode count, matured row/class counts, summed training
weight and training-date audit fields. `episode_start_date` is known at decision
time; to identify original non-overlapping starts, compare it with the index.
Future labels remain in the separate outcome files. Wide probability column names
have the form `logit_term_structure__p_recovery`.

**No row on a non-decline date means no recovery forecast.** Do not replace this
with a zero probability or forward-fill an old probability automatically. Member 4
must explicitly decide what allocation to use when no decline signal is present.
Probability estimates are not portfolio weights. Likewise, do not send a ten-day
cumulative prediction to a backtest that interprets it as a daily realised return.
The original `load_signal()` defaults to horizon 1 and rejects horizon-10 files.

Per-family build reports contain penalty selection, validation candidates,
coverage and evaluation-fit coefficients. Coefficients are in scaled-feature
units; the classifier has one row per class/model/decision. No coefficients are
fabricated for prior fallbacks. The summary reports model-build counts and design
choices, not evaluation performance.

## Verification

```bash
python -m unittest tests.test_models tests.test_model_extensions -v
python -m unittest tests.test_recovery_evaluation tests.test_recovery_diagnostics -v
python -m tests.test_pipeline
python -m tests.test_features
python -m tests.test_events
python -m tests.test_targets
```

The extension checks cover exact horizon compounding, eleven-session maturity,
future-data independence, synthetic recovery/continuation/unresolved paths,
training-only episode weights, probability fallback, convergence errors,
selection boundaries, matched penalties, output reload and file tampering.
Real-data runs also verify that the original one-day numerical artifacts remain
byte-identical and that extension artifacts reproduce deterministically on the
pinned environment. No claim of predictive improvement follows from those checks.
