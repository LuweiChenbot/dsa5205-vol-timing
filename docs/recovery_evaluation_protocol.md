# Frozen recovery-classifier evaluation

Requested after the modelling extension: evaluate the new classifier on 2019–2025,
compare with baselines, and assess whether the validation results hold up without
retuning anything. This separately authorises evaluation; the extension build
itself remains a prediction-only pipeline. No portfolio strategy is evaluated.

## Inputs and unchanged choices

- Read the committed recovery predictions, outcomes and manifest at the existing
  snapshot. Verify their hashes before scoring. Never call fitting or selection.
  Pin the manifest's SHA-256 to commit `4d19649` so regenerating a different bundle
  cannot silently replace the frozen input, even if its internal hashes agree.
- Frozen C values: price=10, VIX level=1, term structure=1, matched term structure=1.
  Preserve features, labels, episode grouping, timing and all prediction files.
- The saved predictions already reflect causal expanding refits. Evaluation here
  only scores those saved probabilities; it does not recompute any model.
- Primary population: every saved decline-trigger decision in 2019–2025. Verify
  full coverage against the original event flags and full panel trading calendar.
- Historical-prior baseline: the saved, training-only, episode-weighted smoothed
  class probabilities. Additional baseline: uniform probabilities (1/3 each).
  No baseline estimates its probabilities from evaluation outcomes.

## Measures fixed before scoring

- Primary metric: three-class natural-log loss, with one total weight per represented
  episode (mean of each episode's mean per-trigger loss), matching validation.
- Secondary metrics: multiclass Brier score (sum across all three classes, range
  0–2), argmax accuracy, macro recall, class recalls, and recovery-versus-rest AUC.
  Retain unresolved outcomes in all these three-class/one-versus-rest metrics.
- For tied maximum probabilities, accuracy/recall use the expected result of
  uniform random tie-breaking; uniform-baseline accuracy is therefore exactly 1/3.
  This is a scoring convention, not a trading rule or a fitted threshold.
- Also report unweighted trigger-level results, decision-year results, and the
  original non-overlapping event-start subset. These are descriptive checks, not
  alternative criteria for selecting a model. Compare the saved design results
  using the same primary metric, explicitly identifying their tuning optimism.
- Report paired differences in episode-mean log loss: each learned model against
  both baselines, plus term structure versus VIX level and price-only. Negative
  differences favour the named model. The matched model remains in all tables;
  identical probabilities do not constitute separate corroborating evidence.
- Approximate 95% percentile intervals use a circular moving-block bootstrap of
  ordered episode losses: block length=5 episodes, 10,000 replicates, seed=5205.
  Use identical sampled episode indices for paired models. No block-length tuning,
  row-wise independence assumption, model refitting or multiplicity-adjusted claim.
  These intervals address some serial dependence, not prior research exposure,
  model selection uncertainty, nonstationarity, or every possible dependence.

## Deliverables

- One scoring command writes metrics, paired intervals, class/episode counts,
  per-row scored forecasts, an input/output audit manifest, and a readable report.
- Verify input bytes remain unchanged after scoring. Fit/scaling/selection code
  is never invoked. Add tests for metric conventions, alignment, weighting,
  bootstrap pairing, bad inputs and the no-refit property.
- Preserve the disclosure that 2019–2025 was used during earlier exploration and
  these models were proposed after earlier evaluation results were seen. Frozen
  parameters do not make this a pristine research holdout.
- Report the actual result, including failure to beat the baselines; do not
  reverse class probabilities, change thresholds, choose new penalties, or switch
  the primary model in response to this evaluation.

Metric reference: https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.log_loss.html

Block-resampling convention: https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html
(implemented directly with NumPy; no additional dependency).
