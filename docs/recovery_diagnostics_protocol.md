# Frozen recovery-model diagnostics

Requested after the frozen 2019–2025 evaluation. Add three diagnostics without
retuning: explain continuation argmax failure, inspect probability calibration,
and assess validation-selection and coefficient stability. These are post hoc,
descriptive analyses. They do not supersede the original primary log-loss result.

## Immutable inputs

Use the recovery bundle from commit `4d19649`, anchored by the same manifest hash
as the existing frozen evaluation. Additionally verify every saved recovery
report against that manifest, including candidate predictions, the original
penalty-selection table and evaluation coefficients. Validate candidate dates,
completed design outcomes, grid coverage and selected-candidate agreement with
the saved validation forecasts. Never fit, transform features, invoke the model
selector, change probabilities, choose new thresholds or write trading signals.
Verify input hashes again after all analysis and output generation.

## 1. Class discrimination and probability distributions

On all 175 evaluation triggers, retain all three outcomes and both baselines.
Give each represented episode total weight one, as in the primary evaluation.
For each class report one-versus-rest ROC AUC (undefined for single-class truth),
binary Brier contribution, average forecast, observed frequency and argmax recall.
For each forecast class and actual class report mean, min, max and weighted
10th/50th/90th percentiles; quantiles use the inverse weighted empirical CDF.
Also summarize each class probability minus the largest competing probability.

Export three-by-three confusion tables with unweighted expected counts,
episode-weighted mass and within-actual-class recall. Exact probability ties
split prediction mass evenly, matching the frozen evaluation. Missing actual
classes have undefined recall, never fabricated zero. A class never winning
argmax does not imply its forecast probabilities are zero or have no ranking value.

## 2. Calibration

For each class and model use the same fixed bins [0,.2), [.2,.4), [.4,.6),
[.6,.8), [.8,1]. Include empty bins explicitly with zero counts/weight and
undefined means. In populated bins compare the episode-weighted mean forecast
with the weighted observed class frequency. Weights are computed on the whole
model sample and are NOT recomputed within a bin or actual class. Include trigger
counts, represented-episode counts and weight mass. Episode counts across bins
need not add to 53 because an episode may span bins.

Summarize signed overall probability bias (forecast minus observed) and weighted
mean absolute bin gap (ECE). ECE is bin-dependent and descriptive, not a proper
scoring rule or a model-selection criterion. Plot reliability curves with a
diagonal reference, preserving the empty-bin gaps. No calibrator is fitted.
Sparse bins and dependent outcomes mean these point estimates are uncertain;
do not infer significant miscalibration or superiority from them.

## 3. Validation selection and coefficient stability

Using only the saved candidates and completed 2017–2018 outcomes, reproduce the
original episode-weighted losses and within-feature-set C rankings. For each of
the ten validation episodes omit that episode from SCORING, then recompute all
candidate losses/ranks with equal remaining-episode weights. Exact ties prefer
smaller C. Record diagnostic winners, winner counts and how often the originally
selected C remains first. The original selected C values stay frozen. Include the
originally selected models' loss differences versus the prior on each retained
sample. No evaluation outcomes participate in this diagnostic.

This is a leave-one-episode-out scoring sensitivity check, NOT refitted
cross-validation or an independent performance estimate: the omitted episode
may already have entered training for later saved forecasts. Neighbouring episodes
can also have overlapping label windows. Do not reinterpret the diagnostic
winners as new selections or use them to replace any forecasts.

Read the saved 2019–2025 coefficients at the frozen C values. Validate exact
model/date/class coverage against fitted forecasts. Export and summarize
continuation-minus-recovery and unresolved-minus-recovery coefficient contrasts,
including intercepts, first/last/min/max, positive/negative/near-zero counts and
nonzero sign changes over successive saved refits (zero tolerance 1e-10).
Contrasts describe pairwise log odds and avoid interpreting a single softmax
coefficient in isolation. Summaries are unweighted descriptions of successive
fitted states, not independent estimates or causal effects. Scalers and clipping
limits change with the expanding training history, so magnitudes across refits
are only approximately comparable in scaled-feature units. Original validation
coefficients were not saved; do not fabricate or refit them.

## Outputs and verification

`python -m src.analysis.recovery_diagnostics` writes a separate
`reports/recovery_diagnostics/` folder containing machine-readable tables,
calibration figure, readable report and manifest with input/source/output hashes.
Keep previous model/evaluation artifacts unchanged. Test weighted calculations,
bin edges and empty bins, ties/missing classes, ranking sensitivity, coefficient
contrasts and sign changes, tamper rejection, and an integration run that fails
if model fitting, scaling or penalty selection is attempted. Reproduce the
artifacts and run the existing relevant tests before updating PR #1.

References: [probability calibration](https://scikit-learn.org/1.7/modules/calibration.html)
and [classification decisions](https://scikit-learn.org/1.7/modules/classification_threshold.html).
