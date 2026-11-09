# Frozen recovery classifier: 2019–2025 evaluation

Evaluated saved probabilities from model commit `4d19649`; no training, scaling,
penalty selection, threshold changes or probability reversal was performed.
Frozen C: {'price': 10.0, 'vix_level': 1.0, 'term_structure': 1.0}. The term-structure model remains the
designated direct-recovery model; this report does not replace it with a test winner.

Earlier feature research examined 2017–2025 outcomes, as disclosed in docs/report_features_sections.md. The present forecasts obey information timing, but 2019–2025 is not a pristine, untouched research holdout.

The classifier extension was also designed after the original one-day evaluation
was seen. This is a frozen-parameter retrospective evaluation, not a pristine
research holdout. It does not assess strategy profitability.

## Primary comparison: all triggers, equal total weight per episode

175 trigger decisions across 53 represented episodes.
Class counts: 54 continuation, 29
unresolved, 92 recovery. Unresolved labels remain a separate class.

| model_id                     |   log_loss |    brier |   accuracy |   macro_recall |   recovery_vs_rest_auc |
|:-----------------------------|-----------:|---------:|-----------:|---------------:|-----------------------:|
| class_prior                  |   0.977701 | 0.578399 |   0.587099 |       0.333333 |               0.444247 |
| logit_price                  |   0.996040 | 0.583840 |   0.584554 |       0.343518 |               0.564427 |
| logit_term_structure         |   1.017874 | 0.604408 |   0.547866 |       0.320653 |               0.519818 |
| logit_term_structure_matched |   1.017874 | 0.604408 |   0.547866 |       0.320653 |               0.519818 |
| logit_vix_level              |   1.017425 | 0.598616 |   0.554680 |       0.318415 |               0.500020 |
| uniform                      |   1.098612 | 0.666667 |   0.333333 |       0.333333 |               0.500000 |

Lower log loss and Brier score are better. Brier sums all three squared probability
errors (0–2 convention). Accuracy uses argmax and expected random tie-breaking;
uniform-baseline accuracy is 1/3. Macro recall averages all three class recalls
and is blank where a class has no observations. AUC ranks recovery against both
other classes combined, with the same episode weights. It is not the older
one-day model's recovery-versus-continuation diagnostic.

## Validation comparison (tuning data)

| model_id                     |   n |   n_episodes |   log_loss |
|:-----------------------------|----:|-------------:|-----------:|
| class_prior                  |  32 |           10 |   1.170779 |
| logit_price                  |  32 |           10 |   1.015801 |
| logit_term_structure         |  32 |           10 |   1.051211 |
| logit_term_structure_matched |  32 |           10 |   1.051211 |
| logit_vix_level              |  32 |           10 |   1.063429 |
| uniform                      |  32 |           10 |   1.098612 |

Validation selected C on these same observations and is therefore optimistic.
Changes in loss levels across periods also reflect different class mixtures and
market conditions; compare each model with its contemporaneous baselines.

## Direct answer for the designated recovery model

Term-structure classifier minus historical prior: +0.040173
log loss (+4.109%); approximate paired interval
[-0.027368, 0.118418]. A positive difference is
worse. This comparison, rather than improvement over equal probabilities alone,
is the main check of whether the validation advantage carries over.

The following expected argmax prediction counts expose class collapse. They use
unweighted trigger counts; only tied probability maxima produce fractional counts.

| model_id                     |   expected_predictions_continuation |   expected_predictions_unresolved |   expected_predictions_recovery |
|:-----------------------------|------------------------------------:|----------------------------------:|--------------------------------:|
| class_prior                  |                               0.000 |                             0.000 |                         175.000 |
| logit_price                  |                               0.000 |                             9.000 |                         166.000 |
| logit_term_structure         |                               0.000 |                            11.000 |                         164.000 |
| logit_term_structure_matched |                               0.000 |                            11.000 |                         164.000 |
| logit_vix_level              |                               0.000 |                             5.000 |                         170.000 |
| uniform                      |                              58.333 |                            58.333 |                          58.333 |

Counts describe fixed argmax classification, not portfolio decisions. No threshold
or class-weight changes were made to improve these numbers.

## Paired uncertainty on the primary loss

| model_id                     | comparator      |   loss_difference |   loss_change_pct |   ci95_lower |   ci95_upper |
|:-----------------------------|:----------------|------------------:|------------------:|-------------:|-------------:|
| logit_price                  | class_prior     |          0.018339 |          1.875711 |    -0.030253 |     0.066398 |
| logit_price                  | uniform         |         -0.102572 |         -9.336504 |    -0.234487 |     0.033605 |
| logit_vix_level              | class_prior     |          0.039724 |          4.062992 |    -0.025427 |     0.118645 |
| logit_vix_level              | uniform         |         -0.081187 |         -7.389950 |    -0.228372 |     0.077574 |
| logit_term_structure         | class_prior     |          0.040173 |          4.108889 |    -0.027368 |     0.118418 |
| logit_term_structure         | uniform         |         -0.080738 |         -7.349104 |    -0.227036 |     0.073541 |
| logit_term_structure_matched | class_prior     |          0.040173 |          4.108889 |    -0.027368 |     0.118418 |
| logit_term_structure_matched | uniform         |         -0.080738 |         -7.349104 |    -0.227036 |     0.073541 |
| logit_term_structure         | logit_price     |          0.021834 |          2.192062 |    -0.017558 |     0.069932 |
| logit_term_structure_matched | logit_vix_level |          0.000449 |          0.044105 |    -0.032333 |     0.033056 |

Differences are named model minus comparator; negative favours the model. The
approximate 95% percentile intervals use 10,000 paired circular
block resamples, with blocks of 5 consecutive episodes and seed
5205. They preserve whole-episode weighting and some adjacent-episode dependence;
they do not assume 175 independent daily observations. They are not
adjusted for multiple comparisons and do not account for earlier research exposure,
model-selection uncertainty or all forms of nonstationarity. An interval crossing
zero does not establish an improvement. Identical matched-model results are not
independent corroboration.

## Original event-start subset

| model_id                     |   n |   log_loss |    brier |   accuracy |   recovery_vs_rest_auc |
|:-----------------------------|----:|-----------:|---------:|-----------:|-----------------------:|
| class_prior                  |  53 |   1.047726 | 0.628141 |   0.509434 |               0.472934 |
| logit_price                  |  53 |   1.027057 | 0.619607 |   0.509434 |               0.591168 |
| logit_term_structure         |  53 |   1.083899 | 0.651815 |   0.471698 |               0.554131 |
| logit_term_structure_matched |  53 |   1.083899 | 0.651815 |   0.471698 |               0.554131 |
| logit_vix_level              |  53 |   1.047999 | 0.636671 |   0.490566 |               0.512821 |
| uniform                      |  53 |   1.098612 | 0.666667 |   0.333333 |               0.500000 |

This descriptive subset uses only decisions equal to their original episode start;
the labels were checked against the original event file. Episode counts for all
triggers can include an episode that started before the evaluation boundary.

## Artifacts and interpretation

`metrics.csv` includes trigger-weighted and episode-weighted results, per-class
recalls and decision-year diagnostics. Each yearly row recomputes episode weights
within its own sample; yearly averages do not simply average to the pooled result.
`paired_loss_intervals.csv`, `episode_counts.csv` and `scored_predictions.csv`
provide the underlying comparisons. The scored file includes future outcomes and
is an analysis artifact, never a trading-signal input.

The manifest records frozen input hashes and verifies that scoring left them
unchanged. Reproduce with `python -m src.analysis.recovery_evaluation`. Protocol:
`docs/recovery_evaluation_protocol.md`. Member 4 should use the original signal
loaders, define allocation and cost rules on design data, and retain baselines in
the backtest even if a model looks best in one reported metric.
