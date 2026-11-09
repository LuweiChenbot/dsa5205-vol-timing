# Exploratory modelling extensions

Exploratory extension designed after the original one-day evaluation was seen. Settings use design data only, but no untouched-holdout claim is made. No new evaluation-performance or trading analysis is performed here.

Earlier feature research examined 2017–2025 outcomes, as disclosed in docs/report_features_sections.md. The present forecasts obey information timing, but 2019–2025 is not a pristine, untouched research holdout.

No evaluation performance metrics or trading results are calculated in this extension.

## Ten-day ridge

Selected settings (design data only):

| feature_set   |   alpha |   n |         mse |       mae | first_decision      | last_decision       | last_label_end      | selected   |
|:--------------|--------:|----:|------------:|----------:|:--------------------|:--------------------|:--------------------|:-----------|
| base          |  100000 | 481 | 0.000592247 | 0.016334  | 2017-01-18 00:00:00 | 2018-12-13 00:00:00 | 2018-12-31 00:00:00 | True       |
| full          |  100000 | 481 | 0.000592523 | 0.0163377 | 2017-01-18 00:00:00 | 2018-12-13 00:00:00 | 2018-12-31 00:00:00 | True       |
| vix           |  100000 | 481 | 0.000592492 | 0.0163375 | 2017-01-18 00:00:00 | 2018-12-13 00:00:00 | 2018-12-31 00:00:00 | True       |
| vix_level     |  100000 | 481 | 0.000592375 | 0.0163371 | 2017-01-18 00:00:00 | 2018-12-13 00:00:00 | 2018-12-31 00:00:00 | True       |

| period     | status                      |    n |
|:-----------|:----------------------------|-----:|
| evaluation | incomplete_outcome_calendar |    1 |
| evaluation | predicted                   | 1759 |
| validation | insufficient_history        |   10 |
| validation | outcome_after_design_end    |   11 |
| validation | predicted                   |  481 |

## Recovery classifier

Selected settings (design data only):

| feature_set    |   C |   n |   n_episodes |   weighted_log_loss | last_label_end      | selected   |
|:---------------|----:|----:|-------------:|--------------------:|:--------------------|:-----------|
| price          |  10 |  32 |           10 |             1.0158  | 2018-12-27 00:00:00 | True       |
| term_structure |   1 |  32 |           10 |             1.05121 | 2018-12-27 00:00:00 | True       |
| vix_level      |   1 |  32 |           10 |             1.06343 | 2018-12-27 00:00:00 | True       |

| period     | status                   |   n |
|:-----------|:-------------------------|----:|
| evaluation | predicted                | 175 |
| validation | outcome_after_design_end |   6 |
| validation | predicted                |  32 |

Evaluation trigger dates: 175. Fallback rows across all fitted model IDs: 0.

Episode weighting reduces concentration; it does not turn overlapping labels into independent samples.

## One-day penalty sensitivity (design only)

| feature_set   |   alpha |   n |         mse |        mae | first_decision      | last_decision       | last_label_end      | selected   |
|:--------------|--------:|----:|------------:|-----------:|:--------------------|:--------------------|:--------------------|:-----------|
| base          |  100000 | 499 | 6.69815e-05 | 0.00520309 | 2017-01-04 00:00:00 | 2018-12-27 00:00:00 | 2018-12-31 00:00:00 | True       |
| full          |  100000 | 499 | 6.69929e-05 | 0.0052039  | 2017-01-04 00:00:00 | 2018-12-27 00:00:00 | 2018-12-31 00:00:00 | True       |
| vix           |  100000 | 499 | 6.69876e-05 | 0.00520338 | 2017-01-04 00:00:00 | 2018-12-27 00:00:00 | 2018-12-31 00:00:00 | True       |
| vix_level     |  100000 | 499 | 6.69888e-05 | 0.00520324 | 2017-01-04 00:00:00 | 2018-12-27 00:00:00 | 2018-12-31 00:00:00 | True       |

The original one-day settings, predictions and performance tables remain unchanged.

Read docs/model_extensions.md for the target definitions, sparse-signal contract, missing-tail handling and Member 4 examples.
