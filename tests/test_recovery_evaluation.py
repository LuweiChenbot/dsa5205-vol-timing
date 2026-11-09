"""Independent metric and frozen-artifact checks for recovery evaluation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.analysis.recovery_evaluation import (evaluate_frozen, metric_row, metrics_table,
                                             paired_intervals, score_rows)
from src.models.extensions.config import MODEL_IDS, MODEL_SETS, OUTPUT_DIR, PROBABILITY_COLUMNS


def toy_predictions():
    dates = pd.bdate_range("2019-01-02", periods=3)
    calendar = pd.bdate_range("2019-01-02", periods=14)
    rows, outcomes = [], []
    for i, date in enumerate(dates):
        episode = dates[0] if i < 2 else date
        for model in MODEL_IDS:
            rows.append({"decision_date": date, "execution_date": calendar[i+1], "label_end_date": calendar[i+11],
                         "episode_start_date": episode, "train_start_date": pd.Timestamp("2018-11-01"),
                         "train_end_date": pd.Timestamp("2018-12-03"),
                         "train_last_label_end_date": pd.Timestamp("2018-12-18"),
                         "model_id": model, "feature_set": "none" if model == "class_prior" else MODEL_SETS[model],
                         "horizon": 10, "C": np.nan if model == "class_prior" else 10. if model == "logit_price" else 1.,
                         "n_train": 12, "n_episodes": 12, "n_continuation": 4, "n_unresolved": 4,
                         "n_recovery": 4, "train_weight_sum": 12.,
                         "fit_status": "prior" if model == "class_prior" else "fit",
                         "p_continuation": .6, "p_unresolved": .3, "p_recovery": .1})
        outcomes.append({"decision_date": date, "execution_date": calendar[i+1], "label_end_date": calendar[i+11],
                         "episode_start_date": episode, "label": i-1})
    return pd.DataFrame(rows), pd.DataFrame(outcomes)


class MetricTests(unittest.TestCase):
    def test_episode_weighted_loss_matches_independent_sklearn_calculation(self):
        predictions, outcomes = toy_predictions()
        scored = score_rows(predictions, outcomes)
        p = scored[scored.model_id == "logit_price"]
        actual = metric_row(p, weighting="episode")
        expected = log_loss([-1, 0, 1], np.tile([.6, .3, .1], (3, 1)), labels=[-1, 0, 1], sample_weight=[.5, .5, 1])
        self.assertAlmostEqual(actual["log_loss"], expected, places=14)
        self.assertAlmostEqual(actual["brier"], .91, places=14)
        self.assertAlmostEqual(actual["accuracy"], .25, places=14)
        self.assertEqual(actual["n_episodes"], 2)

    def test_uniform_uses_expected_random_tie_breaking(self):
        scored = score_rows(*toy_predictions())
        result = metric_row(scored[scored.model_id == "uniform"], weighting="episode")
        self.assertAlmostEqual(result["log_loss"], np.log(3))
        self.assertAlmostEqual(result["brier"], 2/3)
        self.assertAlmostEqual(result["accuracy"], 1/3)
        self.assertAlmostEqual(result["macro_recall"], 1/3)
        self.assertAlmostEqual(result["recovery_vs_rest_auc"], .5)
        for name in ("continuation", "unresolved", "recovery"):
            self.assertAlmostEqual(result[f"expected_predictions_{name}"], 1.)

    def test_outcomes_align_by_date_not_row_order(self):
        predictions, outcomes = toy_predictions()
        expected = score_rows(predictions, outcomes)
        actual = score_rows(predictions, outcomes.iloc[::-1])
        pd.testing.assert_frame_equal(expected, actual)
        for bad in (outcomes.iloc[:-1], pd.concat([outcomes, outcomes.iloc[:1]]),
                    outcomes.assign(label=2), outcomes.assign(execution_date=outcomes.decision_date)):
            with self.assertRaises(ValueError):
                score_rows(predictions, bad)

    def test_three_class_and_trigger_metrics_do_not_silently_drop_unresolved(self):
        scored = score_rows(*toy_predictions())
        table = metrics_table(scored, period="evaluation")
        pooled = table.query("subset == 'all_triggers' and decision_year == 'all' and weighting == 'trigger'")
        self.assertTrue(pooled.n.eq(3).all())
        self.assertTrue(pooled.n_unresolved.eq(1).all())
        self.assertAlmostEqual(pooled.set_index("model_id").at["logit_price", "accuracy"], 1/3)
        starts = table.query("subset == 'event_starts' and decision_year == 'all'")
        self.assertTrue(starts.n.eq(2).all())
        self.assertTrue(starts.macro_recall.isna().all())


class PairedBootstrapTests(unittest.TestCase):
    def data(self):
        losses = {"class_prior": 1., "uniform": 1.1, "logit_price": .8,
                  "logit_vix_level": .9, "logit_term_structure": .85, "logit_term_structure_matched": .85}
        return pd.DataFrame([{"episode_start_date": episode, "model_id": model, "log_loss": loss + episode / 100}
                             for episode in range(8) for model, loss in losses.items()])

    def test_paired_common_noise_cancels_and_sampling_is_reproducible(self):
        a = paired_intervals(self.data(), block_size=3, replicates=200, seed=1)
        b = paired_intervals(self.data(), block_size=3, replicates=200, seed=1)
        pd.testing.assert_frame_equal(a, b)
        row = a.query("model_id == 'logit_price' and comparator == 'class_prior'").iloc[0]
        np.testing.assert_allclose([row.loss_difference, row.ci95_lower, row.ci95_upper], -.2, atol=1e-14)

    def test_full_length_circular_blocks_preserve_the_observed_mean(self):
        data = self.data()
        mask = data.model_id == "logit_price"
        data.loc[mask, "log_loss"] += data.loc[mask, "episode_start_date"] ** 2 / 100
        intervals = paired_intervals(data, block_size=8, replicates=200)
        np.testing.assert_allclose(intervals.ci95_lower, intervals.loss_difference, atol=1e-14)
        np.testing.assert_allclose(intervals.ci95_upper, intervals.loss_difference, atol=1e-14)
        with self.assertRaises(ValueError):
            paired_intervals(data.iloc[1:], block_size=3)


class FrozenArtifactTests(unittest.TestCase):
    def test_real_snapshot_scoring_cannot_fit_or_select_and_does_not_change_inputs(self):
        # This public integration seam reads the exact committed snapshot used by
        # the report. A future accidental fit/selection call must fail this test.
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(LogisticRegression, "fit", side_effect=AssertionError("refit prohibited")), \
             patch("src.models.extensions.recovery.select_cs", side_effect=AssertionError("retuning prohibited")):
            metrics, comparisons = evaluate_frozen(OUTPUT_DIR / "recovery", Path(folder))
            sample = metrics.query("period == 'evaluation' and subset == 'all_triggers' and decision_year == 'all'")
            self.assertTrue(sample.n.eq(175).all())
            self.assertTrue(sample.n_episodes.eq(53).all())
            self.assertEqual(len(comparisons), 10)
            self.assertTrue((Path(folder) / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
