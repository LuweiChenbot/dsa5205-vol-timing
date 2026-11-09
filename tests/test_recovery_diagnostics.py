"""Independent checks of frozen-model diagnostic calculations and no-retuning contract."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.analysis.recovery_diagnostics import (
    audit_reports, candidate_episode_losses, coefficient_diagnostics,
    probability_diagnostics, run_diagnostics, selection_sensitivity, weighted_quantiles,
)
from src.analysis.recovery_evaluation import digest, score_rows
from src import config
from src.features.feature_build import PastScaler
from src.models.extensions.config import OUTPUT_DIR, PROBABILITY_COLUMNS
from tests.test_recovery_evaluation import toy_predictions


class ProbabilityDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.scored = score_rows(*toy_predictions())
        self.tables = probability_diagnostics(self.scored)

    def test_weighted_class_metrics_and_confusion_match_independent_calculations(self):
        rows = self.tables["class_metrics.csv"].query("model_id == 'logit_price'").set_index("class_name")
        continuation = rows.loc["continuation"]
        self.assertAlmostEqual(continuation.observed_frequency, .25)
        self.assertAlmostEqual(continuation.mean_probability, .6)
        self.assertAlmostEqual(continuation.binary_brier, .31)
        self.assertAlmostEqual(continuation.ece_fixed_bins, .35)
        self.assertAlmostEqual(continuation.one_vs_rest_auc, .5)
        confusion = self.tables["confusion_matrices.csv"].query("model_id == 'logit_price'").pivot(
            index="actual_class", columns="predicted_class", values="episode_weight")
        names = ["continuation", "unresolved", "recovery"]
        expected = confusion_matrix([-1, 0, 1], [-1, -1, -1], labels=[-1, 0, 1], sample_weight=[.5, .5, 1.])
        np.testing.assert_allclose(confusion.reindex(index=names, columns=names), expected)

    def test_class_auc_with_nonconstant_scores(self):
        data = self.scored[self.scored.model_id == "logit_price"].copy()
        data[list(PROBABILITY_COLUMNS)] = [[.1, .3, .6], [.6, .2, .2], [.4, .1, .5]]
        table = probability_diagnostics(data)["class_metrics.csv"].set_index("class_name")
        expected = roc_auc_score([1, 0, 0], [.1, .6, .4], sample_weight=[.5, .5, 1.])
        self.assertEqual(table.loc["continuation", "one_vs_rest_auc"], expected)
        self.assertEqual(expected, 0.)

    def test_uniform_ties_and_missing_truth_class(self):
        uniform = self.tables["class_metrics.csv"].query("model_id == 'uniform'")
        np.testing.assert_allclose(uniform.argmax_recall, 1/3)
        np.testing.assert_allclose(uniform.expected_argmax_count, 1.)
        small = probability_diagnostics(self.scored[self.scored.label == -1])
        self.assertTrue(small["class_metrics.csv"].one_vs_rest_auc.isna().all())
        absent = small["class_metrics.csv"].query("class_name != 'continuation'")
        self.assertTrue(absent.argmax_recall.isna().all())
        self.assertTrue(small["confusion_matrices.csv"].query("actual_class != 'continuation'").row_fraction.isna().all())

    def test_bin_boundaries_and_empty_bins_keep_original_weights(self):
        data = self.scored[self.scored.model_id == "logit_price"].copy()
        bins = probability_diagnostics(data)["calibration_bins.csv"].query("class_name == 'continuation'")
        self.assertEqual(bins.n_triggers.tolist(), [0, 0, 0, 3, 0])
        self.assertTrue(bins.loc[bins.n_triggers == 0, "mean_probability"].isna().all())
        self.assertAlmostEqual(bins.loc[bins.n_triggers > 0, "episode_weight"].iloc[0], 2.)
        # The same episode straddles two bins. Its total weight must remain one,
        # rather than becoming one separately in each bin.
        data[list(PROBABILITY_COLUMNS)] = [[0., .5, .5], [.2, .3, .5], [1., 0., 0.]]
        bins = probability_diagnostics(data)["calibration_bins.csv"].query("class_name == 'continuation'")
        np.testing.assert_allclose(bins.episode_weight, [.5, .5, 0., 0., 1.])
        self.assertEqual(bins.n_triggers.tolist(), [1, 1, 0, 0, 1])
        data[list(PROBABILITY_COLUMNS)] = [[.4, .3, .3], [.6, .2, .2], [.8, .1, .1]]
        bins = probability_diagnostics(data)["calibration_bins.csv"].query("class_name == 'continuation'")
        self.assertEqual(bins.n_triggers.tolist(), [0, 0, 1, 1, 1])

    def test_inverse_cdf_quantiles_use_weights_without_interpolation(self):
        np.testing.assert_allclose(weighted_quantiles(np.array([.9, .1, .5]), np.array([1., 8., 1.])), [.1, .1, .5])
        with self.assertRaises(ValueError):
            weighted_quantiles(np.array([.1]), np.array([0.]))


class SelectionSensitivityTests(unittest.TestCase):
    def losses(self):
        dates = pd.date_range("2017-01-01", periods=3, freq="MS")
        rows = []
        for model, feature, c, losses in [
            ("class_prior", "none", np.nan, [1., 1., 1.]),
            ("logit_price", "price", 1., [.5, .5, 1.5]),
            ("logit_price", "price", 10., [1., 1., .1]),
        ]:
            rows.extend({"episode_start_date": date, "model_id": model, "feature_set": feature,
                         "C": c, "log_loss": loss} for date, loss in zip(dates, losses))
        return pd.DataFrame(rows)

    def test_omitting_one_scoring_episode_exposes_unstable_original_winner(self):
        tables = selection_sensitivity(self.losses(), {"price": 10.})
        ranks = tables["validation_candidate_ranks.csv"]
        full = ranks.query("sample == 'full_validation' and feature_set == 'price'")
        self.assertEqual(full.loc[full.rank_within_feature_set == 1, "C"].iloc[0], 10.)
        counts = tables["validation_rank_counts.csv"].set_index("C")
        self.assertEqual(counts.loc[10., "n_ranked_first"], 2)
        self.assertEqual(counts.loc[1., "n_ranked_first"], 1)
        self.assertAlmostEqual(counts.loc[10., "max_loss_minus_prior"], 0.)
        self.assertEqual(ranks.query("sample == 'omit_one_episode'").n_episodes.unique().tolist(), [2])

    def test_exact_ties_prefer_smaller_c_and_incomplete_candidate_fails(self):
        losses = self.losses()
        losses.loc[losses.feature_set == "price", "log_loss"] = .5
        counts = selection_sensitivity(losses, {"price": 1.})["validation_rank_counts.csv"].set_index("C")
        self.assertEqual(counts.loc[1., "n_ranked_first"], 3)
        with self.assertRaises(ValueError):
            selection_sensitivity(losses.iloc[1:], {"price": 1.})

    def test_candidate_grid_and_saved_forecast_alignment(self):
        predictions, outcomes = toy_predictions()
        # Move the synthetic validation period into the toy dates without changing
        # any real settings or the real frozen experiment.
        manifest = {"feature_sets": {"price": ["dec_dd"]}, "selected_penalties": {"price": 10.},
                    "settings": {"classifier_cs": [10.], "periods": {
                        "validation_start": "2019-01-01", "design_end": "2019-12-31"}}}
        candidates = predictions[predictions.model_id.isin(["class_prior", "logit_price"])]
        expected = candidate_episode_losses(candidates, outcomes, manifest, predictions)
        actual = candidate_episode_losses(candidates.iloc[::-1], outcomes.iloc[::-1], manifest, predictions)
        pd.testing.assert_frame_equal(expected, actual)
        bad = candidates.copy()
        bad.loc[bad.model_id == "logit_price", list(PROBABILITY_COLUMNS)] = [.1, .3, .6]
        with self.assertRaisesRegex(ValueError, "differs from frozen"):
            candidate_episode_losses(bad, outcomes, manifest, predictions)
        with self.assertRaisesRegex(ValueError, "same validation dates"):
            candidate_episode_losses(candidates.iloc[1:], outcomes, manifest, predictions)


class CoefficientTests(unittest.TestCase):
    def test_contrasts_and_sign_changes_skip_near_zero_and_validate_coverage(self):
        predictions, _ = toy_predictions()
        predictions = predictions[predictions.model_id == "logit_price"]
        rows = []
        for date, continuation in zip(predictions.decision_date, [-.1, 1e-12, .2]):
            for label in [-1, 0, 1]:
                rows.append({"decision_date": date, "model_id": "logit_price", "C": 10., "class_label": label,
                             "intercept": 3., "dec_dd": 1. + continuation if label == -1 else 1.})
        coefficients = pd.DataFrame(rows)
        summary = coefficient_diagnostics(coefficients, predictions, {"price": ["dec_dd"]})["coefficient_stability.csv"]
        row = summary.query("term == 'dec_dd' and contrast == 'continuation_minus_recovery'").iloc[0]
        self.assertAlmostEqual(row["first"], -.1)
        self.assertAlmostEqual(row["last"], .2)
        self.assertEqual(row.nonzero_sign_changes, 1)
        self.assertEqual(row.n_near_zero, 1)
        intercept = summary.query("term == 'intercept'")
        self.assertTrue(intercept.n_near_zero.eq(3).all())
        with self.assertRaisesRegex(ValueError, "coverage"):
            coefficient_diagnostics(coefficients.iloc[1:], predictions, {"price": ["dec_dd"]})
        with self.assertRaisesRegex(ValueError, "penalties"):
            coefficient_diagnostics(coefficients.assign(C=1.), predictions, {"price": ["dec_dd"]})


class FrozenIntegrationTests(unittest.TestCase):
    def test_report_hash_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "coefficients.csv"
            path.write_text("original")
            manifest = {"report_sha256": {path.name: digest(path)}}
            self.assertEqual(audit_reports(path.parent, manifest)[path], digest(path))
            path.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                audit_reports(path.parent, manifest)

    def test_real_diagnostics_cannot_fit_scale_or_retune_and_reproduce(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(LogisticRegression, "fit", side_effect=AssertionError("refit prohibited")), \
             patch.object(PastScaler, "fit", side_effect=AssertionError("scaling fit prohibited")), \
             patch.object(PastScaler, "transform", side_effect=AssertionError("scaling prohibited")), \
             patch("src.models.extensions.recovery.select_cs", side_effect=AssertionError("retuning prohibited")):
            first, second = Path(folder) / "a", Path(folder) / "b"
            tables = run_diagnostics(first)
            run_diagnostics(second)
            metrics = tables["class_metrics.csv"]
            self.assertTrue(metrics.n_triggers.eq(175).all())
            self.assertTrue(metrics.n_episodes.eq(53).all())
            self.assertEqual(len(tables["calibration_bins.csv"]), 6 * 3 * 5)
            self.assertEqual(tables["validation_rank_counts.csv"].n_omissions.unique().tolist(), [10])
            old = pd.read_csv(config.ROOT / "reports/recovery_evaluation/metrics.csv").query(
                "period == 'evaluation' and subset == 'all_triggers' and decision_year == 'all' and weighting == 'episode'")
            binary_sum = metrics.groupby("model_id").binary_brier.sum().sort_index()
            np.testing.assert_allclose(binary_sum, old.set_index("model_id").brier.sort_index(), rtol=0, atol=1e-14)
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes(), path.name)
            manifest = json.loads((first / "manifest.json").read_text())
            self.assertTrue(manifest["no_refitting_or_retuning"])
            for name, expected in manifest["output_sha256"].items():
                self.assertEqual(digest(first / name), expected)

    def test_frozen_output_locations_are_protected(self):
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            run_diagnostics(OUTPUT_DIR / "recovery")


if __name__ == "__main__":
    unittest.main()
