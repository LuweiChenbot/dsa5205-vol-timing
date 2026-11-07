"""Behavioral checks for timing, estimation, selection and the public handoff."""
from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models.config import FEATURE_SETS, ModelConfig
from src.features.feature_build import PastScaler
from src.models.evaluation import (forecast_metrics, paired_comparisons, score_predictions,
                                   select_alphas, event_diagnostics)
from src.models.handoff import load_predictions, load_signal, validate_predictions, wide_predictions
from src.models.pipeline import run_experiment, write_outputs
from src.models.walk_forward import prepare_data, walk_forward


def sample_data(n=90):
    rng = np.random.default_rng(5205)
    calendar = pd.bdate_range("2018-01-01", periods=n + 2, name="date")
    panel = pd.DataFrame({"spy_tr_cc": rng.normal(0.0003, 0.01, n + 2),
                          "bil_tr_cc": np.full(n + 2, 0.00005)}, index=calendar)
    panel["fwd_spy_xs"] = (panel.spy_tr_cc - panel.bil_tr_cc).shift(-2)
    features = pd.DataFrame(rng.normal(size=(n, 14)), index=calendar[:n],
                            columns=FEATURE_SETS["full"])
    features["dec_cond"] = (np.arange(n) % 4 == 0).astype(float)
    features["ts_backw"] = (np.arange(n) % 3 == 0).astype(float)
    return panel, features


def sample_experiment():
    panel, features = sample_data()
    events = pd.DataFrame({"ev_start": np.arange(len(panel)) % 11 == 0,
                           "fwd_ev_label": np.resize([1., -1., 0.], len(panel))}, index=panel.index)
    dates = features.index
    settings = ModelConfig(train_window=20, alphas=(1.0, 10.0),
                           validation_start=str(dates[21].date()), design_end=str(dates[45].date()),
                           test_start=str(dates[46].date()), test_end=str(dates[-1].date()))
    return panel, features, events, settings


class WalkForwardTests(unittest.TestCase):
    def test_only_completed_labels_enter_a_fixed_window(self):
        panel, features = sample_data()
        data = prepare_data(panel, features)
        dates = features.index[21:25]
        result = walk_forward(data, dates, {"ridge_base": ("base", (1.0,))}, 20)
        first = result.predictions.iloc[0]
        self.assertEqual(first.decision_date, dates[0])
        self.assertEqual(first.train_end_date, features.index[19])
        self.assertEqual(first.train_last_label_end_date, dates[0])
        self.assertEqual(first.n_train, 20)
        self.assertEqual(first.execution_date, features.index[22])
        self.assertEqual(first.label_end_date, features.index[23])

    def test_future_targets_and_features_cannot_change_current_forecast(self):
        panel, features = sample_data()
        date = features.index[40]
        data = prepare_data(panel, features)
        original = walk_forward(data, pd.DatetimeIndex([date]),
                                {"ridge_full": ("full", (10.0,))}, 20)
        # At T, the label for T-1 is still unknown, even though its feature is old.
        data.target.iloc[39:] += 100
        data.features.iloc[41:] += 1000
        changed = walk_forward(data, pd.DatetimeIndex([date]),
                               {"ridge_full": ("full", (10.0,))}, 20)
        pd.testing.assert_frame_equal(original.predictions, changed.predictions)

    def test_warmup_is_reported_not_silently_shortened(self):
        panel, features = sample_data()
        result = walk_forward(prepare_data(panel, features), features.index[20:23],
                              {"ridge_base": ("base", (1.0,))}, 20)
        self.assertEqual(result.coverage.status.tolist(), ["insufficient_history", "predicted", "predicted"])
        self.assertEqual(result.coverage.iloc[0].available_train_rows, 19)

    def test_ridge_agrees_with_augmented_normal_equations(self):
        panel, features = sample_data()
        features["ma200_gap"] = 0.0  # constant columns must remain well-defined
        data = prepare_data(panel, features)
        date = features.index[40]
        result = walk_forward(data, pd.DatetimeIndex([date]), {"ridge_base": ("base", (10.0,))}, 20)
        train = features.iloc[19:39][list(FEATURE_SETS["base"])]
        scaler = PastScaler().fit(train)
        X = np.column_stack([np.ones(20), scaler.transform(train)])
        penalty = np.diag([0., *([10.] * 5)])
        beta = np.linalg.solve(X.T @ X + penalty, X.T @ data.target.iloc[19:39].to_numpy())
        new = np.r_[1., scaler.transform(features.loc[[date], train.columns]).iloc[0].to_numpy()]
        actual = result.predictions.query("model_id == 'ridge_base'").prediction.iloc[0]
        self.assertAlmostEqual(actual, float(new @ beta), places=13)

    def test_trailing_dates_follow_actual_calendar_not_weekdays(self):
        panel, features = sample_data(40)
        days = pd.bdate_range(end="2025-12-31", periods=40).append(pd.DatetimeIndex(["2026-01-02", "2026-01-05"]))
        panel.index, features.index = days, days[:40]
        data = prepare_data(panel, features)
        self.assertEqual(data.calendar.iloc[-1].execution_date, pd.Timestamp("2026-01-02"))
        self.assertEqual(data.calendar.iloc[-1].label_end_date, pd.Timestamp("2026-01-05"))
        self.assertAlmostEqual(data.target.iloc[-1], panel.spy_tr_cc.iloc[-1] - panel.bil_tr_cc.iloc[-1])

    def test_invalid_inputs_fail_instead_of_dropping_rows(self):
        panel, features = sample_data()
        bad_features = [features.iloc[::-1], features.drop(features.index[5]),
                        pd.concat([features.iloc[:1], features]), features.assign(dec_vol=np.nan),
                        features.assign(dec_cond=2), features.assign(fwd_bad=1)]
        for bad in bad_features:
            with self.subTest(case=bad.shape), self.assertRaises(ValueError):
                prepare_data(panel, bad)
        bad_panel = panel.copy()
        bad_panel.loc[features.index[0], "fwd_spy_xs"] += 0.01
        with self.assertRaisesRegex(ValueError, "fwd_spy_xs"):
            prepare_data(bad_panel, features)
        with self.assertRaisesRegex(ValueError, "extend two"):
            prepare_data(panel.iloc[:-1], features)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel, cls.features, cls.events, cls.settings = sample_experiment()
        cls.result = run_experiment(cls.panel, cls.features, cls.events, cls.settings)

    def test_validation_boundary_and_complete_evaluation_coverage(self):
        result = self.result
        self.assertTrue((result.validation_predictions.label_end_date <= pd.Timestamp(self.settings.design_end)).all())
        purged = result.coverage[result.coverage.status == "outcome_after_design_end"]
        self.assertEqual(len(purged), 2)
        self.assertEqual(result.predictions.decision_date.nunique(), 44)
        self.assertEqual(len(result.predictions), 44 * 6)
        self.assertTrue((result.predictions.train_last_label_end_date <= result.predictions.decision_date).all())
        self.assertEqual(result.predictions.iloc[-1].label_end_date, self.panel.index[-1])

    def test_evaluation_outcome_shock_cannot_change_alpha_selection(self):
        panel = self.panel.copy()
        panel.loc[panel.index > self.settings.design_end, "spy_tr_cc"] += 0.2
        panel["fwd_spy_xs"] = (panel.spy_tr_cc - panel.bil_tr_cc).shift(-2)
        changed = run_experiment(panel, self.features, self.events, self.settings)
        self.assertEqual(self.result.selected_alphas, changed.selected_alphas)
        pd.testing.assert_frame_equal(self.result.validation_predictions, changed.validation_predictions)
        # Frozen design permits adapting to newly matured evaluation observations.
        original_late = self.result.predictions.prediction.iloc[-6:].to_numpy()
        self.assertFalse(np.allclose(original_late, changed.predictions.prediction.iloc[-6:].to_numpy()))

    def test_feature_comparison_holds_everything_else_fixed(self):
        self.assertEqual([len(v) for v in FEATURE_SETS.values()], [5, 9, 11, 14])
        self.assertEqual(set(FEATURE_SETS["vix"]) - set(FEATURE_SETS["vix_level"]), {"ts_ratio", "ts_backw"})
        p = self.result.predictions
        level = p[p.model_id == "ridge_vix_level"].set_index("decision_date")
        matched = p[p.model_id == "ridge_vix_matched"].set_index("decision_date")
        pd.testing.assert_series_equal(level.alpha, matched.alpha)
        for column in ("train_start_date", "train_end_date", "train_last_label_end_date"):
            pd.testing.assert_series_equal(level[column], matched[column])
        self.assertEqual(self.result.comparisons.query("segment == 'all'").n.tolist(), [44] * 4)

    def test_alpha_ties_use_stronger_penalty_and_unmatched_dates_fail(self):
        candidates = self.result.validation_candidates.copy()
        candidates["prediction"] = 0.0
        chosen, _ = select_alphas(candidates, self.result.data.outcomes, self.settings.design_end)
        self.assertEqual(chosen, {name: 10.0 for name in FEATURE_SETS})
        row = candidates.index[candidates.model_id == "ridge_base"][0]
        with self.assertRaisesRegex(ValueError, "same unique dates"):
            select_alphas(candidates.drop(row), self.result.data.outcomes, self.settings.design_end)

    def test_incomplete_evaluation_and_insufficient_validation_fail(self):
        from dataclasses import replace
        with self.assertRaisesRegex(ValueError, "insufficient training history"):
            run_experiment(self.panel, self.features, self.events, replace(self.settings, train_window=100))
        with self.assertRaises(ValueError):
            run_experiment(self.panel, self.features.iloc[:-1], self.events, self.settings)

    def test_export_reload_and_member4_execution_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            out, reports = Path(directory) / "signals", Path(directory) / "reports"
            # Rendering is checked in the real-data run; this checks file semantics.
            with patch("src.models.pipeline._plot_results"):
                write_outputs(self.result, self.settings, out, reports)
            loaded = load_predictions(out / "predictions.csv")
            self.assertEqual(len(loaded), len(self.result.predictions))
            self.assertTrue(np.array_equal(loaded.prediction, self.result.predictions.prediction))
            wide = wide_predictions(loaded)
            self.assertEqual(wide.shape, (44, 9))
            self.assertFalse(any(c.startswith("actual") or c.startswith("fwd_") for c in loaded))
            signal = load_signal(path=out / "predictions.csv")
            # A sample rule only tests alignment. Its trading merits are not assessed.
            weights = (signal.prediction > 0).astype(float)
            from src.metrics import apply_weights
            actual = apply_weights(self.panel, weights, cost_bps=0)
            ends = pd.DatetimeIndex(signal.label_end_date)
            explicit = (weights.to_numpy() * self.panel.loc[ends, "spy_tr_cc"].to_numpy() +
                        (1 - weights.to_numpy()) * self.panel.loc[ends, "bil_tr_cc"].to_numpy())
            np.testing.assert_allclose(actual.loc[ends].to_numpy(), explicit, rtol=0, atol=1e-15)
            with self.assertRaisesRegex(ValueError, "unknown model"):
                load_signal("typo", out / "predictions.csv")
            # A removed date or changed number must not pass the saved checksum.
            path = out / "predictions.csv"
            path.write_text(path.read_text().replace("ridge_full", "ridge_typo", 1))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_predictions(path)

    def test_handoff_rejects_bad_timing_duplicates_missing_models_and_outcomes(self):
        original = self.result.predictions
        invalid = [pd.concat([original, original.iloc[:1]], ignore_index=True),
                   original[original.model_id != "ridge_base"], original.assign(actual_return=1),
                   original.assign(prediction=np.nan), original.assign(execution_date=original.decision_date)]
        for bad in invalid:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                validate_predictions(bad)

    def test_metrics_use_rolling_mean_benchmark_and_paired_dates(self):
        dates = pd.bdate_range("2020-01-01", periods=2)
        predictions = pd.DataFrame({"decision_date": list(dates) * 2,
                                    "model_id": ["historical_mean"] * 2 + ["ridge_full"] * 2,
                                    "prediction": [1., 1., 1., 2.], "decline": [1, 0, 1, 0]})
        outcomes = pd.DataFrame({"actual_excess_return": [0., 2.]}, index=dates)
        scored = score_predictions(predictions, outcomes)
        table = forecast_metrics(scored).query("decision_year == 'all' and segment == 'all'").set_index("model_id")
        self.assertEqual(table.at["historical_mean", "oos_r2"], 0.0)
        self.assertEqual(table.at["ridge_full", "oos_r2"], 0.5)
        self.assertEqual(table.at["ridge_full", "mse"], 0.5)
        self.assertEqual(table.at["ridge_full", "mae"], 0.5)
        scores = score_predictions(self.result.predictions, self.result.data.outcomes)
        row = scores.index[scores.model_id == "ridge_vix"][0]
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_comparisons(scores.drop(row))

    def test_event_auc_retains_unresolved_events_in_counts(self):
        p = self.result.predictions
        summary, rows = event_diagnostics(p, self.events)
        self.assertTrue((summary.n_events == summary.n_recovery + summary.n_continuation + summary.n_unresolved).all())
        self.assertTrue((rows.fwd_ev_label == 0).any())
        single_class = self.events.assign(fwd_ev_label=1.0)
        summary, _ = event_diagnostics(p, single_class)
        self.assertTrue(summary.resolved_auc.isna().all())


class ConfigTests(unittest.TestCase):
    def test_invalid_grid_and_splits_fail(self):
        for args in ({"alphas": ()}, {"alphas": (0.,)}, {"alphas": (1., 1.)},
                     {"alphas": (np.nan,)}, {"train_window": 0}, {"train_window": 10.5},
                     {"design_end": "2020-01-01"}, {"test_end": "2018-01-01"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                ModelConfig(**args)


if __name__ == "__main__":
    unittest.main()
