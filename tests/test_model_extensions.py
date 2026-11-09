"""Behavioural checks for exploratory recovery and ten-day models."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.features.events import event_starts
from src.models.config import ModelConfig
from src.models.handoff import load_signal
from src.models.walk_forward import prepare_data, walk_forward
from src.models.extensions.config import ExtensionConfig, FEATURE_SETS, PROBABILITY_COLUMNS
from src.models.extensions.handoff import (load_recovery_predictions, load_recovery_signal,
                                           load_ten_day_signal, validate_recovery)
from src.models.extensions.pipeline import run_extensions, write_extensions
from src.models.extensions.recovery import (class_prior, episode_weights, prepare_recovery,
                                            select_cs, walk_recovery)
from tests.test_models import sample_data


def fixture(n=130):
    panel, features = sample_data(n + 9)
    features = features.iloc[:n].copy()
    features["dec_vol"] = 0.007
    features["dec_cond"] = (np.arange(n) % 3 == 0).astype(float)
    panel["spy_adj_close"] = 100 * (1 + panel.spy_tr_cc).cumprod()
    cond = features.dec_cond.astype(bool).reindex(panel.index, fill_value=False)
    events = pd.DataFrame({"dec_cond": cond, "ev_start": event_starts(cond, 10)}, index=panel.index)
    dates = features.index
    settings = ExtensionConfig(
        periods=ModelConfig(train_window=20, alphas=(1., 10.), validation_start=str(dates[51].date()),
                            design_end=str(dates[89].date()), test_start=str(dates[90].date()),
                            test_end=str(dates[-1].date())),
        ridge_alphas=(1., 10.), classifier_cs=(0.1, 1.), min_episodes=2)
    return panel, features, events, settings


def recovery_fixture():
    panel, features, events, settings = fixture()
    data = prepare_recovery(panel, prepare_data(panel, features, horizon=10), events)
    # Deliberately controlled classes let tests exercise fitting and fallback.
    data.outcomes["label"] = np.resize(np.array([-1., 0., 1.]), len(data.outcomes))
    return data


class HorizonTests(unittest.TestCase):
    def test_ten_day_target_compounds_each_asset_and_waits_eleven_sessions(self):
        panel, features, _, _ = fixture()
        data = prepare_data(panel, features, horizon=10)
        date = features.index[60]
        result = walk_forward(data, pd.DatetimeIndex([date]), {"ridge_base": ("base", (10.,))}, 20)
        row = result.predictions.iloc[0]
        self.assertEqual(row.train_end_date, features.index[49])
        self.assertEqual(row.train_last_label_end_date, date)
        self.assertEqual(row.label_end_date, panel.index[71])
        self.assertEqual(row.horizon, 10)
        spy = np.prod(1 + panel.spy_tr_cc.iloc[62:72]) - 1
        bil = np.prod(1 + panel.bil_tr_cc.iloc[62:72]) - 1
        self.assertAlmostEqual(data.target.loc[date], spy - bil, places=14)

    def test_ten_day_future_labels_and_features_do_not_change_forecasts(self):
        panel, features, _, _ = fixture()
        data = prepare_data(panel, features, horizon=10)
        date = features.index[60]
        args = (data, pd.DatetimeIndex([date]), {"ridge_full": ("full", (10.,))}, 20)
        expected = walk_forward(*args).predictions
        data.target.iloc[50:] += 1000
        data.features.iloc[61:] += 1000
        pd.testing.assert_frame_equal(expected, walk_forward(*args).predictions)

    def test_missing_tail_is_logged_and_not_filled(self):
        panel, features, _, _ = fixture()
        panel = panel.iloc[:-1]
        with self.assertRaisesRegex(ValueError, "horizon"):
            prepare_data(panel, features, horizon=10)
        data = prepare_data(panel, features, horizon=10, allow_incomplete_tail=True)
        self.assertTrue(pd.isna(data.target.iloc[-1]))
        self.assertTrue(pd.isna(data.calendar.label_end_date.iloc[-1]))
        result = walk_forward(data, features.index[-2:], {"ridge_base": ("base", (1.,))}, 20)
        self.assertEqual(result.coverage.status.tolist(), ["predicted", "incomplete_outcome_calendar"])
        self.assertEqual(result.predictions.decision_date.nunique(), 1)

    def test_default_horizon_matches_explicit_one_day(self):
        panel, features, _, _ = fixture()
        a, b = prepare_data(panel, features), prepare_data(panel, features, horizon=1)
        dates = features.index[40:43]
        plan = {"ridge_full": ("full", (1000.,))}
        pd.testing.assert_frame_equal(walk_forward(a, dates, plan, 20).predictions,
                                      walk_forward(b, dates, plan, 20).predictions)
        for horizon in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                prepare_data(panel, features, horizon=horizon)


class RecoveryTests(unittest.TestCase):
    def test_known_barrier_paths_have_three_distinct_labels(self):
        panel, features, events, _ = fixture()
        features["dec_cond"] = 0.
        features.loc[features.index[[0, 12, 24]], "dec_cond"] = 1.
        features["dec_vol"] = 0.005
        panel["spy_adj_close"] = 100.
        panel.loc[panel.index[2], "spy_adj_close"] = 102.
        panel.loc[panel.index[14], "spy_adj_close"] = 98.
        events["dec_cond"] = features.dec_cond.astype(bool).reindex(panel.index, fill_value=False)
        data = prepare_recovery(panel, prepare_data(panel, features, horizon=10), events)
        self.assertEqual(data.outcomes.label.tolist(), [1., -1., 0.])
        self.assertEqual(data.calendar.label_end_date.tolist(), list(panel.index[[11, 23, 35]]))

    def test_labels_and_episodes_before_cutoff_ignore_future_prices_and_triggers(self):
        panel, features, events, _ = fixture()
        before = prepare_recovery(panel, prepare_data(panel, features, horizon=10), events)
        cutoff = features.index[65]
        panel.loc[panel.index > cutoff, "spy_adj_close"] *= 1.7
        features.loc[features.index > cutoff, "dec_cond"] = 1.
        events.loc[events.index > cutoff, "dec_cond"] = True
        after = prepare_recovery(panel, prepare_data(panel, features, horizon=10), events)
        completed = before.outcomes.index[before.outcomes.label_end_date <= cutoff]
        pd.testing.assert_frame_equal(before.outcomes.loc[completed], after.outcomes.loc[completed])

    def test_weights_give_each_matured_episode_one_vote(self):
        groups = pd.Series(["a", "a", "a", "b", "b", "c"])
        weights = episode_weights(groups)
        np.testing.assert_allclose(weights, [1/3, 1/3, 1/3, 1/2, 1/2, 1])
        sums = pd.Series(weights).groupby(groups).sum()
        np.testing.assert_allclose(sums, 1.)
        # Recompute on training rows only, without dilution from future rows.
        np.testing.assert_allclose(episode_weights(groups.iloc[:2]), [.5, .5])

    def test_recovery_forecast_ignores_unmatured_labels_and_future_features(self):
        data = recovery_fixture()
        date = data.features.index[25]
        plan = {"logit_term_structure": ("term_structure", (0.1,))}
        expected = walk_recovery(data, pd.DatetimeIndex([date]), plan, min_episodes=2)
        immature = data.calendar.label_end_date > date
        data.outcomes.loc[immature, "label"] = 1.
        # Old feature rows without completed labels must not fit the scaler either.
        data.features.loc[immature & (data.features.index < date)] += 500
        data.features.loc[data.features.index > date] += 1000
        data.calendar.loc[data.calendar.index > date, "episode_start_date"] = pd.Timestamp("1900-01-01")
        data.outcomes.loc[data.outcomes.index > date, "episode_start_date"] = pd.Timestamp("1900-01-01")
        actual = walk_recovery(data, pd.DatetimeIndex([date]), plan, min_episodes=2)
        pd.testing.assert_frame_equal(expected.predictions, actual.predictions)
        self.assertTrue(expected.predictions.fit_status.eq("fit").any())

    def test_empty_or_missing_class_history_falls_back_to_smoothed_prior(self):
        data = recovery_fixture()
        plan = {"logit_price": ("price", (0.1,))}
        empty = walk_recovery(data, data.features.index[:1], plan, min_episodes=2).predictions
        np.testing.assert_allclose(empty[list(PROBABILITY_COLUMNS)], 1/3)
        self.assertEqual(empty.n_train.tolist(), [0, 0])
        self.assertEqual(empty.fit_status.tolist(), ["prior", "prior_fallback"])
        data.outcomes["label"] = 1.
        result = walk_recovery(data, data.features.index[-1:], plan, min_episodes=2).predictions
        self.assertEqual(result.fit_status.tolist(), ["prior", "prior_fallback"])
        self.assertTrue((result.p_recovery > result.p_continuation).all())
        self.assertTrue((result.p_unresolved > 0).all())

    def test_weighted_prior_uses_episode_mass_not_raw_row_count(self):
        labels = np.array([1, 1, 1, -1])
        p = class_prior(labels, np.array([1/3, 1/3, 1/3, 1.]))
        np.testing.assert_allclose(p, [2/5, 1/5, 2/5])

    def test_convergence_warning_is_not_silently_accepted(self):
        data = recovery_fixture()
        def warn(*args, **kwargs):
            warnings.warn("not converged", ConvergenceWarning)
        with patch.object(LogisticRegression, "fit", warn), self.assertRaises(ConvergenceWarning):
            walk_recovery(data, data.features.index[-1:], {"logit_price": ("price", (0.1,))}, min_episodes=2)

    def test_constant_decline_features_are_excluded(self):
        self.assertEqual([len(v) for v in FEATURE_SETS.values()], [2, 3, 4])
        self.assertEqual(set(FEATURE_SETS["term_structure"]) - set(FEATURE_SETS["vix_level"]), {"ts_ratio"})
        self.assertNotIn("dec_cond", FEATURE_SETS["term_structure"])
        self.assertNotIn("x_cond_ts", FEATURE_SETS["term_structure"])


class ExtensionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel, cls.features, cls.events, cls.settings = fixture()
        cls.result = run_extensions(cls.panel, cls.features, cls.events, cls.settings)

    def test_selection_uses_only_completed_design_labels(self):
        cutoff = pd.Timestamp(self.settings.periods.design_end)
        for family in (self.result.ridge, self.result.recovery):
            self.assertTrue((family.validation_predictions.label_end_date <= cutoff).all())
            self.assertTrue((family.selection.last_label_end <= cutoff).all())
        self.assertTrue((self.result.sensitivity_candidates.label_end_date <= cutoff).all())
        self.assertTrue((self.result.sensitivity_candidates.decision_date <= cutoff).all())

    def test_later_outcome_changes_cannot_change_selected_settings(self):
        panel = self.panel.copy()
        cutoff = self.settings.periods.design_end
        later = panel.index > cutoff
        panel.loc[later, "spy_tr_cc"] += 0.2
        panel.loc[later, "spy_adj_close"] *= 2
        panel["fwd_spy_xs"] = (panel.spy_tr_cc - panel.bil_tr_cc).shift(-2)
        result = run_extensions(panel, self.features, self.events, self.settings)
        for before, after in ((self.result.ridge, result.ridge), (self.result.recovery, result.recovery)):
            self.assertEqual(before.chosen, after.chosen)
            pd.testing.assert_frame_equal(before.validation_predictions, after.validation_predictions)
        pd.testing.assert_frame_equal(self.result.sensitivity_selection, result.sensitivity_selection)

    def test_classifier_selection_ties_choose_stronger_regularisation(self):
        p = self.result.recovery.candidates.copy()
        p[list(PROBABILITY_COLUMNS)] = 1/3
        chosen, _ = select_cs(p, self.result.recovery.outcomes, self.settings.periods.design_end)
        self.assertEqual(chosen, {name: 0.1 for name in FEATURE_SETS})
        row = p.index[p.model_id == "logit_price"][0]
        with self.assertRaisesRegex(ValueError, "identical"):
            select_cs(p.drop(row), self.result.recovery.outcomes, self.settings.periods.design_end)

    def test_exports_roundtrip_with_explicit_horizon_and_no_outcomes(self):
        with tempfile.TemporaryDirectory() as folder:
            out, report = Path(folder) / "signals", Path(folder) / "reports"
            write_extensions(self.result, self.settings, output_dir=out, report_dir=report)
            recovery = load_recovery_signal(path=out / "recovery/predictions.csv")
            returns = load_ten_day_signal(path=out / "ridge_h10/predictions.csv")
            self.assertTrue((recovery.horizon == 10).all())
            self.assertTrue((returns.horizon == 10).all())
            self.assertNotIn("label", recovery)
            self.assertNotIn("actual_excess_return", returns)
            self.assertTrue(recovery.index.isin(self.features.index[self.features.dec_cond == 1]).all())
            p = self.result.recovery.predictions
            before = p.loc[p.model_id == "logit_term_structure", list(PROBABILITY_COLUMNS)].to_numpy()
            np.testing.assert_array_equal(recovery[list(PROBABILITY_COLUMNS)].to_numpy(), before)
            with self.assertRaisesRegex(ValueError, "horizon=1"):
                load_signal(path=out / "ridge_h10/predictions.csv")
            path = out / "recovery/predictions.csv"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_recovery_predictions(path)

    def test_signal_validation_rejects_invalid_probabilities_and_future_labels(self):
        p = self.result.recovery.predictions
        invalid = [p.assign(p_recovery=2.), p.assign(p_unresolved=np.nan), p.assign(label=1),
                   p.assign(train_last_label_end_date=p.label_end_date),
                   p[p.model_id != "class_prior"], pd.concat([p, p.iloc[:1]])]
        for bad in invalid:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                validate_recovery(bad)

    def test_penalties_and_training_dates_match_for_term_structure_controls(self):
        p = self.result.recovery.predictions.set_index(["model_id", "decision_date"])
        a, b = p.loc["logit_vix_level"], p.loc["logit_term_structure_matched"]
        for column in ("C", "n_train", "n_episodes", "train_last_label_end_date"):
            pd.testing.assert_series_equal(a[column], b[column])
        self.assertTrue((a.train_last_label_end_date <= a.index).all())

    def test_invalid_settings_fail(self):
        for changes in ({"classifier_cs": ()}, {"ridge_alphas": (0.,)}, {"min_episodes": 0},
                        {"classifier_cs": (np.inf,)}, {"classifier_cs": (1., 1.)}):
            with self.assertRaises(ValueError):
                replace(self.settings, **changes)


if __name__ == "__main__":
    unittest.main()
