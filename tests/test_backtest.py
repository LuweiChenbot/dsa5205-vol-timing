"""Checks for the Member 4 backtest engine, rules and statistics.

Run: python -m unittest tests.test_backtest -v
"""
import unittest

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest, to_weights
from src.backtest.evaluate import capm_alpha, sharpe_diff_test, stationary_bootstrap_idx
from src.data.build import load_panel
from src.metrics import apply_weights


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = load_panel()
        cls.dates = cls.panel.loc["2019-01-02":"2019-12-31"].index
        rng = np.random.default_rng(0)
        cls.w01 = pd.Series(rng.integers(0, 2, len(cls.dates)).astype(float), index=cls.dates)

    def test_matches_apply_weights_for_binary_weights(self):
        bt = run_backtest(self.panel, self.w01, cost_bps=3.0)
        ref = apply_weights(self.panel, self.w01, cost_bps=3.0)
        common = bt.index[1:].intersection(ref.index)
        # apply_weights only sees weights from t0; it ends one day early (no NaN-free t+2 for last w)
        self.assertGreater(len(common), 240)
        np.testing.assert_allclose(bt.loc[common, "net"], ref.loc[common], atol=1e-15)

    def test_full_weight_is_buy_and_hold_after_entry(self):
        w = pd.Series(1.0, index=self.dates)
        bt = run_backtest(self.panel, w, cost_bps=0.0)
        np.testing.assert_allclose(bt["net"].iloc[1:], self.panel.loc[bt.index[1:], "spy_tr_cc"])
        self.assertAlmostEqual(bt["net"].iloc[0], self.panel.loc[bt.index[0], "bil_tr_cc"])
        bt5 = run_backtest(self.panel, w, cost_bps=5.0)
        self.assertAlmostEqual(bt5["trade"].sum(), 1.0)      # only the entry trade

    def test_dates_follow_next_close_convention(self):
        bt = run_backtest(self.panel, self.w01)
        idx = self.panel.index
        self.assertEqual(bt.index[0], idx[idx.get_loc(self.dates[0]) + 1])
        self.assertEqual(bt.index[-1], idx[idx.get_loc(self.dates[-1]) + 2])

    def test_weight_change_only_affects_later_days(self):
        """w[t] may only move returns dated t+1 (its trade cost) and later."""
        t = self.dates[100]
        w2 = self.w01.copy()
        w2[t] = 1 - w2[t]
        a = run_backtest(self.panel, self.w01, cost_bps=5.0)["net"]
        b = run_backtest(self.panel, w2, cost_bps=5.0)["net"]
        changed = a.index[~np.isclose(a, b, rtol=0, atol=1e-15)]
        idx = self.panel.index
        self.assertEqual(changed.min(), idx[idx.get_loc(t) + 1])
        gross_a = run_backtest(self.panel, self.w01, cost_bps=0.0)["gross"]
        gross_b = run_backtest(self.panel, w2, cost_bps=0.0)["gross"]
        changed_gross = gross_a.index[~np.isclose(gross_a, gross_b, rtol=0, atol=1e-15)]
        self.assertEqual(list(changed_gross), [idx[idx.get_loc(t) + 2]])

    def test_drift_cost_for_fractional_weight(self):
        w = pd.Series(0.5, index=self.dates[:3])
        bt = run_backtest(self.panel, w, cost_bps=10.0)
        rs, rb = bt["spy"].iloc[1], bt["bil"].iloc[1]
        g = 0.5 * rs + 0.5 * rb
        drifted = 0.5 * (1 + rs) / (1 + g)
        self.assertAlmostEqual(bt["trade"].iloc[1], abs(0.5 - drifted))
        self.assertAlmostEqual(bt["net"].iloc[1], g - 10e-4 * abs(0.5 - drifted))
        self.assertAlmostEqual(bt["trade"].iloc[0], 0.5)    # entry from cash

    def test_rejects_gaps(self):
        with self.assertRaises(ValueError):
            run_backtest(self.panel, self.w01.drop(self.dates[5]))


class RuleTests(unittest.TestCase):
    def test_rules(self):
        i = pd.date_range("2020-01-01", periods=4)
        pred = pd.Series([-1e-3, 0.0, 2e-4, 5e-3], index=i)
        vol = pd.Series(0.01, index=i)
        dec = pd.Series([0, 1, 1, 0], index=i)
        self.assertEqual(list(to_weights(pred, "sign")), [0, 0, 1, 1])
        self.assertEqual(list(to_weights(pred, "mv", vol=vol, gamma=5)), [0, 0, 0.4, 1.0])
        self.assertEqual(list(to_weights(pred, "mv", vol=vol, gamma=5, w_max=1.5)), [0, 0, 0.4, 1.5])
        self.assertEqual(list(to_weights(pred, "decline_overlay", decline=dec)), [1, 0, 1, 1])
        with self.assertRaises(ValueError):
            to_weights(pred, "nope")


class StatsTests(unittest.TestCase):
    def test_capm_recovers_parameters(self):
        rng = np.random.default_rng(1)
        x = pd.Series(rng.normal(0, 0.01, 2000))
        a, b, t = capm_alpha(0.001 + 2 * x, x)
        self.assertAlmostEqual(a, 0.001)
        self.assertAlmostEqual(b, 2.0)

    def test_bootstrap_indices_are_wrapped_blocks(self):
        i = stationary_bootstrap_idx(5000, 20, np.random.default_rng(3))
        self.assertTrue(((i >= 0) & (i < 5000)).all())
        steps = (np.diff(i) % 5000) == 1
        self.assertAlmostEqual(steps.mean(), 1 - 1 / 20, delta=0.02)   # mean block ~20

    def test_sharpe_test_identical_and_shifted(self):
        rng = np.random.default_rng(2)
        x = pd.Series(rng.normal(3e-4, 0.01, 1500))
        self.assertEqual(sharpe_diff_test(x, x)["sharpe_diff_p"], 1.0)
        r = sharpe_diff_test(x + 0.002, x, reps=300)
        self.assertGreater(r["sharpe_diff_vs_spy"], 0)
        self.assertLess(r["sharpe_diff_p"], 0.01)


if __name__ == "__main__":
    unittest.main()
