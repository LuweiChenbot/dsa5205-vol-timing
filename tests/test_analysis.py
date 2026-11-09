"""Checks for the OLS + Newey-West interaction test.

Run: python -m unittest tests.test_analysis -v
"""
import unittest

import numpy as np
import pandas as pd

from src.analysis.interaction_test import run, window_end
from src.backtest.evaluate import ols_hac


class OlsHacTests(unittest.TestCase):
    def test_lag0_equals_white_hc0(self):
        rng = np.random.default_rng(4)
        X = np.column_stack([np.ones(500), rng.normal(size=(500, 2))])
        y = X @ np.array([0.1, 2.0, -1.0]) + rng.normal(size=500) * (1 + np.abs(X[:, 1]))
        b, u, V = ols_hac(y, X, 0)
        XtXi = np.linalg.inv(X.T @ X)
        np.testing.assert_allclose(V, XtXi @ (X.T * u**2) @ X @ XtXi)
        np.testing.assert_allclose(b, np.linalg.solve(X.T @ X, X.T @ y))

    def test_hac_widens_se_for_autocorrelated_errors(self):
        rng = np.random.default_rng(5)
        n = 3000
        x = np.convolve(rng.normal(size=n + 9), np.ones(10), "valid")   # MA(9) regressor
        e = np.convolve(rng.normal(size=n + 9), np.ones(10), "valid")   # MA(9) errors
        X = np.column_stack([np.ones(n), x])
        _, _, V0 = ols_hac(x * 0.5 + e, X, 0)
        _, _, V10 = ols_hac(x * 0.5 + e, X, 10)
        self.assertGreater(V10[1, 1], 2 * V0[1, 1])


class InteractionTests(unittest.TestCase):
    def test_window_end(self):
        i = pd.bdate_range("2020-01-01", periods=15)
        e = window_end(i, 10)
        self.assertEqual(e.iloc[0], i[11])
        self.assertTrue(e.iloc[-11:].isna().all())

    def test_design_period_uses_no_test_returns(self):
        tab = run()
        self.assertEqual(len(tab), 2 * 2 * 3)
        d = tab[tab.period == "design_2014_2018"]
        f = tab[tab.period == "full_2014_2025"]
        t = tab[tab.period == "test_2019_2025"]
        # h=10 design rows lose the last 11 decisions of 2018 to the cutoff
        n_design = d.set_index(["spec", "horizon"]).n
        self.assertEqual(n_design.loc[("A_decline_x_ts", 1)] - n_design.loc[("A_decline_x_ts", 10)], 9)
        self.assertTrue((d.n.values + t.n.values <= f.n.values).all())
        self.assertTrue(np.isfinite(tab[["b3", "t_b3", "p_b3"]].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
