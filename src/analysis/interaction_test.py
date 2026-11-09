"""Does the VIX term structure on decline days predict the rebound? (explanatory test)

This is the small OLS + Newey-West test proposed in the feature-stage README
("Suggested Analysis" 1, commit 6565d4d, 2026-11-05), run as written. It is a
secondary, explanatory analysis next to the predictive ridge experiment. It was
added after the 2019-2025 model and backtest results were seen, so it is
reported in full and none of its settings were changed after running it.

Specifications (fixed before running):
  A (primary)   y = b0 + b1 ts_ratio + b2 dec_cond + b3 x_cond_ts
                x_cond_ts = dec_cond * (ts_ratio - 1)
  B (continuous) y = b0 + b1 ts_ratio + b2 dec_mom + b3 x_mom_ts
                x_mom_ts = dec_mom * (ts_ratio - 1)
Targets: fwd_spy_xs_1 and fwd_spy_xs_10 (SPY minus BIL, starting at close t+1).
Newey-West lags: max(h, 5) for an h-day target (overlapping windows for h = 10).
Periods by decision date: design 2014-2018, test 2019-2025, full 2014-2025.
Decisions whose target window ends after the period end are dropped, so
design-period estimates use no 2019 returns.

Hypotheses (two-sided tests):
  b3      : the term-structure slope differs on decline days (A) or with momentum (B)
  b1 + b3 : in A, the total effect of ts_ratio on a decline day
A positive b3 in A means declines that start in backwardation (ts_ratio > 1)
are followed by higher excess returns, i.e. more likely to rebound.

Run: python -m src.analysis.interaction_test
Output: reports/analysis/interaction_tests.csv, reports/analysis/interaction_tests.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.backtest.evaluate import ols_hac
from src.data.build import load_panel
from src.features.feature_build import load_features
from src.features.targets import load_targets

OUT = config.PROCESSED_DIR.parent.parent / "reports" / "analysis"
SPECS = {
    "A_decline_x_ts": ("ts_ratio", "dec_cond", "x_cond_ts"),
    "B_momentum_x_ts": ("ts_ratio", "dec_mom", "x_mom_ts"),
}
HORIZONS = (1, 10)
PERIODS = {
    "design_2014_2018": (config.SAMPLE_START, config.DESIGN_END),
    "test_2019_2025": (config.TEST_START, config.SAMPLE_END),
    "full_2014_2025": (config.SAMPLE_START, config.SAMPLE_END),
}


def window_end(index: pd.DatetimeIndex, h: int) -> pd.Series:
    """Realisation date of an h-day target decided on t: session t+1+h."""
    pos = np.arange(len(index)) + 1 + h
    ends = pd.Series(pd.NaT, index=index)
    ok = pos < len(index)
    ends[ok] = index[pos[ok]]
    return ends


def fit(y: pd.Series, X: pd.DataFrame, lags: int) -> dict:
    Xc = np.column_stack([np.ones(len(X)), X.to_numpy()])
    beta, _, V = ols_hac(y.to_numpy(), Xc, lags)
    se = np.sqrt(np.diag(V))
    df = len(y) - Xc.shape[1]
    p = lambda t: 2 * stats.t.sf(abs(t), df)
    out = {"n": len(y), "lags": lags}
    for k, name in enumerate(["const", *X.columns]):
        out[f"b_{name}"] = beta[k]
        out[f"t_{name}"] = beta[k] / se[k]
    b3 = X.columns[2]
    out["b3"], out["t_b3"], out["p_b3"] = beta[3], beta[3] / se[3], p(beta[3] / se[3])
    comb = beta[1] + beta[3]                         # b1 + b3
    se_c = np.sqrt(V[1, 1] + V[3, 3] + 2 * V[1, 3])
    out["b1_plus_b3"], out["t_b1_plus_b3"], out["p_b1_plus_b3"] = comb, comb / se_c, p(comb / se_c)
    return out


def run() -> pd.DataFrame:
    panel = load_panel()
    feat = load_features()
    targets = load_targets().reindex(feat.index)
    rows = []
    for h in HORIZONS:
        y_all = targets[f"fwd_spy_xs_{h}"]
        ends = window_end(panel.index, h).reindex(feat.index)
        for spec, cols in SPECS.items():
            for period, (a, b) in PERIODS.items():
                d = pd.concat([y_all.rename("y"), feat[list(cols)], ends.rename("end"),
                               feat["dec_cond"].rename("decline")], axis=1).loc[a:b].dropna()
                # design windows must close by DESIGN_END; later periods may use the
                # January 2026 buffer that settles December 2025 decisions
                cutoff = pd.Timestamp(b) if b == config.DESIGN_END else panel.index[-1]
                d = d[d["end"] <= cutoff]
                r = fit(d["y"], d[list(cols)], max(h, 5))
                r.update({"spec": spec, "horizon": h, "period": period,
                          "n_decline_days": int(d["decline"].sum())})
                rows.append(r)
    lead = ["spec", "horizon", "period", "n", "n_decline_days", "lags",
            "b3", "t_b3", "p_b3", "b1_plus_b3", "t_b1_plus_b3", "p_b1_plus_b3"]
    tab = pd.DataFrame(rows)
    return tab[lead + [c for c in tab.columns if c not in lead]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tab = run()
    tab.to_csv(OUT / "interaction_tests.csv", index=False)
    show = tab[["spec", "horizon", "period", "n", "n_decline_days", "b3", "t_b3", "p_b3",
                "b1_plus_b3", "t_b1_plus_b3", "p_b1_plus_b3"]].copy()
    for c in ("b3", "b1_plus_b3"):
        show[c] = show[c].map("{:+.4f}".format)
    for c in ("t_b3", "p_b3", "t_b1_plus_b3", "p_b1_plus_b3"):
        show[c] = show[c].map("{:.2f}".format)
    md = [
        "# Term structure x decline interaction tests",
        "",
        "Generated by `python -m src.analysis.interaction_test`. OLS with Newey-West",
        "standard errors (lags = max(h, 5)); two-sided p-values. Specification A:",
        "y on ts_ratio, dec_cond, x_cond_ts. Specification B: y on ts_ratio, dec_mom,",
        "x_mom_ts. b3 is the interaction; b1 + b3 is the ts_ratio effect on a decline",
        "day (meaningful for A). Coefficients are decimal excess returns per unit of",
        "ts_ratio (A) or per unit of dec_mom x (ts_ratio - 1) (B); ts_ratio has a",
        "standard deviation of roughly 0.07, so multiply by 0.1 for a typical move.",
        "",
        show.to_markdown(index=False),
        "",
        "Reading the signs. A: b3 > 0 supports the rebound hypothesis (declines that",
        "start in backwardation are followed by higher returns). B: on a sharp decline",
        "dec_mom < 0, and in backwardation ts_ratio - 1 > 0, so x_mom_ts < 0; b3 < 0",
        "therefore supports the hypothesis. In B, b1 + b3 is the ts_ratio effect at",
        "dec_mom = +1 and is not of interest. Twelve regressions are reported; at the",
        "5% level about one false positive is expected by chance.",
        "",
        "Secondary, explanatory analysis proposed in the feature-stage README on",
        "2026-11-05 and run unchanged; added after the 2019-2025 forecast and",
        "backtest results were seen. Decline days cluster within episodes, so the",
        "effective sample is the number of episodes, not decline days.",
    ]
    (OUT / "interaction_tests.md").write_text("\n".join(md), encoding="utf-8")
    print(show.to_markdown(index=False))


if __name__ == "__main__":
    main()
