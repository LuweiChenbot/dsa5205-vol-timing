"""Shared performance metrics and the execution helper.

Every strategy and benchmark should be scored with these functions so the
comparison in the report is like for like.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS


def perf_stats(r: pd.Series, rf: pd.Series | None = None) -> dict:
    """Annualised stats for a daily simple-return series.

    ann_return : geometric (CAGR)
    ann_vol    : std of daily returns * sqrt(252)
    sharpe     : mean daily excess return over rf (BIL) / std of excess * sqrt(252)
    max_dd     : worst peak-to-trough fall of the cumulative wealth curve
    """
    r = r.dropna()
    if rf is None:
        xs = r
    else:
        xs = (r - rf.reindex(r.index)).dropna()
    wealth = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    return {
        "ann_return": wealth.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "ann_vol": r.std(ddof=1) * np.sqrt(TRADING_DAYS),
        "sharpe": xs.mean() / xs.std(ddof=1) * np.sqrt(TRADING_DAYS) if xs.std() > 0 else np.nan,
        "max_dd": (wealth / wealth.cummax() - 1).min(),
        "n_days": len(r),
    }


def stats_by_year(r: pd.Series, rf: pd.Series | None = None) -> pd.DataFrame:
    rows = {y: perf_stats(g, rf) for y, g in r.dropna().groupby(r.dropna().index.year)}
    return pd.DataFrame(rows).T


def apply_weights(panel: pd.DataFrame, w: pd.Series, cost_bps: float = 0.0) -> pd.Series:
    """Daily strategy returns under the next-close convention.

    w        : target SPY weight indexed by DECISION date t (rest held in BIL).
    Mechanics: the order is filled at the close of t+1, so the holding over
               close(d) -> close(d+1) is h[d] = w[d-1]. The return dated d is
               h[d-1] * spy_tr_cc[d] + (1 - h[d-1]) * bil_tr_cc[d]
               minus cost_bps * |h[d] - h[d-1]| for the trade done at close d.
    Returns  : series indexed by REALISATION date, comparable with the
               buy-and-hold benchmark returns.
    """
    w = w.reindex(panel.index)
    h = w.shift(1)                       # held after the close of d
    h_prev = h.shift(1)                  # held during close(d-1) -> close(d)
    gross = h_prev * panel["spy_tr_cc"] + (1 - h_prev) * panel["bil_tr_cc"]
    cost = cost_bps / 1e4 * (h - h_prev).abs().fillna(0)
    return (gross - cost).dropna().rename("strategy")


def turnover(w: pd.Series) -> float:
    """Average annual one-way turnover of the SPY weight."""
    return w.diff().abs().mean() * TRADING_DAYS
