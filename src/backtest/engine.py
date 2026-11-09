"""Score-to-weight rules and the execution engine (Member 4).

Timing (DATA.md): a weight w[t] is decided after the close of t, filled at the
close of t+1 and earns close(t+1) -> close(t+2). This module extends
src.metrics.apply_weights with the two conventions that helper leaves open:

* the initial entry trade is charged (the book starts at INITIAL_WEIGHT), and
* turnover is measured against the DRIFTED weight, so holding a fractional
  weight pays for the daily rebalancing back to target.

For weights in {0, 1} there is no drift and the result equals apply_weights
on every day after the first (tested in tests/test_backtest.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import config as bc


def to_weights(pred: pd.Series, rule: str, vol: pd.Series | None = None,
               decline: pd.Series | None = None, gamma: float = bc.GAMMA,
               w_max: float = bc.W_MAX) -> pd.Series:
    """Map decimal excess-return forecasts to SPY weights (decision-date index).

    sign            : 1 if the forecast is positive, else 0 (all BIL).
    mv              : pred / (gamma * vol^2), clipped to [0, w_max]
                      (single-asset mean-variance weight).
    decline_overlay : fully invested by default; on decline-trigger days only,
                      follow the sign rule. Isolates the research question:
                      does the model help decide whether to stay in after a decline?
    """
    if rule == "sign":
        w = (pred > 0).astype(float)
    elif rule == "mv":
        if vol is None:
            raise ValueError("mv rule needs vol")
        w = (pred / (gamma * vol.reindex(pred.index) ** 2)).clip(0.0, w_max)
    elif rule == "decline_overlay":
        if decline is None:
            raise ValueError("decline_overlay rule needs the decline flag")
        d = decline.reindex(pred.index).astype(bool)
        w = pd.Series(np.where(d, (pred > 0).astype(float), 1.0), index=pred.index)
    else:
        raise ValueError(f"unknown rule {rule!r}; choose from {bc.RULES}")
    if w.isna().any():
        raise ValueError("weights contain NaN")
    return w.rename("weight")


def run_backtest(panel: pd.DataFrame, w: pd.Series, cost_bps: float = bc.COST_BPS,
                 initial_weight: float = bc.INITIAL_WEIGHT,
                 returns: tuple[str, str] = bc.RETURNS_MAIN) -> pd.DataFrame:
    """Daily P&L of a SPY/BIL book driven by decision-date weights.

    w must cover a contiguous block of panel sessions. Output rows run from the
    first execution date (t0+1, still holding `initial_weight`) to the last
    label-end date (tN+2), indexed by REALISATION date, with columns:
      net, gross  : daily simple returns after / before costs
      held        : SPY weight held over the day ending on that date
      trade       : |target - drifted weight| traded at that date's close
      spy, bil    : the asset returns used
    """
    idx = panel.index
    pos = idx.get_indexer(w.index)
    if (pos < 0).any() or not (np.diff(pos) == 1).all():
        raise ValueError("weights must cover contiguous panel sessions")
    start, n = pos[0] + 1, len(w)
    dates = idx[start:start + n + 1]               # execution day 0 .. last label end
    if len(dates) != n + 1:
        raise ValueError("panel too short: last decision has no realised return")
    rs = panel[returns[0]].to_numpy()[start:start + n + 1]
    rb = panel[returns[1]].to_numpy()[start:start + n + 1]
    if np.isnan(rs).any() or np.isnan(rb).any():
        raise ValueError("missing asset returns in the holding window")

    target = np.append(w.to_numpy(dtype=float), np.nan)   # set at close of dates[k]
    held = np.concatenate([[initial_weight], target[:-1]])  # during the day ending dates[k]
    gross = held * rs + (1 - held) * rb
    drifted = held * (1 + rs) / (1 + gross)
    trade = np.abs(target - drifted)
    trade[-1] = 0.0                                 # no trade after the last label end
    net = gross - cost_bps / 1e4 * trade
    return pd.DataFrame({"net": net, "gross": gross, "held": held, "trade": trade,
                         "spy": rs, "bil": rb}, index=dates)
