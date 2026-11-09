"""Performance metrics, significance checks and event analysis for backtests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.backtest import config as bc
from src.metrics import perf_stats

T = config.TRADING_DAYS


def strategy_stats(bt: pd.DataFrame, spy_bt: pd.DataFrame, inference: bool = True) -> dict:
    """perf_stats plus trading and benchmark-relative statistics.

    bt, spy_bt: outputs of run_backtest over the SAME realisation dates.
    inference : add CAPM alpha (Newey-West) and the bootstrap Sharpe test.
    """
    if not bt.index.equals(spy_bt.index):
        raise ValueError("strategy and benchmark must share realisation dates")
    years = len(bt) / T
    out = perf_stats(bt["net"], rf=bt["bil"])
    out.update({
        "avg_weight": bt["held"].mean(),
        "pct_days_in_spy": (bt["held"] > 0).mean(),
        "turnover_pa": bt["trade"].sum() / years,
        "cost_drag_pa": (bt["gross"] - bt["net"]).mean() * T,
    })
    if not inference:
        return out
    a, b, t_a = capm_alpha(bt["net"] - bt["bil"], spy_bt["net"] - bt["bil"])
    out.update({"alpha_pa": a * T, "alpha_t_nw": t_a, "beta": b})
    out.update(sharpe_diff_test(bt["net"] - bt["bil"], spy_bt["net"] - bt["bil"]))
    return out


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS of y on X (caller adds the constant) with a Newey-West (Bartlett) covariance.

    Returns (coefficients, residuals, covariance). No small-sample correction.
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    xu = X * u[:, None]
    S = xu.T @ xu
    for L in range(1, lags + 1):
        g = xu[L:].T @ xu[:-L]
        S += (1 - L / (lags + 1)) * (g + g.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    return beta, u, XtX_inv @ S @ XtX_inv


def capm_alpha(y: pd.Series, x: pd.Series, lags: int = bc.NW_LAGS) -> tuple[float, float, float]:
    """OLS y = a + b x with a Newey-West (Bartlett) t-statistic for a."""
    X = np.column_stack([np.ones(len(x)), x.to_numpy()])
    yv = y.to_numpy()
    beta, u, V = ols_hac(yv, X, lags)
    if u.std() < 1e-6 * max(yv.std(), 1e-12):      # (near-)exact fit, e.g. SPY on itself
        return beta[0], beta[1], np.nan
    se_a = np.sqrt(V[0, 0])
    return beta[0], beta[1], (beta[0] / se_a if se_a > 0 else np.nan)


def _sharpe(x: np.ndarray) -> float:
    s = x.std(ddof=1)
    return x.mean() / s * np.sqrt(T) if s > 0 else 0.0


def stationary_bootstrap_idx(n: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices with mean block length `block`."""
    new_block = rng.random(n) < 1 / block
    new_block[0] = True
    starts = rng.integers(n, size=n)
    block_id = np.cumsum(new_block) - 1
    first = np.flatnonzero(new_block)              # position where each block begins
    offset = np.arange(n) - first[block_id]
    return (starts[first][block_id] + offset) % n


def sharpe_diff_test(xs_a: pd.Series, xs_b: pd.Series, reps: int = bc.BOOT_REPS,
                     block: float = bc.BOOT_BLOCK, seed: int = bc.SEED) -> dict:
    """Paired stationary-bootstrap test of Sharpe(a) - Sharpe(b).

    Days are resampled jointly so the correlation between the two strategies is
    kept. The two-sided p-value uses the bootstrap distribution centred at the
    observed difference (null: difference = 0).
    """
    a, b = xs_a.to_numpy(), xs_b.to_numpy()
    diff = _sharpe(a) - _sharpe(b)
    if np.allclose(a, b):
        return {"sharpe_diff_vs_spy": 0.0, "sharpe_diff_lo": 0.0, "sharpe_diff_hi": 0.0,
                "sharpe_diff_p": 1.0}
    rng = np.random.default_rng(seed)
    boots = np.empty(reps)
    for r in range(reps):
        i = stationary_bootstrap_idx(len(a), block, rng)
        boots[r] = _sharpe(a[i]) - _sharpe(b[i])
    p = np.mean(np.abs(boots - diff) >= abs(diff))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"sharpe_diff_vs_spy": diff, "sharpe_diff_lo": lo, "sharpe_diff_hi": hi,
            "sharpe_diff_p": p}


def period_stats(bt: pd.DataFrame, periods: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows = {}
    for name, (a, b) in periods.items():
        part = bt.loc[a:b]
        rows[name] = perf_stats(part["net"], rf=part["bil"]) | {"avg_weight": part["held"].mean()}
    return pd.DataFrame(rows).T


def yearly_stats(bt: pd.DataFrame, min_days: int = 20) -> pd.DataFrame:
    """Calendar years by realisation date. A short trailing stub (the January
    buffer that settles December decisions) is folded into the previous year."""
    years = sorted(set(bt.index.year))
    periods = {str(y): (f"{y}-01-01", f"{y}-12-31") for y in years}
    last = str(years[-1])
    if len(years) > 1 and len(bt.loc[last]) < min_days:
        prev = str(years[-2])
        periods[prev] = (periods[prev][0], periods.pop(last)[1])
    return period_stats(bt, periods)


def event_rows(bt: pd.DataFrame, spy_bt: pd.DataFrame, w: pd.Series,
               events: pd.DataFrame, panel_index: pd.DatetimeIndex,
               h: int = bc.EVENT_WINDOW) -> pd.DataFrame:
    """One row per decline event inside the decision window.

    w_event   : weight decided on the event day t (first trade after the trigger)
    w_window  : mean weight decided over t .. t+h-1
    strat_ret : strategy net return over the h holding days t+2 .. t+h+1
    spy_ret   : SPY buy-and-hold return over the same days
    Labels are outcomes (fwd_ prefix) and are used for evaluation only.
    """
    ev = events[events["ev_start"]].loc[w.index.min():w.index.max()]
    rows = []
    for t, e in ev.iterrows():
        p = panel_index.get_loc(t)
        hold_dates = panel_index[p + 2:p + 2 + h]
        dec_dates = panel_index[p:p + h]
        if hold_dates[-1] > bt.index[-1] or dec_dates[-1] > w.index[-1]:
            continue                                  # window runs past the sample
        rows.append({
            "event_date": t,
            "label": int(e["fwd_ev_label"]),
            "w_event": w.loc[t],
            "w_window": w.loc[dec_dates].mean(),
            "strat_ret": (1 + bt.loc[hold_dates, "net"]).prod() - 1,
            "spy_ret": (1 + spy_bt.loc[hold_dates, "net"]).prod() - 1,
        })
    return pd.DataFrame(rows)


def event_summary(rows: pd.DataFrame) -> dict:
    """Separation: mean weight in recoveries minus in continuations (>0 is useful)."""
    rec, con = rows[rows.label == 1], rows[rows.label == -1]
    return {
        "n_events": len(rows), "n_recovery": len(rec), "n_continuation": len(con),
        "w_event_recovery": rec.w_event.mean(), "w_event_continuation": con.w_event.mean(),
        "w_event_separation": rec.w_event.mean() - con.w_event.mean(),
        "w_window_separation": rec.w_window.mean() - con.w_window.mean(),
        "excess_vs_spy_recovery": (rec.strat_ret - rec.spy_ret).mean(),
        "excess_vs_spy_continuation": (con.strat_ret - con.spy_ret).mean(),
        "excess_vs_spy_all": (rows.strat_ret - rows.spy_ret).mean(),
    }
