"""Creating the decline events and their recovery / continuation labels.

Trigger on day t (both must hold, using only data dated t or earlier):
  dec_dd  = adj_close[t] / max(adj_close[t-19 .. t]) - 1                <= -S
  dec_mom = (adj_close[t] / adj_close[t-5] - 1) / (dec_vol[t] * sqrt(5)) <= -P
  dec_vol = std of SPY daily total returns over t-19 .. t

Events: a trigger opens an observation window of H=10 trading days (the window size
is left to our decision), and triggers inside that window are ignored, so it won't create overlaps. 

Labels (outcomes, hence the fwd_ prefix). 
The entry is the close of t+1 (next-close execution, see DATA.md), then the SPY total-return close 
is followed over t+2 .. t+1+H against barriers set at t:
+/- ev_barrier = K * dec_vol[t] * sqrt(H) denotes the upper and lower barriers, and we make the following
assignments,
  +1 recovery      the close reaches the upper barrier first
  -1 continuation  the close reaches the lower barrier first
   0 unresolved    neither barrier within H days
fwd_ev_label_bin resolves the 0-case by taking the sign of the 10-day return.

Role: models are trained on every trading day, while the events and labels here are for evaluation only. 
They mainly helps to answer the question if out-of-sample predictions made on event days separate recoveries from continuations?

Run:  python -m src.features.events
Output: data/processed/events.csv (+ events.parquet), indexed by date like the panel,
        columns dec_cond, ev_start, ev_barrier, fwd_ev_label, fwd_ev_label_bin, fwd_ev_days
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data.build import load_panel

S = 0.02          # close at least 2% below the 20-day high
P = 1.0           # 5-day return at least 1 sd below zero
H = 10            # observation window (number of trading days)
K = 0.75          # barrier half-width in units of dec_vol * sqrt(H)

DD_LOOKBACK = 20
MOM_DAYS = 5
VOL_WINDOW = 20


def decline_signals(panel: pd.DataFrame) -> pd.DataFrame:
    a = panel["spy_adj_close"]
    vol = panel["spy_tr_cc"].rolling(VOL_WINDOW).std() # find the rolling volatility in past 20 days
    return pd.DataFrame({
        "dec_dd": a / a.rolling(DD_LOOKBACK).max() - 1,
        "dec_mom": (a / a.shift(MOM_DAYS) - 1) / (vol * np.sqrt(MOM_DAYS)),
        "dec_vol": vol,
    })


def event_starts(cond: pd.Series, h: int) -> pd.Series:
    """First trigger starts an event; triggers in the next h days are skipped."""
    out = np.zeros(len(cond), dtype=bool)
    nxt = 0
    for i in np.flatnonzero(cond.to_numpy()):
        if i >= nxt:
            out[i] = True
            nxt = i + h + 1
    return pd.Series(out, index=cond.index)


def barrier_labels(panel: pd.DataFrame, starts: pd.Series, width: pd.Series, h: int) -> pd.DataFrame:
    spy = panel["spy_adj_close"].to_numpy()
    n = len(spy)
    cols = {c: np.full(n, np.nan) for c in ["fwd_ev_label", "fwd_ev_label_bin", "fwd_ev_days"]}
    for i in np.flatnonzero(starts.to_numpy()):
        if i + 1 + h >= n:                      # window runs past the end of the data
            continue
        w = width.iloc[i]
        path = spy[i + 2:i + 2 + h] / spy[i + 1] - 1
        up = np.flatnonzero(path >= w)
        dn = np.flatnonzero(path <= -w)
        j_up = up[0] if len(up) else h # find the first day the path crosses the upper barrier
        j_dn = dn[0] if len(dn) else h
        label, j = (1, j_up) if j_up < j_dn else (-1, j_dn) if j_dn < j_up else (0, h - 1)
        cols["fwd_ev_label"][i] = label
        cols["fwd_ev_label_bin"][i] = label if label != 0 else (1 if path[-1] > 0 else -1)
        cols["fwd_ev_days"][i] = j + 1
    return pd.DataFrame(cols, index=panel.index)


def build(panel: pd.DataFrame, s: float = S, p: float = P, h: int = H, k: float = K) -> pd.DataFrame:
    sig = decline_signals(panel) # this creates the dec_dd, dec_mom, and dec_vol columns
    sig["dec_cond"] = (sig["dec_dd"] <= -s) & (sig["dec_mom"] <= -p) # boolean series
    sig["ev_start"] = event_starts(sig["dec_cond"], h) # clears out overlapping events to get separete ones
    sig["ev_barrier"] = (k * sig["dec_vol"] * np.sqrt(h)).where(sig["ev_start"])
    ev = sig[["dec_cond", "ev_start", "ev_barrier"]]   # dec_dd/mom/vol live in the features file
    return ev.join(barrier_labels(panel, sig["ev_start"], sig["ev_barrier"], h))


def load_events() -> pd.DataFrame:
    path = config.PROCESSED_DIR / "events.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.read_csv(config.PROCESSED_DIR / "events.csv", parse_dates=["date"], index_col="date")


def main() -> None:
    panel = load_panel()
    ev = build(panel)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ev.to_csv(config.PROCESSED_DIR / "events.csv", float_format="%.10g")
    try:
        ev.to_parquet(config.PROCESSED_DIR / "events.parquet")
    except ImportError:
        pass

    e = ev[panel["in_sample"] & ev["ev_start"]]
    print(f"S={S:.0%} P={P} H={H} K={K}: {len(e)} events in sample, "
          f"{e['fwd_ev_label'].isna().sum()} without a full window")
    print(e["fwd_ev_label"].value_counts().sort_index().rename({-1: "continuation", 0: "unresolved",
                                                                  1: "recovery"}).to_string())
    # Barrier width only sets the labelling scale; shown for the chosen events.
    rows = {}
    for kk in [0.5, 0.75, 1.0, 1.25, 1.5]:
        lab = build(panel, k=kk).loc[e.index, "fwd_ev_label"].dropna()
        rows[kk] = lab.value_counts(normalize=True).reindex([1, 0, -1], fill_value=0).round(2).to_dict()
    print("\nlabel shares by barrier width K (1 recovery, 0 unresolved, -1 continuation)")
    print(pd.DataFrame(rows).T.rename_axis("K").to_markdown())


if __name__ == "__main__":
    main()
