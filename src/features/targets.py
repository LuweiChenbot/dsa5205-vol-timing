"""Multi-day prediction targets.

fwd_spy_xs_{h}[t] = SPY total return minus BIL total return, compounded over the h
holding days that follow a decision at t: close(t+1) -> close(t+1+h). 
h = 1 will simply reproduces the panel's fwd_spy_xs.

Overlap: consecutive h-day targets share h-1 days. A model refitted at date T may
only use rows with t <= T - 1 - h (the label is realised at close t+1+h), and
validation folds need a gap of at least h + 1 days.

Run:  python -m src.features.targets
Output: data/processed/targets.csv (+ targets.parquet), indexed by date like the panel
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data.build import load_panel

HORIZONS = (1, 5, 10)


def forward_return(r: pd.Series, h: int) -> pd.Series:
    """Compound daily returns dated t+2 .. t+1+h onto row t.

    Each window is summed on its own (no running sum), so a row depends only on
    the returns inside its window, bit for bit.
    """
    lr = np.log1p(r.to_numpy())
    out = np.full(len(lr), np.nan)
    if len(lr) >= h + 2:
        out[:len(lr) - h - 1] = np.lib.stride_tricks.sliding_window_view(lr[2:], h).sum(axis=1)
    return pd.Series(np.expm1(out), index=r.index)


def build_targets(panel: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    return pd.DataFrame({
        f"fwd_spy_xs_{h}": forward_return(panel["spy_tr_cc"], h) - forward_return(panel["bil_tr_cc"], h)
        for h in horizons
    })


def load_targets() -> pd.DataFrame:
    path = config.PROCESSED_DIR / "targets.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.read_csv(config.PROCESSED_DIR / "targets.csv", parse_dates=["date"], index_col="date")


def main() -> None:
    panel = load_panel()
    t = build_targets(panel)
    out = config.PROCESSED_DIR
    t.to_csv(out / "targets.csv", float_format="%.10g")
    try:
        t.to_parquet(out / "targets.parquet")
    except ImportError:
        pass
    s = t[panel["in_sample"]]
    print(f"wrote {list(t.columns)} to {out.relative_to(config.ROOT)}; "
          f"in-sample NaN: {s.isna().sum().to_dict()}")


if __name__ == "__main__":
    main()
