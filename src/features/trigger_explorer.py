"""Sweep the decline-event parameters (s, p, H) and count events per combination
to decide an appropriate one.

s : drawdown threshold, close <= (1 - s) x 20-day high
p : momentum threshold, 5-day return / (20-day vol x sqrt 5) <= -p, representing a z-score of the 5-day return
H : observation window; triggers within H days of an event start are skipped

Run:  python -m src.features.trigger_explorer
Output: reports/features/trigger_sweep.csv
"""
from __future__ import annotations

from itertools import product

import pandas as pd

from src import config
from src.data.build import load_panel
from src.features import events

S_GRID = [0.02, 0.03, 0.04, 0.05]
P_GRID = [0.5, 1.0, 1.5, 2.0]
H_GRID = [5, 10, 15, 20]
OUT = config.REPORT_DIR.parent / "features" / "trigger_sweep.csv"


def sweep(panel: pd.DataFrame) -> pd.DataFrame:
    sig = events.decline_signals(panel)
    ins = panel["in_sample"]
    rows = []
    for s, p in product(S_GRID, P_GRID):
        cond = (sig["dec_dd"] <= -s) & (sig["dec_mom"] <= -p)
        for h in H_GRID:
            st = events.event_starts(cond, h)[ins]
            per_year = st.groupby(st.index.year).sum()
            rows.append({"s": s, "p": p, "H": h, "trigger_days": int(cond[ins].sum()),
                         "events": int(st.sum()), "per_year": per_year.mean(),
                         "zero_years": int((per_year == 0).sum()),
                         "min_year": int(per_year.min()), "max_year": int(per_year.max())})
    return pd.DataFrame(rows)


def main() -> None:
    df = sweep(load_panel())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.4g")

    # events (years with no event), one row per (s, p), one column per H
    cell = df["events"].astype(str) + " (" + df["zero_years"].astype(str) + ")"
    table = df.assign(cell=cell).pivot(index=["s", "p"], columns="H", values="cell")
    table.columns = [f"H={h}" for h in table.columns]
    table.insert(0, "trigger_days", df.groupby(["s", "p"])["trigger_days"].first())
    table.index = [f"s={s:.0%} p={p:g}" + ("  <- chosen" if (s, p) == (events.S, events.P) else "")
                   for s, p in table.index]
    print("events in sample, (years with no event)")
    print(table.to_markdown())
    print(f"\nchosen: S={events.S:.0%} P={events.P} H={events.H}; full table in {OUT.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
