"""Leakage and logic checks for the decline events, on the processed panel.

Run:  python -m tests.test_events
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.build import load_panel
from src.features import events


def main() -> None:
    panel = load_panel()
    full = events.build(panel)
    known = [c for c in full.columns if not c.startswith("fwd_")]

    # 1. Columns known at t do not change when everything after a cut date is removed.
    for frac in (0.3, 0.6, 0.9):
        cut = panel.index[int(len(panel) * frac)]
        part = events.build(panel.loc[:cut])
        pd.testing.assert_frame_equal(full.loc[:cut, known], part[known], check_exact=True)

        # 2. Labels use closes up to t+1+H only: windows that end by the cut are unchanged,
        #    windows that run past it are left empty.
        pos = np.flatnonzero(part["ev_start"])
        done = part.index[pos[pos + 1 + events.H < len(part)]]
        open_ = part.index[pos[pos + 1 + events.H >= len(part)]]
        fwd = [c for c in full.columns if c.startswith("fwd_")]
        pd.testing.assert_frame_equal(full.loc[done, fwd], part.loc[done, fwd], check_exact=True)
        assert part.loc[open_, fwd].isna().all().all()

    # 3. Event windows never overlap.
    pos = np.flatnonzero(full["ev_start"])
    assert (np.diff(pos) > events.H).all()

    # 4. Events only on trigger days, labels only on events.
    assert not (full["ev_start"] & ~full["dec_cond"]).any()
    assert full.loc[~full["ev_start"], "fwd_ev_label"].isna().all()

    # 5. Label agrees with the return at the barrier touch.
    e = full[full["ev_start"]].dropna(subset=["fwd_ev_label"])
    spy = panel["spy_adj_close"]
    for t, row in e.iterrows():
        i = panel.index.get_loc(t)
        r = spy.iloc[i + 1 + int(row["fwd_ev_days"])] / spy.iloc[i + 1] - 1
        if row["fwd_ev_label"] == 1:
            assert r >= row["ev_barrier"]
        elif row["fwd_ev_label"] == -1:
            assert r <= -row["ev_barrier"]
        else:
            assert int(row["fwd_ev_days"]) == events.H

    print(f"{len(e)} labelled events checked. ALL TESTS PASSED")


if __name__ == "__main__":
    main()
