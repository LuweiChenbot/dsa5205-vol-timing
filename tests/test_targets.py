"""Timing checks for the multi-day targets.

Run:  python -m tests.test_targets
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.build import load_panel
from src.features import targets


def main() -> None:
    panel = load_panel()
    s = panel["in_sample"]

    # 1. h = 1 reproduces the panel's one-day target.
    one = targets.build_targets(panel, horizons=(1,))["fwd_spy_xs_1"]
    np.testing.assert_allclose(one, panel["fwd_spy_xs"], rtol=0, atol=1e-12)

    t = targets.build_targets(panel)
    for h in targets.HORIZONS:
        col = t[f"fwd_spy_xs_{h}"]
        # 2. Direct compounding of the returns dated t+2 .. t+1+h, checked on a few rows.
        for i in np.flatnonzero(s)[::500]:
            spy = np.prod(1 + panel["spy_tr_cc"].iloc[i + 2:i + 2 + h]) - 1
            bil = np.prod(1 + panel["bil_tr_cc"].iloc[i + 2:i + 2 + h]) - 1
            assert abs(col.iloc[i] - (spy - bil)) < 1e-12
        # 3. Nothing dated t+1 or earlier enters: changing the return at t+1 leaves row t unchanged.
        bumped = panel.copy()
        k = int(np.flatnonzero(s)[1000])
        bumped.iloc[k + 1, bumped.columns.get_loc("spy_tr_cc")] += 0.05
        assert targets.build_targets(bumped)[f"fwd_spy_xs_{h}"].iloc[k] == col.iloc[k]
        # 4. Every in-sample decision date has a full window, unless it runs past the panel end.
        fits = np.arange(len(panel)) + 1 + h < len(panel)
        assert col[s & fits].notna().all() and col[~fits].isna().all(), h

    pd.testing.assert_frame_equal(targets.load_targets(), t, check_freq=False)
    print(f"horizons {targets.HORIZONS} checked. ALL TESTS PASSED")


if __name__ == "__main__":
    main()
