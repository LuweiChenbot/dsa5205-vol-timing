"""Leakage and coverage checks for the model features, on the processed panel.

Run:  python -m tests.test_features
"""
from __future__ import annotations

import pandas as pd

from src.data.build import load_panel
from src.features import feature_build as fb


def main() -> None:
    panel = load_panel()
    s = panel["in_sample"]
    feat = fb.build_features(panel)

    # 1. Every in-sample day has every feature, and no outcome column slipped in.
    missing = feat.loc[s].isna().sum()
    assert missing.sum() == 0, missing[missing > 0]
    assert not any(c.startswith("fwd_") for c in feat.columns)

    # 2. Feature sets are nested and cover exactly the saved columns.
    sets = fb.FEATURE_SETS
    assert set(sets["base"]) < set(sets["vix"]) < set(sets["full"]) == set(feat.columns)
    assert not set(fb.DROPPED) & set(feat.columns)

    # 3. Values up to a cut date do not change when everything after it is removed.
    for frac in (0.25, 0.5, 0.75, 0.95):
        cut = panel.index[int(len(panel) * frac)]
        pd.testing.assert_frame_equal(feat.loc[:cut], fb.build_features(panel.loc[:cut]), check_exact=True)

    # 4. The saved file matches a fresh build.
    pd.testing.assert_frame_equal(fb.load_features(), feat, check_freq=False)

    print(f"{feat.shape[1]} features x {s.sum()} sample days checked. ALL TESTS PASSED")


if __name__ == "__main__":
    main()
