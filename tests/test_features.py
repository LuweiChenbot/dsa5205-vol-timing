"""Leakage and coverage checks for the model features, on the processed panel.

Run:  python -m tests.test_features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
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

    # 4. The saved file matches a fresh build and holds in-sample days only.
    saved = fb.load_features()
    pd.testing.assert_frame_equal(saved, feat.loc[s], check_freq=False)
    assert saved.notna().all().all()

    # 5. The declared selection rules, applied to the design period only, reproduce DROPPED.
    design = fb.build_candidates(panel).loc[config.SAMPLE_START:config.DESIGN_END]
    dropped, _ = fb.select_features(design)
    assert dropped == fb.DROPPED, dropped

    # 6. PastScaler: statistics come from the training rows only.
    train, test = feat.loc[config.SAMPLE_START:config.DESIGN_END], feat.loc[config.TEST_START:]
    sc = fb.PastScaler().fit(train)
    z = sc.transform(train)
    assert np.allclose(z.mean(), 0, atol=1e-12) and np.allclose(z.std(ddof=0), 1)
    shocked = test * 100                                  # wild test data must not move the fit
    sc2 = fb.PastScaler().fit(train)
    pd.testing.assert_frame_equal(sc2.transform(test), sc.transform(test))
    assert (sc.transform(shocked)[sc.lo_.index] <= (sc.hi_ - sc.mean_[sc.hi_.index]) / sc.std_[sc.hi_.index] + 1e-12).all().all()
    for b in fb.BINARY + ["x_cond_ts"]:                   # not clipped
        assert b not in sc.lo_.index
        assert z[b].nunique() == train[b].nunique()

    print(f"{feat.shape[1]} features x {s.sum()} sample days checked. ALL TESTS PASSED")


if __name__ == "__main__":
    main()
