"""Model features for every trading day: indicators and interaction terms.

A row dated t uses only data dated t or earlier. No scaling or winsorising here;
fit those inside each training window. Definitions, the collinearity check behind
DROPPED and the leakage checks are walked through in src/features/features.ipynb.

Run:  python -m src.features.feature_build
Output: data/processed/features.csv (+ features.parquet), data/processed/feature_sets.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.data.build import load_panel
from src.features import events

# Candidates removed after the collinearity check (see the notebook, section 3).
DROPPED = ["ts_chg5", "x_cond_vrp", "x_cond_vixchg"]

FEATURE_SETS = {"base": ["dec_dd", "dec_mom", "dec_vol", "dec_cond", "ma200_gap"]}
FEATURE_SETS["vix"] = FEATURE_SETS["base"] + ["vix_log", "vix_z", "vix_chg5", "ts_ratio", "ts_backw", "vrp"]
FEATURE_SETS["full"] = FEATURE_SETS["vix"] + ["x_cond_ts", "x_dd_ts", "x_mom_backw"]


def decline_features(panel: pd.DataFrame) -> pd.DataFrame:
    f = events.decline_signals(panel)                          # dec_dd, dec_mom, dec_vol
    f["dec_cond"] = ((f["dec_dd"] <= -events.S) & (f["dec_mom"] <= -events.P)).astype(float)
    a = panel["spy_adj_close"]
    f["ma200_gap"] = a / a.rolling(200).mean() - 1
    return f


def vix_features(panel: pd.DataFrame) -> pd.DataFrame:
    vix, ratio = panel["vix_close"], panel["vix_ratio"]
    return pd.DataFrame({
        "vix_log": np.log(vix),
        "vix_z": (vix - vix.rolling(252).mean()) / vix.rolling(252).std(),
        "vix_chg5": np.log(vix / vix.shift(5)),
        "ts_ratio": ratio,
        "ts_backw": (ratio > 1).astype(float),
        "ts_chg5": ratio - ratio.shift(5),
    })


def interactions(f: pd.DataFrame) -> pd.DataFrame:
    ts = f["ts_ratio"] - 1          # distance from backwardation; a fixed centre, so no leakage
    return pd.DataFrame({
        "x_cond_ts": f["dec_cond"] * ts,
        "x_dd_ts": f["dec_dd"] * ts,
        "x_mom_backw": f["dec_mom"] * f["ts_backw"],
        "x_cond_vrp": f["dec_cond"] * f["vrp"],
        "x_cond_vixchg": f["dec_cond"] * f["vix_chg5"],
    })


def build_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    """All 17 candidate features, before pruning."""
    f = decline_features(panel).join(vix_features(panel))
    f["vrp"] = panel["vix_close"] / 100 - f["dec_vol"] * np.sqrt(config.TRADING_DAYS)
    return f.join(interactions(f))


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """The final features, in the order of FEATURE_SETS['full']."""
    return build_candidates(panel)[FEATURE_SETS["full"]]


def save(feat: pd.DataFrame) -> None:
    out = config.PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out / "features.csv", float_format="%.10g")
    try:
        feat.to_parquet(out / "features.parquet")
    except ImportError:
        pass
    (out / "feature_sets.json").write_text(json.dumps(FEATURE_SETS, indent=2))
    print(f"wrote {len(feat.columns)} features x {len(feat)} rows to {out.relative_to(config.ROOT)}")


def load_features() -> pd.DataFrame:
    path = config.PROCESSED_DIR / "features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.read_csv(config.PROCESSED_DIR / "features.csv", parse_dates=["date"], index_col="date")


def main() -> None:
    save(build_features(load_panel()))


if __name__ == "__main__":
    main()
