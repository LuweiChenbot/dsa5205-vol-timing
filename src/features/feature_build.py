"""Model features for every trading day: indicators and interaction terms.

A row dated t uses only data dated t or earlier. The saved features are raw: no
scaling or winsorising, because those must be fitted inside each training window.
Use PastScaler for that. Definitions, the selection rules behind DROPPED and the
leakage checks are walked through in src/features/features.ipynb.

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

# ------------------------------------------------------------ selection rules
# Declared before looking at any outcome, applied to the design period only
# (config.SAMPLE_START .. config.DESIGN_END). select_features() reproduces DROPPED.
RESCALE_CORR = 0.8      # rule 1: drop an interaction this correlated with one of its own parts
DUPLICATE_CORR = 0.9    # rule 2: of two candidates this correlated, drop the later-listed one
MATERIAL = 0.1          # rule 3: |x| above this share of its maximum counts as material variation
EPISODE_GAP = 10        # material days more than this many trading days apart are separate episodes
MIN_EPISODES = 15       # rule 3: keep a term-structure interaction with at least this many episodes
                        # in the design period (about three a year)
INTERACTION_PARTS = {
    "x_cond_ts": ("dec_cond", "ts_ratio"),
    "x_dd_ts": ("dec_dd", "ts_ratio"),
    "x_mom_backw": ("dec_mom", "ts_backw"),
    "x_cond_vrp": ("dec_cond", "vrp"),
    "x_cond_vixchg": ("dec_cond", "vix_chg5"),
}
TS_INTERACTIONS = ["x_cond_ts", "x_dd_ts", "x_mom_backw"]   # rule 3 applies to these

DROPPED = ["x_cond_vrp", "x_cond_vixchg", "ts_chg5", "x_dd_ts"]

FEATURE_SETS = {"base": ["dec_dd", "dec_mom", "dec_vol", "dec_cond", "ma200_gap"]}
FEATURE_SETS["vix"] = FEATURE_SETS["base"] + ["vix_log", "vix_z", "vix_chg5", "ts_ratio", "ts_backw", "vrp"]
FEATURE_SETS["full"] = FEATURE_SETS["vix"] + ["x_cond_ts", "x_mom_backw"]

BINARY = ["dec_cond", "ts_backw"]


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


def material_episodes(x: pd.Series) -> int:
    idx = np.flatnonzero((x.abs() > MATERIAL * x.abs().max()).to_numpy())
    return 0 if len(idx) == 0 else 1 + int((np.diff(idx) > EPISODE_GAP).sum())


def select_features(cand: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Apply the three selection rules to design-period candidates.

    Returns the dropped columns in order and a log with the reason for each decision.
    """
    corr = cand.corr().abs()
    dropped, log = [], []
    for x, parts in INTERACTION_PARTS.items():                      # rule 1
        c = corr.loc[x, list(parts)].max()
        if c > RESCALE_CORR:
            dropped.append(x)
            log.append((x, "drop", f"rule 1: |corr| {c:.2f} with its own part"))
    kept = [c for c in cand.columns if c not in dropped]
    for i, a in enumerate(kept):                                   # rule 2
        for b in kept[i + 1:]:
            if a not in dropped and b not in dropped and corr.loc[a, b] > DUPLICATE_CORR:
                dropped.append(b)
                log.append((b, "drop", f"rule 2: |corr| {corr.loc[a, b]:.2f} with {a}"))
    for x in [x for x in TS_INTERACTIONS if x not in dropped]:     # rule 3
        eps = material_episodes(cand[x])
        keep = eps >= MIN_EPISODES
        if not keep:
            dropped.append(x)
        log.append((x, "keep" if keep else "drop", f"rule 3: {eps} material episodes (min {MIN_EPISODES})"))
    return dropped, pd.DataFrame(log, columns=["feature", "decision", "reason"])


class PastScaler:
    """Winsorise and standardise with statistics fitted on training rows only.

    fit() sees only the training window; transform() applies the same clip bounds,
    means and standard deviations to later rows. Continuous features are clipped
    at the training quantiles q and 1 - q. Binary features and interactions are
    not clipped: they are mostly 0, so in a calm window a quantile bound could be 0
    and wipe out every 1.
    """

    def __init__(self, q: float = 0.01):
        self.q = q

    def fit(self, X: pd.DataFrame) -> "PastScaler":
        clip = [c for c in X.columns if c not in BINARY and not c.startswith("x_")]
        self.lo_ = X[clip].quantile(self.q)
        self.hi_ = X[clip].quantile(1 - self.q)
        Xc = self._clip(X)
        self.mean_ = Xc.mean()
        self.std_ = Xc.std(ddof=0).replace(0.0, 1.0)
        return self

    def _clip(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.lo_.index] = X[self.lo_.index].clip(self.lo_, self.hi_, axis=1)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return (self._clip(X) - self.mean_) / self.std_

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


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
