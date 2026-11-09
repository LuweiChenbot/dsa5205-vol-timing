"""One entry point per team member, for the end-to-end notebook and the recording.

    from src.walkthrough import member1, member2, member3, member4
    member1.build_data()        # Sai: clean panel, benchmarks, 33 checks
    member2.extract_features()  # Luwei Chen: features, targets, decline events
    member3.train_models()      # Navneet Melarkode: walk-forward ridge
    member4.run_backtest()      # Hieu Hoang: trades, costs, statistics, interaction test
    show(some_function)         # print the real source for the code walk-through

The upstream outputs are a frozen, hash-checked snapshot. Members 2 and 3
therefore recompute their stage in memory and compare it with the committed
files instead of overwriting them. A refit on another machine differs in the
last floating-point digits, and the frozen-snapshot guards rightly reject that.
To regenerate files on purpose, use the README commands.
"""
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import config


def _same(name: str, new: pd.DataFrame, saved: pd.DataFrame, tol: float = 1e-8) -> float:
    """Compare a recomputed table with the committed one; return the largest gap."""
    new = new.reindex(index=saved.index, columns=saved.columns)
    num = saved.select_dtypes("number").columns
    gap = float(np.nanmax(np.abs(new[num].to_numpy(float) - saved[num].to_numpy(float)))) if len(num) else 0.0
    same_nan = new[num].isna().equals(saved[num].isna())
    ok = gap <= tol and same_nan and new.drop(columns=num).equals(saved.drop(columns=num))
    print(f"  {name:<12} {saved.shape[0]:>5} rows x {saved.shape[1]:>2} cols   "
          f"max gap {gap:.1e}   {'matches committed snapshot' if ok else 'DIFFERS'}")
    if not ok:
        raise AssertionError(f"{name}: recomputed values differ from the committed file")
    return gap


def show(obj, start: str | None = None, stop: str | None = None) -> None:
    """Display the real source of a function, optionally only the lines from the
    first one containing `start` to the first later one containing `stop`."""
    from IPython.display import Markdown, display
    lines, first = inspect.getsourcelines(obj)
    i = next(k for k, line in enumerate(lines) if start in line) if start else 0
    j = next(k for k, line in enumerate(lines) if k >= i and stop in line) + 1 if stop else len(lines)
    path = Path(inspect.getsourcefile(obj)).resolve().relative_to(config.ROOT).as_posix()
    code = textwrap.dedent("".join(lines[i:j]))
    display(Markdown(f"`{path}`, lines {first + i}-{first + j - 1}\n```python\n{code}```"))


def build_data(download: bool = False) -> pd.DataFrame:
    """Data stage.

    The raw Yahoo/FRED CSVs are not committed (see .gitignore), only the panel built
    from them. So by default this rebuilds only when data/raw/ holds them; otherwise
    it loads the committed panel and its saved check results. download=True fetches
    fresh data first; vendor revisions can then change the panel and its hashes.
    """
    from src.data import benchmarks, build, fetch, verify
    if download:
        print("== fetch"); fetch.main()
    elif not any(config.RAW_DIR.glob("*.csv")):
        panel = build.load_panel()
        print(f"No raw CSVs in {config.RAW_DIR}; using the committed panel "
              f"({len(panel)} rows, {panel.index[0]:%Y-%m-%d} to {panel.index[-1]:%Y-%m-%d}) "
              "and the saved checks. Pass download=True to rebuild from source.")
        return pd.read_csv(config.REPORT_DIR / "checks.csv")
    print("== build"); build.main()
    print("== benchmarks"); benchmarks.main()
    print("== verify")
    try:
        verify.main()
    except SystemExit as stop:          # verify always exits; 0 means every check passed
        if stop.code:
            raise RuntimeError("data verification failed; see reports/data/verification.md")
    return pd.read_csv(config.REPORT_DIR / "checks.csv")


def extract_features() -> pd.DataFrame:
    """Feature stage, recomputed from the panel and checked against the committed files."""
    from src.data.build import load_panel
    from src.features import events, feature_build, targets
    panel = load_panel()
    feats = feature_build.sample_features(panel)
    print("Recomputing from the panel:")
    _same("features", feats, feature_build.load_features())
    _same("targets", targets.build_targets(panel), targets.load_targets())
    _same("events", events.build(panel), events.load_events())
    return feats


def train_models() -> pd.DataFrame:
    """Model stage: rerun the frozen walk-forward ridge in memory and compare forecasts."""
    from src.data.build import load_panel
    from src.features.events import load_events
    from src.features.feature_build import load_features
    from src.models.config import ModelConfig
    from src.models.handoff import load_predictions, wide_predictions
    from src.models.pipeline import run_experiment
    print("Refitting six models walk-forward (about a minute)...", flush=True)
    result = run_experiment(load_panel(), load_features(), load_events(), ModelConfig())
    print(f"Frozen alphas: {result.selected_alphas}")
    saved = wide_predictions(load_predictions())        # hash-checked against the manifest
    _same("forecasts", wide_predictions(result.predictions), saved, tol=1e-10)
    return result.metrics


def run_backtest() -> pd.DataFrame:
    """Backtest stage: allocation rules, costs, significance tests, interaction test."""
    from src.analysis import interaction_test
    from src.backtest import pipeline
    print("== backtest"); pipeline.main()
    print("\n== interaction test"); interaction_test.main()
    return pd.read_csv(config.ROOT / "reports" / "backtest" / "strategy_metrics.csv")


member1 = SimpleNamespace(name="Sai", role="Data", build_data=build_data)
member2 = SimpleNamespace(name="Luwei Chen", role="Features", extract_features=extract_features)
member3 = SimpleNamespace(name="Navneet Melarkode", role="Models", train_models=train_models)
member4 = SimpleNamespace(name="Hieu Hoang", role="Backtest and evaluation", run_backtest=run_backtest)
