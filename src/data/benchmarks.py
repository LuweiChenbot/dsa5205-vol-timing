"""Benchmark return series and their metrics over the study sample.

  spy_bh : buy-and-hold SPY, daily total return (investable benchmark)
  bil_bh : buy-and-hold BIL, daily total return (cash leg and risk-free rate)
  mix_50 : SPY/BIL 50/50 rebalanced daily. Example only: replace 0.5 with the
           strategy's average SPY weight for an exposure-matched comparison
           via static_mix(panel, w).

All series are indexed by realisation date, the same way apply_weights()
indexes strategy returns.

Run:  python -m src.data.benchmarks
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.data.build import load_panel
from src.metrics import perf_stats, stats_by_year


def static_mix(panel: pd.DataFrame, w: float) -> pd.Series:
    return (w * panel["spy_tr_cc"] + (1 - w) * panel["bil_tr_cc"]).rename(f"mix_{int(round(w * 100))}")


def benchmark_returns(panel: pd.DataFrame) -> pd.DataFrame:
    s = panel[panel["in_sample"]]
    return pd.DataFrame({
        "spy_bh": s["spy_tr_cc"],
        "bil_bh": s["bil_tr_cc"],
        "mix_50": static_mix(s, 0.5),
    })


def main() -> None:
    panel = load_panel()
    b = benchmark_returns(panel)
    rf = b["bil_bh"]
    b.to_csv(config.PROCESSED_DIR / "benchmarks.csv", float_format="%.10g")

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({c: perf_stats(b[c], rf) for c in b}).T
    table.to_csv(config.REPORT_DIR / "benchmark_metrics.csv", float_format="%.6f")
    by_year = pd.concat({c: stats_by_year(b[c], rf)[["ann_return", "sharpe"]] for c in ["spy_bh", "bil_bh"]},
                        axis=1)
    by_year.to_csv(config.REPORT_DIR / "benchmark_by_year.csv", float_format="%.6f")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c, lab in [("spy_bh", "SPY buy-and-hold (total return)"),
                   ("bil_bh", "BIL buy-and-hold (total return)"),
                   ("mix_50", "50/50 SPY/BIL, daily rebalanced")]:
        ax.plot((1 + b[c]).cumprod(), label=lab, lw=1.3)
    ax.set_title(f"Benchmarks, growth of $1 ({config.SAMPLE_START[:4]} to {config.SAMPLE_END[:4]})")
    ax.set_ylabel("Wealth"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(config.REPORT_DIR / "fig_benchmarks_cumulative.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.float_format", "{:.4f}".format)
    print(table)


if __name__ == "__main__":
    main()
