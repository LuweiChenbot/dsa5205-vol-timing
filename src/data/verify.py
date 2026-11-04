"""Data-quality and leakage checks on the raw snapshot and the built panel.

Writes reports/data/verification.md, checks.csv and figures.
Exits with code 1 if any check FAILs (WARNs are for human review).

Run:  python -m src.data.verify
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data.build import load_etf, load_panel

CHECKS: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str, level: str = "FAIL") -> None:
    CHECKS.append((name, "PASS" if ok else level, detail))


# Reference closes for well-known volatility events (Cboe VIX close).
# Used as WARN-level spot checks that the right file was parsed.
VIX_SPOT = {"2015-08-24": 40.74, "2018-02-05": 37.32, "2020-03-16": 82.69, "2024-08-05": 38.57}


def load_dtb3():
    """FRED 3-month T-bill rate (decimal) and its in-sample yearly average, or (None, None)."""
    path = config.RAW_DIR / "DTB3_fred.csv"
    if not path.exists():
        return None, None
    f = pd.read_csv(path)
    f.columns = ["date", "dtb3"]
    f["date"] = pd.to_datetime(f["date"])
    f["dtb3"] = pd.to_numeric(f["dtb3"], errors="coerce") / 100
    f = f.set_index("date")["dtb3"]
    f = f[(f.index >= config.SAMPLE_START) & (f.index <= config.SAMPLE_END)]
    return f, f.groupby(f.index.year).mean()


def main() -> None:
    p = load_panel()
    s = p[p["in_sample"]]
    tables: dict[str, pd.DataFrame] = {}
    dtb3, dtb3_year = load_dtb3()

    # ---------------------------------------------------------------- calendar
    days = s.groupby(s.index.year).size()
    check("trading days per year in [248, 254]", days.between(248, 254).all(),
          ", ".join(f"{y}:{n}" for y, n in days.items()))
    check("sample starts and ends in configured window",
          s.index.min() <= pd.Timestamp(config.SAMPLE_START) + pd.Timedelta(days=7)
          and s.index.max() >= pd.Timestamp(config.SAMPLE_END) - pd.Timedelta(days=7),
          f"{s.index.min().date()} -> {s.index.max().date()}")
    check("dates strictly increasing, no duplicates",
          p.index.is_monotonic_increasing and not p.index.duplicated().any(), "")

    # ---------------------------------------------------------------- prices
    for t in ["SPY", "BIL"]:
        raw = load_etf(t)
        raw = raw[(raw.index >= config.SAMPLE_START) & (raw.index <= config.SAMPLE_END)]
        check(f"{t} prices positive", (raw[["Open", "High", "Low", "Close"]] > 0).all().all(), "")
        bad = (raw["High"] < raw["Low"]).sum()
        check(f"{t} High >= Low", bad == 0, f"{bad} violations")
        bad = ((raw["Close"] > raw["High"] * 1.0001) | (raw["Close"] < raw["Low"] * 0.9999)).sum()
        check(f"{t} Close within [Low, High]", bad == 0, f"{bad} violations", "WARN")
        # Yahoo's Close is already split-adjusted, so a split must not show up
        # as a jump in the return series. (BIL had a 1-for-2 reverse split on
        # 2017-11-30, reported by Yahoo as ratio 0.5.)
        splits = raw.loc[raw["Stock Splits"] > 0, "Stock Splits"]
        jumps = p.loc[p.index.isin(splits.index), f"{t.lower()}_tr_cc"].abs()
        check(f"{t} splits already reflected in prices (|return| < 5% on split dates)",
              bool((jumps < 0.05).all()),
              "no splits" if splits.empty else "; ".join(
                  f"{d.date()} ratio {v:g}, return that day {p.at[d, f'{t.lower()}_tr_cc']:+.4%}"
                  for d, v in splits.items()))

        divs = raw.loc[raw["Dividends"] > 0, "Dividends"]
        per_year = divs.groupby(divs.index.year).size().reindex(days.index, fill_value=0)
        detail = ", ".join(f"{y}:{n}" for y, n in per_year.items())
        if t == "SPY":
            check("SPY pays 4 dividends every year", bool((per_year == 4).all()), detail, "WARN")
        else:
            # Informational only. A T-bill ETF distributes only when bill income
            # exceeds its expense ratio, so zero-rate years (2014-16, 2020-21)
            # have few or no payouts, and timing is irregular even at normal
            # rates (none in Apr-May 2022, two in Dec 2019). The economic test
            # of BIL is its total return vs the T-bill rate, checked below.
            check("BIL dividend count per year (informational; see BIL vs T-bill check)",
                  True, detail)

        # Our total return vs Yahoo Adj Close implied return
        ours = p[f"{t.lower()}_tr_cc"]
        vendor = p[f"{t.lower()}_vendor_adj_close"].pct_change()
        diff = (ours - vendor)[p["in_sample"]].dropna().abs()
        is_div = p.loc[diff.index, f"{t.lower()}_div"] > 0
        check(f"{t} total return matches Yahoo Adj Close (tol {config.TR_TOLERANCE:g})",
              (diff > config.TR_TOLERANCE).sum() == 0,
              f"max |diff| {diff.max():.2e} (div days {diff[is_div].max():.2e}, "
              f"other days {diff[~is_div].max():.2e}); mean {diff.mean():.2e}; "
              f"{(diff > config.TR_TOLERANCE).sum()} days over tolerance")
        check(f"{t} adjusted close reproduces total return exactly",
              np.allclose(p[f"{t.lower()}_adj_close"].pct_change()[p['in_sample']].dropna(),
                          ours[p["in_sample"]].dropna(), atol=1e-8), "CSV stores 10 significant digits")

        yearly = pd.DataFrame({
            "total_return": (1 + s[f"{t.lower()}_tr_cc"]).groupby(s.index.year).prod() - 1,
            "price_return": (1 + s[f"{t.lower()}_pr_cc"]).groupby(s.index.year).prod() - 1,
            "dividends_paid": s[f"{t.lower()}_div"].groupby(s.index.year).sum(),
            "n_dividends": (s[f"{t.lower()}_div"] > 0).groupby(s.index.year).sum(),
        })
        yearly["dividend_contribution"] = yearly["total_return"] - yearly["price_return"]
        tables[f"{t} yearly total vs price return"] = yearly

    big = s.loc[s["spy_tr_cc"].abs() > config.SPY_MOVE_FLAG, ["spy_tr_cc", "vix_close"]]
    check(f"SPY daily moves above {config.SPY_MOVE_FLAG:.0%} (review against known events)",
          True, "; ".join(f"{d.date()} {r:+.2%}" for d, r in big["spy_tr_cc"].items()) or "none")
    check("BIL daily |return| < 0.5% (T-bill ETF sanity)",
          (s["bil_tr_cc"].abs() < 0.005).all(), f"max {s['bil_tr_cc'].abs().max():.4%}", "WARN")

    # ---------------------------------------------------------------- volatility indices
    for c in ["vix_close", "vix3m_close"]:
        n_na = s[c].isna().sum()
        n_ff = s[f"{c}_ffilled"].sum()
        check(f"{c} no missing values after ffill", n_na == 0, f"{n_na} NaN, {n_ff} forward-filled")
        check(f"{c} within [5, 100]", s[c].dropna().between(5, 100).all(),
              f"min {s[c].min():.2f}, max {s[c].max():.2f}")
    for d, ref in VIX_SPOT.items():
        ts = pd.Timestamp(d)
        if ts in s.index:
            v = s.at[ts, "vix_close"]
            check(f"VIX close on {d} ~ {ref}", abs(v - ref) < 0.5, f"got {v:.2f}", "WARN")
    back = (s["vix_ratio"] > 1).mean()
    check("VIX/VIX3M ratio plausible (median 0.8 to 0.95)", 0.8 <= s["vix_ratio"].median() <= 0.95,
          f"median {s['vix_ratio'].median():.3f}; backwardation on {back:.1%} of days", "WARN")

    # ---------------------------------------------------------------- leakage / timing
    idx = np.arange(len(p) - 2)
    check("fwd_spy_tr[t] == spy_tr_cc[t+2] (next-close convention)",
          np.allclose(p["fwd_spy_tr"].values[idx], p["spy_tr_cc"].values[idx + 2], equal_nan=True), "")
    check("fwd_bil_tr[t] == bil_tr_cc[t+2]",
          np.allclose(p["fwd_bil_tr"].values[idx], p["bil_tr_cc"].values[idx + 2], equal_nan=True), "")
    check("every in-sample decision date has a forward return", s["fwd_spy_tr"].notna().all(),
          f"{s['fwd_spy_tr'].isna().sum()} missing")
    corr_same = s["spy_tr_cc"].corr(s["fwd_spy_tr"])
    check("same-day and forward SPY returns are different series (|corr| < 0.3)",
          abs(corr_same) < 0.3, f"corr {corr_same:.3f}")

    # ---------------------------------------------------------------- BIL vs T-bill rate
    if dtb3 is not None:
        cmp_ = pd.DataFrame({
            "BIL_total_return": (1 + s["bil_tr_cc"]).groupby(s.index.year).prod() - 1,
            "DTB3_avg_rate": dtb3_year,
        })
        cmp_["gap"] = cmp_["BIL_total_return"] - cmp_["DTB3_avg_rate"]
        tables["BIL vs 3-month T-bill rate (FRED DTB3)"] = cmp_
        check("BIL annual return within 1pp of average T-bill rate", (cmp_["gap"].abs() < 0.01).all(),
              f"max gap {cmp_['gap'].abs().max():.2%} (BIL has a fee of ~0.14% a year)", "WARN")

    # ---------------------------------------------------------------- figures
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot((1 + s["spy_tr_cc"]).cumprod(), label="SPY total return (dividends reinvested)", lw=1.3)
    ax.plot((1 + s["spy_pr_cc"]).cumprod(), label="SPY price return", lw=1.3)
    ax.set_title("SPY: total return vs price return, growth of $1")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(config.REPORT_DIR / "fig_spy_total_vs_price.png", dpi=150); plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(s["vix_close"], label="VIX (30-day)", lw=0.9)
    a1.plot(s["vix3m_close"], label="VIX3M (3-month)", lw=0.9)
    a1.set_ylabel("Index level"); a1.legend(); a1.grid(alpha=0.3)
    a1.set_title("Cboe implied volatility indices and term structure")
    a2.plot(s["vix_ratio"], lw=0.8, color="tab:purple")
    a2.axhline(1, color="k", lw=0.8, ls="--")
    a2.set_ylabel("VIX / VIX3M"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(config.REPORT_DIR / "fig_vix_term_structure.png", dpi=150); plt.close(fig)

    # ---------------------------------------------------------------- write report
    df = pd.DataFrame(CHECKS, columns=["check", "status", "detail"])
    df.to_csv(config.REPORT_DIR / "checks.csv", index=False)
    lines = ["# Data verification report", "",
             f"Sample {s.index.min().date()} to {s.index.max().date()}, {len(s)} trading days.", "",
             "| Check | Status | Detail |", "|---|---|---|"]
    lines += [f"| {c} | {st} | {d} |" for c, st, d in CHECKS]
    for title, t in tables.items():
        lines += ["", f"## {title}", "", t.to_markdown(floatfmt=".4f") if _has_tabulate() else
                  "```\n" + t.to_string(float_format=lambda x: f"{x:.4f}") + "\n```"]
    (config.REPORT_DIR / "verification.md").write_text("\n".join(lines) + "\n")

    counts = df["status"].value_counts().to_dict()
    for c, st, d in CHECKS:
        print(f"[{st:4s}] {c}" + (f"  ({d})" if d else ""))
    print(f"\n{counts}. Report: {config.REPORT_DIR / 'verification.md'}")
    sys.exit(1 if counts.get("FAIL", 0) else 0)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
