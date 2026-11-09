"""Member 4: predictions -> weights -> post-cost backtests, metrics and figures.

Run:  python -m src.backtest.pipeline
Reads the frozen model signals (data/processed/models/) and the panel; writes
everything to reports/backtest/. No settings are tuned here: all choices live
in src/backtest/config.py and were fixed on 2017-2018 validation predictions.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.backtest import config as bc
from src.backtest.engine import run_backtest, to_weights
from src.backtest.evaluate import (event_rows, event_summary, period_stats,
                                   strategy_stats, yearly_stats)
from src.data.build import load_panel
from src.features.events import load_events
from src.features.feature_build import load_features
from src.models.config import MODEL_DIR, MODEL_IDS
from src.models.handoff import load_predictions, wide_predictions

OUT = config.PROCESSED_DIR.parent.parent / "reports" / "backtest"
PRIMARY = f"{bc.PRIMARY_MODEL}|{bc.PRIMARY_RULE}"
MV = f"{bc.PRIMARY_MODEL}|mv"
OVERLAY = f"{bc.PRIMARY_MODEL}|decline_overlay"
# Reference categorical palette (dataviz skill, validated light mode); SPY is neutral ink.
C_SPY, C1, C2, C3 = "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
GRID = "#e4e3df"


def load_inputs():
    panel = load_panel()
    vol = load_features()["dec_vol"]
    events = load_events()
    test = wide_predictions(load_predictions())
    valid = wide_predictions(load_predictions(MODEL_DIR / "validation_predictions.csv"))
    outcomes = pd.read_csv(MODEL_DIR / "outcomes.csv", parse_dates=["decision_date"],
                           index_col="decision_date")
    return panel, vol, events, test, valid, outcomes


def gamma_audit(valid: pd.DataFrame, vol: pd.Series) -> float:
    """Median pred/var of the primary model on validation (design data only)."""
    return float((valid[bc.PRIMARY_MODEL] / vol.reindex(valid.index) ** 2).median())


def all_weights(wide: pd.DataFrame, vol: pd.Series, w_max: float = bc.W_MAX) -> dict:
    out = {}
    for m in MODEL_IDS:
        for rule in bc.RULES:
            out[f"{m}|{rule}"] = to_weights(wide[m], rule, vol=vol, decline=wide["decline"],
                                            w_max=w_max)
    return out


def benchmarks(panel, dates, avg_w, cost, returns=bc.RETURNS_MAIN) -> dict:
    """Same engine, same dates: SPY buy-and-hold, BIL, exposure-matched static mix."""
    return {name: run_backtest(panel, pd.Series(w, index=dates), cost, returns=returns)
            for name, w in {"spy_bh": 1.0, "bil_bh": 0.0, "static_mix": avg_w}.items()}


def metrics_table(panel, weights, cost, inference=True, returns=bc.RETURNS_MAIN):
    dates = next(iter(weights.values())).index
    bts = {k: run_backtest(panel, w, cost, returns=returns) for k, w in weights.items()}
    avg_w = bts[PRIMARY]["held"].iloc[1:].mean()
    bts |= benchmarks(panel, dates, avg_w, cost, returns)
    spy = bts["spy_bh"]
    rows = {k: strategy_stats(bt, spy, inference) for k, bt in bts.items()}
    tab = pd.DataFrame(rows).T
    tab.index.name = "strategy"
    return tab, bts


def split_key(tab: pd.DataFrame) -> pd.DataFrame:
    parts = tab.index.to_series().str.split("|", expand=True)
    tab = tab.copy()
    tab.insert(0, "model_id", parts[0])
    tab.insert(1, "rule", parts[1] if parts.shape[1] > 1 else None)
    return tab


def cumulative_sse(test: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Goyal-Welch plot data: cumsum of (benchmark SE - model SE). Rising = model better."""
    y = outcomes.loc[test.index, "actual_excess_return"]
    se_hm = (y - test["historical_mean"]) ** 2
    models = [m for m in MODEL_IDS if m != "historical_mean"]
    return pd.DataFrame({m: (se_hm - (y - test[m]) ** 2).cumsum() for m in models})


# ---------------------------------------------------------------- figures
def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#a9a8a2")
    ax.tick_params(colors="#52514e", labelsize=9)


def fig_cumulative(bts, path):
    lines = [("spy_bh", "SPY buy-and-hold", C_SPY, "-"),
             (PRIMARY, "ridge_full, sign rule (primary)", C1, "-"),
             (OVERLAY, "ridge_full, decline overlay", C2, "-"),
             ("static_mix", "Static SPY/BIL mix, same avg. weight", C3, "--")]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.3]})
    for key, label, c, ls in lines:
        wealth = (1 + bts[key]["net"]).cumprod()
        a1.plot(wealth.index, wealth, color=c, lw=2 if key != "spy_bh" else 1.6, ls=ls, label=label)
        a1.annotate(f"{wealth.iloc[-1]:.2f}", (wealth.index[-1], wealth.iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", fontsize=8, color="#0b0b0b", va="center")
        dd = wealth / wealth.cummax() - 1
        a2.plot(dd.index, dd * 100, color=c, lw=1.2, ls=ls)
    a1.set_yscale("log")
    a1.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 1.5, 2, 2.5, 3, 3.5]))
    a1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    a1.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a1.set_ylabel("Growth of $1 (log scale)", fontsize=9)
    a1.set_title(f"Out-of-sample wealth after {bc.COST_BPS:g} bps costs, 2019-2025 decisions",
                 fontsize=11, loc="left")
    a1.legend(frameon=False, fontsize=8.5, loc="upper left")
    a2.set_ylabel("Drawdown (%)", fontsize=9)
    for a in (a1, a2):
        _style(a)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_yearly(yearly, path):
    years = yearly["spy_bh"].index
    x = np.arange(len(years))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title in ((axes[0], "ann_return", "Annual return (%)"),
                           (axes[1], "sharpe", "Sharpe ratio")):
        for j, (key, label, c) in enumerate((("spy_bh", "SPY buy-and-hold", C_SPY),
                                             (PRIMARY, "ridge_full, sign", C1),
                                             (OVERLAY, "ridge_full, decline overlay", C2))):
            v = yearly[key][col] * (100 if col == "ann_return" else 1)
            ax.bar(x + (j - 1) * 0.27, v, width=0.25, color=c, label=label, edgecolor="white", linewidth=1)
        ax.axhline(0, color="#a9a8a2", lw=0.8)
        ax.set_xticks(x, list(years), fontsize=8)
        ax.set_title(title, fontsize=10, loc="left")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_cum_sse(cs, path):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for m, c in (("ridge_full", C1), ("ridge_vix", C2), ("ridge_base", C3)):
        ax.plot(cs.index, cs[m] * 1e4, color=c, lw=1.8, label=m)
    ax.axhline(0, color="#a9a8a2", lw=0.8)
    ax.set_ylabel("Cumulative SSE gain vs\nhistorical mean (x1e-4)", fontsize=9)
    ax.set_title("Out-of-sample forecast accuracy over time (rising = beats the rolling mean)",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_weights(panel, weights, test, path):
    w = weights[f"{bc.PRIMARY_MODEL}|mv"]
    px = panel.loc[w.index, "spy_adj_close"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                 gridspec_kw={"height_ratios": [1.4, 1]})
    a1.plot(px.index, px, color=C_SPY, lw=1.2)
    dec = test.index[test["decline"] == 1]
    a1.scatter(dec, px.loc[dec], s=10, color=C2, zorder=3, label="decline-trigger day")
    a1.set_ylabel("SPY (total-return adj.)", fontsize=9)
    a1.legend(frameon=False, fontsize=8)
    a2.fill_between(w.index, 0, w, color=C1, alpha=0.35, linewidth=0)
    a2.plot(w.index, w.rolling(21).mean(), color=C1, lw=1.5, label="21-day average")
    a2.set_ylim(0, 1.02)
    a2.set_ylabel("SPY weight\n(ridge_full, mv)", fontsize=9)
    a2.legend(frameon=False, fontsize=8, loc="lower left")
    for a in (a1, a2):
        _style(a)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_costs(costs, path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for key, label, c in ((PRIMARY, "ridge_full, sign", C1),
                          (MV, "ridge_full, mean-variance", C2),
                          (OVERLAY, "ridge_full, decline overlay", C3),
                          ("spy_bh", "SPY buy-and-hold", C_SPY)):
        s = costs[costs.strategy == key].set_index("cost_bps")["sharpe"]
        ax.plot(s.index, s, marker="o", ms=5, color=c, lw=2, label=label)
    ax.set_xlabel("One-way cost (bps per unit of weight traded)", fontsize=9)
    ax.set_ylabel("Sharpe ratio", fontsize=9)
    ax.set_title("Cost sensitivity, 2019-2025", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def fmt(tab: pd.DataFrame, cols: list[str]) -> str:
    t = tab[cols].copy()
    for c in cols:
        if c in {"ann_return", "ann_vol", "max_dd", "avg_weight", "pct_days_in_spy", "alpha_pa",
                 "cost_drag_pa"}:
            t[c] = (t[c].astype(float) * 100).map("{:.2f}%".format)
        elif c not in {"model_id", "rule"}:
            t[c] = t[c].astype(float).map("{:.3f}".format)
    return t.to_markdown()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, vol, events, test, valid, outcomes = load_inputs()
    g = gamma_audit(valid, vol)
    print(f"gamma audit: median validation pred/var of {bc.PRIMARY_MODEL} = {g:.2f} "
          f"(frozen GAMMA = {bc.GAMMA})")

    # 1. design period (validation predictions, alphas chosen on that same period)
    w_val = all_weights(valid, vol)
    val_tab, _ = metrics_table(panel, w_val, bc.COST_BPS, inference=False)
    split_key(val_tab).to_csv(OUT / "validation_metrics.csv")

    # 2. out-of-sample evaluation at the primary cost
    w_test = all_weights(test, vol)
    tab, bts = metrics_table(panel, w_test, bc.COST_BPS)
    split_key(tab).to_csv(OUT / "strategy_metrics.csv")
    pd.DataFrame({k: bt["net"] for k, bt in bts.items()}).to_csv(OUT / "daily_returns.csv")
    pd.DataFrame(w_test).to_csv(OUT / "weights.csv")

    # 3. cost sensitivity
    costs = []
    for c in bc.COST_GRID_BPS:
        t, _ = metrics_table(panel, w_test, c, inference=False)
        costs.append(split_key(t).assign(cost_bps=c).reset_index())
    costs = pd.concat(costs, ignore_index=True)
    costs.to_csv(OUT / "cost_sensitivity.csv", index=False)

    # 4. robustness: next-open execution; leverage cap 1.5 for the mv rule
    rob = []
    t, _ = metrics_table(panel, w_test, bc.COST_BPS, inference=False, returns=bc.RETURNS_NEXT_OPEN)
    rob.append(split_key(t).assign(variant="next_open_execution"))
    w_lev = {k: v for k, v in all_weights(test, vol, w_max=bc.W_MAX_LEVERED).items() if k.endswith("|mv")}
    w_lev[PRIMARY] = w_test[PRIMARY]
    t, _ = metrics_table(panel, w_lev, bc.COST_BPS, inference=False)
    rob.append(split_key(t.drop(index=PRIMARY)).assign(variant=f"mv_wmax_{bc.W_MAX_LEVERED}"))
    pd.concat(rob).to_csv(OUT / "robustness.csv")

    # 5. by year and sub-period
    yearly = {k: yearly_stats(bts[k]) for k in bts}
    pd.concat(yearly, names=["strategy", "year"]).to_csv(OUT / "yearly_metrics.csv")
    sub = pd.concat({k: period_stats(bts[k], bc.SUBPERIODS) for k in bts}, names=["strategy", "period"])
    sub.to_csv(OUT / "subperiod_metrics.csv")

    # 6. decline events: does the strategy hold more SPY into recoveries than continuations?
    ev_all, ev_sum = [], {}
    for k, w in w_test.items():
        rows = event_rows(bts[k], bts["spy_bh"], w, events, panel.index)
        ev_all.append(rows.assign(strategy=k))
        ev_sum[k] = event_summary(rows)
    pd.concat(ev_all).to_csv(OUT / "event_trades.csv", index=False)
    ev_tab = split_key(pd.DataFrame(ev_sum).T)
    ev_tab.to_csv(OUT / "event_summary.csv")

    # 7. figures
    cs = cumulative_sse(test, outcomes)
    fig_cumulative(bts, OUT / "fig_cumulative.png")
    fig_yearly(yearly, OUT / "fig_yearly.png")
    fig_cum_sse(cs, OUT / "fig_oos_r2_cumulative.png")
    fig_weights(panel, w_test, test, OUT / "fig_weights.png")
    fig_costs(costs, OUT / "fig_cost_sensitivity.png")

    # 8. summary
    main_cols = ["ann_return", "ann_vol", "sharpe", "max_dd", "avg_weight", "turnover_pa",
                 "alpha_pa", "alpha_t_nw", "sharpe_diff_vs_spy", "sharpe_diff_p"]
    sign_rows = [f"{m}|sign" for m in MODEL_IDS]
    mv_rows = [f"{m}|mv" for m in MODEL_IDS]
    ov_rows = [f"{m}|decline_overlay" for m in MODEL_IDS]
    bench = ["spy_bh", "bil_bh", "static_mix"]
    cost_pivot = costs.pivot(index="strategy", columns="cost_bps", values="sharpe")
    cost_pivot.columns = [f"{c:g} bps" for c in cost_pivot.columns]
    ev_cols = ["n_events", "n_recovery", "n_continuation", "w_event_recovery",
               "w_event_continuation", "w_event_separation", "excess_vs_spy_all"]
    md = [
        "# Backtest and evaluation (Member 4)",
        "",
        "Generated by `python -m src.backtest.pipeline`. Decisions 2019-01-02 to 2025-12-31,",
        "next-close execution, returns realised 2019-01-03 to 2026-01-05. All strategies,",
        f"benchmarks included, run through the same engine at {bc.COST_BPS:g} bps per unit traded,",
        "starting from cash (entry trade charged), drift-aware turnover. Sharpe uses BIL as rf.",
        "",
        f"Gamma audit (design data): median validation pred/var of {bc.PRIMARY_MODEL} = {g:.2f};",
        f"frozen GAMMA = {bc.GAMMA}. Primary strategy, declared before the test run: `{PRIMARY}`.",
        "",
        "## Benchmarks",
        "",
        fmt(tab.loc[bench], main_cols[:6]),
        "",
        f"`static_mix` holds the primary strategy's average SPY weight every day.",
        "",
        "## Sign rule (w = 1 if forecast > 0 else 0)",
        "",
        fmt(tab.loc[sign_rows], main_cols),
        "",
        f"## Mean-variance rule (w = clip(pred / ({bc.GAMMA:g} x vol^2), 0, 1))",
        "",
        fmt(tab.loc[mv_rows], main_cols),
        "",
        "## Decline overlay (fully invested except on trigger days, where the sign rule applies)",
        "",
        fmt(tab.loc[ov_rows], main_cols),
        "",
        "alpha_pa / alpha_t_nw: CAPM alpha versus SPY excess returns with Newey-West",
        f"({bc.NW_LAGS} lags) t-statistic. sharpe_diff_p: two-sided p-value of the Sharpe",
        f"difference versus SPY, paired stationary bootstrap ({bc.BOOT_REPS} reps, mean block",
        f"{bc.BOOT_BLOCK} days, seed {bc.SEED}).",
        "",
        "## Cost sensitivity (Sharpe)",
        "",
        cost_pivot.loc[[PRIMARY, MV, OVERLAY, "ridge_base|sign", "ridge_vix|sign",
                        "spy_bh"]].round(3).to_markdown(),
        "",
        "## Decline events in the test period",
        "",
        "Weight decided on the event day, averaged over recoveries (+1) and continuations (-1).",
        "Positive separation means the strategy held more SPY into declines that recovered.",
        "",
        ev_tab.loc[sign_rows + mv_rows, ev_cols].astype(float).round(3).to_markdown(),
        "",
        "## Figures",
        "",
        "- `fig_cumulative.png`: wealth and drawdown versus SPY and the exposure-matched mix",
        "- `fig_yearly.png`: annual return and Sharpe by calendar year",
        "- `fig_oos_r2_cumulative.png`: cumulative squared-error gain over the historical mean",
        "- `fig_weights.png`: SPY weight of ridge_full (mv rule) with decline-trigger days",
        "- `fig_cost_sensitivity.png`: Sharpe as costs rise",
        "",
        "Validation-period (design) results: `validation_metrics.csv`. Robustness",
        "(next-open execution, leverage cap 1.5): `robustness.csv`. By year and sub-period:",
        "`yearly_metrics.csv`, `subperiod_metrics.csv`.",
    ]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT}")
    print(fmt(tab.loc[bench + [PRIMARY, f'{bc.PRIMARY_MODEL}|mv'], ], main_cols))


if __name__ == "__main__":
    main()
