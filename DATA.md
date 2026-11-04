# Data handover: read before building features or backtests

## 1. Timing convention (agreed: next close)

| Event | Time (US Eastern) |
|---|---|
| SPY and BIL close on day t | 16:00 |
| VIX and VIX3M official close on day t | 16:15 |
| **Decision** for day t | after 16:15 on day t, using only data dated t or earlier |
| **Execution** | market-on-close order at 16:00 on day **t+1** |
| **Holding period** | close(t+1) to close(t+2) |

Because the VIX close is struck 15 minutes after the SPY close, a signal that
uses VIX on day t cannot trade at SPY's close on day t. Doing so would be
look-ahead. This is why execution is at t+1 and not at t.

**Rule for everyone: a row dated t may use any column without the `fwd_`
prefix. `fwd_` columns are labels and outcomes only. Never use them as features.**

## 2. Files

| File | What it is |
|---|---|
| `data/processed/panel.parquet` / `panel.csv` | One row per SPY trading day, 2012-01-03 to mid-Jan 2026 (rows before 2014 are warm-up for feature lookbacks) |
| `data/processed/benchmarks.csv` | Daily benchmark returns over the sample, indexed by realisation date |
| `data/processed/missing_log.csv` | Every alignment issue: missing, forward-filled or dropped dates |
| `reports/data/verification.md` | All data checks, with yearly total vs price return tables |

Load the panel with:

```python
from src.data.build import load_panel
p = load_panel()                 # DatetimeIndex 'date'
s = p[p["in_sample"]]            # 2014-01-02 .. 2025-12-31
```

## 3. Column dictionary

Returns are simple (not log) daily returns in decimals.

### Known at decision time t (usable as features)

| Column | Meaning |
|---|---|
| `in_sample` | True for 2014-01-01 to 2025-12-31 |
| `spy_close`, `bil_close` | Raw close (split-adjusted by Yahoo, not dividend-adjusted) |
| `spy_open`, `bil_open` | Raw open |
| `spy_adj_close`, `bil_adj_close` | Total-return back-adjusted close, anchored to the last raw close. Use this for price-based features (moving averages, drawdowns) |
| `spy_adj_open`, `bil_adj_open` | Open adjusted with the same factor as that day's close |
| `spy_vendor_adj_close`, `bil_vendor_adj_close` | Yahoo's Adj Close, kept only for verification |
| `spy_volume`, `bil_volume` | Shares traded |
| `spy_div`, `bil_div` | Cash dividend per share with ex-date t (0 otherwise) |
| `spy_split`, `bil_split` | Split ratio on t (0 = none) |
| `spy_tr_cc`, `bil_tr_cc` | Total return close(t-1) to close(t): (P_t + D_t) / P_{t-1} - 1 |
| `spy_pr_cc`, `bil_pr_cc` | Price return close(t-1) to close(t) |
| `spy_tr_oo`, `bil_tr_oo` | Total return open(t-1) to open(t) |
| `spy_xs_cc` | `spy_tr_cc - bil_tr_cc` |
| `vix_close`, `vix3m_close` | Cboe index closes (16:15 ET) |
| `vix_close_ffilled`, `vix3m_close_ffilled` | True if that value was forward-filled from t-1 |
| `vix_ratio` | `vix_close / vix3m_close`. Above 1 means backwardation (near-term stress) |

### Outcomes for a decision made at t (labels only)

| Column | Meaning |
|---|---|
| `fwd_spy_tr` | SPY total return close(t+1) to close(t+2), i.e. `spy_tr_cc` two rows ahead |
| `fwd_bil_tr` | BIL total return close(t+1) to close(t+2) |
| `fwd_spy_xs` | `fwd_spy_tr - fwd_bil_tr`, the default prediction target |
| `fwd_spy_tr_oo`, `fwd_bil_tr_oo` | Same under the alternative next-open convention (open t+1 to open t+2), for robustness checks only |

Multi-day targets (for example the 5- or 21-day return after a decline) should
be built by compounding `spy_tr_cc` starting at row t+2, so they follow the
same execution lag.

## 4. Cleaning rules

- Master calendar: SPY trading days. BIL, VIX and VIX3M are joined onto it.
- Cboe values on days SPY did not trade are dropped (logged).
- A missing VIX or VIX3M value is forward-filled for at most 1 day and flagged.
  Longer gaps stay NaN (logged). None occur in the 2014 to 2025 sample if
  `verify` passes.
- Non-positive index levels are treated as missing.
- No winsorising of returns: large SPY moves (above 8% in a day) are listed in
  `verification.md` and checked against known events, not removed.
  Winsorising or clipping of features is a feature-stage decision and must be
  fitted on training data only.

## 5. Executing a strategy (use this, do not re-implement)

```python
from src.metrics import apply_weights, perf_stats
r = apply_weights(s, w, cost_bps=1.0)   # w: SPY weight indexed by DECISION date t
perf_stats(r, rf=s["bil_tr_cc"])        # r is indexed by REALISATION date
```

`apply_weights` holds `w[t]` over close(t+1) to close(t+2), puts the remainder
in BIL, and charges `cost_bps` per unit of weight traded at the close of t+1.
The test suite checks that `w = 1` reproduces buy-and-hold SPY exactly and that
`w[t]` never touches a return dated earlier than t+2.

## 6. Benchmarks

| Name | Definition |
|---|---|
| `spy_bh` | Buy-and-hold SPY, total return (required investable benchmark) |
| `bil_bh` | Buy-and-hold BIL, total return. This is also the risk-free rate for Sharpe ratios |
| `mix_50` | 50/50 SPY/BIL rebalanced daily. Example only: call `static_mix(panel, w_bar)` with the strategy's average SPY weight for an exposure-matched comparison |

Sharpe ratio = mean daily excess return over BIL / its std x sqrt(252).
