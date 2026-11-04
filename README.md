# DSA5205 Project 2: Implied-volatility market timing (SPY vs BIL)

Does option-implied volatility, and in particular the VIX term structure, help
separate market declines that recover from declines that keep falling? The
strategy allocates daily between SPY (S&P 500 ETF) and BIL (T-bill ETF).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run commands

Run every command from the repository root.

| Step | Command | Output |
|---|---|---|
| Data stage, all steps | `python -m src.data.pipeline` | everything below |
| 1. Download raw data | `python -m src.data.fetch` | `data/raw/*.csv`, `data/raw/manifest.json` |
| 2. Clean and align | `python -m src.data.build` | `data/processed/panel.csv`, `panel.parquet`, `missing_log.csv` |
| 3. Benchmarks | `python -m src.data.benchmarks` | `data/processed/benchmarks.csv`, `reports/data/benchmark_*.csv`, `fig_benchmarks_cumulative.png` |
| 4. Verify | `python -m src.data.verify` | `reports/data/verification.md`, `checks.csv`, figures |
| Rebuild from the saved snapshot | `python -m src.data.pipeline --offline` | same as above, no network |
| Offline test on mock data | `python -m tests.test_pipeline` | prints `ALL TESTS PASSED` |

Expected output: `build` reports about 3,000 in-sample trading days
(2014-01-02 to 2025-12-31). `verify` exits with code 0 and prints no `FAIL`
lines. Any `WARN` lines are written to `reports/data/verification.md` for review.

## Layout

```
src/config.py          dates, tickers, URLs, timing convention, cleaning thresholds
src/metrics.py         perf_stats, stats_by_year, apply_weights (execution), turnover
src/data/fetch.py      downloads (Yahoo Finance, Cboe, FRED)
src/data/build.py      alignment, total returns, forward returns -> panel
src/data/verify.py     data-quality and leakage checks
src/data/benchmarks.py buy-and-hold benchmarks and metrics
tests/test_pipeline.py offline test on synthetic data
DATA.md                data dictionary and timing convention (read this first)
```

## Data sources

| Series | Source |
|---|---|
| SPY, BIL daily OHLCV, dividends, splits | Yahoo Finance via `yfinance`: https://finance.yahoo.com/quote/SPY/history/, https://finance.yahoo.com/quote/BIL/history/ |
| VIX daily close | Cboe: https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv |
| VIX3M daily close | Cboe: https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv |
| 3-month T-bill rate (optional cross-check) | FRED DTB3: https://fred.stlouisfed.org/series/DTB3 |

Raw downloads are not committed. `fetch` downloads them from the sources above and
records the download time, row counts and SHA-256 hashes in
`data/raw/manifest.json`. Yahoo occasionally revises history, so a fresh
download can differ slightly from the snapshot used in the report.

The processed files in `data/processed/` (panel, benchmarks, missing log) are
committed, so after cloning you can load them directly without running `fetch`.
Rerunning the pipeline overwrites them; only commit the result if the group
agrees to move to a new snapshot.

## Reproducibility

- Fixed sample and download windows in `src/config.py`. All paths are relative to the repo root.
- `build`, `benchmarks` and `verify` are deterministic given `data/raw/`.
- Test seed: `SEED = 5205` in `tests/test_pipeline.py`.
- Package versions are pinned in `requirements.txt`.
