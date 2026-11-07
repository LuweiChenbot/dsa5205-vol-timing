# DSA5205 Project 2: Implied-Volatility-Aware Market Timing (SPY vs BIL)

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
| 5. Build model features for every trading day | run `src/features/features.ipynb` | `data/processed/features.csv`, `features.parquet`, `feature_sets.json` |
| Rebuild from the saved snapshot (optional) | `python -m src.data.pipeline --offline` | same as above, no network |
| Offline test on mock data (optional)| `python -m tests.test_pipeline` | prints `ALL TESTS PASSED` |
| Build decline events and labels (optional) | `python -m src.features.events` | `data/processed/events.csv`, `events.parquet` |
| Event leakage test (optional)| `python -m tests.test_events` | prints `ALL TESTS PASSED` |
| 6. Ridge models and walk-forward validation | `python -m src.models.pipeline` | `data/processed/models/`, `reports/models/summary.md` |
| Model and handoff tests | `python -m unittest tests.test_models -v` | timing, selection, forecasts and export checks |

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

src/features/feature_build.py  features, feature sets, selection rules, PastScaler
src/features/features.ipynb  walk-through of the features, selection and checks
src/features/targets.py  1-, 5- and 10-day targets
src/features/events.py decline trigger, non-overlapping events, recovery/continuation labels
src/features/trigger_explorer.py  sweep of the event parameters (s, p, H)

tests/test_pipeline.py offline test on synthetic data
tests/test_features.py leakage checks for the features and PastScaler
tests/test_events.py   leakage checks for the event columns
tests/test_targets.py  timing checks for the targets
src/models/           ridge fitting, walk-forward selection, diagnostics and Member 4 loader
tests/test_models.py   model leakage, numerical and handoff checks

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

The processed files in `data/processed/` (panel, benchmarks, missing log, events, features) are
committed, so after cloning you can load them directly without running `fetch`.
Rerunning will the pipeline overwrites them.

## Features, events and scaling

### Final feature set

The prepared features are held in `data/processed/features.parquet`, and each row is indexed by the decision date t. The file covers the in-sample days only (2014-01-02 to 2025-12-31); the 2012–13 warm-up rows are left out. `data/processed/feature_sets.json` splits the whole feature set into three categories, allowing further evaluation on term-structure-specific features.

| Set | Features |
|---|---|
| `base` (5) | `dec_dd`, `dec_mom`, `dec_vol`, `dec_cond`, `ma200_gap` |
| `vix` (11) | `base` + `vix_log`, `vix_z`, `vix_chg5`, `ts_ratio`, `ts_backw`, `vrp` |
| `full` (14) | `vix` + `x_cond_ts`, `x_mom_backw`, `x_mom_ts` |

| Feature | Definition |
|---|---|
| `dec_dd` | close / 20-day high − 1 |
| `dec_mom` | 5-day return / (20-day vol × √5) |
| `dec_vol` | std of daily SPY total returns, 20 days |
| `dec_cond` | 1 if `dec_dd` ≤ −2% and `dec_mom` ≤ −1 (decline trigger) |
| `ma200_gap` | close / 200-day moving average − 1 |
| `vix_log` | log VIX |
| `vix_z` | VIX z-score over 252 days |
| `vix_chg5` | 5-day log change of VIX |
| `ts_ratio` | VIX / VIX3M (above 1 = backwardation) |
| `ts_backw` | 1 if `ts_ratio` > 1 |
| `vrp` | VIX/100 − annualised 20-day realised vol |
| `x_cond_ts` | `dec_cond` × (`ts_ratio` − 1) |
| `x_mom_backw` | `dec_mom` × `ts_backw` |
| `x_mom_ts` | `dec_mom` × (`ts_ratio` − 1) |

We specify lookbacks, rationale and the selection rules for each feature in `src/features/features.ipynb`. You may load the prepared features into your model by doing the following,

```python
import json, pandas as pd
from src import config
from src.data.build import load_panel
from src.features.feature_build import load_features

panel, feat = load_panel(), load_features()
sets = json.loads((config.PROCESSED_DIR / "feature_sets.json").read_text())
X, y = feat[sets["full"]], panel.loc[feat.index, "fwd_spy_xs"]
```

### Decline events (for evaluation)

Our primary prediction target is the one-day forawrd excess return,
$$\text{fwd\_spy\_xs}=(Close^{SPY}_{t+2}/Close^{SPY}_{t+1} - 1)-(Close^{BIL}_{t+2}/Close^{BIL}_{t+1} - 1)$$

This target is mainly a vehicle for training the model on every trading day, and we do not judge the model by how accurately it forecasts returns in absolute terms. Instead, our research question is comparative, answering if adding term-structure features to a baseline model help distinguish declines that are followed by a rebound form those that continue on days when a decline is triggered.

To this end, we prepared  `data/processed/events.csv` and it marks decline events and relevant metrics according to our prescribed definition (see `src/features/events.py`). Then, we could compare the model's prediction on each marked event with our lebels.

A brief guide on how we portrait an event:
- An event starts on a day with `dec_cond` = 1, and triggers in the next 10 trading
  days are ignored. (for definition of a trigger, see `src/features/events.py`)
- From the close of t+1, the SPY close is followed for 10 trading days against
  barriers of ±0.75 × 20-day vol × √10. (the barriers is a function of triggered day t, so it is not a fixed term)
- `fwd_ev_label`: +1 recovery (return path touches upper barrier first), −1 continuation (lower barrier is touched first), 0 neither. 
- `fwd_ev_days`: days until a barrier was reached starting from t.
- Together, we created 88 events in 2014–2025 (45 recovery, 31 continuation, 12 unresolved).

To load them, do:
```python
from src.features.events import load_events
ev = load_events()
ev = ev[ev["ev_start"] & panel["in_sample"]]
```

### Scaling

Features are saved unscaled at first. Inside each refit, we fit `PastScaler` on the
training rows only and apply it to both training and prediction rows:

When training on your model, please apply the scaling with the following:

```python
from src.features.feature_build import PastScaler
sc = PastScaler().fit(X_train)
X_train_z, X_new_z = sc.transform(X_train), sc.transform(X_new)
```

## Models and validation

The implemented model stage compares price-only ridge, VIX-level ridge, ridge
with term structure, and the full interaction model, against a rolling historical
mean. A fifth ridge model holds the VIX-level penalty fixed when adding term
structure. Daily fits use exactly 756 completed labels; alpha is selected on
2017–2018 validation labels realised by 2018-12-31, then frozen for 2019–2025.
`ridge_full` is the predeclared primary model.

Run `python -m src.models.pipeline` from the root. No downloads are needed.
See [the model protocol and Member 4 guide](docs/models_and_validation.md) for
the methodology, file schemas and complete examples, and
[the generated results](reports/models/summary.md) for the actual comparisons.

```python
from src.models.handoff import load_signal
signal = load_signal("ridge_full")
# Indexed by decision date; includes prediction, execution_date and label_end_date.
```

Signals contain decimal excess-return predictions, not portfolio weights or
probabilities. Outcomes are separate files. The final 2025 decision earns a
return ending on 2026-01-05, so retain the panel's January buffer in backtests.

**Research-history limitation:** earlier feature exploration examined 2017–2025
outcomes, as documented in `docs/report_features_sections.md`. The implemented
walk-forward forecasts respect information timing, but 2019–2025 is not a
pristine, untouched research holdout. Negative forecast or trading results
should be reported, not used to retune the experiment.

### Feature-stage handover and possible extensions

The suggestions below are optional extensions from the feature stage. They are
not part of the implemented one-day ridge experiment above.

**Rebuild and check** (just for a fresh build of feature sets)

```bash
python -m src.features.feature_build
python -m src.features.targets
python -m src.features.events
python -m tests.test_features
python -m tests.test_events
python -m tests.test_targets
```

**Targets.** The main label is `fwd_spy_xs` already in the panel: SPY minus BIL from
close(t+1) to close(t+2). Additionally, the script `src/features/targets.py` writes
`data/processed/targets.csv` with `fwd_spy_xs_1`, `fwd_spy_xs_5` and
`fwd_spy_xs_10`. They are the same excess return over 1, 5 and 10 days from close(t+1).
So, `fwd_spy_xs_1` equals the panel's `fwd_spy_xs`, and the 5- and 10-day targets can be used for robustness check, namely train&evaluating your model on them in case the 1 day return series is too noisy.

**Points to keep when training:**

- A refit on day T can use rows t ≤ T − 2 for `fwd_spy_xs`, and t ≤ T − 1 − h
  for an h-day target. Cross-validation folds need a gap of h + 1 days.
- Design choices (grids, selection rules) should use **2014–2018** only
  (`config.DESIGN_END`). Keep **2019–2025** (`config.TEST_START`) for the final
  evaluation.
- Comparing `base`, `vix` and `full` shows what the VIX information adds.

**Suggested Analysis** (closer to the research question than predicting
returns inside the event window):

1. *Test the statistical significance of the interaction coefficients.*
   - OLS of `fwd_spy_xs` on a small model, e.g. `ts_ratio`, `dec_cond`,
     `x_cond_ts` (or `dec_mom` with `x_mom_ts` / `x_mom_backw`), with
     Newey–West standard errors (statsmodels `cov_type="HAC"`; at least h lags
     for an h-day target). statsmodels is not yet in `requirements.txt`.
   - Test b3 (the interaction) and b1 + b3 (the term-structure effect on
     decline days).
   - Ridge gives no standard errors, and the 14 features are correlated, so a
     small model gives more usable standard errors.
   - The effective sample is roughly 37–71 episodes, so power is low; a
     non-significant result is still reportable.
2. *Check if the shape of the path after trigger days aligns with our predictions.*
   - For each test-period event day T, predict the cumulative excess return for
     h = 1, …, 10 using only data at T (one model per horizon; set `HORIZONS`
     in `targets.py` to `range(1, 11)`), and compare with the realised path.
   - Possible measures: correlation of predicted and realised paths per event,
     sign agreement at day 10, and average paths by label as a figure.
   - With strong shrinkage the predicted paths tend to be flat, so this may
     work better with a small model or the classifier below.
3. *Classification.*
   - Predict the barrier label (±1) instead of the return.
   - Event start days give 88 labels (about 35 in 2014–2018), which is few.
     Labelling every trigger day (289) with the same barrier rule gives more;
     those windows overlap, so leave an 11-day gap between training and test
     rows. This needs a small change in `events.py` (not done yet).
   - Logistic regression with strong L2 and 2–4 features; report AUC and
     accuracy against the base rate (about 51% recoveries), with and without
     the VIX features. The predicted probability can be mapped to a weight.
   - Unresolved windows (label 0): use `fwd_ev_label_bin` or drop them; report
     both.

**Open questions for Parts 3 and 4:** whether the SPY weight may exceed 1, and
the default weight on days without a trigger. In exploratory runs, declines
that start in backwardation were followed by stronger rebounds than those in
contango, but both had non-negative average returns, so a weight capped at 1
leaves little room to gain from the signal.

## Reproducibility

- Fixed sample and download windows in `src/config.py`. All paths are relative to the repo root.
- `build`, `benchmarks` and `verify` are deterministic given `data/raw/`.
- Test seed: `SEED = 5205` in `tests/test_pipeline.py`.
- Package versions are pinned in `requirements.txt`.
