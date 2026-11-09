# Backtest and evaluation (Member 4): report draft for §4.3, §4.8, §4.9

Run from the repository root (offline, about 10 seconds):

```bash
python -m src.backtest.pipeline          # writes reports/backtest/
python -m unittest tests.test_backtest -v
```

Inputs: the frozen model signals in `data/processed/models/` (loaded through
`src.models.handoff`, which checks their SHA-256 against the manifest), the
panel, `dec_vol` from the features and the decline events. Settings:
`src/backtest/config.py`. Generated numbers: `reports/backtest/summary.md`.

## 4.3 Benchmarks

All benchmarks run through the same engine, on the same realisation dates
(2019-01-03 to 2026-01-05), with the same costs and starting from cash:

| Benchmark | Definition | Why |
|---|---|---|
| `spy_bh` | 100% SPY, total return | Required investable benchmark |
| `bil_bh` | 100% BIL | Cash leg and risk-free rate for Sharpe ratios |
| `static_mix` | Constant SPY weight equal to the primary strategy's average weight (84%) | Exposure-matched: separates timing skill from simply holding less equity |
| `historical_mean` signal | The rolling 756-day mean forecast passed through the same rules | Forecast baseline: no conditioning information at all |

Under the sign rule the historical mean is always positive, so it reproduces
buy-and-hold exactly; under the mean-variance rule it becomes a pure
volatility-managed portfolio (weight ∝ 1/σ², Moreira and Muir 2017). This makes
the comparison "VIX information versus volatility timing alone" explicit.

## 4.8 From predictions to positions

**Score-to-weight rules.** Three long-only rules, all declared in
`src/backtest/config.py` before the 2019–2025 run. Weights are SPY weights; the
rest is held in BIL. No shorting and no leverage in the main run.

| Rule | Weight on decision day t | Idea |
|---|---|---|
| `sign` (**primary**) | 1 if forecast > 0, else 0 | Simplest threshold: hold equity only when its expected excess return is positive |
| `mv` | clip(forecast / (γ · σ²ₜ), 0, 1), γ = 5, σₜ = 20-day SPY volatility (`dec_vol`) | Single-asset mean-variance weight; Kelly et al. (2024) style π = ŷ scaled by risk |
| `decline_overlay` | 1 on normal days; sign rule on decline-trigger days only (`dec_cond` = 1) | The research question as a trade: after a decline, stay in or step out? |

**Primary strategy:** `ridge_full` with the `sign` rule, declared before the
test run. The primary model was already fixed by the model stage.

**Design choices made on 2017–2018 validation predictions only:**

- γ = 5: the median of ridge_full's forecast / σ² over validation decisions is
  4.84, so the median validation day is about fully invested. The pipeline
  recomputes and prints this as an audit.
- Validation forecasts were not degenerate: ridge models predicted a positive
  excess return on 69–87% of validation days.
- Validation backtests (`reports/backtest/validation_metrics.csv`) already
  showed that the sign and mv rules underperformed buy-and-hold on design data.
  The ridge_full sign rule had a Sharpe of 0.22 against SPY's 0.50. We
  nevertheless kept the predeclared primary rule rather than choosing a
  different one, and report the result.

**Execution timing.** Decide after 16:15 ET on day t (once VIX and VIX3M close),
send a market-on-close order for t+1 and earn close(t+1) → close(t+2). The
engine (`src/backtest/engine.py`) is tested to agree exactly with the shared
`src.metrics.apply_weights` for 0/1 weights. A weight change on t cannot affect
any return dated before t+1 (its trade cost) or the gross return before t+2
(`tests/test_backtest.py`).

**Costs.** 5 bps per unit of SPY weight traded (one-way). This is conservative
for SPY market-on-close orders, where the spread is about 0.3 bp. The engine
adds two conventions the shared helper leaves open:
1. The book starts in cash, so the initial entry trade is charged for every
   strategy and benchmark alike.
2. Turnover is measured against the drifted weight, so fractional weights pay
   for daily rebalancing back to target.

Sensitivity: 0, 1, 5, 10 and 20 bps.

**Risk limits.** Weight in [0, 1]; no leverage or shorting. A cap of 1.5
(borrowing at the BIL rate) is a robustness check only.

## 4.9 Out-of-sample evaluation

**Training–testing correspondence.** For each decision day T in 2019–2025, the
model stage refits scaler and ridge on the latest 756 fully realised labels
(features dated s enter only once their outcome date s+2 ≤ T), predicts T, and
rolls forward. Alphas, feature sets, the primary model, the allocation rules,
γ and costs were all frozen beforehand. Each forecast is then turned into a
weight, executed at close T+1 and charged costs. There are 1,760 decisions per
strategy.

**Main results, 5 bps** (`reports/backtest/summary.md`):

| Strategy | Ann. return | Ann. vol | Sharpe | Max DD | Avg. weight | Turnover/yr | ΔSharpe vs SPY (p) |
|---|---|---|---|---|---|---|---|
| SPY buy-and-hold | 17.8% | 19.8% | 0.80 | −33.7% | 100% | 0.1 | – |
| Static mix (84% SPY) | 15.5% | 16.7% | 0.80 | −29.0% | 84% | 0.4 | – |
| **ridge_full, sign (primary)** | 13.2% | 17.0% | 0.67 | −28.9% | 84% | 25.3 | −0.13 (0.35) |
| ridge_full, mean-variance | 7.5% | 12.4% | 0.44 | −28.9% | 66% | 35.4 | −0.36 (0.14) |
| ridge_full, decline overlay | 18.5% | 18.3% | 0.88 | −25.6% | 99% | 5.3 | +0.08 (0.45) |
| ridge_base, decline overlay | 18.9% | 18.8% | 0.88 | −26.7% | 99% | 1.9 | +0.08 (0.56) |
| historical mean, mean-variance (vol-timing only) | 12.0% | 12.1% | 0.79 | −20.7% | 81% | 4.4 | −0.01 (0.96) |

The p-value is from a paired stationary bootstrap of the Sharpe difference
(2,000 replications, mean block of 21 days, seed 5205). CAPM alphas against SPY
with Newey–West (5-lag) t-statistics are in `strategy_metrics.csv`. Across all
18 model–rule strategies the largest |t| is 1.02. `static_mix` shows a large t
only because it is almost an exact multiple of SPY, so its small
rebalancing-cost alpha has a near-zero standard error. Ignore that one.

**Required plots** (`reports/backtest/`):
1. `fig_oos_r2_cumulative.png`: cumulative squared-error gain over the rolling
   historical mean (Goyal–Welch style). ridge_full ends ahead (OOS R² = +0.23%);
   ridge_vix and ridge_base end behind. Most of the movement comes from a few
   high-volatility days (March 2020, April 2025).
2. `fig_cumulative.png`: growth of $1 and drawdowns against SPY and the
   exposure-matched mix.
3. `fig_yearly.png`: annual return and Sharpe by calendar year.
   `subperiod_metrics.csv` splits the sample into 2019–2021 and 2022–2025.

Also: `fig_weights.png` (the SPY weight path with decline days marked) and
`fig_cost_sensitivity.png`.

**Robustness** (`robustness.csv`, `cost_sensitivity.csv`):
- *Costs:* the sign and mv rules trade 25–35 times a year, so costs matter.
  ridge_full's sign-rule Sharpe goes from 0.74 at 0 bps to 0.44 at 20 bps. The
  overlay trades about 5 times a year and is almost cost-insensitive (0.89 to
  0.84). Even at zero cost the sign rule is below SPY (0.74 vs 0.80).
- *Next-open execution:* conclusions unchanged. The primary rule's Sharpe is
  0.56 against SPY's 0.81; the overlay's is 0.75.
- *Leverage cap 1.5 for mv:* does not help (ridge_full mv: 0.41).
- *Sub-periods:* the overlay's advantage comes entirely from 2019–2021, mostly
  the March 2020 crash (Sharpe 1.47 vs 1.17). In 2022–2025 it slightly trails
  SPY (0.42 vs 0.47).

**Decline events** (`event_summary.csv`, `event_trades.csv`): 53 test-period
events (27 recoveries, 20 continuations, 6 unresolved). If the VIX features
separated the two types, the strategy would hold more SPY into recoveries than
into continuations. It does the opposite. Under the sign rule, ridge_full's
event-day weight is 0.78 in recoveries against 0.95 in continuations
(separation −0.17). ridge_base is about neutral (+0.01).

## Explanatory test: does the term structure predict rebounds after declines?

The ridge experiment asks whether VIX information improves *forecasts and
trades*. The research question is narrower: does the term structure *on
decline days* separate rebounds from continuations? We test it directly with
the small regression proposed in the feature-stage README (commit 6565d4d,
5 Nov 2026), run unchanged (`python -m src.analysis.interaction_test`,
`reports/analysis/interaction_tests.md`):

- **A:** excess return on `ts_ratio`, `dec_cond` and `x_cond_ts` =
  `dec_cond` × (`ts_ratio` − 1). b3 > 0 would mean declines starting in
  backwardation are followed by higher returns.
- **B:** the continuous version with `dec_mom` and `x_mom_ts`. Here b3 < 0
  supports the hypothesis, because both terms flip sign on a sharp decline in
  backwardation.
- Both are fitted by OLS with Newey–West errors, for 1- and 10-day targets,
  over 2014–2018, 2019–2025 and 2014–2025.

| Spec, horizon | Design 2014–18: b3 (p) | Test 2019–25: b3 (p) | Full: b3 (p) |
|---|---|---|---|
| A, 1 day | +0.012 (0.48) | −0.038 (0.21) | −0.015 (0.42) |
| A, 10 days | +0.034 (0.34) | −0.148 (0.12) | −0.059 (0.35) |
| B, 1 day | −0.008 (0.04) | +0.005 (0.46) | −0.001 (0.74) |
| B, 10 days | −0.025 (0.04) | +0.056 (0.07) | +0.017 (0.43) |

Both specifications point the hypothesised way in 2014–2018. Spec B is
significant at 5% there, and this is the pattern the feature stage saw while
exploring. In 2019–2025 **both flip sign**, and over the full sample neither is
significant. Twelve regressions were run, so one or two p-values below 0.05 are
expected by chance.

The relationship is not stable over time, which matches the backtest: the
ridge models learnt the 2014–2018 pattern and it reversed out of sample. The
effective sample is the number of decline episodes (about 88), not the 289
decline days, so the power is low.

Caveat: this analysis was proposed before any model or backtest result, but
after the feature stage's exploratory look at declines in backwardation, and it
was run after the test results were known. It is reported in full.

**The model stage's frozen recovery classifier agrees**
(`reports/recovery_evaluation/report.md`). It is an exploratory extension,
designed after the one-day results were seen, that predicts the barrier label
directly on all 175 test-period trigger days. The term-structure classifier
does *worse* than simply predicting the historical class mix: log loss
+4.1%, approximate 95% interval [−2.7%, +11.8%]. The price-only classifier
ranks recoveries slightly better (recovery-vs-rest AUC 0.56 vs 0.52). Separately,
a design-only sensitivity check with a wider penalty grid selects α = 100,000
for every feature set, which is almost pure shrinkage to the historical mean
(`reports/models/extensions/summary.md`). The regression test, the classifier
and the backtest all point the same way: no stable information in the term
structure about rebounds after declines.

## Interpretation (what worked, what did not)

- **No evidence that VIX information improves timing after costs.** Every
  ridge model under the sign and mv rules has a lower Sharpe than buy-and-hold,
  and none of the differences is statistically significant. Matching exposure
  (`static_mix`, Sharpe 0.80) shows the shortfall is poor timing, not just
  lower equity exposure.
- **Forecast accuracy does not translate into trading gains.** ridge_full has
  the only positive OOS R² (+0.23%), yet the lowest sign-rule Sharpe among the
  ridge models. A tiny R² gain concentrated in a few volatile days does not
  survive daily trading at 25 one-way units a year.
- **The decline overlay is the one positive result, and it is fragile.** It is
  ahead of SPY on Sharpe (0.88 vs 0.80) and drawdown (−25.6% vs −33.7%), but:
  - the result is not significant (p = 0.45);
  - the edge comes from 2020 alone;
  - the price-only model performs identically, so the gain is not attributable
    to the term structure.

  It was declared before the test run but is not the primary strategy, and we
  do not promote it after the fact.
- **Volatility timing alone** (historical mean with the mv rule) roughly
  matches SPY's Sharpe with a much smaller drawdown (−20.7%). This is consistent
  with Moreira and Muir (2017), and it is a stronger, simpler baseline than any
  of the VIX models.
- **Event analysis contradicts the hypothesis in this sample.** The model
  reduces exposure more often before declines that later recover. With 47
  resolved events the power is low, so this is "no evidence for", not
  "evidence against".
- **Limitations:**
  - 2019–2025 is not a pristine holdout (see the model-stage disclosure).
  - All alphas landed at the top of the grid (1000), so forecasts are heavily
    shrunk.
  - The overlay depends on a single crash.
  - Costs are a flat per-unit charge with no market impact.
