# Draft report text: labels, features and decline events (§4.2, §4.5 design split)

Numbers come from the processed snapshot in `data/processed/` and from
`src/features/features.ipynb`, `src/features/trigger_explorer.py` and
`src/features/events.py`. Data cleaning of prices and VIX is covered in
`docs/report_data_sections.md` (§4.2 Data cleaning).

## 4.2 Labels

**Label and horizon.** The label is the next-day excess return of SPY over BIL,
`fwd_spy_xs`, with horizon H = 1 trading day. The decision for day t is made
after the 16:15 ET VIX close and uses only data dated t or earlier. The order is
filled at the close of t+1, and the position is held from close(t+1) to
close(t+2), so the label is the total return of SPY minus that of BIL over
exactly that holding period. A model refitted on day T may only train on rows
t ≤ T − 2, because later labels are not yet realised.

**Secondary targets.** For robustness checks we also build 5- and 10-day
targets, `fwd_spy_xs_5` and `fwd_spy_xs_10`, by compounding the same daily
returns from close(t+1) to close(t+1+h). They overlap from one day to the next,
so training on day T may only use rows t ≤ T − 1 − h, and validation folds need
a gap of at least h + 1 days.

**Decline events (evaluation only).** To ask whether the term structure
separates declines that recover from declines that continue, we define decline
events and give each a recovery or continuation label. They are not the training
sample, because there are too few of them (see below); the model is trained on
every trading day and its out-of-sample predictions on event days are then
compared with the event labels.

- *Trigger.* Day t is a decline day when both hold: the close is at least
  s = 2% below its 20-day high, and the 5-day return is at least p = 1 standard
  deviation below zero, using 20-day realised volatility scaled by √5.
- *Non-overlapping events.* A trigger opens an observation window of
  H = 10 trading days; triggers inside the window are ignored. Whether a day
  starts an event depends only on earlier days.
- *Choosing s, p and H.* We swept s ∈ {2, 3, 4, 5}%, p ∈ {0.5, 1, 1.5, 2} and
  H ∈ {5, 10, 15, 20} (64 combinations) and counted events per year; no returns
  were looked at. With a severity floor of s ≥ 2% and p ≥ 1 (a real pull-back
  and at least a one-sigma sell-off), s = 2% and p = 1 give the most events and
  no year without one. H = 10 lets a barrier be reached in most windows (H = 5
  is too short) without cutting the event count as much as H = 20.
- *Labels.* Entry is the close of t+1. Barriers are set at t at
  ± K × 20-day volatility × √H with K = 0.75 (a median of ±2.0%, range 1.1% to
  7.6%). The event is a recovery (+1) if the close reaches the upper barrier
  first, a continuation (−1) if it reaches the lower barrier first, and
  unresolved (0) if neither happens within H days. K was set from the
  random-walk benchmark, under which about 15–20% of windows end unresolved
  at K = 0.75 and about 35% at K = 1. A binary label resolves the 0s by the
  sign of the H-day return.
- *Result.* 88 events in 2014–2025: 45 recoveries, 31 continuations,
  12 unresolved, between 2 and 11 per year.

## 4.2 Features

All features are computed for every trading day from data dated t or earlier.
Table [X] lists each feature with its formula, lookback window and economic
rationale (from `features.ipynb`, section 2).

| Group | Feature | Lookback | Why it might help |
|---|---|---|---|
| Decline state | `dec_dd` close / 20-day high − 1 | 20 d | depth of the pull-back; short-term losses partly reverse |
| | `dec_mom` 5-day return / (20-day vol × √5) | 5 d / 20 d | speed of the sell-off relative to recent volatility |
| | `dec_vol` std of daily returns | 20 d | recent realised risk (volatility-managed portfolios) |
| | `dec_cond` decline trigger flag | 20 d | marks the regime the study is about |
| Trend | `ma200_gap` close / 200-day MA − 1 | 200 d | pull-backs in bull and bear markets differ |
| VIX level | `vix_log` log VIX | same day | level of priced fear |
| | `vix_z` VIX z-score | 252 d | fear relative to the past year |
| | `vix_chg5` 5-day log change of VIX | 5 d | fear rising or fading |
| Term structure | `ts_ratio` VIX / VIX3M | same day | > 1 means acute short-term stress, typical near a capitulation low |
| | `ts_backw` 1 if `ts_ratio` > 1 | same day | discrete backwardation regime |
| Implied vs realised | `vrp` VIX/100 − annualised 20-day vol | 20 d | price of protection; negative in crashes |
| Interactions | `x_cond_ts` `dec_cond` × (`ts_ratio` − 1) | 20 d | lets the term-structure slope differ on decline days |
| | `x_mom_backw` `dec_mom` × `ts_backw` | 5 d / 20 d | lets the speed of a sell-off matter differently in backwardation, the clearest picture of capitulation |

**Reading the interaction.** In a linear model
y = b1·TS + b2·D + b3·(D × TS) + …, the term-structure slope is b1 on normal
days and b1 + b3 on decline days. b3 ≠ 0 means the term structure works
differently during declines, and b1 + b3 is its effect on decline days, which
is the research question. The ratio enters as `ts_ratio − 1`, a fixed centre
(distance from backwardation): the raw ratio sits around 0.9, which would make
`dec_cond × ts_ratio` almost identical to `dec_cond`, and centring on a sample
mean would leak future data.

**Feature selection.** We started from 17 candidates and kept 13 using three
rules that were declared before looking at any outcome and that use feature
values from the design period (2014–2018) only:

1. drop an interaction whose |corr| with one of its own parts exceeds 0.8
   (removes `dec_cond × vrp`, 0.85, and `dec_cond × vix_chg5`, 0.83: VIX almost
   always rises in a decline, so these are nearly rescaled `dec_cond`);
2. of two candidates with |corr| above 0.9, drop the later one (removes the
   5-day change of `ts_ratio`, 0.95 with the 5-day change of VIX);
3. keep a term-structure interaction only if it has at least 15 separate
   episodes of material variation in the design period, about three a year
   (|x| above 10% of its maximum; episodes more than 10 trading days apart).
   `dec_cond` × (`ts_ratio` − 1) has 25 episodes and `dec_mom` × `ts_backw` 17,
   so both are kept; `dec_dd` × (`ts_ratio` − 1) has 9 and is dropped. The
   interactions are informative only on the few days when both parts are
   active (in-sample the trigger coincides with backwardation on 114 days that
   fall into about 37 episodes), so an interaction needs enough such episodes
   to be estimated at all. The two kept interactions are not redundant: their
   correlation is −0.64 in the design period.

`select_features()` in `src/features/feature_build.py` applies the rules and
the test suite checks that it reproduces the final list.

**Feature sets.** Three nested sets let the modelling stage test what the VIX
information adds: `base` (the 5 price-based features), `vix` (`base` plus 6 VIX,
term-structure and implied-vs-realised features, 11 in total) and `full` (`vix`
plus the two interactions, 13). Each step adds few parameters, which matters because
daily returns are very noisy: as a rule of thumb, each extra feature costs about
1/n of out-of-sample R², and a plausible true R² for daily index returns is well
below 1%.

## 4.2 Scaling and cleaning of features

**Missing data.** No in-sample day has a missing feature. The longest lookback
is 252 days, covered by the 2012–13 warm-up rows, which are never used for
training or evaluation.

**Outliers.** The saved features are raw; nothing is deleted or clipped,
because the most extreme days (for example `vix_z` of 11.8 on 5 February 2018,
or a 29% drawdown from the 20-day high on 16 March 2020) are exactly the
declines we study. Clipping and scaling are fitted inside each training window
by `PastScaler`: continuous features are clipped at the 1% and 99% quantiles of
the training rows and standardised with the training mean and standard
deviation, and the same constants are applied unchanged to the rows being
predicted. The binary flags and the interaction are only standardised, not
clipped, because they are mostly 0 and a quantile bound in a calm window could
be 0.

Fitting the scaler on 2014–2018 and applying it to 2019–2025 shows why fitting
on past data matters: the test-period mean of `vix_log` is 1.06 training
standard deviations above the training mean, because VIX was higher on average
after 2019.

## 4.5 Design and test periods

Design choices (feature selection rules, parameter grids, selection criteria)
may use only the design period, 2 January 2014 to 31 December 2018
(`config.DESIGN_END`). The test period, 2 January 2019 to 31 December 2025
(`config.TEST_START`), is evaluated once, at the end. The design period
contains the 2015–16 and 2018 sell-offs; the test period contains 2020, 2022
and 2025.

## Leakage checks

- *Features.* Rebuilding all features on the panel cut at 25%, 50%, 75% and 95%
  of its length gives values identical, bit for bit, to the full build up to
  each cut (`tests/test_features.py`).
- *Events.* The same truncation test for the trigger and event flags; labels of
  events whose window ends before the cut are unchanged, and those whose window
  runs past it are left empty; no two event windows overlap
  (`tests/test_events.py`).
- *Targets.* The h-day target at t equals the direct compounding of returns
  dated t+2 to t+1+h, and changing the return dated t+1 leaves row t unchanged
  (`tests/test_targets.py`).
- *Scaling.* `PastScaler` statistics depend only on the training rows; the test
  checks that extreme test data does not move them.

## Design log (for §4.7: what was tried)

For transparency, the iterations behind the final features:

1. Trigger parameters: 64 (s, p, H) combinations, judged on event counts only;
   5 values of K, judged on the share of unresolved labels.
2. Features: a first set of 17 candidates; 14 after dropping three collinear
   candidates using full-sample (2014–2025) correlations; 12 after limiting the
   interactions to one; 13 in the final version, which keeps two.
3. Before the design/test split was fixed, we ran two exploratory rolling-ridge
   probes on 2017–2025 out-of-sample returns, and an earlier version kept only
   `dec_mom × ts_backw`, partly because its coefficient sign was stable in that
   probe. Because this used test-period outcomes, we replaced it with
   outcome-free rules applied to the design period only. A first version of
   rule 3 kept a single interaction and picked `dec_cond × (ts_ratio − 1)`. We
   then changed rule 3 to the 15-episode minimum, which keeps both. That
   threshold was set after seeing the episode counts (which use no returns),
   and keeping `dec_mom × ts_backw` is not fully independent of the earlier
   probe. The test period 2019–2025 was therefore seen once in exploration,
   and the results in §4.9 should be read with that in mind.
