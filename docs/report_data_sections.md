# Draft report text: data, timing and benchmarks (§4.1, §4.2 cleaning, §4.3)

Numbers come from the data snapshot downloaded on 4 Nov 2026
(`data/raw/manifest.json`), `reports/data/verification.md` and
`reports/data/benchmark_metrics.csv`.

## 4.1 Markets, assets and timing

**Assets.** The strategy allocates between two exchange-traded funds: the SPDR
S&P 500 ETF Trust (SPY), as the equity asset, and the SPDR Bloomberg 1-3 Month
T-Bill ETF (BIL), as the cash asset. Both are liquid US-listed ETFs that a
retail investor can hold. The Cboe Volatility Index (VIX, 30-day implied
volatility of the S&P 500) and the Cboe 3-Month Volatility Index (VIX3M) are
used as signals only. They are not traded.

**Frequency and calendar.** Daily, on the NYSE trading calendar (US Eastern
time). The base currency is USD, so no FX conversion is needed.

**Sample.** The evaluation sample runs from 2 January 2014 to 31 December 2025,
which is 3,018 trading days. Data is downloaded from 3 January 2012 to give
features with long lookbacks a warm-up period, and to 15 January 2026 so that
decisions made on the last sample days have realised outcomes.

**Decision and execution time.** SPY closes at 16:00 ET, but the official VIX
and VIX3M closes are struck at 16:15 ET. A signal that uses the day-t VIX close
is therefore not available at SPY's day-t closing auction, and trading at that
close would introduce look-ahead bias. We decide after 16:15 ET on day t and
execute with a market-on-close order at 16:00 ET on day t+1. The position earns
the close-to-close total return from t+1 to t+2. As a robustness check we also
report results with execution at the next open (open t+1 to open t+2).

**Benchmark.** Buy-and-hold SPY, with dividends reinvested (see §4.3).

**Data sources.** Daily open, high, low, close, volume, cash dividends and
splits for SPY and BIL come from Yahoo Finance via the `yfinance` Python
package (version 1.7.0). Daily VIX and VIX3M closes come from Cboe's public
index history files (cdn.cboe.com). The 3-month Treasury bill secondary market
rate (FRED series DTB3) is used only to cross-check BIL. All data is public and
is downloaded by `src/data/fetch.py`. The download time and SHA-256 hashes of
the snapshot used in this report are recorded in `data/raw/manifest.json`.

## 4.2 Data cleaning (contribution to the features section)

**Total returns.** Yahoo's raw close is split-adjusted but not
dividend-adjusted. We compute daily total returns from the raw close and the
cash dividend on the ex-dividend date, R_t = (P_t + D_t)/P_{t-1} - 1, and build
a back-adjusted price series by compounding these returns, anchored to the
latest raw close. Our SPY total returns match those implied by Yahoo's Adjusted
Close to within 6.2e-07 on average. The largest gap on any day is 2.5e-04, and
it occurs only on ex-dividend dates (on other days the largest gap is 9.9e-07).
The gap arises because Yahoo applies the multiplicative factor 1 - D_t/P_{t-1}
to prior prices, which differs from the additive formula above by about
R_t x D_t/P_{t-1}. For BIL the largest gap is 2.4e-06.

SPY paid four dividends in every sample year and had no splits. Dividends added
between 1.3 and 2.4 percentage points a year to SPY's return, 1.9 points on
average (Figure [X], `fig_spy_total_vs_price.png`).

BIL had a 1-for-2 reverse split on 30 November 2017. Yahoo's prices are already
adjusted for it, and the BIL return on that day is 0.00%, so no correction is
needed. BIL paid 12 monthly distributions in normal-rate years but few or none
when Treasury bill yields were near zero (2014 to 2016 and 2020 to 2021),
because the fund's income did not exceed its expense ratio. In those years BIL's
total return was slightly negative (-0.07% in 2014, -0.13% in 2015 and -0.10% in
2021), which the cash leg of the strategy correctly reflects. BIL's annual total
return tracks the average 3-month T-bill rate to within 0.25 percentage points
in every year except 2022 (0.62 points). That year rates rose quickly, and the
calendar-year average of a rising rate overstates what a rolling 1-3 month bill
portfolio earned.

**Alignment.** The SPY trading calendar is the master calendar. Cboe publishes
VIX values on 27 days when the NYSE was closed: US exchange holidays from
mid-2022 onward, when Cboe extended VIX calculation to its global trading hours,
plus the national day of mourning on 9 January 2025. These values are dropped.
No VIX or VIX3M value was missing on any SPY trading day, so the forward-fill
rule (at most one day, flagged) was never triggered
(`data/processed/missing_log.csv`).

**Outliers.** Returns are not winsorised. The five daily SPY moves larger than
8% (12, 13, 16 and 24 March 2020, and 9 April 2025) were checked against known
market events (the COVID-19 crash and the April 2025 tariff pause rally) and
kept, because they are genuine and are exactly the episodes the strategy
targets. VIX closes on known stress dates match published values: 40.74 on
24 August 2015, 37.32 on 5 February 2018, 82.69 on 16 March 2020 and 38.57 on
5 August 2024. Over the sample the median VIX/VIX3M ratio is 0.88, and the term
structure is inverted (ratio above 1) on 7.6% of days (`fig_vix_term_structure.png`).

## 4.3 Benchmarks

1. **Buy-and-hold SPY** (total return, dividends reinvested). This is the main
   investable benchmark.
2. **Buy-and-hold BIL** (total return). This is the cash alternative and the
   risk-free rate in all Sharpe ratios.
3. **Exposure-matched static mix.** SPY and BIL at fixed weights equal to the
   strategy's average SPY weight, rebalanced daily. With BIL as the risk-free
   rate, any static mix has exactly the same Sharpe ratio as SPY (0.72), because
   its excess return is a constant fraction of SPY's. A Sharpe ratio above
   SPY's can therefore only come from timing, not from holding less equity.
   The mix also gives a fair comparison for return and drawdown at the same
   average exposure.

All benchmarks and strategies are scored with the same code
(`src/metrics.py: perf_stats`) and on the same realisation dates.

| 2014 to 2025 | Ann. return | Ann. vol | Sharpe | Max drawdown |
|---|---|---|---|---|
| SPY buy-and-hold | 13.46% | 17.34% | 0.72 | -33.70% |
| BIL buy-and-hold | 1.68% | 0.27% | n/a | -0.24% |
| 50/50 SPY/BIL (example mix) | 7.82% | 8.67% | 0.72 | -17.85% |

Yearly returns and Sharpe ratios are in `reports/data/benchmark_by_year.csv`.
SPY's worst years were 2022 (-18.2%) and 2018 (-4.6%), and its maximum
drawdown came in the February-March 2020 crash. The 2022 decline was a slow,
rate-driven bear market, while 2020 was a fast volatility shock, so the sample
contains both kinds of decline the strategy aims to tell apart.
