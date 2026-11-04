# Data verification report

Sample 2014-01-02 to 2025-12-31, 3018 trading days.

| Check | Status | Detail |
|---|---|---|
| trading days per year in [248, 254] | PASS | 2014:252, 2015:252, 2016:252, 2017:251, 2018:251, 2019:252, 2020:253, 2021:252, 2022:251, 2023:250, 2024:252, 2025:250 |
| sample starts and ends in configured window | PASS | 2014-01-02 -> 2025-12-31 |
| dates strictly increasing, no duplicates | PASS |  |
| SPY prices positive | PASS |  |
| SPY High >= Low | PASS | 0 violations |
| SPY Close within [Low, High] | PASS | 0 violations |
| SPY splits already reflected in prices (|return| < 5% on split dates) | PASS | no splits |
| SPY pays 4 dividends every year | PASS | 2014:4, 2015:4, 2016:4, 2017:4, 2018:4, 2019:4, 2020:4, 2021:4, 2022:4, 2023:4, 2024:4, 2025:4 |
| SPY total return matches Yahoo Adj Close (tol 0.0005) | PASS | max |diff| 2.52e-04 (div days 2.52e-04, other days 9.92e-07); mean 6.17e-07; 0 days over tolerance |
| SPY adjusted close reproduces total return exactly | PASS | CSV stores 10 significant digits |
| BIL prices positive | PASS |  |
| BIL High >= Low | PASS | 0 violations |
| BIL Close within [Low, High] | PASS | 0 violations |
| BIL splits already reflected in prices (|return| < 5% on split dates) | PASS | 2017-11-30 ratio 0.5, return that day +0.0000% |
| BIL dividend count per year (informational; see BIL vs T-bill check) | PASS | 2014:0, 2015:0, 2016:2, 2017:12, 2018:12, 2019:12, 2020:5, 2021:0, 2022:9, 2023:12, 2024:12, 2025:12 |
| BIL total return matches Yahoo Adj Close (tol 0.0005) | PASS | max |diff| 2.35e-06 (div days 2.35e-06, other days 1.17e-06); mean 1.48e-07; 0 days over tolerance |
| BIL adjusted close reproduces total return exactly | PASS | CSV stores 10 significant digits |
| SPY daily moves above 8% (review against known events) | PASS | 2020-03-12 -9.57%; 2020-03-13 +8.55%; 2020-03-16 -10.94%; 2020-03-24 +9.06%; 2025-04-09 +10.50% |
| BIL daily |return| < 0.5% (T-bill ETF sanity) | PASS | max 0.0765% |
| vix_close no missing values after ffill | PASS | 0 NaN, 0 forward-filled |
| vix_close within [5, 100] | PASS | min 9.14, max 82.69 |
| vix3m_close no missing values after ffill | PASS | 0 NaN, 0 forward-filled |
| vix3m_close within [5, 100] | PASS | min 11.85, max 72.98 |
| VIX close on 2015-08-24 ~ 40.74 | PASS | got 40.74 |
| VIX close on 2018-02-05 ~ 37.32 | PASS | got 37.32 |
| VIX close on 2020-03-16 ~ 82.69 | PASS | got 82.69 |
| VIX close on 2024-08-05 ~ 38.57 | PASS | got 38.57 |
| VIX/VIX3M ratio plausible (median 0.8 to 0.95) | PASS | median 0.880; backwardation on 7.6% of days |
| fwd_spy_tr[t] == spy_tr_cc[t+2] (next-close convention) | PASS |  |
| fwd_bil_tr[t] == bil_tr_cc[t+2] | PASS |  |
| every in-sample decision date has a forward return | PASS | 0 missing |
| same-day and forward SPY returns are different series (|corr| < 0.3) | PASS | corr 0.067 |
| BIL annual return within 1pp of average T-bill rate | PASS | max gap 0.62% (BIL has a fee of ~0.14% a year) |

## SPY yearly total vs price return

|   date |   total_return |   price_return |   dividends_paid |   n_dividends |   dividend_contribution |
|-------:|---------------:|---------------:|-----------------:|--------------:|------------------------:|
|   2014 |         0.1346 |         0.1129 |           3.8360 |        4.0000 |                  0.0217 |
|   2015 |         0.0125 |        -0.0081 |           4.2060 |        4.0000 |                  0.0206 |
|   2016 |         0.1200 |         0.0964 |           4.5390 |        4.0000 |                  0.0236 |
|   2017 |         0.2170 |         0.1938 |           4.8020 |        4.0000 |                  0.0232 |
|   2018 |        -0.0456 |        -0.0635 |           5.1010 |        4.0000 |                  0.0179 |
|   2019 |         0.3122 |         0.2879 |           5.6190 |        4.0000 |                  0.0244 |
|   2020 |         0.1837 |         0.1616 |           5.6910 |        4.0000 |                  0.0221 |
|   2021 |         0.2874 |         0.2704 |           5.7150 |        4.0000 |                  0.0171 |
|   2022 |        -0.1817 |        -0.1948 |           6.3200 |        4.0000 |                  0.0131 |
|   2023 |         0.2619 |         0.2429 |           6.6330 |        4.0000 |                  0.0190 |
|   2024 |         0.2489 |         0.2330 |           7.0660 |        4.0000 |                  0.0158 |
|   2025 |         0.1772 |         0.1635 |           7.2810 |        4.0000 |                  0.0136 |

## BIL yearly total vs price return

|   date |   total_return |   price_return |   dividends_paid |   n_dividends |   dividend_contribution |
|-------:|---------------:|---------------:|-----------------:|--------------:|------------------------:|
|   2014 |        -0.0007 |        -0.0007 |           0.0000 |        0.0000 |                  0.0000 |
|   2015 |        -0.0013 |        -0.0013 |           0.0000 |        0.0000 |                  0.0000 |
|   2016 |         0.0011 |         0.0004 |           0.0600 |        2.0000 |                  0.0007 |
|   2017 |         0.0069 |        -0.0000 |           0.6280 |       12.0000 |                  0.0069 |
|   2018 |         0.0174 |         0.0007 |           1.5160 |       12.0000 |                  0.0167 |
|   2019 |         0.0203 |        -0.0003 |           1.8710 |       12.0000 |                  0.0206 |
|   2020 |         0.0040 |         0.0010 |           0.2740 |        5.0000 |                  0.0030 |
|   2021 |        -0.0010 |        -0.0010 |           0.0000 |        0.0000 |                  0.0000 |
|   2022 |         0.0140 |         0.0004 |           1.2360 |        9.0000 |                  0.0136 |
|   2023 |         0.0494 |        -0.0009 |           4.5010 |       12.0000 |                  0.0503 |
|   2024 |         0.0519 |         0.0004 |           4.5970 |       12.0000 |                  0.0515 |
|   2025 |         0.0415 |        -0.0005 |           3.7710 |       12.0000 |                  0.0420 |

## BIL vs 3-month T-bill rate (FRED DTB3)

|   date |   BIL_total_return |   DTB3_avg_rate |     gap |
|-------:|-------------------:|----------------:|--------:|
|   2014 |            -0.0007 |          0.0003 | -0.0010 |
|   2015 |            -0.0013 |          0.0005 | -0.0018 |
|   2016 |             0.0011 |          0.0032 | -0.0021 |
|   2017 |             0.0069 |          0.0093 | -0.0025 |
|   2018 |             0.0174 |          0.0194 | -0.0020 |
|   2019 |             0.0203 |          0.0206 | -0.0003 |
|   2020 |             0.0040 |          0.0035 |  0.0004 |
|   2021 |            -0.0010 |          0.0005 | -0.0014 |
|   2022 |             0.0140 |          0.0202 | -0.0062 |
|   2023 |             0.0494 |          0.0507 | -0.0013 |
|   2024 |             0.0519 |          0.0497 |  0.0022 |
|   2025 |             0.0415 |          0.0407 |  0.0008 |
