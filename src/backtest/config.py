"""Predeclared backtest settings (Member 4). Frozen before the 2019-2025 run.

Every choice below was fixed using the 2017-2018 validation predictions only
(design data). Nothing here may be changed after looking at test results; a
change must be logged as a deviation in docs/backtest_and_evaluation.md.
"""
from __future__ import annotations

# ---------------------------------------------------------------- allocation rules
# All rules are long-only in SPY, remainder in BIL (no shorting, no leverage in
# the main run). Weights are indexed by DECISION date t and executed at close t+1.
RULES = ("sign", "mv", "decline_overlay")
PRIMARY_RULE = "sign"
PRIMARY_MODEL = "ridge_full"          # predeclared by the model stage

# "mv": mean-variance weight w = clip(pred / (GAMMA * dec_vol^2), 0, W_MAX).
# dec_vol is the 20-day std of SPY daily total returns, known at t.
# GAMMA calibration (design data only): the median of ridge_full's
# pred / dec_vol^2 over the 2017-2018 validation decisions is 4.84, rounded to
# 5, so the median validation day is roughly fully invested. The pipeline
# recomputes this median and prints it as an audit check.
GAMMA = 5.0
W_MAX = 1.0                           # no leverage in the main run
W_MAX_LEVERED = 1.5                   # robustness only; borrowing at the BIL rate

# ---------------------------------------------------------------- execution and costs
# Next-close execution (DATA.md): decide after 16:15 ET on t, market-on-close
# order at t+1, earn close(t+1) -> close(t+2). The next-open convention is a
# robustness check only.
RETURNS_MAIN = ("spy_tr_cc", "bil_tr_cc")
RETURNS_NEXT_OPEN = ("spy_tr_oo", "bil_tr_oo")

# Cost per unit of SPY weight traded (one-way), charged on drift-aware turnover.
# 5 bps is conservative for SPY market-on-close orders (spread ~0.3 bp plus
# commission and impact); the grid is reported as a sensitivity check.
COST_BPS = 5.0
COST_GRID_BPS = (0.0, 1.0, 5.0, 10.0, 20.0)
INITIAL_WEIGHT = 0.0                  # start in BIL; the entry trade is charged

# ---------------------------------------------------------------- evaluation
SUBPERIODS = {                        # by realisation date
    "2019-2021": ("2019-01-01", "2021-12-31"),
    "2022-2026": ("2022-01-01", "2026-12-31"),
}
NW_LAGS = 5                           # Newey-West lags for the CAPM alpha regression
BOOT_REPS = 2000                      # stationary bootstrap of the Sharpe difference
BOOT_BLOCK = 21                       # mean block length (trading days)
SEED = 5205
EVENT_WINDOW = 10                     # matches events.H
