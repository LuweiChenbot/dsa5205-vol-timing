"""Single source of truth for paths, dates, tickers and the timing convention.

All paths are relative to the repository root, so the code runs from any clone.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# VT_WORK_DIR lets the test suite run the pipeline on mock data without
# touching the real snapshot. Leave it unset for normal use.
_WORK = Path(os.environ.get("VT_WORK_DIR", ROOT))
RAW_DIR = _WORK / "data" / "raw"
PROCESSED_DIR = _WORK / "data" / "processed"
REPORT_DIR = _WORK / "reports" / "data"

# ---------------------------------------------------------------- dates
# Download starts two years before the study sample so that features with
# lookbacks of up to ~500 trading days have a warm-up period.
DOWNLOAD_START = "2012-01-01"
# A few days past the sample end so decisions made on the last sample days
# still have realised next-close returns (needed for t+2).
DOWNLOAD_END = "2026-01-16"          # yfinance end date is exclusive
SAMPLE_START = "2014-01-01"
SAMPLE_END = "2025-12-31"

# ---------------------------------------------------------------- assets
ETFS = ["SPY", "BIL"]                # traded assets (Yahoo Finance)
VOL_INDICES = {                      # Cboe volatility indices (not tradable, signals only)
    "VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "VIX3M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
}
# Optional cross-check of BIL against the 3-month T-bill secondary market rate.
FRED_DTB3_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"

# ---------------------------------------------------------------- timing
# Decision: after the close of day t, using information known by then
#   (SPY close at 16:00 ET, VIX/VIX3M close at 16:15 ET).
# Execution: market-on-close order at the close of day t+1.
# Return earned by the decision taken on day t: close(t+1) -> close(t+2).
EXECUTION = "next_close"

# ---------------------------------------------------------------- cleaning
MAX_FFILL_DAYS = 1                   # longest gap in VIX/VIX3M we forward-fill
SPY_MOVE_FLAG = 0.08                 # |daily total return| above this is listed for review
TR_TOLERANCE = 5e-4                  # max allowed gap between our TR and Yahoo Adj Close TR

TRADING_DAYS = 252
