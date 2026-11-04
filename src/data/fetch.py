"""Download raw data and save untouched snapshots to data/raw/.

Sources
  SPY, BIL : Yahoo Finance via yfinance (raw OHLC, Adj Close, dividends, splits)
  VIX, VIX3M: Cboe daily index history CSVs
  DTB3     : FRED 3-month T-bill rate (optional cross-check, failure is not fatal)

Run:  python -m src.data.fetch
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from src import config

USER_AGENT = "Mozilla/5.0 (research; DSA5205 course project)"


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_yahoo(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(ticker).history(
        start=config.DOWNLOAD_START,
        end=config.DOWNLOAD_END,
        interval="1d",
        auto_adjust=False,   # keep raw Close and vendor Adj Close side by side
        actions=True,        # Dividends and Stock Splits columns
    )
    if df.empty:
        raise RuntimeError(f"Yahoo returned no rows for {ticker}")
    needed = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"{ticker}: Yahoo response missing columns {missing}")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]]


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "python": sys.version.split()[0], "files": {}}
    try:
        import yfinance
        manifest["yfinance"] = yfinance.__version__
    except ImportError:
        sys.exit("yfinance is not installed: pip install -r requirements.txt")

    for t in config.ETFS:
        df = fetch_yahoo(t)
        path = config.RAW_DIR / f"{t}_yahoo.csv"
        df.to_csv(path)
        manifest["files"][path.name] = {"source": f"Yahoo Finance ({t})", "rows": len(df),
                                        "first": str(df.index.min().date()),
                                        "last": str(df.index.max().date()),
                                        "sha256": _sha256(path)}
        print(f"{t:6s} {len(df):5d} rows  {df.index.min().date()} -> {df.index.max().date()}")

    for name, url in config.VOL_INDICES.items():
        content = _http_get(url)
        path = config.RAW_DIR / f"{name}_cboe.csv"
        path.write_bytes(content)
        n = len(pd.read_csv(io.BytesIO(content)))
        manifest["files"][path.name] = {"source": url, "rows": n, "sha256": _sha256(path)}
        print(f"{name:6s} {n:5d} rows  (full Cboe history)")

    try:
        content = _http_get(config.FRED_DTB3_URL)
        path = config.RAW_DIR / "DTB3_fred.csv"
        path.write_bytes(content)
        manifest["files"][path.name] = {"source": config.FRED_DTB3_URL, "sha256": _sha256(path)}
        print("DTB3   saved (optional cross-check)")
    except Exception as e:  # optional source
        print(f"DTB3   skipped ({e.__class__.__name__}); BIL cross-check will be omitted")

    (config.RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Raw snapshot and manifest written to {config.RAW_DIR}")


if __name__ == "__main__":
    main()
