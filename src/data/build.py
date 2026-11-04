"""Clean and align raw data into one daily panel.

Reads only data/raw/ (never the network), so a rebuild from the same snapshot is
byte-for-byte reproducible.

Output: data/processed/panel.csv (+ panel.parquet if pyarrow is installed)
        data/processed/missing_log.csv

Run:  python -m src.data.build
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


# --------------------------------------------------------------------- loaders
def load_etf(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(config.RAW_DIR / f"{ticker}_yahoo.csv", parse_dates=["Date"], index_col="Date")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def load_cboe(name: str) -> pd.Series:
    df = pd.read_csv(config.RAW_DIR / f"{name}_cboe.csv")
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["DATE"]).drop_duplicates("DATE", keep="last").set_index("DATE").sort_index()
    s = pd.to_numeric(df["CLOSE"], errors="coerce")
    s[s <= 0] = np.nan                     # non-positive index levels are data errors
    s.name = f"{name.lower()}_close"
    return s


# --------------------------------------------------------------------- returns
def etf_returns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Total and price returns from the raw (split-adjusted, dividend-unadjusted) close.

    Total return follows the spec's Appendix B:  R_t = (P_t + D_t) / P_{t-1} - 1
    on the ex-dividend date. The back-adjusted close is the cumulative total return
    anchored to the last raw close, so A_t / A_{t-1} - 1 equals R_t exactly.
    """
    p, d, o = df["Close"], df["Dividends"].fillna(0.0), df["Open"]
    tr = (p + d) / p.shift(1) - 1
    pr = p / p.shift(1) - 1
    growth = (1 + tr.fillna(0)).cumprod().where(p.notna())
    last = p.last_valid_index()
    adj_close = growth / growth.loc[last] * p.loc[last]
    adj_open = o * adj_close / p          # same adjustment factor as that day's close
    out = pd.DataFrame({
        f"{prefix}_open": o,
        f"{prefix}_close": p,
        f"{prefix}_adj_open": adj_open,
        f"{prefix}_adj_close": adj_close,
        f"{prefix}_vendor_adj_close": df["Adj Close"],
        f"{prefix}_volume": df["Volume"],
        f"{prefix}_div": d,
        f"{prefix}_split": df["Stock Splits"].fillna(0.0),
        f"{prefix}_tr_cc": tr,                         # close(t-1) -> close(t), known at t
        f"{prefix}_pr_cc": pr,
        f"{prefix}_tr_oo": adj_open / adj_open.shift(1) - 1,   # open(t-1) -> open(t)
    })
    return out


# --------------------------------------------------------------------- build
def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    spy, bil = load_etf("SPY"), load_etf("BIL")
    vols = {n: load_cboe(n) for n in config.VOL_INDICES}

    # Master calendar: SPY trading days.
    cal = spy.index
    log = []

    for name, df in {"BIL": bil}.items():
        for dt in cal.difference(df.index):
            log.append((dt, name, "missing on SPY trading day"))
        for dt in df.index.difference(cal):
            log.append((dt, name, "extra date not in SPY calendar (dropped)"))

    panel = etf_returns(spy, "spy").join(etf_returns(bil.reindex(cal), "bil"))
    # BIL gaps (if any) stay NaN and are listed in missing_log.csv.

    for name, s in vols.items():
        s_in = s[(s.index >= cal.min()) & (s.index <= cal.max())]
        for dt in s_in.index.difference(cal):
            log.append((dt, name, "Cboe value on non-SPY day (dropped)"))
        aligned = s.reindex(cal)
        missing = aligned.isna()
        filled = aligned.ffill(limit=config.MAX_FFILL_DAYS)
        col = s.name
        panel[col] = filled
        panel[f"{col}_ffilled"] = missing & filled.notna()
        for dt in cal[missing & filled.notna()]:
            log.append((dt, name, f"missing, forward-filled (limit {config.MAX_FFILL_DAYS} day)"))
        for dt in cal[filled.isna()]:
            log.append((dt, name, "missing, left NaN (gap longer than ffill limit)"))

    # Signal-side helper already known at the decision time t (16:15 ET close).
    panel["vix_ratio"] = panel["vix_close"] / panel["vix3m_close"]   # >1 = backwardation

    # ---------------------------------------------------------------- forward returns
    # Decision at close t, executed at close t+1, held to close t+2.
    # The return on that holding is the close-to-close return dated t+2.
    k = 2
    panel["fwd_spy_tr"] = panel["spy_tr_cc"].shift(-k)
    panel["fwd_bil_tr"] = panel["bil_tr_cc"].shift(-k)
    panel["fwd_spy_xs"] = panel["fwd_spy_tr"] - panel["fwd_bil_tr"]
    # Alternative (next-open) convention kept for robustness checks:
    # decision t, execute at open t+1, hold to open t+2 -> open-to-open return dated t+2.
    panel["fwd_spy_tr_oo"] = panel["spy_tr_oo"].shift(-k)
    panel["fwd_bil_tr_oo"] = panel["bil_tr_oo"].shift(-k)

    panel["spy_xs_cc"] = panel["spy_tr_cc"] - panel["bil_tr_cc"]     # known at t
    panel["in_sample"] = (panel.index >= config.SAMPLE_START) & (panel.index <= config.SAMPLE_END)
    panel.index.name = "date"

    log_df = pd.DataFrame(log, columns=["date", "series", "issue"]).sort_values(["date", "series"])
    return panel, log_df


COLUMN_ORDER_HEAD = ["in_sample", "spy_adj_close", "spy_tr_cc", "bil_tr_cc", "spy_xs_cc",
                     "vix_close", "vix3m_close", "vix_ratio",
                     "fwd_spy_tr", "fwd_bil_tr", "fwd_spy_xs"]


def main() -> None:
    panel, log_df = build()
    cols = COLUMN_ORDER_HEAD + [c for c in panel.columns if c not in COLUMN_ORDER_HEAD]
    panel = panel[cols]
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(config.PROCESSED_DIR / "panel.csv", float_format="%.10g")
    try:
        panel.to_parquet(config.PROCESSED_DIR / "panel.parquet")
    except ImportError:
        pass
    log_df.to_csv(config.PROCESSED_DIR / "missing_log.csv", index=False)
    s = panel[panel["in_sample"]]
    print(f"panel: {len(panel)} rows ({panel.index.min().date()} -> {panel.index.max().date()}), "
          f"{len(s)} in sample; {len(log_df)} alignment issues logged")


def load_panel() -> pd.DataFrame:
    """Convenience loader for teammates: pd.DataFrame indexed by date."""
    path = config.PROCESSED_DIR / "panel.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.read_csv(config.PROCESSED_DIR / "panel.csv", parse_dates=["date"], index_col="date")


if __name__ == "__main__":
    main()
