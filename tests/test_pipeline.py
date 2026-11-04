"""Offline end-to-end test of the data stage on synthetic data.

Writes mock raw files in the exact Yahoo (yfinance) and Cboe formats into a
temporary directory, runs build, benchmarks and verify, and asserts the key
invariants (total-return maths, timing alignment, forward-fill policy).

Run:  python -m tests.test_pipeline
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

SEED = 5205


def make_mock(raw: Path) -> dict:
    rng = np.random.default_rng(SEED)
    hol = USFederalHolidayCalendar().holidays("2011-01-01", "2026-12-31")
    days = pd.bdate_range("2012-01-03", "2026-01-15").difference(hol)

    def etf(start_px, mu, sigma, div_months, div_yield):
        r = rng.normal(mu / 252, sigma / np.sqrt(252), len(days))
        close = start_px * np.cumprod(1 + r)
        df = pd.DataFrame(index=days)
        df["Close"] = close
        df["Open"] = close * (1 + rng.normal(0, sigma / np.sqrt(252) / 3, len(days)))
        df["High"] = np.maximum(df["Open"], df["Close"]) * (1 + abs(rng.normal(0, 0.002, len(days))))
        df["Low"] = np.minimum(df["Open"], df["Close"]) * (1 - abs(rng.normal(0, 0.002, len(days))))
        df["Volume"] = rng.integers(1e6, 1e8, len(days))
        df["Dividends"] = 0.0
        # first trading day on/after the 15th of each dividend month is the ex-date
        for y in range(days[0].year, days[-1].year + 1):
            for m in div_months:
                cand = days[(days.year == y) & (days.month == m) & (days.day >= 15)]
                if len(cand):
                    i = days.get_loc(cand[0])
                    df.iloc[i, df.columns.get_loc("Dividends")] = df["Close"].iloc[i - 1] * div_yield
        # Price drops by the dividend on the ex-date (raw close is ex-dividend)
        drop = df["Dividends"].cumsum()
        for c in ["Open", "High", "Low", "Close"]:
            df[c] = df[c] - drop
        # Yahoo-style Adj Close: multiplicative factor (1 - D_t / P_{t-1}) on prior history
        factor = pd.Series(1.0, index=days)
        d = df["Dividends"]
        f_t = (1 - d / df["Close"].shift(1)).fillna(1.0)
        factor = f_t[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
        df["Adj Close"] = df["Close"] * factor
        df["Stock Splits"] = 0.0
        df.index.name = "Date"
        return df[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]]

    spy = etf(130, 0.10, 0.17, [3, 6, 9, 12], 0.004)
    bil = etf(45.7, 0.015, 0.002, list(range(1, 13)), 0.0012)
    bil["High"] = bil[["Open", "Close"]].max(axis=1) * 1.00005
    bil["Low"] = bil[["Open", "Close"]].min(axis=1) * 0.99995
    raw.mkdir(parents=True, exist_ok=True)
    spy.to_csv(raw / "SPY_yahoo.csv")
    bil.to_csv(raw / "BIL_yahoo.csv")

    # Cboe-style files: longer history, MM/DD/YYYY dates
    vdays = pd.bdate_range("2010-01-04", "2026-01-15").difference(hol)
    x = np.empty(len(vdays)); x[0] = np.log(16)
    for i in range(1, len(vdays)):
        x[i] = x[i - 1] + 0.03 * (np.log(16) - x[i - 1]) + rng.normal(0, 0.07)
    vix = np.exp(x)
    vix3m = vix * 0.9 + 0.1 * 18 + 1.0   # mostly contango, inverted when VIX spikes
    removed = pd.Timestamp("2019-06-12")  # one missing day -> forward-fill test
    extra = pd.Timestamp("2016-07-04")    # Cboe value on an exchange holiday -> dropped
    for name, lvl in [("VIX", vix), ("VIX3M", vix3m)]:
        df = pd.DataFrame({"DATE": vdays, "OPEN": lvl, "HIGH": lvl * 1.03, "LOW": lvl * 0.97, "CLOSE": lvl})
        if name == "VIX":
            df = df[df["DATE"] != removed]
        df = pd.concat([df, pd.DataFrame({"DATE": [extra], "OPEN": [15.0], "HIGH": [15.0],
                                          "LOW": [15.0], "CLOSE": [15.0]})]).sort_values("DATE")
        df["DATE"] = df["DATE"].dt.strftime("%m/%d/%Y")
        df.to_csv(raw / f"{name}_cboe.csv", index=False, float_format="%.2f")
    return {"removed": removed, "extra": extra}


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="vt_mock_"))
    os.environ["VT_WORK_DIR"] = str(work)
    try:
        info = make_mock(work / "data" / "raw")
        from src.data import benchmarks, build, verify
        from src.metrics import apply_weights, perf_stats

        build.main()
        benchmarks.main()
        try:
            verify.main()
        except SystemExit as e:
            print(f"verify exit code {e.code} (mock data: WARNs on VIX spot checks are expected)")
            assert e.code == 0, "verify reported FAIL on mock data"

        p = build.load_panel()
        # 1. total return identity on ex-dividend days
        spy_raw = build.load_etf("SPY")
        ex = spy_raw.index[spy_raw["Dividends"] > 0][5]
        prev = spy_raw.index[spy_raw.index.get_loc(ex) - 1]
        expect = (spy_raw.at[ex, "Close"] + spy_raw.at[ex, "Dividends"]) / spy_raw.at[prev, "Close"] - 1
        assert abs(p.at[ex, "spy_tr_cc"] - expect) < 1e-12
        # 2. forward-fill policy
        assert bool(p.at[info["removed"], "vix_close_ffilled"]) is True
        assert info["extra"] not in p.index
        # 3. timing: forward label is two rows ahead
        i = 1000
        assert p["fwd_spy_tr"].iloc[i] == p["spy_tr_cc"].iloc[i + 2]
        # 4. apply_weights: constant weight 1 reproduces buy-and-hold exactly; 0 gives BIL
        s = p[p["in_sample"]]
        r1 = apply_weights(s, pd.Series(1.0, index=s.index))
        assert np.allclose(r1.values, s["spy_tr_cc"].loc[r1.index].values)
        r0 = apply_weights(s, pd.Series(0.0, index=s.index))
        assert np.allclose(r0.values, s["bil_tr_cc"].loc[r0.index].values)
        # 5. apply_weights uses w[t] only from t+2 onward (no look-ahead)
        w = pd.Series(0.0, index=s.index); w.iloc[500] = 1.0
        r = apply_weights(s, w, cost_bps=0)
        d = s.index
        assert np.isclose(r.loc[d[502]], s["spy_tr_cc"].loc[d[502]])
        assert np.isclose(r.loc[d[501]], s["bil_tr_cc"].loc[d[501]])
        # 6. costs charged on the execution date (close of t+1)
        rc = apply_weights(s, w, cost_bps=10)
        assert np.isclose(r.loc[d[501]] - rc.loc[d[501]], 0.001)
        print(perf_stats(r1, s["bil_tr_cc"]))
        print("\nALL TESTS PASSED")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
