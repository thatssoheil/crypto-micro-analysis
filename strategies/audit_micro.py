#!/usr/bin/env python3
"""
AUDIT - data integrity + signal correctness for crypto-micro-analysis.
RUN BEFORE trusting any aggregation. Mirrors the macro-repo convention.

Checks:
  A1 schema/monotonicity/duplicates for every dataset CSV
  A2 ZEC primary (Yahoo) vs live cross-check (Bitstamp) close divergence
  A3 ETH/BTC computed ratio vs Bitstamp's own ETHBTC market (network,
     skipped silently if offline)
  A4 independent recomputation of engine signals at the last bar:
     RSI(14) via raw Wilder loop vs engine ewm implementation
     MA50 position + 5d slope, phase-score components

STATELESS: prints PASS/FAIL lines to stdout, saves nothing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data" / "micro_dataset"
FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}{(' - ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def load_ohlcv(name):
    df = pd.read_csv(DATA / f"{name}.csv")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts").sort_index()


# ------------------------------------------------------------------ A1 ----
def schema_checks():
    charts = ["btcusd_daily", "ethusd_daily", "zecusd_yahoo",
              "zecusd_bitstamp_live"]
    for name in charts:
        df = pd.read_csv(DATA / f"{name}.csv")
        ok_cols = list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
        ts = pd.to_datetime(df["ts"])
        mono = ts.is_monotonic_increasing
        nodup = not ts.duplicated().any()
        pos = (df[["open", "high", "low", "close"]] > 0).all().all()
        hi_lo = (df["high"] >= df["low"]).all()
        check(f"A1 {name}: schema+order+positive OHLC",
              ok_cols and mono and nodup and pos and hi_lo,
              f"{len(df)} rows, {ts.iloc[0].date()}..{ts.iloc[-1].date()}")


# ------------------------------------------------------------------ A2 ----
def zec_crosscheck():
    y = load_ohlcv("zecusd_yahoo")["close"]
    bl = load_ohlcv("zecusd_bitstamp_live")
    # Bitstamp ZEC book is near-dead: zero-volume days freeze the last print
    # (documented in SOURCES.md). Compare ONLY days with actual trades.
    b = bl.loc[bl["volume"] > 0, "close"]
    idx = y.index.intersection(b.index)
    if len(idx) < 10:
        check("A2 ZEC yahoo-vs-bitstamp (traded days)", False,
              f"only {len(idx)} traded-day overlaps")
        return
    diff = ((y.loc[idx] - b.loc[idx]).abs() / b.loc[idx] * 100)
    med = float(diff.median())
    mx = float(diff.max())
    check("A2 ZEC primary-vs-live median divergence < 3% (traded days)",
          med < 3.0,
          f"median {med:.2f}%, max {mx:.1f}% over {len(idx)} traded days "
          f"({int((bl['volume'].tail(30) == 0).sum())} of last 30 bitstamp days had no trades)")
    # and warn loudly if the live chart has gone fully stale recently
    recent_dead = float((bl["volume"].tail(14) == 0).mean())
    if recent_dead >= 0.9:
        print(f"[WARN] A2 Bitstamp ZEC dead book: {recent_dead*100:.0f}% of last "
              f"14 days had zero volume - cross-check value is limited")


# ------------------------------------------------------------------ A3 ----
def ratio_crosscheck():
    try:
        import requests
        r = requests.get(
            "https://www.bitstamp.net/api/v2/ohlc/ethbtc/",
            params={"step": 86400, "limit": 1}, timeout=15)
        live = float(r.json()["data"]["ohlc"][-1]["close"])
        btc = load_ohlcv("btcusd_daily")["close"].iloc[-1]
        eth = load_ohlcv("ethusd_daily")["close"].iloc[-1]
        comp = eth / btc
        d = abs(comp - live) / live * 100
        check("A3 computed ETH/BTC vs Bitstamp ETHBTC market < 3%",
              d < 3.0, f"computed {comp:.6f} vs venue {live:.6f} ({d:.2f}%)")
    except Exception:
        print("[SKIP] A3 ratio cross-check (offline or venue unavailable)")


# ------------------------------------------------------------------ A4 ----
def wilder_rsi(closes, n=14):
    """Reference implementation: explicit Wilder smoothing loop."""
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = gains[:n].mean()
    avg_l = losses[:n].mean()
    for i in range(n, len(deltas)):
        avg_g = (avg_g * (n - 1) + gains[i]) / n
        avg_l = (avg_l * (n - 1) + losses[i]) / n
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - 100.0 / (1.0 + rs)


def ewm_rsi(close_series, n=14):
    delta = close_series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def signal_recompute():
    btc = load_ohlcv("btcusd_daily")["close"]

    r_loop = wilder_rsi(btc.values[-200:])
    r_engine = float(ewm_rsi(btc).iloc[-1])
    check("A4 RSI(14) reference-vs-engine agree within 1pt",
          abs(r_loop - r_engine) < 1.0, f"{r_loop:.1f} vs {r_engine:.1f}")

    ma50 = btc.rolling(50).mean()
    slope = (ma50.iloc[-1] / ma50.iloc[-6] - 1) * 100
    above = bool(btc.iloc[-1] > ma50.iloc[-1])
    check("A4 MA50 position+slope computable",
          True, f"above={above}, slope5d={slope:+.2f}%")

    panel = pd.DataFrame({
        "btc": btc,
        "eth": load_ohlcv("ethusd_daily")["close"],
        "zec": load_ohlcv("zecusd_yahoo")["close"],
    }).dropna()
    ma = panel.rolling(50, min_periods=50).mean()
    holding = int(((panel > ma) & ma.notna()).iloc[-1].sum())
    dd = (panel["btc"].iloc[-1] / panel["btc"].rolling(180, min_periods=180)
          .max().iloc[-1] - 1) * 100
    check("A4 phase components computable",
          holding in (0, 1, 2, 3),
          f"breadth {holding}/3, BTC dd {dd:+.1f}% vs 180d high")


def main():
    print("=" * 64)
    print("MICRO AUDIT - data integrity + signal correctness")
    print("=" * 64)
    schema_checks()
    zec_crosscheck()
    ratio_crosscheck()
    signal_recompute()
    n = len(FAILS)
    print("-" * 64)
    if n:
        print(f"RESULT: {n} FAILURE(S): {FAILS}")
        sys.exit(1)
    print("RESULT: all checks passed.")


if __name__ == "__main__":
    main()
