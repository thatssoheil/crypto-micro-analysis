#!/usr/bin/env python3
"""
MICRO DATA DUMP - the raw numbers behind the micro regime engine, per asset.

Presentation only: this script recomputes NOTHING on its own. It imports the live
engine's own helpers (load_ohlcv, rsi, ma_slope) so every figure shown here is the
same figure the engine scored from. Read-only and STATELESS (prints to stdout,
saves nothing, never writes the dataset).

Usage:
  ./.venv/bin/python strategies/micro_data.py
  ./.venv/bin/python strategies/micro_data.py --json    # machine-readable
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from micro_regime import load_ohlcv, load_fng, rsi, ma_slope  # noqa: E402

ASSETS = (("BTC", "btcusd_daily"), ("ETH", "ethusd_daily"), ("ZEC", "zecusd_yahoo"))
DD_WIN, MA_FAST, MA_SLOW = 180, 50, 200


def asset_row(name, df):
    c = df["close"]
    close = float(c.iloc[-1])
    ma50 = float(c.rolling(MA_FAST).mean().iloc[-1])
    ma200 = float(c.rolling(MA_SLOW).mean().iloc[-1])
    high180 = float(c.rolling(DD_WIN).max().iloc[-1])
    low180 = float(c.rolling(DD_WIN).min().iloc[-1])
    dd180 = close / high180 - 1
    # ma_slope returns (score, message) - the raw % only lives inside the message,
    # so recompute it with the engine's EXACT formula and take the score from the
    # engine, keeping this dump consistent with what the engine scored.
    slope_score, slope_msg = ma_slope(df, MA_FAST, 5)
    _ma = c.rolling(MA_FAST).mean()
    slope_pct = float(_ma.iloc[-1] / _ma.iloc[-1 - 5] - 1) * 100
    ret = lambda n: (close / float(c.iloc[-1 - n]) - 1) if len(c) > n else float("nan")
    lr = np.log(c / c.shift(1)).dropna().tail(30)
    vol30 = float(lr.std() * np.sqrt(365)) if len(lr) > 5 else float("nan")
    return {
        "asset": name,
        "date": str(c.index[-1].date()),
        "close": close,
        "ma50": ma50,
        "ma200": ma200,
        "high180": high180,
        "low180": low180,
        "vs_ma50_pct": (close / ma50 - 1) * 100,
        "vs_ma200_pct": (close / ma200 - 1) * 100,
        "ma50_slope_5d_pct": slope_pct,
        "ma50_slope_score": slope_score,
        "ma50_slope_note": slope_msg,
        "rsi14": float(rsi(c, 14).iloc[-1]),
        "ret30d_pct": ret(30) * 100,
        "ret90d_pct": ret(90) * 100,
        "vol30d_ann_pct": vol30 * 100,
        "dd180_pct": dd180 * 100,
        # how far price must FALL to reach each alert level (0 = already through it)
        "buffer_to_ma50_pct": (close - ma50) / close * 100,
        "buffer_to_ma200_pct": (close - ma200) / close * 100,
        "buffer_to_dd20_pts": (dd180 + 0.20) * 100,
    }


def main():
    rows = [asset_row(n, load_ohlcv(f)) for n, f in ASSETS]
    fng = load_fng()

    if "--json" in sys.argv:
        print(json.dumps({"assets": rows,
                          "fng": {"last": float(fng.iloc[-1]),
                                  "date": str(fng.index[-1].date())}}, indent=2))
        return

    df = pd.DataFrame(rows)
    asof = df["date"].iloc[0]
    print(f"MICRO DATA DUMP - data through {asof}")
    print("=" * 118)
    print(f"{'asset':5} {'close':>11} {'MA50':>11} {'MA200':>11} {'vsMA50':>7} {'vsMA200':>8} "
          f"{'slope5d':>8} {'RSI':>4} {'ret30d':>7} {'ret90d':>7} {'vol30d':>7} {'dd180':>7}")
    for r in rows:
        print(f"{r['asset']:5} {r['close']:>11,.2f} {r['ma50']:>11,.2f} {r['ma200']:>11,.2f} "
              f"{r['vs_ma50_pct']:>+6.1f}% {r['vs_ma200_pct']:>+7.1f}% "
              f"{(r['ma50_slope_5d_pct'] or 0):>+7.2f}% {r['rsi14']:>4.0f} "
              f"{r['ret30d_pct']:>+6.1f}% {r['ret90d_pct']:>+6.1f}% "
              f"{r['vol30d_ann_pct']:>6.1f}% {r['dd180_pct']:>+6.1f}%")

    print("\nDISTANCE TO ALERT LEVELS (how far price must fall)")
    print(f"{'asset':5} {'180d high':>11} {'drop to MA50':>13} {'drop to MA200':>14} {'dd20 room':>15}")
    for r in rows:
        ma200_note = "n/a (rule 2)" if r["asset"] == "ZEC" else f"{r['buffer_to_ma200_pct']:>13.1f}%"
        print(f"{r['asset']:5} {r['high180']:>11,.2f} {r['buffer_to_ma50_pct']:>12.1f}% "
              f"{ma200_note:>14} {r['buffer_to_dd20_pts']:>14.1f}pp")

    print("\nCROSS-PAIRS (computed from aligned USD closes)")
    import micro_regime as mr
    for label, num, den in (("ETH/BTC", "ethusd_daily", "btcusd_daily"),
                            ("ZEC/BTC", "zecusd_yahoo", "btcusd_daily"),
                            ("ZEC/ETH", "zecusd_yahoo", "ethusd_daily")):
        n, d = load_ohlcv(num), load_ohlcv(den)
        idx = n.index.intersection(d.index)
        ratio = (n.loc[idx, "close"] / d.loc[idx, "close"]).dropna()
        mom20 = ratio / ratio.rolling(20).mean() - 1
        mom60 = ratio / ratio.rolling(60).mean() - 1
        mn, mx = ratio.rolling(DD_WIN).min(), ratio.rolling(DD_WIN).max()
        band = (ratio.iloc[-1] - mn.iloc[-1]) / (mx.iloc[-1] - mn.iloc[-1])
        print(f"  {label:8} ratio {ratio.iloc[-1]:.6f} | 20d {mom20.iloc[-1]*100:+.2f}% vs 60d MA "
              f"| 60d {mom60.iloc[-1]*100:+.2f}% vs 60d MA | band {band*100:.0f}% of 180d range")

    print(f"\nF&G: {float(fng.iloc[-1]):.0f} ({fng.index[-1].date()})")
    print("\nrecent closes (last 7 sessions)")
    frames = {n: load_ohlcv(f)["close"].tail(7) for n, f in ASSETS}
    dates = frames["BTC"].index
    print("  date        " + "".join(f"{n:>14}" for n, _ in ASSETS))
    for i, dt in enumerate(dates):
        cells = ""
        for n, _ in ASSETS:
            v = frames[n]
            cells += f"{float(v.loc[dt]):>14,.2f}" if dt in v.index else f"{'-':>14}"
        print(f"  {dt.date()}  {cells}")


if __name__ == "__main__":
    main()
