#!/usr/bin/env python3
"""
MICRO REGIME ENGINE - short-horizon companion to the macro verdict.

Answers, per asset (BTC / ETH / ZEC):
  1. Is the asset in a tradeable trend right now?      -> trend health
  2. Which asset is the strongest place to sit?        -> cross-pair RS
  3. Where are we in the bull cycle?                   -> phase score

Signals (each -1 / 0 / +1 unless noted), grouped by what they mean:
  TREND HEALTH per asset: close vs MA50, MA50 slope up/down, RSI(14)
    regime (>70 hot, <35 washed out, else neutral).
  CROSS-PAIR RELATIVE STRENGTH: ETH/BTC and ZEC/BTC 20d vs 60d MA
    (short momentum over medium trend) + ratio distance from its own 180d
    high/low band.
  CYCLE PHASE: BTC drawdown from 180d high + F&G contrarian band +
    breadth (how many of the 3 assets hold their own MA50).

Output: full report to stdout. STATELESS: consumes local data only,
saves nothing, re-run anytime.

Crosses are computed as USD-close ratios on aligned dates - never fetched.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data" / "micro_dataset"


def load_ohlcv(name):
    p = DATA / f"{name}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts").sort_index()


def load_fng():
    p = DATA / "fear_greed.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date")["value"].astype(float).sort_index()


def rsi(series, n=14):
    """Wilder-smoothed RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50)


def ma_slope(df, n=50, lookback=5):
    """+1 if MA50 rising over `lookback` days, -1 falling, 0 flat (<0.15%)."""
    ma = df["close"].rolling(n).mean()
    chg = (ma.iloc[-1] / ma.iloc[-1 - lookback] - 1) * 100
    if chg > 0.15:
        return 1, f"MA50 rising (+{chg:.1f}%/5d)"
    if chg < -0.15:
        return -1, f"MA50 falling ({chg:.1f}%/5d)"
    return 0, f"MA50 flat ({chg:+.1f}%/5d)"


# --------------------------------------------------------------- per-asset --
def trend_block(name, df):
    """Trend health for one asset. Returns (score -3..+3, [lines])."""
    lines, score = [], 0
    c = df["close"]
    ma50 = c.rolling(50).mean().iloc[-1]
    cur = c.iloc[-1]

    s, d = (1, f"above MA50 {ma50:.0f}") if cur > ma50 else (-1, f"below MA50 {ma50:.0f}")
    score += s
    lines.append(f"  price vs MA50   {d}  [{s:+d}]")

    s, d = ma_slope(df)
    score += s
    lines.append(f"  MA50 slope      {d}  [{s:+d}]")

    r = rsi(c).iloc[-1]
    if r > 70:
        s, d = -1, "overbought - trim zone"
    elif r < 35:
        s, d = 1, "oversold - washout"
    else:
        s, d = 0, ""
    score += s
    extra = f" ({d})" if d else ""
    lines.append(f"  RSI(14)         {r:.0f}{extra}  [{s:+d}]")
    return score, lines


# ------------------------------------------------------------------ ratios --
def ratio_block(pair_name, num_df, den_df):
    """Relative strength of pair_name = num/den, computed on aligned closes.
    Returns dict with score components or None if insufficient overlap."""
    a = num_df["close"]; b = den_df["close"]
    idx = a.index.intersection(b.index)
    if len(idx) < 200:
        return None
    ratio = (a.loc[idx] / b.loc[idx]).dropna()
    if len(ratio) < 200:
        return None
    ma20 = ratio.rolling(20).mean().iloc[-1]
    ma60 = ratio.rolling(60).mean().iloc[-1]
    hi = ratio.rolling(180).max().iloc[-1]
    lo = ratio.rolling(180).min().iloc[-1]
    cur = ratio.iloc[-1]
    pos_in_band = (cur - lo) / (hi - lo) if hi > lo else 0.5

    mom = 1 if ma20 > ma60 else (-1 if ma20 < ma60 else 0)
    band = 1 if pos_in_band > 0.66 else (-1 if pos_in_band < 0.33 else 0)
    return {
        "ratio": ratio,
        "cur": cur, "ma20": ma20, "ma60": ma60, "pos": pos_in_band,
        "mom": mom, "band": band,
        "score": mom + band,
        "mom_d": ("20d above 60d MA" if mom > 0 else
                  "20d below 60d MA" if mom < 0 else "flat"),
        "band_d": (f"{pos_in_band*100:.0f}% of 180d range",
                   "upper third - leading", "lower third - lagging"),
    }


# ------------------------------------------------------------------- phase --
def phase_score(btc, eth, zec, fng):
    """Cycle-phase estimate: where in the bull run are we?
    Returns (score -3..+3, phase label, [lines])."""
    lines = []
    score = 0.0
    w = []

    # BTC depth of drawdown from its 180d high (the core bull-health tell)
    c = btc["close"]
    dd = (c.iloc[-1] / c.rolling(180).max().iloc[-1] - 1) * 100
    if dd > -10:
        s = 1; d = f"{dd:+.0f}% vs 180d high - uptrend intact"
    elif dd > -25:
        s = 0; d = f"{dd:+.0f}% vs 180d high - correction territory"
    else:
        s = -1; d = f"{dd:+.0f}% vs 180d high - deep drawdown"
    score += s * 2.0; w.append(2.0)
    lines.append(f"  BTC drawdown     {d}  [{s:+d} x2.0]")

    # Fear & Greed contrarian bands (macro repo proved it trims tops/bottoms)
    v = fng.iloc[-1]
    if v >= 75:
        s = -1; d = "extreme greed - distribution risk"
    elif v <= 25:
        s = 1; d = "fear - accumulation zone"
    else:
        s = 0; d = "neutral band"
    score += s * 1.0; w.append(1.0)
    lines.append(f"  F&G {v:.0f}          {d}  [{s:+d} x1.0]")

    # Breadth: how many assets hold their own MA50
    holding = sum(
        1 for df in (btc, eth, zec)
        if df is not None and len(df) >= 55
        and df["close"].iloc[-1] > df["close"].rolling(50).mean().iloc[-1]
    )
    s = {3: 1, 2: 0, 1: -1, 0: -1}[holding]
    d = f"{holding}/3 assets above MA50"
    score += s * 1.5; w.append(1.5)
    lines.append(f"  Breadth          {d}  [{s:+d} x1.5]")

    norm = score / sum(w) * 3
    if norm >= 1.5:
        label = "PHASE 1 - BULL EXPANSION (ride trends, rotate to leaders)"
    elif norm >= 0.5:
        label = "EARLY BULL - accumulate pullbacks, respect stops"
    elif norm <= -1.5:
        label = "PHASE 2 - DISTRIBUTION/BREAKDOWN (cash, no new longs)"
    elif norm <= -0.5:
        label = "LATE CYCLE - tighten stops, trim into strength"
    else:
        label = "TRANSITION - mixed signals, half exposure max"
    return round(norm, 1), label, lines


# -------------------------------------------------------------------- main --
def main():
    btc = load_ohlcv("btcusd_daily")
    eth = load_ohlcv("ethusd_daily")
    zec = load_ohlcv("zecusd_yahoo")
    fng = load_fng()
    if btc is None or eth is None:
        raise SystemExit("dataset missing - run build_micro_dataset.py first")

    today = btc.index[-1].strftime("%Y-%m-%d")
    print("=" * 64)
    print(f"MICRO REGIME ENGINE - data through {today}")
    print("=" * 64)

    # ---- trend health per asset
    blocks = {}
    slopes = {}
    for name, df in [("BTC", btc), ("ETH", eth), ("ZEC", zec)]:
        if df is None:
            continue
        s, lines = trend_block(name, df)
        blocks[name] = s
        ma = df["close"].rolling(50).mean()
        slopes[name] = float((ma.iloc[-1] / ma.iloc[-6] - 1) * 100)
        px = df['close'].iloc[-1]
        print(f"\n-- {name}  (${px:,.2f})  trend score {s:+d}/3")
        print("\n".join(lines))

    # ---- cross-pair relative strength
    print("\n-- cross-pair relative strength")
    for pname, num, den, dlabel in [
            ("ETH/BTC", eth, btc, "ETH vs BTC"), ("ZEC/BTC", zec, btc, "ZEC vs BTC"),
            ("ZEC/ETH", zec, eth, "ZEC vs ETH")]:
        rb = ratio_block(pname, num, den)
        if rb is None:
            print(f"  {pname}: insufficient aligned history (<200d)")
            continue
        print(f"  {pname:8s} {rb['cur']:.6f} | mom[{rb['mom']:+d}] {rb['mom_d']}; "
              f"band[{rb['band']:+d}] at {rb['band_d'][0]} ({rb['band_d'][1]})"
              f" -> RS {rb['score']:+d}")

    # ---- rotation hint (score first, MA50 5d slope as tie-break)
    ranked = sorted(blocks.items(),
                    key=lambda kv: (-kv[1], -slopes.get(kv[0], 0.0)))
    print("\n-- rotation ranking (trend score, slope tie-break): "
          + " > ".join(f"{kv[0]} ({kv[1]:+d}, slope {slopes[kv[0]]:+.1f}%/5d)"
                       for kv in ranked))

    # ---- cycle phase
    ps, plabel, plines = phase_score(btc, eth, zec, fng)
    print("\n-- cycle phase score")
    print("\n".join(plines))
    print(f"\n>>> PHASE SCORE {ps:+.1f}  ->  {plabel}")
    print("(micro engine = tactical overlay; macro repo owns the cash decision)")


if __name__ == "__main__":
    main()
