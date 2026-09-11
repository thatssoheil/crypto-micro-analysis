#!/usr/bin/env python3
"""
FUNDING PROBE - perpetual funding as a crowding/positioning reading.

Thesis under test: crowd positioning (visible through perp funding) turns BEFORE
spot trend does, so it could add lead time the price-only engine cannot have.

HARD LIMITATION, read first: Kraken's public funding history only goes back ~1
year (verified: 2025-09-10 onward, 8771 hourly rows). That is far too short for a
multi-cycle lead-time study, so this script deliberately does NOT claim a verdict.
It reports:
  1. what the data can support today (current readings + percentiles - useful now),
  2. a descriptive check (do funding extremes precede moves, with the tiny n shown),
  3. an A/B of a funding-based de-risking rule over that single year, flagged as
     n=1-year evidence and NOT adopted on this basis.

Funding is fetched live from Kraken's public API - keyless, nothing is written to
the dataset. STATELESS: prints to stdout.
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from micro_backtest import load_panel                              # noqa: E402

PAIRS = {"btc": "PF_XBTUSD", "eth": "PF_ETHUSD", "zec": "PF_ZECUSD"}
UA = {"User-Agent": "Mozilla/5.0"}
FUND_URL = "https://futures.kraken.com/derivatives/api/v4/historicalfundingrates?symbol="
Z_WINDOW = 90        # rolling window for the funding z-score
EXTREME_Z = 2.0      # |z| above this counts as crowding


def fetch_funding(symbol):
    req = urllib.request.Request(FUND_URL + symbol, headers=UA)
    data = json.load(urllib.request.urlopen(req, timeout=30))
    if data.get("result") != "success":
        raise RuntimeError(f"kraken funding: {data.get('result')}")
    rows = [(pd.Timestamp(r["timestamp"]), float(r["relativeFundingRate"]))
            for r in data.get("rates", [])]
    s = pd.Series(dict(rows)).sort_index()
    return s  # hourly, decimal (e.g. 0.0001 = 0.01%)


def daily_mean(s):
    return s.resample("1D").mean().dropna()


def main():
    panel = load_panel()
    print("FUNDING PROBE - perp funding as a crowding reading (Kraken, keyless)")
    print("=" * 96)

    data, rows = {}, []
    for asset, symbol in PAIRS.items():
        try:
            h = fetch_funding(symbol)
        except Exception as exc:
            print(f"  {asset}: fetch failed ({exc})")
            continue
        d = daily_mean(h)
        data[asset] = d
        span = f"{d.index[0].date()} .. {d.index[-1].date()}"
        z = (d - d.rolling(Z_WINDOW, min_periods=20).mean()) / d.rolling(Z_WINDOW, min_periods=20).std()
        pct = float((d <= d.iloc[-1]).mean())
        print(f"  {asset.upper():4} {symbol:11} {len(d):>4} daily obs  {span}")
        print(f"       now {d.iloc[-1]*100:+.4f}%/day  |  90d z {z.iloc[-1]:+.2f}  "
              f"|  percentile {pct*100:.0f}%  |  range "
              f"{d.min()*100:+.4f}% .. {d.max()*100:+.4f}%")
        rows.append({"asset": asset, "obs": len(d), "last_pct": d.iloc[-1]*100,
                     "z": z.iloc[-1], "pctile": pct*100,
                     "min_pct": d.min()*100, "max_pct": d.max()*100})

    if not data:
        return

    print("\nCOVERAGE LIMIT: Kraken funding history is ~1 year (from 2025-09-10).")
    print("Events in that window are too few for a lead-time verdict, so the checks")
    print("below are DESCRIPTIVE and are not adoption evidence.\n")

    print("forward returns after extreme funding (|z| >= %.1f), same window:" % EXTREME_Z)
    for asset, d in data.items():
        z = (d - d.rolling(Z_WINDOW, min_periods=20).mean()) / d.rolling(Z_WINDOW, min_periods=20).std()
        close = panel[asset]
        idx = d.index.intersection(close.index)
        c = close.reindex(idx).ffill()
        fwd7 = (c.shift(-7) / c - 1) * 100
        fwd30 = (c.shift(-30) / c - 1) * 100
        hi = z.loc[idx] >= EXTREME_Z
        lo = z.loc[idx] <= -EXTREME_Z
        print(f"  {asset.upper():4} high-funding days n={int(hi.sum()):<3} "
              f"fwd7 {fwd7[hi].mean() if hi.any() else float('nan'):>+6.2f}%  "
              f"fwd30 {fwd30[hi].mean() if hi.any() else float('nan'):>+6.2f}%   |   "
              f"low-funding days n={int(lo.sum()):<3} "
              f"fwd7 {fwd7[lo].mean() if lo.any() else float('nan'):>+6.2f}%  "
              f"fwd30 {fwd30[lo].mean() if lo.any() else float('nan'):>+6.2f}%")
        print(f"       (all days: fwd7 {fwd7.mean():+.2f}%  fwd30 {fwd30.mean():+.2f}%)")

    print("\nA/B over the funding window only - MA50 filter vs MA50 + funding de-risk")
    print("(de-risk = flat when funding z >= %.1f).  n = 1 YEAR: not adoption evidence." % EXTREME_Z)
    rets = panel.pct_change()
    for asset, d in data.items():
        close = panel[asset]
        idx = close.index.intersection(d.index)
        c = close.reindex(idx).ffill()
        z = (d - d.rolling(Z_WINDOW, min_periods=20).mean()) / d.rolling(Z_WINDOW, min_periods=20).std()
        base = (c > c.rolling(50).mean()).astype(float)
        guarded = base.where(~(z.reindex(idx) >= EXTREME_Z), 0.0)
        r = rets[asset].reindex(idx).ffill()
        for label, pos in (("ma50 only", base), ("ma50+funding", guarded)):
            p = pos.shift(1).fillna(0.0)
            eq = (1.0 + p * r.fillna(0.0)).cumprod()
            tot = (eq.iloc[-1] - 1) * 100
            rr = eq.pct_change().dropna()
            sh = (rr.mean() / rr.std() * np.sqrt(365)) if rr.std() > 0 else 0.0
            dd = (eq / eq.cummax() - 1).min() * 100
            print(f"  {asset.upper():4} {label:14} total {tot:>8.1f}%  sharpe {sh:>5.2f}  maxDD {dd:>7.1f}%")


if __name__ == "__main__":
    main()
