#!/usr/bin/env python3
"""
WAVETREND STUDY - LazyBear's WaveTrend Oscillator (WTO), tested honestly.

Published definition (LazyBear's original Pine, UNFITTED - parameters are the
author's defaults, deliberately NOT tuned here):
    ap  = hlc3
    esa = EMA(ap, n1=10)
    d   = EMA(|ap - esa|, n1=10)
    ci  = (ap - esa) / (0.015 * d)
    wt1 = EMA(ci, n2=21)
    wt2 = SMA(wt1, 4)
    overbought / oversold = +/-60 (LazyBear's original levels)

Documented usage is mean-reversion: a wt1/wt2 cross inside the OS zone is a buy
signal, inside the OB zone a sell signal. (Divergence is the other half of the
method but it is discretionary, so it is NOT tested here - it cannot be made
falsifiable and this repo does not accept interpretive signals.)

What "how well does it work" means here, measured four ways:
  1. forward returns after each signal vs the unconditional baseline (same window),
  2. hit rate (share of signals followed by a positive 30d return) vs base rate,
  3. lead-time / false-positive rate at major turns (same event definitions and
     machinery as the alert study),
  4. A/B: MA50 alone vs MA50 + WTO exit overlay.

Parameters are the published defaults precisely BECAUSE tuning them on this data
would be curve-fitting - the CUSUM episode showed a grid winner can vanish
out-of-sample. Any tuned variant needs its own OOS test.

STATELESS: prints to stdout, writes nothing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from micro_regime import load_ohlcv                          # noqa: E402
from micro_backtest import sim, report                       # noqa: E402
from micro_alerts import find_events, leadtime_table         # noqa: E402

ASSETS = (("BTC", "btcusd_daily"), ("ETH", "ethusd_daily"), ("ZEC", "zecusd_yahoo"))
N1, N2, WT2_LEN = 10, 21, 4
OB, OS = 60.0, -60.0


def wavetrend(df, n1=N1, n2=N2):
    """LazyBear WTO. Returns (wt1, wt2) aligned to df's index."""
    ap = (df["high"] + df["low"] + df["close"]) / 3.0
    esa = ap.ewm(span=n1, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d)
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(WT2_LEN).mean()          # Pine sma(wt1, 4)
    return wt1, wt2


def signals(wt1, wt2):
    """The two published signal families, both edge-triggered."""
    up = (wt1 > wt2)
    dn = (wt1 < wt2)
    cross_up = up & ~up.shift(1, fill_value=False)
    cross_dn = dn & ~dn.shift(1, fill_value=False)

    sig = pd.DataFrame(index=wt1.index)
    # classic: line cross while inside the zone
    sig["buy_cross_in_OS"] = (cross_up & (wt1 < OS)).fillna(False)
    sig["sell_cross_in_OB"] = (cross_dn & (wt1 > OB)).fillna(False)
    # looser published variant: level cross only
    sig["buy_on_OS_exit"] = ((wt1 > OS) & (wt1.shift(1) <= OS)).fillna(False)
    sig["sell_on_OB_exit"] = ((wt1 < OB) & (wt1.shift(1) >= OB)).fillna(False)
    return sig


def main():
    print("WAVETREND (LazyBear) STUDY")
    print(f"published params: n1={N1} n2={N2} wt2=SMA({WT2_LEN}) OB/OS=+/-{OB:.0f} "
          f"- UNFITTED, tested as published")
    print("=" * 104)

    rows, ab_rows = [], []
    for asset, fname in ASSETS:
        df = load_ohlcv(fname)
        close = df["close"]
        wt1, wt2 = wavetrend(df)
        sig = signals(wt1, wt2)
        rets = close.pct_change()

        print(f"\n===== {asset}  ({len(df)} bars, {close.index[0].date()} .. {close.index[-1].date()})")
        print(f"  now: wt1 {wt1.iloc[-1]:+7.2f}  wt2 {wt2.iloc[-1]:+7.2f}  "
              f"zone {'OVERBOUGHT' if wt1.iloc[-1] > OB else 'OVERSOLD' if wt1.iloc[-1] < OS else 'neutral'}")

        fwd7 = (close.shift(-7) / close - 1) * 100
        fwd30 = (close.shift(-30) / close - 1) * 100
        base7, base30 = fwd7.mean(), fwd30.mean()
        print(f"  baseline forward returns: 7d {base7:+.2f}%   30d {base30:+.2f}%  "
              f"(hit rate {(fwd30 > 0).mean()*100:.0f}%)")
        for col in sig.columns:
            m = sig[col]
            n = int(m.sum())
            if n == 0:
                print(f"  {col:18} no signals")
                continue
            f7, f30 = fwd7[m], fwd30[m]
            hit = (f30 > 0).mean() * 100
            print(f"  {col:18} n={n:<4} fwd7 {f7.mean():+6.2f}%  fwd30 {f30.mean():+7.2f}%  "
                  f"hit30 {hit:>3.0f}%   (edge vs base 7d {f7.mean()-base7:+5.2f} / "
                  f"30d {f30.mean()-base30:+6.2f})")

        # lead-time / FP at major turns
        lt = leadtime_table(close, sig, asset)
        for r in lt:
            print(f"    lead-time {r['trigger']:18} fires {r['fires']:<4} warned "
                  f"{r['warned']:>3}/{r['events']:<4} lead_med {r['lead_med']}  fp_rate {r['fp_rate']}")
        rows += lt

        # A/B: does a WTO exit overlay improve the validated MA50 filter?
        base_pos = (close > close.rolling(50).mean()).astype(float)
        sell = sig["sell_cross_in_OB"] | sig["sell_on_OB_exit"]
        # hold until an OB sell signal fires, then go flat until price reclaims MA50
        overlay = base_pos.copy()
        flat_until = None
        for ts in close.index:
            if flat_until is not None and ts <= flat_until:
                overlay.loc[ts] = 0.0
            elif bool(sell.get(ts, False)):
                overlay.loc[ts] = 0.0
                nxt = close.index[close.index.get_loc(ts) + 1:]
                reclaim = nxt[base_pos.reindex(nxt).fillna(0).to_numpy() == 1]
                flat_until = reclaim[0] if len(reclaim) else close.index[-1]
        for label, pos in (("ma50 only", base_pos), ("ma50 + wto exit", overlay)):
            eq, flips = sim(pos, rets)
            st = report(eq, flips, f"  {label}")
            ab_rows.append({"asset": asset, "rule": label, **st})

    print("\n" + "=" * 104)
    if ab_rows:
        ab = pd.DataFrame(ab_rows)
        print("A/B SUMMARY (mean across assets)")
        print(ab.groupby("rule").agg(total=("total", "mean"), maxdd=("maxdd", "mean"),
                                     sharpe=("sharpe", "mean")).to_string(
            float_format=lambda v: f"{v:.3f}"))
    print("\nRead: a signal is only useful here if it beats the BASELINE forward return")
    print("with a hit rate above the base rate, AND survives the lead-time/FP test.")
    print("Divergence-based usage is deliberately untested: it is interpretive.")


if __name__ == "__main__":
    main()
