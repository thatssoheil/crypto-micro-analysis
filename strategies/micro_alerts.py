#!/usr/bin/env python3
"""
MICRO ALERT TRIGGERS - candidate early-warning triggers for BTC/ETH/ZEC, plus the
historical evidence that decides whether each has earned a place in monitoring.

Why this exists: the owner wants to be notified BEFORE a big move. This repo's own
tested facts say the phase score CONFIRMS breakdowns with a 10-31 day lag and does
NOT call tops (+1.67 at the Nov-2021 top = FAIL). So a trigger only gets wired into
monitoring if it demonstrates (a) measurable lead time before historical turns and
(b) an acceptable false-positive rate. This script measures both from the committed
dataset only - it does not assert that any trigger works.

STATELESS (repo convention): regenerates from the committed dataset, prints to
stdout, never saves results.

Usage:
  ./.venv/bin/python strategies/micro_alerts.py             # current trigger state
  ./.venv/bin/python strategies/micro_alerts.py --leadtime  # lead-time / FP evidence

Trigger set (built only from components the repo has already validated):
  ma50_cross_down   close crosses below MA50          (golden rule 1: MA50 is the edge)
  ma50_cross_up     close crosses back above MA50
  ma200_breach      close crosses below MA200         (BTC/ETH only - rule 2: the
                                                       200d floor HURTS ZEC, so ZEC
                                                       is evaluated without it)
  dd20_new          drawdown from the 180d high first passes -20%
  rs_*_decile_new   cross-pair RS band enters the upper/lower decile
  fng_extreme_new   Fear & Greed enters <=20 or >=80

Events ("a big move", stated so the test is falsifiable):
  TOP    : a 90d high from which price falls >=25% within the next 90 days
  BOTTOM : a 90d low from which price rises >=30% within the next 90 days
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data" / "micro_dataset"

MA_FAST, MA_SLOW, DD_WIN = 50, 200, 180
TOP_DROP, BOT_RALLY, EVENT_FWD = 0.25, 0.30, 90
LEAD_LOOKBACK = 60   # a firing warns an event if it lands within this many days before
HIT_WINDOW = 30      # a firing is a false positive if no event lands within this many days after


def load(name):
    df = pd.read_csv(DATA / f"{name}.csv")
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    return df


def load_fng():
    df = pd.read_csv(DATA / "fear_greed.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").set_index("date").sort_index()["value"]


def build_triggers(close, use_ma200=True):
    """Boolean trigger series for one price series. Every value uses only data
    available at that close (no lookahead)."""
    t = pd.DataFrame(index=close.index)
    ma50 = close.rolling(MA_FAST).mean()
    ma200 = close.rolling(MA_SLOW).mean()

    below = close < ma50
    t["ma50_cross_down"] = (below & ~below.shift(1, fill_value=False)).fillna(False)
    t["ma50_cross_up"] = (~below & below.shift(1, fill_value=False)).fillna(False)

    if use_ma200:
        b200 = close < ma200
        t["ma200_breach"] = (b200 & ~b200.shift(1, fill_value=False)).fillna(False)

    dd = close / close.rolling(DD_WIN).max() - 1
    d20 = dd < -0.20
    t["dd20_new"] = (d20 & ~d20.shift(1, fill_value=False)).fillna(False)
    return t


def build_cross_triggers(numer, denom, lo=0.10, hi=0.90, win=DD_WIN):
    """Cross-pair RS band triggers. The cross is COMPUTED from aligned USD closes
    (repo rule: crosses are never fetched separately), so both legs share one clock."""
    idx = numer.index.intersection(denom.index)
    ratio = (numer.loc[idx] / denom.loc[idx]).dropna()
    mn = ratio.rolling(win).min()
    mx = ratio.rolling(win).max()
    band = ((ratio - mn) / (mx - mn)).replace([np.inf, -np.inf], np.nan)

    low = (band < lo).fillna(False)
    high = (band > hi).fillna(False)
    t = pd.DataFrame(index=ratio.index)
    t["rs_low_decile_new"] = (low & ~low.shift(1, fill_value=False)).fillna(False)
    t["rs_high_decile_new"] = (high & ~high.shift(1, fill_value=False)).fillna(False)
    return t, ratio, band


def build_fng_trigger(fng):
    lo = fng <= 20
    hi = fng >= 80
    extreme = lo | hi
    return pd.DataFrame({"fng_extreme_new":
                         (extreme & ~extreme.shift(1, fill_value=False)).fillna(False)})


def find_events(series, drop=TOP_DROP, rally=BOT_RALLY, fwd=EVENT_FWD):
    """Positions of major tops and bottoms in `series` (see module docstring)."""
    fwd_min = series.shift(-1)[::-1].rolling(fwd, min_periods=1).min()[::-1]
    fwd_max = series.shift(-1)[::-1].rolling(fwd, min_periods=1).max()[::-1]
    top_flag = ((fwd_min / series - 1) <= -drop).fillna(False).to_numpy()
    bot_flag = ((fwd_max / series - 1) >= rally).fillna(False).to_numpy()
    arr = series.to_numpy()

    def collapse(flags, want_max):
        events, i, n = [], 0, len(flags)
        while i < n:
            if not flags[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            seg = arr[i:j + 1]
            events.append(i + (int(np.argmax(seg)) if want_max else int(np.argmin(seg))))
            i = j + 1
        return events

    return collapse(top_flag, True), collapse(bot_flag, False)


def leadtime_table(series, trig, label):
    """One row per trigger: how often it warned, with how much lead, and its FP rate."""
    tops, bots = find_events(series)
    events = sorted(set(tops) | set(bots))
    rows = []
    for col in trig.columns:
        fire_idx = np.flatnonzero(trig[col].to_numpy())
        leads, missed = [], 0
        for ev in events:
            prior = fire_idx[(fire_idx >= max(0, ev - LEAD_LOOKBACK)) & (fire_idx < ev)]
            if len(prior):
                leads.append(int(ev - prior[-1]))
            else:
                missed += 1
        fps = sum(1 for f in fire_idx
                  if not any(f < ev <= f + HIT_WINDOW for ev in events))
        rows.append({
            "series": label, "trigger": col, "events": len(events), "fires": len(fire_idx),
            "warned": len(events) - missed, "missed": missed,
            "lead_med": int(np.median(leads)) if leads else None,
            "lead_min": int(np.min(leads)) if leads else None,
            "lead_max": int(np.max(leads)) if leads else None,
            "fp": fps,
            "fp_rate": round(fps / len(fire_idx), 2) if len(fire_idx) else None})
    return rows


def print_state(btc, eth, zec, fng):
    print("MICRO ALERT STATE (from committed dataset)")
    print(f"  as of {btc.index[-1].date()}")
    for name, close, use200 in (("BTC", btc, True), ("ETH", eth, True), ("ZEC", zec, False)):
        tg = build_triggers(close, use200)
        ma50 = close.rolling(MA_FAST).mean().iloc[-1]
        ma200 = close.rolling(MA_SLOW).mean().iloc[-1]
        dd = close.iloc[-1] / close.rolling(DD_WIN).max().iloc[-1] - 1
        firing = [c for c in tg.columns if bool(tg[c].iloc[-1])]
        print(f"  {name}: close {close.iloc[-1]:,.2f} | MA50 {ma50:,.2f} "
              f"({'above' if close.iloc[-1] > ma50 else 'BELOW'}) | MA200 {ma200:,.2f} "
              f"({'above' if close.iloc[-1] > ma200 else 'BELOW'}) | dd180 {dd*100:+.1f}%")
        print(f"      firing today: {', '.join(firing) if firing else 'none'}")
    for lbl, num, den in (("ZEC/BTC", zec, btc), ("ZEC/ETH", zec, eth)):
        _, _, band = build_cross_triggers(num, den)
        print(f"  RS {lbl}: band {band.iloc[-1]*100:.0f}% of 180d range"
              f"{'  <-- UPPER DECILE' if band.iloc[-1] > 0.90 else ''}"
              f"{'  <-- LOWER DECILE' if band.iloc[-1] < 0.10 else ''}")
    print(f"  F&G: {fng.iloc[-1]:.0f}")


def main():
    btc = load("btcusd_daily")["close"]
    eth = load("ethusd_daily")["close"]
    zec = load("zecusd_yahoo")["close"]
    fng = load_fng()

    if "--leadtime" not in sys.argv:
        print_state(btc, eth, zec, fng)
        return

    print("ALERT TRIGGER LEAD-TIME EVIDENCE")
    print(f"  events: TOP = 90d high then -{TOP_DROP:.0%} within {EVENT_FWD}d | "
          f"BOTTOM = 90d low then +{BOT_RALLY:.0%} within {EVENT_FWD}d")
    print(f"  a firing WARNS an event if it lands within {LEAD_LOOKBACK}d before it; "
          f"it is a FALSE POSITIVE if no event follows within {HIT_WINDOW}d\n")

    rows = []
    for label, close, use200 in (("BTC", btc, True), ("ETH", eth, True), ("ZEC", zec, False)):
        rows += leadtime_table(close, build_triggers(close, use200), label)

    # F&G is market-wide: judge it against BTC's price events.
    fng_t = build_fng_trigger(fng).reindex(btc.index).ffill().fillna(False)
    rows += leadtime_table(btc, fng_t.astype(bool), "BTC (F&G)")

    # RS triggers: the thing that "moves" is the ratio itself.
    for lbl, num, den in (("ZEC/BTC", zec, btc), ("ZEC/ETH", zec, eth)):
        ct, ratio, _ = build_cross_triggers(num, den)
        rows += leadtime_table(ratio, ct, lbl)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\n  verdict rule: a trigger earns monitoring only if warned > missed "
          "AND fp_rate is low (a noisy trigger that fires constantly is useless).")


if __name__ == "__main__":
    main()
