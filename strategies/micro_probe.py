#!/usr/bin/env python3
"""
MICRO INTRADAY PROBE - live prices vs the committed daily levels, for BTC/ETH/ZEC.

Why: the daily trigger layer can only confirm; it sees one close per day. To be
warned BEFORE a move you need to know where price sits against the levels
intraday. This probe compares a live quote against levels computed from the
committed dataset.

Venue: Kraken public ticker - keyless, and the ONLY venue probed here that has a
live ZEC book (Bitstamp's ZEC ticker returns nothing, as this repo's AGENTS.md
already documents). Using one venue for all three keeps a single clock.

IMPORTANT CAVEAT: the levels come from the committed daily series (Bitstamp for
BTC/ETH, Yahoo for ZEC) while the live leg is Kraken, so a breach here is
PROVISIONAL - it is a heads-up, not a confirmed signal. The daily close confirms
it. This is deliberate: an intraday pierce that recovers is noise, not a trend break.

STATELESS (repo convention): prints to stdout, saves nothing.

Usage:
  ./.venv/bin/python strategies/micro_probe.py               # full state
  ./.venv/bin/python strategies/micro_probe.py --alert-only  # only alerts (empty = quiet)
"""
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent.parent / "data" / "micro_dataset"
MA_FAST, MA_SLOW, DD_WIN = 50, 200, 180
PROXIMITY = 0.02          # within 2% above a level = "approaching"
KRAKEN = "https://api.kraken.com/0/public/Ticker?pair="
UA = {"User-Agent": "Mozilla/5.0"}
PAIRS = {"BTC": "XXBTZUSD", "ETH": "XETHZUSD", "ZEC": "XZECZUSD"}


def load(name):
    df = pd.read_csv(DATA / f"{name}.csv")
    df["ts"] = pd.to_datetime(df["ts"])
    return df.drop_duplicates("ts").set_index("ts").sort_index()["close"]


def live_prices():
    """Kraken public ticker for the trio. Returns {asset: last} (may be partial)."""
    url = KRAKEN + ",".join(PAIRS.values())
    req = urllib.request.Request(url, headers=UA)
    data = json.load(urllib.request.urlopen(req, timeout=25))
    if data.get("error"):
        raise RuntimeError(f"kraken: {data['error']}")
    res = data["result"]
    out = {}
    for asset, pair in PAIRS.items():
        key = next((k for k in res if k.endswith(pair[1:]) or k == pair), None)
        if key:
            out[asset] = float(res[key]["c"][0])
    return out


def levels(close):
    last = float(close.iloc[-1])
    ma50 = float(close.rolling(MA_FAST).mean().iloc[-1])
    ma200 = float(close.rolling(MA_SLOW).mean().iloc[-1])
    high180 = float(close.rolling(DD_WIN).max().iloc[-1])
    return {"close": last, "ma50": ma50, "ma200": ma200, "high180": high180,
            "dd180": last / high180 - 1}


def main():
    alert_only = "--alert-only" in sys.argv
    series = {"BTC": load("btcusd_daily"), "ETH": load("ethusd_daily"),
              "ZEC": load("zecusd_yahoo")}
    use_ma200 = {"BTC": True, "ETH": True, "ZEC": False}   # rule 2: no 200d floor on ZEC

    try:
        live = live_prices()
    except Exception as exc:                                  # never crash the watcher
        print(f"ALERT: micro probe could not reach Kraken ({exc})")
        return

    alerts, lines = [], []
    for asset, close in series.items():
        if asset not in live:
            lines.append(f"  {asset}: no live quote")
            continue
        L = levels(close)
        px = live[asset]
        as_of = close.index[-1].date()
        lines.append(f"  {asset}: live {px:,.2f} | close({as_of}) {L['close']:,.2f} | "
                     f"MA50 {L['ma50']:,.2f} | MA200 {L['ma200']:,.2f} | "
                     f"180d high {L['high180']:,.2f} | dd180 {L['dd180']*100:+.1f}%")

        # 1. provisional trend break: the COMMITTED close was above a level, live is below
        if L["close"] >= L["ma50"] > px:
            alerts.append(f"ALERT {asset}: live {px:,.2f} is BELOW its MA50 {L['ma50']:,.2f} "
                          f"(committed close was above) - provisional trend break, "
                          f"confirm at the daily close")
        if use_ma200[asset] and L["close"] >= L["ma200"] > px:
            alerts.append(f"ALERT {asset}: live {px:,.2f} is BELOW its MA200 {L['ma200']:,.2f} "
                          f"- provisional floor breach, confirm at the daily close")

        # 2. approaching a level from above
        for lbl, lv in (("MA50", L["ma50"]), ("MA200", L["ma200"]), ("180d high", L["high180"])):
            if lv > px >= lv * (1 - PROXIMITY):
                alerts.append(f"ALERT {asset}: live {px:,.2f} is within {PROXIMITY:.0%} of its "
                              f"{lbl} {lv:,.2f}")

        # 3. intraday drawdown threshold crossed
        live_dd = px / L["high180"] - 1
        if live_dd <= -0.20 and L["dd180"] > -0.20:
            alerts.append(f"ALERT {asset}: live drawdown {live_dd*100:.1f}% from 180d high "
                          f"(committed close was {L['dd180']*100:.1f}%) - -20% level crossed")

    if alert_only:
        for a in alerts:
            print(a)
        return

    print(f"MICRO INTRADAY PROBE (Kraken live vs committed daily levels)")
    for ln in lines:
        print(ln)
    print()
    if alerts:
        for a in alerts:
            print(a)
    else:
        print("  no alerts - all assets clear of their levels")


if __name__ == "__main__":
    main()
