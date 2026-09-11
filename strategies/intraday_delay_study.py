#!/usr/bin/env python3
"""
INTRADAY DELAY STUDY - does finer sampling actually cut detection delay?

The daily filter study showed 10.7d (tops) / 16.7d (bottoms) median delay for MA50
on daily closes, and that better DAILY estimators cannot buy much of it back. The
open question is sampling: same family, same horizon, finer bars.

Fair-comparison design (the whole point):
  daily : close vs MA50 on daily bars          -> 50-day horizon
  4h    : close vs MA(300) on 4h bars          -> 300 * 4h = 50 days, SAME horizon
  4h-fast: close vs MA(150) on 4h bars         -> 25-day horizon (what a lower-lag
                                                  setting actually costs)
Comparing only horizons and sampling this way isolates the effect of bar size from
the effect of smoothing.

Events are computed on the DAILY committed close (same definition as every other
study here) and both filters are timed against those same turns, so the delays are
directly comparable. Whipsaws are normalised to changes-per-year, since a 4h filter
gets 6x as many bars to flip on.

Data: Bitstamp public OHLC at step=14400 (4h), paginated forward. Keyless.
ZEC is intentionally absent: Bitstamp's ZEC market only has 4h bars from 2026-07,
nowhere near enough history for an event study. BTC/ETH only.

STATELESS: prints to stdout, writes nothing.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from micro_backtest import load_panel, sim                     # noqa: E402
from micro_alerts import find_events                           # noqa: E402

UA = {"User-Agent": "Mozilla/5.0"}
STEP = 14400           # 4h bars
PAIRS = {"btc": "btcusd", "eth": "ethusd"}
START = "2021-01-01"
END = "2026-09-11"


def get(url):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=30))


def fetch_4h(pair, start=START, end=END):
    """Paginate Bitstamp 4h OHLC forward from `start`."""
    cur = int(pd.Timestamp(start).timestamp())
    stop = int(pd.Timestamp(end).timestamp())
    rows = []
    while cur < stop:
        url = (f"https://www.bitstamp.net/api/v2/ohlc/{pair}/"
               f"?step={STEP}&limit=1000&start={cur}")
        try:
            d = get(url)
        except Exception as exc:
            print(f"    fetch error at {pd.Timestamp(cur, unit='s').date()}: "
                  f"{type(exc).__name__}", file=sys.stderr)
            break
        oh = d.get("data", {}).get("ohlc", [])
        if not oh:
            break
        for r in oh:
            rows.append((pd.Timestamp(int(r["timestamp"]), unit="s", tz="UTC"),
                         float(r["close"])))
        last = int(oh[-1]["timestamp"])
        if last <= cur or len(oh) < 1000:
            break
        cur = last + STEP
        time.sleep(0.05)
    s = pd.Series(dict(rows)).sort_index()
    return s[~s.index.duplicated()]


def flips(pos):
    return np.flatnonzero(np.diff(pos.astype(int).to_numpy()) != 0)


def delay_days(event_ts, pos, bar_days):
    """Days from a daily event until this filter flips, or None.

    Events before the series start, and the MA warm-up, are skipped rather than
    treated as an instant flip - otherwise they read as 0-day delay.
    """
    if len(pos.index) == 0 or event_ts < pos.index[0]:
        return None
    prior = pos.asof(event_ts)
    if prior is None or pd.isna(prior):
        return None
    target = 1 - int(prior)                  # the flip we are waiting for
    after = pos.loc[pos.index >= event_ts]
    after = after[after.notna()]
    if len(after) == 0:
        return None
    nz = np.flatnonzero(after.astype(int).to_numpy() == target)
    if len(nz) == 0:
        return None
    return (after.index[nz[0]] - event_ts).total_seconds() / 86400.0


def main():
    panel = load_panel()
    print("INTRADAY DELAY STUDY - finer bars, same horizon (Bitstamp 4h, keyless)")
    print(f"window {START} .. {END}; events from the committed daily close\n")

    summary = []
    for asset, pair in PAIRS.items():
        print(f"===== {asset.upper()}  ({pair} 4h)")
        bars = fetch_4h(pair)
        if bars is None or len(bars) < 2000:
            print("   insufficient 4h history - skipped\n")
            continue
        print(f"   4h bars: {len(bars)}  {bars.index[0].date()} .. {bars.index[-1].date()}")

        daily = panel[asset]
        daily_pos = (daily > daily.rolling(50).mean()).astype(float)
        f4_same = (bars > bars.rolling(300).mean()).astype(float)     # 50-day horizon
        f4_fast = (bars > bars.rolling(150).mean()).astype(float)     # 25-day horizon

        tops, bots = find_events(daily)
        events = [daily.index[t] for t in tops] + [daily.index[b] for b in bots]

        print(f"   {'filter':22} {'delay_med':>10} {'delay_n':>8} {'changes/yr':>11}")
        for label, pos, bar_days in (("daily MA50", daily_pos, 1.0),
                                     ("4h MA(300) = 50d", f4_same, 1 / 6),
                                     ("4h MA(150) = 25d", f4_fast, 1 / 6)):
            ds = [d for d in (delay_days(e, pos, bar_days) for e in events) if d is not None]
            years = (pos.index[-1] - pos.index[0]).days / 365.25
            cpy = len(flips(pos)) / max(years, 1e-9)
            med = float(np.median(ds)) if ds else float("nan")
            print(f"   {label:22} {med:>9.1f}d {len(ds):>8} {cpy:>11.1f}")
            summary.append({"asset": asset, "filter": label, "delay_med": med,
                            "n": len(ds), "changes_per_year": cpy})
        print()

    if summary:
        df = pd.DataFrame(summary)
        print("SUMMARY (mean across assets)")
        print(df.groupby("filter").agg(delay_med=("delay_med", "mean"),
                                       n=("n", "mean"),
                                       changes_per_year=("changes_per_year", "mean")
                                       ).to_string(float_format=lambda v: f"{v:.2f}"))
    print("\nVerdict rule: intraday sampling earns its place ONLY if it cuts delay")
    print("materially without a proportional explosion in changes-per-year.")


if __name__ == "__main__":
    main()
