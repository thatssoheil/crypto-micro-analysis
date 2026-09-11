#!/usr/bin/env python3
"""
FUNDING STUDY - does perp funding carry LEAD time over spot? (Deribit, 2019+)

Companion to funding_probe.py (which is a live snapshot from Kraken). This one is
the historical evidence: Deribit's public funding history for BTC-PERPETUAL and
ETH-PERPETUAL reaches back to 2019-06 (~7 years), enough for a real event study.

ZEC is NOT here: Deribit lists no ZEC perpetual, and Kraken holds only ~1 year of
ZEC funding, so ZEC funding cannot be validated yet - it is accumulate-and-revisit.
That asymmetry is stated rather than papered over.

Tests, using the same event definitions and lead-time/FP machinery as the alert
study (imported, not re-implemented, so the numbers stay comparable):
  TOP    = 90d high then -25% within 90d
  BOTTOM = 90d low then +30% within 90d
  trigger: funding z-score (90d) >= +2 (long crowding) or <= -2 (short crowding)

Then an A/B over the full window: MA50 filter vs MA50 + funding de-risk.

STATELESS: prints to stdout, writes nothing to the repo.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from micro_backtest import load_panel, sim, report          # noqa: E402
from micro_alerts import find_events, leadtime_table        # noqa: E402

UA = {"User-Agent": "Mozilla/5.0"}
URL = ("https://www.deribit.com/api/v2/public/get_funding_rate_history"
       "?instrument_name={inst}&start_timestamp={s}&end_timestamp={e}&count=1000")
INSTRUMENTS = {"btc": "BTC-PERPETUAL", "eth": "ETH-PERPETUAL"}
START = pd.Timestamp("2019-06-01", tz="UTC")
Z_WINDOW, EXTREME_Z = 90, 2.0
CHUNK_DAYS = 40          # 40d * 24 points/day = 960 < count cap 1000


def fetch_deribit(inst, start, end):
    """Hourly funding series for one instrument, chunked to respect count=1000."""
    out = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=CHUNK_DAYS), end)
        url = URL.format(inst=inst, s=int(cur.timestamp() * 1000),
                         e=int(nxt.timestamp() * 1000))
        d = None
        for attempt in range(3):
            try:
                d = json.load(urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=10))
                break
            except Exception as exc:
                # Intermittent URLError/rate-limit: back off and retry, because a
                # silently skipped chunk leaves a hole in the funding series.
                if attempt == 2:
                    print(f"    fetch FAILED {cur.date()}: {type(exc).__name__}",
                          file=sys.stderr)
                else:
                    time.sleep(1.5 * (attempt + 1))
        if d is None:
            cur = nxt
            continue
        for r in (d.get("result") or []):
            out.append((pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"),
                        float(r["interest_1h"]) if "interest_1h" in r
                        else float(r.get("interest", 0.0))))
        cur = nxt
        time.sleep(0.05)
    if not out:
        return None
    s = pd.Series(dict(out)).sort_index()
    return s[~s.index.duplicated()]


def main():
    panel = load_panel()
    end = pd.Timestamp.utcnow().tz_localize("UTC") if panel.index.tz is None else panel.index[-1]
    print("FUNDING STUDY - Deribit perp funding as a leading/crowding signal")
    print(f"window {START.date()} .. {end.date()}\n")

    study_rows, ab_rows = [], []
    wanted = [a for a in sys.argv[1:] if a in INSTRUMENTS]
    items = {a: i for a, i in INSTRUMENTS.items() if not wanted or a in wanted}
    for asset, inst in items.items():
        print(f"===== {asset.upper()}  ({inst})")
        h = fetch_deribit(inst, START, end)
        if h is None or len(h) < 5000:
            print("   insufficient funding history - skipped\n")
            continue
        # interest_1h is a PER-HOUR rate; a daily cost figure is 24x the hourly mean.
        daily = (h.resample("1D").mean() * 24).dropna()
        z = ((daily - daily.rolling(Z_WINDOW, min_periods=20).mean())
             / daily.rolling(Z_WINDOW, min_periods=20).std())
        close = panel[asset]
        idx = daily.index.intersection(close.index)
        print(f"   funding: {len(daily)} daily obs, {daily.index[0].date()} .. {daily.index[-1].date()}")
        print(f"   last {daily.iloc[-1]*100:+.4f}%/day  90d z {z.iloc[-1]:+.2f}  "
              f"percentile {float((daily <= daily.iloc[-1]).mean())*100:.0f}%")

        trig = pd.DataFrame({
            "funding_z_high": (z.reindex(idx) >= EXTREME_Z).fillna(False),
            "funding_z_low": (z.reindex(idx) <= -EXTREME_Z).fillna(False)}, index=idx)
        sub_close = close.reindex(idx)
        rows = leadtime_table(sub_close, trig, asset.upper())
        for r in rows:
            study_rows.append(r)
            print(f"   {r['trigger']:16} fires {r['fires']:>4}  warned {r['warned']:>3}/"
                  f"{r['events']:<4} lead_med {r['lead_med']}  fp_rate {r['fp_rate']}")

        # A/B on the SAME dates: MA50 vs MA50 + go flat while long-crowded
        rets = panel.pct_change()
        base = (sub_close > sub_close.rolling(50).mean()).astype(float)
        guarded = base.where(~(z.reindex(idx) >= EXTREME_Z).fillna(False), 0.0)
        r = rets[asset].reindex(idx).ffill()
        print()
        for label, pos in (("ma50 only", base), ("ma50+funding-de-risk", guarded)):
            p = pos.shift(1).fillna(0.0)
            eq = (1.0 + p * r.fillna(0.0)).cumprod()
            tot = (eq.iloc[-1] - 1) * 100
            rr = eq.pct_change().dropna()
            sh = (rr.mean() / rr.std() * np.sqrt(365)) if rr.std() > 0 else 0.0
            dd = (eq / eq.cummax() - 1).min() * 100
            flips = int((p.diff().fillna(0).abs() > 0).sum())
            print(f"   {label:22} total {tot:>9.1f}%  sharpe {sh:>5.2f}  maxDD {dd:>7.1f}%  changes {flips}")
            ab_rows.append({"asset": asset, "rule": label, "total": tot, "sharpe": sh, "maxdd": dd})
        print()

    if study_rows:
        print("LEAD-TIME SUMMARY (across assets with sufficient history)")
        print(pd.DataFrame(study_rows)[["series", "trigger", "fires", "warned", "missed",
                                        "lead_med", "lead_min", "fp_rate"]].to_string(index=False))
    print("\nZEC: no Deribit perp and only ~1 year of Kraken funding - NOT validated here.")
    print("Verdict rule: funding earns a place only if it warns events with a lead AND")
    print("a low false-positive rate, AND the A/B beats the plain MA50 filter.")


if __name__ == "__main__":
    main()
