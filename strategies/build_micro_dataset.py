#!/usr/bin/env python3
"""
MICRO DATASET BUILDER - fetches the locally-owned dataset for crypto-micro-analysis.

Charts (data/micro_dataset/):
  btcusd_daily_bitstamp.csv   BTC/USD daily OHLCV  (Bitstamp, 2011+)
  ethusd_daily_bitstamp.csv   ETH/USD daily OHLCV  (Bitstamp, 2015+)
  zecusd_daily_bitstamp.csv   ZEC/USD daily OHLCV  (Bitstamp, 2016+)
  fear_greed.csv              Fear & Greed index   (alternative.me, 2018+)

Crosses (ETH/BTC, ZEC/BTC) are COMPUTED as USD-close ratios at analysis time -
never fetched separately - so every series shares one source and one clock.

STATELESS by repo convention: this script only writes dataset CSVs + manifest
(the input). It never writes results. Re-running pulls whatever is new.
Keyless sources only - no API keys required.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

DATA = Path(__file__).parent.parent / "data" / "micro_dataset"
DATA.mkdir(parents=True, exist_ok=True)
MANIFEST = DATA / "manifest.json"

FAILED = []


def get(url, params=None, tries=3):
    """GET with retry + backoff. Returns Response or None."""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30,
                             headers={"User-Agent": "crypto-micro-analysis/1.0"})
            if r.status_code == 200:
                return r
            print(f"    HTTP {r.status_code} (try {i+1}/{tries})")
        except requests.RequestException as e:
            print(f"    {type(e).__name__}: {e} (try {i+1}/{tries})")
        time.sleep(2 * (i + 1))
    return None


def save_csv(name, rows, header):
    p = DATA / f"{name}.csv"
    rows = sorted(set(tuple(r) for r in rows))
    with open(p, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"    saved {p.name}: {len(rows)} rows "
          f"({rows[0][0][:10]} .. {rows[-1][0][:10]})")
    return {"rows": len(rows), "first": rows[0][0], "last": rows[-1][0]}


def bitstamp_ohlc(pair, name, max_iters=100):
    """Walk back full daily history. Bitstamp 'start' is a FROM-filter: each
    call returns up to 1000 candles from start; set start = oldest - 1000*step.
    (Same proven approach as crypto-macro-analysis build_macro_dataset.py.)"""
    step = 86400
    rows, start = [], None
    for _ in range(max_iters):
        url = f"https://www.bitstamp.net/api/v2/ohlc/{pair}/?step={step}&limit=1000"
        if start:
            url += f"&start={start}"
        r = get(url)
        if not r:
            break
        ohlc = r.json()["data"]["ohlc"]
        if not ohlc:
            break
        for o in ohlc:
            rows.append([
                datetime.fromtimestamp(int(o["timestamp"]), tz=timezone.utc)
                .strftime("%Y-%m-%d %H:%M:%S"),
                o["open"], o["high"], o["low"], o["close"], o["volume"],
            ])
        oldest = int(ohlc[0]["timestamp"])
        if len(ohlc) < 1000:
            break
        start = oldest - step * 1000
        time.sleep(0.6)
    if not rows:
        FAILED.append(name)
        return None
    meta = save_csv(name, rows, ["ts", "open", "high", "low", "close", "volume"])
    return {"source": "Bitstamp v2 ohlc", **meta}


def yahoo_daily(symbol, name, period1):
    """Full-history daily OHLCV from Yahoo Finance v8 chart API (keyless).
    Used for ZEC/USD because Bitstamp delisted/relisted the pair (its ZEC
    history starts 2026-03 only). Drops None rows (no-trade days)."""
    import time as _time
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{symbol}?period1={period1}&period2={int(_time.time())}&interval=1d&events=history")
    r = get(url)
    if not r:
        FAILED.append(name)
        return None
    q = r.json()["chart"]["result"][0]
    ts = q.get("timestamp") or []
    quote = q["indicators"]["quote"][0]
    rows = []
    for i, t in enumerate(ts):
        o, h, l, c, v = (quote.get(k, [None])[i] if quote.get(k) else None
                         for k in ("open", "high", "low", "close", "volume"))
        if None in (o, h, l, c):
            continue
        rows.append([datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                     o, h, l, c, v or 0])
    if not rows:
        FAILED.append(name)
        return None
    meta = save_csv(name, rows, ["ts", "open", "high", "low", "close", "volume"])
    return {"source": f"Yahoo v8 chart {symbol}", **meta}


def fear_greed():
    # limit=3000 silently truncated the history: the source holds 3141 rows back to
    # 2018-02-01, so the oldest 141 days (2018-02-01..2018-06-24) were dropped on
    # every rebuild. 10000 covers the full series.
    r = get("https://api.alternative.me/fng/", params={"limit": 10000, "format": "json"})
    if not r:
        FAILED.append("fear_greed")
        return None
    rows = [[datetime.fromtimestamp(int(d["timestamp"]), tz=timezone.utc).strftime("%Y-%m-%d"),
             d["value"], d["value_classification"]]
            for d in r.json()["data"]]
    meta = save_csv("fear_greed", rows, ["date", "value", "classification"])
    return {"source": "alternative.me fng", **meta}


def print_coverage():
    """Report calendar coverage per dataset so holes are VISIBLE, never silent.

    Upstream sources have real gaps (alternative.me is missing 2024-10-26; Yahoo
    returns a null bar for ZEC on 2026-09-10). Those are honest gaps in the
    source, not fetch bugs - this report makes them explicit every run.
    """
    print("== coverage ==")
    for p in sorted(DATA.glob("*.csv")):
        rows = p.read_text().strip().splitlines()
        if len(rows) < 2:
            print(f"    {p.name}: EMPTY")
            continue
        dates = []
        for line in rows[1:]:
            try:
                dates.append(datetime.strptime(line.split(",")[0][:10], "%Y-%m-%d").date())
            except ValueError:
                pass
        if not dates:
            continue
        span = (dates[-1] - dates[0]).days + 1
        have = len(set(dates))
        missing = span - have
        flag = "  <-- gap" if missing else ""
        print(f"    {p.name}: {have} rows, {dates[0]} .. {dates[-1]}, span {span}d, "
              f"missing {missing}{flag}")


def main():
    print("=== micro dataset build ===")
    manifest = {}
    # ZEC: Yahoo primary (full history; Bitstamp delisted/relisted the pair).
    print("== Bitstamp BTCUSD daily ==")
    m = bitstamp_ohlc("btcusd", "btcusd_daily")
    if m:
        manifest["btcusd_daily"] = m
    print("== Bitstamp ETHUSD daily ==")
    m = bitstamp_ohlc("ethusd", "ethusd_daily")
    if m:
        manifest["ethusd_daily"] = m
    print("== Yahoo ZEC-USD daily ==")
    m = yahoo_daily("ZEC-USD", "zecusd_yahoo", period1=1461360000)  # 2016-04-23, zcash launch era
    if m:
        manifest["zecusd_yahoo"] = m
    print("== Bitstamp ZECUSD daily (live cross-check) ==")
    m = bitstamp_ohlc("zecusd", "zecusd_bitstamp_live")
    if m:
        manifest["zecusd_bitstamp_live"] = m

    print("== Fear & Greed ==")
    m = fear_greed()
    if m:
        manifest["fear_greed"] = m

    manifest["_generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Preserve existing chart entries not rebuilt here, but ONLY when their CSV
    # still exists on disk. Without this guard a renamed/removed dataset lingers
    # in the manifest forever - a phantom "zecusd_daily" entry survived the rename
    # to zecusd_bitstamp_live and made the manifest untrustworthy as an index.
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        for k, v in old.items():
            if k == "_generated":
                continue
            if (DATA / f"{k}.csv").exists():
                manifest.setdefault(k, v)
            else:
                print(f"    manifest: dropped stale entry '{k}' (no {k}.csv on disk)")

    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {MANIFEST.name}")

    print_coverage()

    if FAILED:
        print(f"FAILED charts: {FAILED}")
        raise SystemExit(1)
    print("=== OK ===")


if __name__ == "__main__":
    main()
