#!/usr/bin/env python3
"""
CLUSTER-DIFFUSION BACKTEST v2 - on TRUE pair candles from Binance Vision archive
(keyless zip-per-day bucket, verified reachable 2026-08-29; the API host is
geo-blocked but data.binance.vision is not).

Claim under test: after a leader 4x+ re-rates its cluster vs BTC, peers whose OWN
pair already sits ABOVE its pair-200dMA (XMR-state) beat BTC over next 90/180d,
while below-MA peers (DASH-state) do not.

Zero lookahead: state at t from candles <= t only. Data cached to
crypto-micro-analysis/data/binance_pairs/<PAIR>.csv (append-only dataset).
"""
import zipfile, io, os, sys, time, urllib.request, datetime
import statistics as st
from pathlib import Path

CACHE = Path('/home/thatssoheil/projects/crypto-micro-analysis/data/binance_pairs')
CACHE.mkdir(parents=True, exist_ok=True)
UA = {'User-Agent': 'Mozilla/5.0'}
DAY = 86400

CLUSTERS = {
    'privacy': ['ZECBTC','XMRBTC','DASHBTC'],
    'l1':      ['SOLBTC','AVAXBTC','NEARBTC','INJBTC','SUIBTC','APTBTC','ATOMBTC','LTCBTC','ETCBTC'],
    'defi':    ['UNIBTC','AAVEBTC','LINKBTC','CRVBTC','FILBTC','AXSBTC'],
    'ai':      ['TAOBTC','FETBTC'],
    'memes':   ['DOGEBTC','WIFBTC'],
}
ALL = sorted({p for v in CLUSTERS.values() for p in v})

def fetch_pair(pair):
    """monthly kline zips (1 req/pair-month), walking BACK from current month and
    stopping when a cached month is reached; caches as csv ts,close."""
    f = CACHE / f"{pair}.csv"
    have = {}
    if f.exists():
        for line in f.read_text().splitlines()[1:]:
            ts, c = line.split(','); have[int(ts)] = float(c)
    cached_months = {datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m") for t in have}
    y, m = 2026, 8
    stop = False
    while (y, m) >= (2019, 1) and not stop:
        stamp = f"{y}-{m:02d}"
        # always re-fetch the two most recent months (revisions/incomplete); stop at older cached
        if stamp in cached_months and (y, m) < (2026, 7):
            break
        url = f"https://data.binance.vision/data/spot/monthly/klines/{pair}/1d/{pair}-1d-{stamp}.zip"
        try:
            req = urllib.request.Request(url, headers=UA)
            raw = urllib.request.urlopen(req, timeout=20).read()
            z = zipfile.ZipFile(io.BytesIO(raw))
            txt = z.read(z.namelist()[0]).decode()
            for row in txt.strip().splitlines():
                parts = row.split(',')
                if not parts or not parts[0].strip().isdigit(): continue
                tsv = int(parts[0])
                while tsv > 4e9: tsv //= 1000   # ms -> s, us -> s, ns -> s
                have[tsv] = float(parts[4])
        except Exception:
            pass  # 404 = not listed that month
        m -= 1
        if m == 0: y, m = y-1, 12
    rows = sorted(have.items())
    f.write_text("ts,close\n" + "\n".join(f"{t},{c}" for t, c in rows))
    return dict(have)

print("== fetching pair candles (Binance Vision) ==", flush=True)
S = {}
for p in ALL:
    try:
        S[p] = fetch_pair(p)
        if S[p]:
            d0 = datetime.datetime.utcfromtimestamp(min(S[p])).date()
            d1 = datetime.datetime.utcfromtimestamp(max(S[p])).date()
            print(f"  {p:9s} {len(S[p]):5d}d  {d0} -> {d1}", flush=True)
        else:
            print(f"  {p:9s} EMPTY", flush=True)
    except KeyboardInterrupt: raise
print("fetch done", flush=True)

def series(p):
    days = sorted(S[p]); return days, [S[p][t] for t in days]

def idx_le(days, t):
    lo, hi = 0, len(days)
    while lo < hi:
        mid = (lo+hi)//2
        if days[mid] <= t: lo = mid+1
        else: hi = mid
    return lo-1

def val(p, t, off=0):
    days, vals = series(p)
    i = idx_le(days, t+off)
    if i < 0 or days[i] > t+off+3*DAY: return None, None
    # require the day be within 3d of target (avoid stale edges)
    if abs(days[i] - (t+off)) > 3*DAY: return None, None
    return days[i], vals[i]

def ma200(p, t):
    days, vals = series(p)
    i = idx_le(days, t)
    if i+1 < 200: return None
    return st.mean(vals[i-199:i+1])

# ---------- leader events ----------
def trail_chg(p, t, n=180):
    days, vals = series(p)
    past = [(d, v) for d, v in zip(days, vals) if t - n*DAY < d <= t]
    if len(past) < 120: return None
    return past[-1][1] / past[0][1]

print("\n== leader events (180d pair >=4x, cluster max, 365d dedupe per cluster) ==", flush=True)
events = []
for cname, members in CLUSTERS.items():
    last_t = 0
    for t_ref in range(1546300800, 1787800000, 5*DAY):
        chgs = {p: trail_chg(p, t_ref) for p in members if p in S and S[p]}
        chgs = {p: c for p, c in chgs.items() if c}
        if not chgs: continue
        lead = max(chgs, key=chgs.get)
        if chgs[lead] >= 4 and t_ref - last_t >= 300*DAY:
            events.append((cname, lead, t_ref, chgs[lead]))
            last_t = t_ref
for c, p, t, ch in events:
    print(f"  {c:7s} {p:8s} at {datetime.datetime.utcfromtimestamp(t).date()}  (180d pair {ch:.1f}x)")

# ---------- outcomes ----------
conf90, conf180, lag90, lag180 = [], [], [], []
per_event = []
print("\n== per-event peer outcomes (fwd pair return vs BTC) ==", flush=True)
for c, lead, t, _ in events:
    line = [f"  {c:7s} {datetime.datetime.utcfromtimestamp(t).date()} lead {lead:8s}"]
    _, lp = val(lead, t); _, lf90 = val(lead, t, 90*DAY)
    if lp and lf90:
        line.append(f"[leader fwd90 {lf90/lp-1:+.0%}]")
    for p in CLUSTERS[c]:
        if p == lead or p not in S or not S[p]: continue
        _, pv = val(p, t); _, f90 = val(p, t, 90*DAY); _, f180 = val(p, t, 180*DAY)
        m = ma200(p, t)
        if not pv or not f90 or m is None: continue
        r90 = f90/pv-1; r180 = (f180/pv-1) if f180 else None
        cf = pv > m
        (conf90 if cf else lag90).append(r90)
        if r180 is not None: (conf180 if cf else lag180).append(r180)
        line.append(f"{p:8s} {'CONF' if cf else 'lagger'} {r90:+7.0%}/90d" + (f" {r180:+7.0%}/180d" if r180 is not None else ""))
    print("  " + "  ".join(line), flush=True)

def summ(name, r90, r180):
    if not r90: print(f"  {name:30s} n=0"); return
    beat90 = sum(1 for x in r90 if x > 0)/len(r90)*100
    l = f"  {name:30s} n={len(r90):3d}  med {st.median(r90)*100:+6.0f}%  mean {st.mean(r90)*100:+6.0f}%  beatBTC {beat90:.0f}%"
    if r180:
        l += f"   | n180={len(r180)} med {st.median(r180)*100:+6.0f}% beat {sum(1 for x in r180 if x>0)/len(r180)*100:.0f}%"
    print(l)

print(f"\n=== AGGREGATE: {len(events)} leader events ===")
summ("CONFIRMED peers (pair>200MA)", conf90, conf180)
summ("lagger peers (pair<200MA)", lag90, lag180)
print("\n(done)")
