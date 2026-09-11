#!/usr/bin/env python3
"""
VOL-TARGET INCREMENTAL STUDY - does a vol overlay add anything ON TOP of MA50?

The previous study (sizing_study.py) found a long-only vol-target overlay
improved Sharpe (0.82 -> 0.91), maxDD (-89.9% -> -77.0%) and return at once.
But the MA50 close-filter already de-risks into crashes, so the overlay may be
REDUNDANT - the same job done twice. This study isolates the increment.

Pre-registered BEFORE results:
  Q1 does the vol overlay still help once MA50 filtering is in place?
  Q2 is the benefit concentrated in the SAME episodes MA50 protects (2018,
     Feb-Mar 2020, 2022) or in others?
  Q3 how much do the two mechanisms overlap in time?
  Q4 does the overlay's own vol estimate need to come from the filtered book, or
     is the underlying asset vol enough?

Protocol (repo contract):
  - aligned daily closes BTC/ETH/ZEC from panel start
  - MA50: long when close > MA50, decided at t close, applied to t+1 (shift(1))
  - each asset carries weight 1/3 when long, 0 when flat -> gross <= 1
  - vol overlay: scale = min(1, target / trailing annualised vol), decided at
    monthly rebalance from data through that close, applied forward
  - VOL_WIN = 60 pre-registered; targets 30/40/50
  - cash earns 0%; long-only spot, no leverage
  - VT-A scales by the FILTERED book's own trailing vol
  - VT-B scales by the UNFILTERED equal-weight book's trailing vol
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data" / "micro_dataset"
ASSETS = ["BTC", "ETH", "ZEC"]
VOL_WIN = 60
TRADING_DAYS = 365
EPISODES = {
    "2018 bear": ("2018-01-01", "2018-12-31"),
    "2020 covid crash": ("2020-02-19", "2020-03-23"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
    "2025-26 ZEC regime": ("2025-01-01", "2026-09-11"),
}


def close(name):
    df = pd.read_csv(DATA / f"{name}.csv")
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")["close"].sort_index()


def panel():
    return pd.DataFrame({
        "BTC": close("btcusd_daily"),
        "ETH": close("ethusd_daily"),
        "ZEC": close("zecusd_yahoo"),
    }).dropna()


def monthly_dates(idx):
    """First trading day of each calendar month (groupby on the INDEX itself
    raises 'unhashable type: Index' in this pandas version - group positions)."""
    ym = idx.year.values * 100 + idx.month.values
    first = pd.Series(np.arange(len(idx))).groupby(ym).min().values
    return idx[first]


def ann_vol(series):
    return float(series.std() * np.sqrt(TRADING_DAYS))


def metrics(pr):
    pr = pr.dropna()
    eq = (1 + pr).cumprod()
    yrs = len(pr) / TRADING_DAYS
    vol = ann_vol(pr) * 100
    return dict(
        total=float(eq.iloc[-1] - 1) * 100,
        cagr=((float(eq.iloc[-1]) ** (1 / yrs) - 1) * 100) if yrs > 0 else np.nan,
        vol=vol,
        sharpe=(float(pr.mean() / pr.std() * np.sqrt(TRADING_DAYS)) if pr.std() > 0 else np.nan),
        maxdd=float((eq / eq.cummax() - 1).min()) * 100,
        posyrs=f"{int((pr.groupby(pr.index.year).apply(lambda s: (1 + s).prod() - 1) > 0).sum())}"
               f"/{pr.groupby(pr.index.year).ngroups}",
    )


def run():
    p = panel()
    rets = p.pct_change().dropna()
    ma50 = p.rolling(50, min_periods=50).mean()
    long_sig = (p > ma50).shift(1)            # executed position, t+1
    w_ew = pd.DataFrame(1 / 3, index=p.index, columns=ASSETS)
    w_bh = w_ew.reindex(rets.index).fillna(1 / 3)
    w_ma = (long_sig / 3).reindex(rets.index).fillna(0.0)      # 1/3 per asset when long

    print("=" * 104)
    print("VOL-TARGET INCREMENTAL STUDY - does the overlay add anything beyond MA50?")
    print(f"panel {p.index[0].date()} .. {p.index[-1].date()}  ({len(rets)} return days)")
    print("=" * 104)

    pr_bh = (w_bh.shift(1) * rets).sum(axis=1).dropna()
    pr_ma = (w_ma.shift(1) * rets).sum(axis=1).dropna()

    # trailing vol of each base book, as known at time t (no look-ahead)
    vol_ma = (w_ma.shift(1) * rets).sum(axis=1).rolling(VOL_WIN, min_periods=VOL_WIN).std() * np.sqrt(TRADING_DAYS)
    vol_ew = (w_bh.shift(1) * rets).sum(axis=1).rolling(VOL_WIN, min_periods=VOL_WIN).std() * np.sqrt(TRADING_DAYS)

    def overlay(base_w, volbase, target):
        sc = pd.Series(1.0, index=base_w.index)
        for d in monthly_dates(base_w.index):
            v = volbase.asof(d)
            if pd.notna(v) and v > 0:
                sc.loc[d:] = min(1.0, target / float(v))
        return base_w.mul(sc, axis=0)

    rows, series = {}, {}
    series["buy&hold EW"] = pr_bh
    series["MA50 filter"] = pr_ma
    rows["buy&hold EW (no filter)"] = metrics(pr_bh)
    rows["MA50 filter only"] = metrics(pr_ma)

    for tgt in (0.30, 0.40, 0.50):
        wA = overlay(w_ma, vol_ma, tgt)                 # VT-A: vol from filtered book
        pA = (wA.shift(1) * rets).sum(axis=1).dropna()
        rows[f"MA50 + VT{int(tgt*100)} (vol=filtered book)"] = metrics(pA)
        series[f"MA50 + VT{int(tgt*100)}A"] = pA

    wB = overlay(w_ma, vol_ew, 0.40)                    # VT-B: vol from unfiltered EW
    pB = (wB.shift(1) * rets).sum(axis=1).dropna()
    rows["MA50 + VT40 (vol=unfiltered EW)"] = metrics(pB)
    series["MA50 + VT40B"] = pB

    wVTonly = overlay(w_bh, vol_ew, 0.40)               # overlay with NO filter, reference
    pVTonly = (wVTonly.shift(1) * rets).sum(axis=1).dropna()
    rows["VT40 only (no MA50 filter)"] = metrics(pVTonly)
    series["VT40 only"] = pVTonly

    print("\n--- Q1: does the overlay still help on top of MA50? ---")
    hdr = f"{'scheme':36} {'total%':>10} {'CAGR%':>7} {'vol%':>6} {'sharpe':>7} {'maxDD%':>8} {'posYr':>6}"
    print(hdr)
    print("-" * len(hdr))
    for k, m in rows.items():
        print(f"{k:36} {m['total']:>10.1f} {m['cagr']:>7.1f} {m['vol']:>6.1f} "
              f"{m['sharpe']:>7.2f} {m['maxdd']:>8.1f} {m['posyrs']:>6}")

    ma_sh = rows["MA50 filter only"]["sharpe"]
    ma_dd = rows["MA50 filter only"]["maxdd"]
    ma_tot = rows["MA50 filter only"]["total"]
    print(f"\n  basis MA50 filter: sharpe {ma_sh:.2f}, maxDD {ma_dd:.1f}%, total {ma_tot:.1f}%")
    worst_inc = None
    for k, m in rows.items():
        if k.startswith("MA50 + VT"):
            d_sh = m["sharpe"] - ma_sh
            print(f"  {k:36} delta sharpe {d_sh:+.2f} | delta maxDD {m['maxdd']-ma_dd:+.1f}pp "
                  f"| delta total {m['total']-ma_tot:+.1f}pp")
            if worst_inc is None or d_sh > worst_inc[1]:
                worst_inc = (k, d_sh)
    if worst_inc:
        print(f"  -> best incremental variant: {worst_inc[0]} ({worst_inc[1]:+.2f} sharpe)")

    print("\n--- Q2: episode-level attribution (total return per window) ---")
    ep = {}
    for label, (a, b) in EPISODES.items():
        row = {}
        for nm in ("buy&hold EW", "MA50 filter only", "MA50 + VT40A", "VT40 only"):
            src = pr_bh if nm == "buy&hold EW" else series.get(nm, pr_ma)
            sl = src.loc[a:b]
            row[nm] = float((1 + sl).prod() - 1) * 100 if len(sl) else np.nan
        ep[label] = row
    print(pd.DataFrame(ep).T.round(1).to_string())

    print("\n--- Q3: do the two mechanisms overlap? ---")
    n_long = w_ma.reindex(rets.index).fillna(0).gt(0).sum(axis=1)
    wA = overlay(w_ma, vol_ma, 0.40)
    grossA = wA.sum(axis=1)
    vt_on = grossA < 0.99
    ma_on = n_long < 3
    both = (vt_on & ma_on)
    print(f"  days with MA50 partially/fully out : {ma_on.mean()*100:5.1f}%")
    print(f"  days with vol overlay engaged      : {vt_on.mean()*100:5.1f}%")
    print(f"  days BOTH engaged (redundant)      : {both.mean()*100:5.1f}%")
    print(f"  vt engaged while MA50 fully long   : {(vt_on & ~ma_on).mean()*100:5.1f}%  <- the incremental part")
    print(f"  MA50 out while vt not engaged      : {(ma_on & ~vt_on).mean()*100:5.1f}%")

    print("\n--- Q4: current gross exposure each scheme would run (2026-09-11) ---")
    for nm, w in (("MA50 filter only", w_ma), ("MA50 + VT40A", wA),
                  ("MA50 + VT40B", wB), ("VT40 only", wVTonly)):
        last = w.iloc[-1]
        print(f"  {nm:22} gross {last.sum():.2f}  " +
              "  ".join(f"{a} {last[a]*100:4.1f}%" for a in ASSETS))


if __name__ == "__main__":
    run()
