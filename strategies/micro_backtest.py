#!/usr/bin/env python3
"""
MICRO BACKTEST - A/B evidence for every micro-engine claim (repo golden rule).

Questions answered on real data (common history of all 3 assets -> today):
  Q1 Does an MA50 trend filter beat buy-and-hold on each asset?
  Q2 Does adding an MA200 hard floor (macro-repo convention) help or hurt?
  Q3 Does cross-pair RS rotation (hold the strongest asset, cash when none
     hold their MA50) beat just holding BTC?

Conventions mirror crypto-macro-analysis: true daily closes, signal at close t
drives the t->t+1 return (shifted execution, no look-ahead), no cost model,
STATELESS - prints to stdout, saves nothing.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data" / "micro_dataset"


def load_panel():
    """Aligned close-price panel: columns btc, eth, zec on shared dates."""
    cols = {}
    for key, name in [("btcusd_daily", "btc"), ("ethusd_daily", "eth"),
                      ("zecusd_yahoo", "zec")]:
        df = pd.read_csv(DATA / f"{key}.csv")
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        cols[name] = df.set_index("ts")["close"].sort_index()
    return pd.DataFrame(cols).dropna()


def sim(pos, ret):
    """Equity curve from a position series decided at the prior close.
    Returns (equity, number of executed position changes)."""
    p = pos.shift(1).fillna(0.0)
    eq = (1.0 + p * ret.fillna(0.0)).cumprod()
    flips = int((p.diff().fillna(0.0).abs() > 0).sum())
    return eq, flips


def sim_weights(W, rets):
    """Same as sim() but W is a per-asset weight DataFrame (rotation)."""
    We = W.shift(1).fillna(0.0)
    pr = (We * rets.fillna(0.0)).sum(axis=1)
    eq = (1.0 + pr).cumprod()
    switches = int((W.diff().fillna(0.0).abs().sum(axis=1) > 0).sum())
    return eq, switches


def report(eq, flips, label):
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] - 1.0
    cagr = eq.iloc[-1] ** (1.0 / yrs) - 1.0
    r = eq.pct_change().dropna()
    vol = r.std() * np.sqrt(365) if len(r) > 2 else 0.0
    sharpe = (r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1.0).min()
    print(f"{label:22s} total {total*100:>11.1f}%  cagr {cagr*100:>6.1f}%  "
          f"vol {vol*100:>5.1f}%  sharpe {sharpe:>4.2f}  maxDD {dd*100:>6.1f}%  "
          f"changes {flips}")
    return {"label": label, "total": total, "cagr": cagr, "maxdd": dd,
            "sharpe": sharpe}


# ---------------------------------------------------------------- Q1 + Q2 --
def per_asset_trend(panel, ma=50, floor=None):
    """MA filter on one asset: in when close > MA, out otherwise.
    floor: optional MA whose loss forces flat even if above the fast MA."""
    sigs = {}
    for c in panel.columns:
        px = panel[c]
        s = (px > px.rolling(ma).mean()).astype(float)
        if floor:
            hard = px > px.rolling(floor).mean()
            s = (s.astype(bool) & hard).astype(float)
        sigs[c] = s
    return pd.DataFrame(sigs)


# --------------------------------------------------------------------- Q3 --
def rs_rotation(panel, fast=20, slow=60):
    """Hold the single strongest asset by RS ratio momentum; cash if none
    hold their MA50. Strength = asset's own 20d/60d MA of price (a proxy for
    ratio-vs-everything momentum that needs no pairwise explosion)."""
    px = panel
    strength = px.rolling(fast).mean() / px.rolling(slow).mean() - 1.0
    healthy = px > px.rolling(50).mean()

    W = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    st = strength.iloc[:, :]
    for i in range(len(px)):
        row = st.iloc[i]
        if row.isna().all():
            continue
        best = row.idxmax()
        if healthy[best].iloc[i] and row[best] == row.max():
            W.loc[W.index[i], best] = 1.0
    return W


def main():
    panel = load_panel()
    rets = panel.pct_change()
    span = f"{panel.index[0].date()} .. {panel.index[-1].date()} ({len(panel)} days)"
    print("=" * 78)
    print(f"MICRO BACKTEST - aligned 3-asset panel {span}")
    print("=" * 78)

    print("\nQ1/Q2 - trend filters per asset (A/B vs buy-and-hold)")
    print("-" * 78)
    for c in panel.columns:
        bh_eq = (1.0 + rets[c].fillna(0)).cumprod()
        report(bh_eq, 0, f"{c.upper()} buy&hold")
        eq, fl = sim(per_asset_trend(panel[[c]], 50)[c], rets[c])
        report(eq, fl, f"{c.upper()} MA50")
        eq2, fl2 = sim(per_asset_trend(panel[[c]], 50, floor=200)[c], rets[c])
        report(eq2, fl2, f"{c.upper()} MA50+200floor")
        print()

    print("Q3 - cross-pair RS rotation vs holding BTC")
    print("-" * 78)
    btc_bh = (1.0 + rets["btc"].fillna(0)).cumprod()
    report(btc_bh, 0, "BTC buy&hold")
    W = rs_rotation(panel)
    eq3, sw = sim_weights(W, rets)
    report(eq3, sw, "RS rotation (1 asset)")

    # exposure stats
    expo = W.shift(1).fillna(0).sum(axis=1)
    print(f"\nrotation exposure: mean {expo.mean()*100:.0f}% of days invested, "
          f"cash {(expo==0).mean()*100:.0f}% of days")


if __name__ == "__main__":
    main()
