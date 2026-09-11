#!/usr/bin/env python3
"""
SIZING STUDY - does volatility-aware sizing beat fixed weights?

Pre-registered BEFORE looking at results (do not re-fit after):
  Q1 does inverse-vol weighting beat equal notional on risk-adjusted return?
  Q2 does risk parity (ERC, correlation-aware) beat inverse-vol?
  Q3 does a long-only vol target (de-risk toward cash when realized vol exceeds
     target) cut drawdown enough to justify its return cost?
  Q4 how does the owner's actual fixed book (59.8/34.5/5.7) compare - is its
     concentration actually a problem, or already near risk-parity?

Protocol (mirrors the repo contract):
  - aligned daily closes, BTC/ETH/ZEC, panel start = first common date
  - weights decided at the close of day t from data UP TO t, applied to t+1
    returns via shift(1) - no look-ahead
  - monthly rebalance (first trading day of the month)
  - long-only, weights sum to 1; the unfunded part is CASH earning 0%
  - VOL_WIN=60 pre-registered default; 30 and 90 reported only as robustness
  - no leverage (spot book): vol targeting can only REDUCE exposure
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data" / "micro_dataset"
ASSETS = ["BTC", "ETH", "ZEC"]
VOL_WIN = 60
BOOK = np.array([0.598, 0.345, 0.057])   # owner's actual book, 2026-09-11
TRADING_DAYS = 365


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


def erc_weights(cov, iters=800):
    """Equal-risk-contribution weights (multiplicative fixed point)."""
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):
        mrc = cov @ w
        rc = w * mrc
        w = w * (rc.mean() / np.maximum(rc, 1e-18)) ** 0.5
        s = w.sum()
        if s <= 0:
            return np.ones(n) / n
        w = w / s
    return w


def schedule(rets, weight_fn):
    """Monthly-rebalanced weight matrix; weights from data through the rb close."""
    rb = rets.groupby([rets.index.year, rets.index.month]).head(1).index
    w = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
    for d in rb:
        w.loc[d:] = weight_fn(rets.loc[:d])
    return w.ffill().dropna()


def w_equal(rets):
    return np.ones(3) / 3


def w_invvol(rets, win=VOL_WIN):
    sd = rets.tail(win).std().values
    if not np.all(np.isfinite(sd)) or sd.sum() == 0:
        return np.ones(3) / 3
    iv = 1.0 / np.maximum(sd, 1e-9)
    return iv / iv.sum()


def w_erc(rets, win=90):
    if len(rets) < win:
        return np.ones(3) / 3
    cov = rets.tail(win).cov().values
    if not np.all(np.isfinite(cov)):
        return np.ones(3) / 3
    return erc_weights(cov)


def w_book(rets):
    return BOOK.copy()


def vt_scaler(rets, base_fn, win=VOL_WIN, target=0.40):
    """Long-only vol target: scale the base book so trailing vol <= target."""
    w0 = base_fn(rets)
    sd = rets.tail(win).std().values
    if not np.all(np.isfinite(sd)):
        return 1.0
    realized = float(np.sqrt(w0 @ np.diag(sd ** 2) @ w0) * np.sqrt(TRADING_DAYS))
    if realized <= 0:
        return 1.0
    return float(min(1.0, target / realized))


def metrics(pr, w):
    pr = pr.dropna()
    eq = (1 + pr).cumprod()
    yrs = len(pr) / TRADING_DAYS
    total = float(eq.iloc[-1] - 1) * 100
    cagr = ((eq.iloc[-1]) ** (1 / yrs) - 1) * 100 if yrs > 0 else np.nan
    vol = float(pr.std() * np.sqrt(TRADING_DAYS)) * 100
    sharpe = float(pr.mean() / pr.std() * np.sqrt(TRADING_DAYS)) if pr.std() > 0 else np.nan
    dd = float((eq / eq.cummax() - 1).min()) * 100
    yr = pr.groupby(pr.index.year).apply(lambda s: float((1 + s).prod() - 1) * 100)
    return dict(total=total, cagr=cagr, vol=vol, sharpe=sharpe, maxdd=dd,
                worst_year=float(yr.min()), pos_years=f"{int((yr > 0).sum())}/{len(yr)}")


def run():
    p = panel()
    rets = p.pct_change().dropna()
    print("=" * 100)
    print("SIZING STUDY - vol-aware vs fixed weights (monthly rebalance, shifted execution)")
    print(f"panel {p.index[0].date()} .. {p.index[-1].date()}  ({len(rets)} return days)")
    print("=" * 100)

    schemes = {
        "equal notional (1/3)": lambda r: schedule(rets, w_equal),
        f"inverse-vol ({VOL_WIN}d)": lambda r: schedule(rets, lambda x: w_invvol(x)),
        "risk parity ERC (90d)": lambda r: schedule(rets, lambda x: w_erc(x)),
        "OWNER book 59.8/34.5/5.7": lambda r: schedule(rets, w_book),
    }
    # vol-target overlays: scale the base weights by a trailing-vol cap
    for nm, base in (("equal notional (1/3)", w_equal), ("risk parity ERC (90d)", lambda x: w_erc(x))):
        def mk(base=base, nm=nm):
            def f(r):
                w = schedule(rets, base)
                sc = pd.Series(index=w.index, dtype=float)
                rb = rets.groupby([rets.index.year, rets.index.month]).head(1).index
                for d in rb:
                    sc.loc[d] = vt_scaler(rets.loc[:d], base)
                return w.mul(sc.ffill().fillna(1.0), axis=0)
            return f
        schemes[f"VT40 + {nm}"] = mk()

    rows = {}
    for name, fn in schemes.items():
        w = fn(rets)
        pr = (w.shift(1) * rets).sum(axis=1).dropna()
        m = metrics(pr, w)
        # turnover: mean half-sum of |dw| per rebalance
        dw = w.diff().abs().sum(axis=1) / 2
        rb = w.index[w.ne(w.shift()).any(axis=1)]
        m["turnover_pm"] = float(dw.loc[rb].mean()) * 100 if len(rb) else 0.0
        m["gross_now"] = float(w.iloc[-1].sum())
        gross = w.sum(axis=1)
        m["frac_derisk"] = float((gross < 0.99).mean()) * 100
        rows[name] = m

    print("\n--- headline (pre-registered: VOL_WIN=60, target 40%, long-only) ---")
    hdr = f"{'scheme':28} {'total%':>10} {'CAGR%':>7} {'vol%':>6} {'sharpe':>7} {'maxDD%':>8} {'worstYr%':>9} {'posYr':>6} {'turn%':>6} {'deRisk%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for k, m in rows.items():
        print(f"{k:28} {m['total']:>10.1f} {m['cagr']:>7.1f} {m['vol']:>6.1f} "
              f"{m['sharpe']:>7.2f} {m['maxdd']:>8.1f} {m['worst_year']:>9.1f} "
              f"{m['pos_years']:>6} {m['turnover_pm']:>6.1f} {m['frac_derisk']:>8.1f}")

    print("\n--- robustness: inverse-vol window (30 / 60 / 90) ---")
    for win in (30, 60, 90):
        w = schedule(rets, lambda x, win=win: w_invvol(x, win))
        pr = (w.shift(1) * rets).sum(axis=1).dropna()
        m = metrics(pr, w)
        print(f"  inv-vol {win:>2}d   total {m['total']:>9.1f}%  vol {m['vol']:>5.1f}%  "
              f"sharpe {m['sharpe']:.2f}  maxDD {m['maxdd']:>6.1f}%")

    print("\n--- robustness: vol target level (30 / 40 / 50), on ERC base ---")
    for tgt in (0.30, 0.40, 0.50):
        w = schedule(rets, lambda x: w_erc(x))
        sc = pd.Series(index=w.index, dtype=float)
        rb = rets.groupby([rets.index.year, rets.index.month]).head(1).index
        for d in rb:
            sc.loc[d] = vt_scaler(rets.loc[:d], lambda x: w_erc(x), target=tgt)
        w = w.mul(sc.ffill().fillna(1.0), axis=0)
        pr = (w.shift(1) * rets).sum(axis=1).dropna()
        m = metrics(pr, w)
        print(f"  VT{int(tgt*100)} on ERC   total {m['total']:>9.1f}%  vol {m['vol']:>5.1f}%  "
              f"sharpe {m['sharpe']:.2f}  maxDD {m['maxdd']:>6.1f}%  gross now {float(w.iloc[-1].sum()):.2f}")

    print("\n--- what each scheme says to hold TODAY (2026-09-11 book = $1,272.63) ---")
    tot = 1272.63
    for name, fn in schemes.items():
        w = fn(rets)
        last = w.iloc[-1]
        book = sorted(((a, float(last[a])) for a in ASSETS), key=lambda x: -x[1])
        print(f"  {name:28} " + "  ".join(f"{a} {v*100:4.1f}% (${v*tot:>7.2f})" for a, v in book))

    print("\n--- per-year returns, main comparison ---")
    yr = {}
    for name in ("equal notional (1/3)", f"inverse-vol ({VOL_WIN}d)",
                 "risk parity ERC (90d)", "OWNER book 59.8/34.5/5.7", "VT40 + risk parity ERC (90d)"):
        w = schemes[name](rets)
        pr = (w.shift(1) * rets).sum(axis=1).dropna()
        yr[name] = pr.groupby(pr.index.year).apply(lambda s: float((1 + s).prod() - 1) * 100)
    yt = pd.DataFrame(yr).round(1)
    print(yt.to_string())


if __name__ == "__main__":
    run()
