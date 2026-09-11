#!/usr/bin/env python3
"""
FILTER DELAY STUDY - how much detection delay is available on the SAME daily data?

Question: the validated edge is the MA50 close filter (AGENTS.md rule 1), but any
moving average is lagging by construction. Before reaching for new data sources
(funding, options), measure what a better trend estimator buys on the existing
dataset - and what it costs in false alarms. Delay vs false alarms is a trade-off
curve, not a free lunch, so every candidate is reported on both axes.

Filters compared (all long/flat, all trailing-only, shifted execution via sim()):
  ma50      close > MA50                      (the incumbent baseline)
  ema50     close > EMA50
  kama      close > Kaufman adaptive MA (ER 10, fast 2, slow 30)
  hma       close > Hull MA(50)               (built to be low-lag)
  kalman    level+slope Kalman filter, long while slope > 0
  cusum     two-sided CUSUM change-point on log returns, expanding-window z

Metrics per asset x filter: the standard repo metrics (total, CAGR, vol, Sharpe,
maxDD, changes) PLUS the two that answer the question:
  delay_off / delay_on  - median trading days from a major TOP/BOTTOM to the
                          filter actually flipping (lower = less lag)
  whipsaw               - share of position changes reversed within 30 days
                          (lower = fewer false alarms)

Events reuse the alert study's definition so the numbers are comparable:
  TOP = 90d high then -25% within 90d; BOTTOM = 90d low then +30% within 90d.

STATELESS: prints to stdout, writes nothing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from micro_backtest import load_panel, sim, report          # noqa: E402
from micro_alerts import find_events                        # noqa: E402

WHIP_WIN = 30   # a change reversed within this many days counts as a whipsaw


# ------------------------------------------------------------------ filters --
def ema50(s):
    return s.ewm(span=50, adjust=False).mean()


def kama(s, er=10, fast=2, slow=30):
    change = (s - s.shift(er)).abs()
    vol = s.diff().abs().rolling(er).sum()
    er_ = (change / vol).fillna(0.0)
    sc = (er_ * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    out = np.empty(len(s))
    out[0] = s.iloc[0]
    vals = s.to_numpy()
    scv = sc.to_numpy()
    for i in range(1, len(s)):
        out[i] = out[i - 1] + scv[i] * (vals[i] - out[i - 1])
    return pd.Series(out, index=s.index)


def _wma(s, n):
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)


def hma(s, n=50):
    return _wma(2 * _wma(s, n // 2) - _wma(s, n), int(np.sqrt(n)))


def kalman_slope(s, q=1e-4, r=1e-2):
    """2-state (level, slope) Kalman filter, written with explicit scalars.
    Returns the slope series; long while slope > 0."""
    x0, x1 = float(s.iloc[0]), 0.0
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0
    out = np.empty(len(s))
    for i, z in enumerate(s.to_numpy()):
        # predict: x = F x, P = F P F^T + Q  with F = [[1, 1], [0, 1]]
        x0 = x0 + x1
        p00, p01, p10, p11 = (p00 + p01 + p10 + p11 + q, p01 + p11,
                              p10 + p11, p11 + q)
        # update
        gain_denom = p00 + r
        k0, k1 = p00 / gain_denom, p10 / gain_denom
        y = z - x0
        x0 = x0 + k0 * y
        x1 = x1 + k1 * y
        p00, p01, p10, p11 = ((1 - k0) * p00, (1 - k0) * p01,
                              p10 - k1 * p00, p11 - k1 * p01)
        out[i] = x1
    return pd.Series(out, index=s.index)


def cusum_pos(s, k=0.5, h=5.0):
    """Two-sided CUSUM on expanding-window standardised log returns.
    Expanding stats keep it trailing-only (no full-sample peek)."""
    lr = np.log(s / s.shift(1))
    mu = lr.expanding(min_periods=30).mean()
    sd = lr.expanding(min_periods=30).std()
    z = ((lr - mu) / sd).to_numpy()
    state, sp, sn = 1, 0.0, 0.0
    out = np.ones(len(z))
    for i, v in enumerate(z):
        if not np.isfinite(v):
            out[i] = state
            continue
        sp = max(0.0, sp + v - k)
        sn = min(0.0, sn + v + k)
        if sp > h:
            state, sp, sn = 1, 0.0, 0.0
        elif sn < -h:
            state, sp, sn = 0, 0.0, 0.0
        out[i] = state
    return pd.Series(out, index=s.index)


FILTERS = {
    "ma50":   lambda s: (s > s.rolling(50).mean()).astype(float),
    "ema50":  lambda s: (s > ema50(s)).astype(float),
    "kama":   lambda s: (s > kama(s)).astype(float),
    "hma":    lambda s: (s > hma(s)).astype(float),
    "kalman": lambda s: (kalman_slope(s) > 0).astype(float),
    "cusum":  cusum_pos,
}


# ------------------------------------------------------------------- metrics --
def delay_stats(pos, tops, bots):
    """Median trading days from a major turn to the filter actually flipping.

    Only events where the filter HELD the position are counted - if it was already
    flat at the top (or already long at the bottom) there is no flip to be late
    for, and counting it as 0 days would understate every filter's lag.
    Returns (delay_off, n_off, delay_on, n_on, n_tops, n_bots).
    """
    p = pos.astype(int).to_numpy()
    off, on = [], []
    for t in tops:
        if p[t] != 1:
            continue
        nz = np.flatnonzero(p[t:] == 0)
        if len(nz):
            off.append(int(nz[0]))
    for b in bots:
        if p[b] != 0:
            continue
        nz = np.flatnonzero(p[b:] == 1)
        if len(nz):
            on.append(int(nz[0]))
    med = lambda a: float(np.median(a)) if a else float("nan")   # noqa: E731
    return med(off), len(off), med(on), len(on), len(tops), len(bots)


def whipsaw_rate(pos, win=WHIP_WIN):
    p = pos.astype(int).to_numpy()
    ch = np.flatnonzero(np.diff(p) != 0)
    if len(ch) < 2:
        return 0.0, len(ch)
    rev = sum(1 for i in ch if np.any((ch > i) & (ch <= i + win)))
    return rev / len(ch), len(ch)


def cusum_sweep(panel, rets):
    """CUSUM delay/false-alarm frontier across a (k, h) grid.

    One arbitrary (k, h) point proves nothing - the whole point of CUSUM is the
    delay-vs-false-alarm trade-off, so show the curve and where MA50 sits on it.
    """
    print("CUSUM PARAMETER SWEEP (mean across assets) - delay vs noise frontier")
    print(f"  {'k':>5} {'h':>4} {'total':>10} {'sharpe':>7} {'maxdd':>7} "
          f"{'changes':>8} {'whipsaw':>8} {'delay_off':>10}")
    out = []
    for k in (0.25, 0.5, 1.0, 1.5):
        for h in (2.0, 3.0, 5.0, 8.0):
            agg = {"total": [], "sharpe": [], "maxdd": [], "changes": [],
                   "whipsaw": [], "delay_off": []}
            for asset in ("btc", "eth", "zec"):
                close = panel[asset]
                pos = cusum_pos(close, k=k, h=h).fillna(0.0)
                eq, flips = sim(pos, rets[asset])
                yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
                r = eq.pct_change().dropna()
                sh = (r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0
                tops, bots = find_events(close)
                d_off, n_off, _, _, _, _ = delay_stats(pos, tops, bots)
                wr, _ = whipsaw_rate(pos)
                agg["total"].append(eq.iloc[-1] - 1)
                agg["sharpe"].append(sh)
                agg["maxdd"].append((eq / eq.cummax() - 1).min())
                agg["changes"].append(flips)
                agg["whipsaw"].append(wr)
                if np.isfinite(d_off):
                    agg["delay_off"].append(d_off)
            m = lambda key: float(np.mean(agg[key])) if agg[key] else float("nan")   # noqa: E731
            print(f"  {k:>5.2f} {h:>4.0f} {m('total')*100:>9.0f}% {m('sharpe'):>7.2f} "
                  f"{m('maxdd')*100:>6.0f}% {m('changes'):>8.0f} {m('whipsaw')*100:>7.1f}% "
                  f"{m('delay_off'):>9.1f}d")
            out.append({"k": k, "h": h, **{key: m(key) for key in agg}})
    print()
    return out


def main():
    panel = load_panel()
    rets = panel.pct_change()
    rows = []

    print("FILTER DELAY STUDY - same daily data, different trend estimators")
    print(f"panel {panel.index[0].date()} .. {panel.index[-1].date()} "
          f"({len(panel)} rows, aligned BTC/ETH/ZEC)")
    print("whipsaw = share of changes reversed within "
          f"{WHIP_WIN}d; delay = median days from a major turn to the flip\n")

    for asset in ("btc", "eth", "zec"):
        close = panel[asset]
        r = rets[asset]
        tops, bots = find_events(close)
        print(f"===== {asset.upper()}  (events: {len(tops)} tops, {len(bots)} bottoms)")
        for name, fn in FILTERS.items():
            pos = fn(close).fillna(0.0)
            eq, flips = sim(pos, r)
            stats = report(eq, flips, f"  {name}")
            d_off, n_off, d_on, n_on, _, _ = delay_stats(pos, tops, bots)
            wr, _ = whipsaw_rate(pos)
            print(f"  {'':22s} delay_off {d_off:>5.1f}d (n={n_off:<3d}) "
                  f"delay_on {d_on:>5.1f}d (n={n_on:<3d}) whipsaw {wr*100:>5.1f}%")
            rows.append({"asset": asset, "filter": name, "whipsaw": wr,
                         "delay_off": d_off, "delay_on": d_on, **stats})
        print()

    df = pd.DataFrame(rows)
    print("SUMMARY - mean across the three assets")
    agg = df.groupby("filter").agg(
        total=("total", "mean"), maxdd=("maxdd", "mean"), sharpe=("sharpe", "mean"),
        changes=("label", "size"), whipsaw=("whipsaw", "mean"),
        delay_off=("delay_off", "mean"), delay_on=("delay_on", "mean"))
    print(agg.to_string(float_format=lambda v: f"{v:.3f}"))




def eq_over(pos, ret, start):
    """Equity from `start` onward; the position series is computed on the FULL
    history so filter state carries in, but returns accumulate out-of-sample only.
    No look-ahead: every filter value at t uses data up to t only."""
    p = pos.shift(1).fillna(0.0)
    ts = pd.Timestamp(start, tz=pos.index.tz) if getattr(pos.index, 'tz', None) else pd.Timestamp(start)
    m = pos.index >= ts
    return (1.0 + p[m] * ret[m].fillna(0.0)).cumprod()


def stat(eq):
    r = eq.pct_change().dropna()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    return {"total": eq.iloc[-1] - 1,
            "cagr": eq.iloc[-1] ** (1 / yrs) - 1,
            "sharpe": (r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0,
            "maxdd": (eq / eq.cummax() - 1).min()}


def oos_test(panel, split="2022-01-01"):
    """Select (k,h) in-sample, then judge it on unseen data against MA50.

    A single grid point already flagged CUSUM(0.25, 2) as best - but it was the best
    of a 16-cell grid, so only out-of-sample performance settles whether that was
    signal or selection bias."""
    rets = panel.pct_change()
    grid = [(k, h) for k in (0.25, 0.5, 1.0, 1.5) for h in (2.0, 3.0, 5.0, 8.0)]
    print("CUSUM OUT-OF-SAMPLE TEST  (in-sample .. %s | out-of-sample %s ..)" % (split, split))
    print("grid searched: %d (k,h) cells - selection bias is real, that is the point" % len(grid))
    print()

    scores = {}
    for k, h in grid:
        shs = []
        for asset in ("btc", "eth", "zec"):
            close = panel[asset]
            pos = cusum_pos(close, k=k, h=h).fillna(0.0)
            eq, _ = sim(pos.loc[:split], rets[asset].loc[:split])
            shs.append(stat(eq)["sharpe"])
        scores[(k, h)] = float(np.mean(shs))
    best = max(scores, key=scores.get)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
    print("  in-sample ranking (mean sharpe): " + "  ".join(
        "k=%.2f,h=%.0f:%.2f" % (k, h, v) for (k, h), v in ranked))
    print("  selected: k=%.2f, h=%.0f" % (best[0], best[1]))
    print()

    print("  %-5s %-10s %10s %8s %7s %8s" % ("asset", "rule", "total", "cagr", "sharpe", "maxDD"))
    verdict = []
    for asset in ("btc", "eth", "zec"):
        close, ret = panel[asset], rets[asset]
        cand = cusum_pos(close, k=best[0], h=best[1]).fillna(0.0)
        base = (close > close.rolling(50).mean()).astype(float)
        for name, pos in (("cusum", cand), ("ma50", base)):
            st = stat(eq_over(pos, ret, split))
            print("  %-5s %-10s %9.0f%% %7.1f%% %7.2f %7.1f%%" % (
                asset, name, st['total'] * 100, st['cagr'] * 100, st['sharpe'], st['maxdd'] * 100))
            verdict.append({"asset": asset, "rule": name, **st})
    v = pd.DataFrame(verdict)
    piv = v.pivot(index="asset", columns="rule", values="sharpe")
    wins = int((piv["cusum"] > piv["ma50"]).sum())
    print()
    print("  OOS sharpe: cusum beats ma50 on %d/3 assets" % wins)
    print("  VERDICT: " + ("candidate survives - still needs walk-forward before adoption"
                           if wins >= 3 else
                           "does NOT survive out-of-sample - reject as selection bias"))

if __name__ == "__main__":
    if "--cusum-sweep" in sys.argv:
        _panel = load_panel()
        cusum_sweep(_panel, _panel.pct_change())
    elif "--oos" in sys.argv:
        oos_test(load_panel())
    else:
        main()
