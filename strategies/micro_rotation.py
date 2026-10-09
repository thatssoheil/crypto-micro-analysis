#!/usr/bin/env python3
"""
MICRO ROTATION ENGINE - pair-chart rotation calls between an alt and BTC.

Owner request (2026-10-09): "a fast trade machine that tells me to convert
zec and eth into btc and vice versa via analyzing the pairbtc charts... on a
daily manner."

WHAT IT DOES
  For ETH/BTC and ZEC/BTC, computes a daily-close rotation state:
      hold ALT  when pair close > its own k-day MA AND alt close > its own m-day MA
      hold BTC  otherwise        (optional cash gate: BTC below its own 50d MA)
  Part 1 prints the A/B evidence for every variant vs its benchmarks WITH a
  cost model (the "fast" question answered with numbers, not vibes);
  Part 2 prints the live call: current state per variant, switch levels,
  distance to every trigger.

WHY PAIRS (standing rule): an alt's USD move = pair leg x BTC leg - the pair
IS the trade. Ratios are always computed from aligned committed closes, never
fetched (repo rule 7).

SPEED vs COST: the k grid (10/20/50) is the delay-vs-noise frontier. The
macro-repo study measured ~12d median delay for a 50d ratio rule and 5-6d
for 20d, with the fast variant cost-fragile; this script re-measures on both
pairs with a real per-flip cost instead of asserting it.

PRIOR (golden rule 2): the 200d MA floor HURTS ZEC as a direct filter - the
m=200 rows for ZEC are kept only as evidence; expect them to lose.

CONVENTIONS (house): daily closes, signal at close t drives the t->t+1
return (shift(1) execution), cost 0.4% per executed switch (two taker legs;
also charged for the to-cash leg - conservative), start $1, STATELESS stdout
only. Window = the repo's aligned common panel (2017-11-09+, ZEC binds).

SCOPE: this is the SELECTOR layer (which asset to sit in). Whether to be in
crypto at all is the macro repo's verdict; reading it stays manual (repos are
never auto-coupled).

Run: ./.venv/bin/python strategies/micro_rotation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from micro_backtest import load_panel  # noqa: E402

COST = 0.004          # per executed switch: sell one asset + buy the other
KS = (10, 20, 50)     # pair-MA speeds (delay vs noise frontier)
MS = (None, 50, 200)  # alt absolute gate (None = pair-only, the "mirage" row)
ALTS = ("eth", "zec")
WHIP_WIN = 30         # a switch reversed within this many days = whipsaw


# ------------------------------------------------------------------ engine --
def alt_on_series(panel, alt, k, m):
    """Daily target state: True = hold ALT, False = hold BTC."""
    ratio = panel[alt] / panel["btc"]
    rel = ratio > ratio.rolling(k).mean()
    if m is None:
        return rel.fillna(False)
    absg = panel[alt] > panel[alt].rolling(m).mean()
    return (rel & absg.fillna(False)).fillna(False)


def confirm(raw, n):
    """Causal n-day confirmation: a state change is kept only after n
    consecutive raw days agree on the new state (no lookahead)."""
    vals = raw.to_numpy(dtype=bool)
    out = np.empty(len(vals), dtype=bool)
    cur = run_val = vals[0]
    run = 1
    out[0] = cur
    for i in range(1, len(vals)):
        v = vals[i]
        if v == run_val:
            run += 1
        else:
            run_val = v
            run = 1
        if v != cur and run >= n:
            cur = v
        out[i] = cur
    return pd.Series(out, index=raw.index)


def simulate(panel, alt, k, m, cash_gate=False, conf=1):
    """Rotation sim on close-to-close returns with shift(1) execution.

    Weights decided at close t are held over (t, t+1]; a switch executed on
    day u pays COST on day u. cash_gate: when the state is BTC but BTC is
    below its 50d MA, hold cash instead. conf: keep a state change only after
    n consecutive raw closes agree.
    """
    on = alt_on_series(panel, alt, k, m)
    if conf > 1:
        on = confirm(on, conf)
    w_alt = on.astype(float)
    w_btc = (~on).astype(float)
    if cash_gate:
        ma50 = panel["btc"].rolling(50).mean()
        btc_ok = (panel["btc"] > ma50) | ma50.isna()
        w_btc = w_btc * btc_ok.astype(float)
    W = pd.DataFrame({"alt": w_alt, "btc": w_btc}, index=panel.index)
    We = W.shift(1).fillna(0.0)
    rets = panel.pct_change()
    pr = We["alt"] * rets[alt].fillna(0.0) + We["btc"] * rets["btc"].fillna(0.0)
    sw = We.diff().abs().sum(axis=1) > 1e-12
    eq_net = (1.0 + pr - COST * sw.astype(float)).cumprod()
    eq_gross = (1.0 + pr).cumprod()

    yrs = max((panel.index[-1] - panel.index[0]).days / 365.25, 1e-9)
    sw_dates = list(sw[sw].index)
    wh = sum(1 for i, t in enumerate(sw_dates[:-1])
             if (sw_dates[i + 1] - t).days <= WHIP_WIN)
    dd = (eq_net / eq_net.cummax() - 1.0).min()
    cut = panel.index[-1] - pd.Timedelta(days=365)
    return {
        "eq": eq_net, "total": eq_net.iloc[-1] - 1.0,
        "total_gross": eq_gross.iloc[-1] - 1.0, "dd": float(dd),
        "flips": len(sw_dates), "flips_1y": sum(1 for t in sw_dates if t >= cut),
        "whip": wh / max(len(sw_dates), 1),
        "alt_share": float(We["alt"].sum()) / len(We),
        "yrs": yrs, "sw_dates": sw_dates,
        "state_last": bool(on.iloc[-1]),
    }


def bhold(panel, col):
    eq = (1.0 + panel[col].pct_change().fillna(0.0)).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {"eq": eq, "total": eq.iloc[-1] - 1.0, "dd": dd,
            "flips": 0, "flips_1y": 0, "whip": 0.0, "alt_share": 0.0,
            "yrs": max((panel.index[-1] - panel.index[0]).days / 365.25, 1e-9)}


def mline(label, r, star=""):
    cagr = ((1.0 + r["total"]) ** (1.0 / r["yrs"]) - 1.0) * 100
    fy = r["flips"] / r["yrs"]
    print(f"  {label:22s} net {r['total']*100:>11,.0f}%  gross {r['total_gross']*100 if 'total_gross' in r else r['total']*100:>11,.0f}%  "
          f"cagr {cagr:>6.1f}%  maxDD {r['dd']*100:>6.1f}%  flips {r['flips']:>3d} "
          f"({fy:>4.1f}/yr, {r['flips_1y']} last12m)  whip {r['whip']*100:>3.0f}%  alt {r['alt_share']*100:>3.0f}%{star}")


# ----------------------------------------------------------------- yearly ---
def yearly_returns(eq):
    ends = eq.resample("YE").last()
    vals = [1.0] + list(ends.values)
    return {int(y): vals[i + 1] / vals[i] - 1.0 for i, y in enumerate(ends.index.year)}


def main():
    panel = load_panel()
    span = f"{panel.index[0].date()} .. {panel.index[-1].date()} ({len(panel)} days)"
    print("=" * 100)
    print(f"MICRO ROTATION ENGINE - pair-driven ALT<->BTC toggles, aligned panel {span}")
    print(f"cost {COST*100:.1f}% per switch, shift(1) execution, $1 start | ratios from aligned closes")
    print("=" * 100)

    for alt in ALTS:
        ratio = panel[alt] / panel["btc"]
        a = alt.upper()
        print(f"\n### {a}/BTC  (current {ratio.iloc[-1]:.5f})")
        print("-" * 100)
        btc_r = bhold(panel, "btc")
        alt_r = bhold(panel, alt)
        # BTC MA50 filter benchmark (the repo incumbent for "whether to be in")
        ma50 = panel["btc"].rolling(50).mean()
        posf = (panel["btc"] > ma50).astype(float)
        pf = posf.shift(1).fillna(0.0)
        rb = panel["btc"].pct_change().fillna(0.0)
        swf = pf.diff().abs().fillna(0.0) > 1e-12
        eqf = (1.0 + pf * rb - COST * swf.astype(float)).cumprod()
        btc_filter = {"eq": eqf, "total": eqf.iloc[-1] - 1.0,
                      "total_gross": ((1.0 + pf * rb).cumprod()).iloc[-1] - 1.0,
                      "dd": float((eqf / eqf.cummax() - 1.0).min()),
                      "flips": int(swf.sum()), "flips_1y": int(swf[swf.index >= panel.index[-1] - pd.Timedelta(days=365)].sum()),
                      "whip": 0.0, "alt_share": 0.0,
                      "yrs": max((panel.index[-1] - panel.index[0]).days / 365.25, 1e-9)}
        # whipsaw for the filter: recompute properly
        swfd = list(swf[swf].index)
        btc_filter["whip"] = sum(1 for i, t in enumerate(swfd[:-1])
                                 if (swfd[i + 1] - t).days <= WHIP_WIN) / max(len(swfd), 1)
        mline(f"{a} buy&hold", alt_r)
        mline("BTC buy&hold", btc_r)
        mline("BTC MA50 filter", btc_filter)
        print("  " + "-" * 96)
        results = {}
        for k in KS:
            for m in MS:
                r = simulate(panel, alt, k, m)
                results[(k, m, False)] = r
                tag = " !" if (alt == "zec" and m == 200) else ""
                mline(f"k{k} m{m if m else 'none'}", r, star=tag)
            r = simulate(panel, alt, k, 50, cash_gate=True)
            results[(k, 50, True)] = r
            mline(f"k{k} m50 +cash", r)
            r = simulate(panel, alt, k, 50, conf=3)
            results[(k, 50, "conf3")] = r
            mline(f"k{k} m50 +conf3", r)
            if k == 50:
                for cf in (2, 5):
                    r = simulate(panel, alt, k, 50, conf=cf)
                    mline(f"k{k} m50 +conf{cf}", r)
            print()
        print(f"  ! m=200 on ZEC kept as evidence only (golden rule 2: the 200d floor hurts ZEC)")

        # ---- speed frontier (m=50 fixed)
        print(f"\n  SPEED FRONTIER ({a}, m=50 fixed):")
        for k in KS:
            r = results[(k, 50, False)]
            print(f"    k={k:<3d} net {r['total']*100:>11,.0f}%  flips {r['flips']:>3d} "
                  f"({r['flips']/r['yrs']:>4.1f}/yr)  whip {r['whip']*100:>3.0f}%  "
                  f"maxDD {r['dd']*100:>6.1f}%")

        # ---- yearly table for the slow (k50 m50) and fast (k20 m50) toggles
        yr_rows = {"BTC hold": btc_r["eq"], "BTC MA50flt": btc_filter["eq"],
                   f"{a} hold": alt_r["eq"],
                   "k50 m50": results[(50, 50, False)]["eq"],
                   "k50 conf3": results[(50, 50, "conf3")]["eq"],
                   "k20 m50": results[(20, 50, False)]["eq"]}
        yrs_all = yearly_returns(btc_r["eq"]).keys()
        print(f"\n  YEARLY RETURNS % (net of costs):")
        hdr = "    " + "".join(f"{y:>7d}" for y in yrs_all)
        print(hdr)
        for lab, eq in yr_rows.items():
            yr = yearly_returns(eq)
            print(f"    {lab:<10s}" + "".join(f"{yr[y]*100:>7.0f}" for y in yrs_all))

    # ------------------------------------------------------------- readout --
    print("\n" + "=" * 100)
    print(f"MACHINE CALL - as of {panel.index[-1].date()} close")
    print("=" * 100)
    for alt in ALTS:
        a = alt.upper()
        ratio = panel[alt] / panel["btc"]
        rr = ratio.tail(750)
        pctile = float(rr.rank(pct=True).iloc[-1]) * 100
        d30 = float(ratio.pct_change(30).iloc[-1]) * 100
        print(f"\n{a}/BTC {ratio.iloc[-1]:.5f}   ({a} {panel[alt].iloc[-1]:,.2f} / BTC {panel['btc'].iloc[-1]:,.0f})   "
              f"30d {d30:+.1f}%   percentile of ~3y: {pctile:.0f}%")
        ma50b = panel["btc"].rolling(50).mean().iloc[-1]
        btc_gap = (panel["btc"].iloc[-1] / ma50b - 1.0) * 100
        print(f"  BTC context: {panel['btc'].iloc[-1]:,.0f} vs MA50 {ma50b:,.0f} ({btc_gap:+.1f}%) "
              f"[cash gate {'ON (in BTC only)' if btc_gap > 0 else 'OFF (would sit cash when in BTC state)'}]")
        for k in KS:
            on = alt_on_series(panel, alt, k, 50)
            st = bool(on.iloc[-1])
            ma_k = ratio.rolling(k).mean().iloc[-1]
            gap_k = (ratio.iloc[-1] / ma_k - 1.0) * 100
            ma_m = panel[alt].rolling(50).mean().iloc[-1]
            gap_m = (panel[alt].iloc[-1] / ma_m - 1.0) * 100
            state = f"HOLD {a}" if st else "HOLD BTC"
            print(f"  k{k:<3d} {state:<9s} | pair {gap_k:+6.2f}% vs {k}dMA ({ma_k:.6f})  "
                  f"| {a} {gap_m:+6.2f}% vs 50dMA ({ma_m:,.0f})")
        # flip levels for the k=50 machine + last flip
        on50 = alt_on_series(panel, alt, 50, 50)
        ma_k = float(ratio.rolling(50).mean().iloc[-1])
        ma_m = float(panel[alt].rolling(50).mean().iloc[-1])
        if bool(on50.iloc[-1]):
            print(f"  k50 flip: rotate {a}->BTC if pair closes < {ma_k:.6f} or {a} closes < {ma_m:,.0f}")
        else:
            print(f"  k50 flip: rotate BTC->{a} if pair closes > {ma_k:.6f} AND {a} closes > {ma_m:,.0f}")
        on50c = bool(confirm(on50, 3).iloc[-1])
        print(f"  k50 +3d-confirm state: {'HOLD ' + a if on50c else 'HOLD BTC'} "
              f"(a flip needs 3 consecutive closes past the line)")
        r50 = simulate(panel, alt, 50, 50)
        if r50["sw_dates"]:
            last_sw = r50["sw_dates"][-1]
            pair_then = ratio.loc[:last_sw].iloc[-1]
            print(f"  k50 last switch {last_sw.date()} | pair then {pair_then:.5f} -> now {ratio.iloc[-1]:.5f} "
                  f"({(ratio.iloc[-1]/pair_then-1)*100:+.1f}%) | flips last 12m: {r50['flips_1y']}")


if __name__ == "__main__":
    main()
