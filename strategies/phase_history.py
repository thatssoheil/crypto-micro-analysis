#!/usr/bin/env python3
"""
PHASE HISTORY VALIDATOR - checks the cycle-phase score against the two known
cycle tops (Dec 2017, Nov 2021) and the bear bottoms that followed.

A phase engine cannot call the exact day. What it MUST do to be useful:
  T1 at a cycle top the score must NOT read full risk-on (>= +1.5);
  T2 within 45 days AFTER the top it must drop below -0.5 (exit zone);
  B1 at a bear bottom it must already be recovering (>= -0.5);
  B2 within 45 days after the bottom it must cross back above +0.5.
F&G starts 2018-06, so the 2017 top is scored on available signals only
(macro-repo rule: unavailable series are excluded from both sums).

STATELESS: consumes local dataset, prints to stdout, saves nothing.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data" / "micro_dataset"


def load_panel():
    cols = {}
    for key, name in [("btcusd_daily", "btc"), ("ethusd_daily", "eth"),
                      ("zecusd_yahoo", "zec")]:
        df = pd.read_csv(DATA / f"{key}.csv")
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        cols[name] = df.set_index("ts")["close"].sort_index()
    return pd.DataFrame(cols).dropna()


def load_fng():
    df = pd.read_csv(DATA / "fear_greed.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    s = df.set_index("date")["value"].astype(float).sort_index()
    return s


def phase_series(panel, fng):
    """Daily phase score, computed only from info available at each close.
    NaN-safe: a component without its full lookback window stays NaN and is
    EXCLUDED from both numerator and denominator (macro-repo rule) - it is
    never silently coerced into a risk-off vote."""
    btc = panel["btc"]

    # component 1: BTC drawdown vs trailing 180d high (weight 2.0)
    dd = btc / btc.rolling(180, min_periods=180).max() - 1.0
    s_dd = pd.Series(np.select([dd > -0.10, dd > -0.25], [1.0, 0.0],
                               default=np.nan), index=btc.index)

    # component 2: breadth - assets above their own MA50 (weight 1.5)
    # only counted once every asset HAS an MA50 (else NaN)
    ma50 = panel.rolling(50, min_periods=50).mean()
    above = ((panel > ma50) & ma50.notna()).sum(axis=1)
    have_ma = ma50.notna().all(axis=1)
    s_br = pd.Series(np.where(have_ma,
                              above.map({3: 1.0, 2: 0.0, 1: -1.0, 0: -1.0}),
                              np.nan), index=panel.index)

    # component 3: F&G contrarian (weight 1.0, only where it exists)
    fg = fng.reindex(fng.index.union(panel.index)).ffill().reindex(panel.index)
    s_fg = pd.Series(np.where(fg >= 75, -1.0,
                              np.where((fg.notna()) & (fg <= 25), 1.0, 0.0)),
                     index=panel.index)

    comps = [(s_dd, 2.0), (s_br, 1.5), (s_fg, 1.0)]
    num = pd.Series(0.0, index=panel.index)
    den = pd.Series(0.0, index=panel.index)
    for s, w in comps:
        ok = s.notna()
        num += (s.fillna(0.0) * w).where(ok, 0.0)
        den += pd.Series(w, index=panel.index).where(ok, 0.0)
    score = (num / den.replace(0.0, np.nan) * 3.0).round(2)
    return score


def extreme_date(close, start, end, kind="max"):
    w = close.loc[start:end]
    return (w.idxmax() if kind == "max" else w.idxmin())


def check(score, panel, label, top_start, top_end, bot_start, bot_end):
    btc = panel["btc"]
    top = extreme_date(btc, top_start, top_end, "max")
    bot = extreme_date(btc, bot_start, bot_end, "min")
    out = [f"\n== {label} =="]

    sc_top = float(score.asof(top))
    ok_t1 = sc_top < 1.5
    out.append(f"TOP {top.date()}  BTC ${btc.asof(top):,.0f}   score {sc_top:+.2f} "
               f"-> T1 {'PASS' if ok_t1 else 'FAIL'} (must not be >= +1.5)")

    after = score.loc[top:top + pd.Timedelta(days=45)]
    exit_day = after[after < -0.5].index.min()
    if exit_day is not None:
        lag = (exit_day - top).days
        out.append(f"     exit-zone (< -0.5) reached {lag}d after top "
                   f"-> T2 PASS")
    else:
        out.append("     never reached exit zone within 45d -> T2 FAIL")

    sc_bot = float(score.asof(bot))
    ok_b1 = sc_bot >= -0.5
    out.append(f"BOTTOM {bot.date()}  BTC ${btc.asof(bot):,.0f}  score {sc_bot:+.2f} "
               f"-> B1 {'PASS' if ok_b1 else 'FAIL'}")

    rec = score.loc[bot:bot + pd.Timedelta(days=45)]
    rec_day = rec[rec > 0.5].index.min()
    if rec_day is not None and not pd.isna(rec_day):
        out.append(f"     risk-on (> +0.5) regained {(rec_day-bot).days}d after bottom"
                   f" -> B2 PASS")
    else:
        out.append("     no risk-on signal within 45d of bottom -> B2 FAIL")
    print("\n".join(out))


def main():
    panel = load_panel()
    score = phase_series(panel, load_fng())
    cur = score.iloc[-1]
    print("=" * 64)
    print(f"PHASE VALIDATOR - {panel.index[0].date()} .. {panel.index[-1].date()}")
    print("=" * 64)
    check(score, panel, "2017-2018 cycle",
          "2017-10-01", "2018-01-31", "2018-11-01", "2019-01-31")
    check(score, panel, "2021-2022 cycle",
          "2021-09-01", "2021-12-31", "2022-10-01", "2023-01-31")
    print(f"\ncurrent score: {cur:+.2f} ({score.index[-1].date()})")


if __name__ == "__main__":
    main()
