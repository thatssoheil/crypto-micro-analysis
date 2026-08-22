# crypto-micro-analysis

Short-horizon investment research on **BTC / ETH / ZEC** and their cross-pairs
(ETH/BTC, ZEC/BTC): trend health, relative strength, and where we are in the
bull cycle. Companion to [crypto-macro-analysis](https://github.com/thatssoheil/crypto-macro-analysis)
- there the macro engine decides **whether** to be in crypto at all (HOLD/CASH);
here the micro engine decides **which asset to sit in** and **when trends are
intact or broken**, on a days-to-weeks horizon.

> **For agents/AI:** read [`AGENTS.md`](AGENTS.md) first - golden rules,
> backtest conventions, engine math, data pitfalls (operational contract).

**The core validated edges** (reproduce with `strategies/micro_backtest.py`,
2017-11 .. today):
- An **MA50 close-filter beats buy-and-hold on all three assets** with much
  lower drawdowns (BTC +2622% vs +977%; ETH +2887% vs +651%; ZEC +1458% vs +203%).
- Adding an **MA200 hard floor** cuts max drawdown further and lifts Sharpe on
  BTC/ETH - but costs ZEC half its filtered return (ZEC's recovery rallies come
  from below its 200d MA).
- **Cross-pair RS rotation** picks the right asset more often than holding BTC,
  but is a selector, not a return driver.

**Stateless by design:** code + fetched data only. Every run regenerates results
from the dataset and prints to stdout - nothing saved, nothing read back.

## Quickstart

```bash
git clone https://github.com/thatssoheil/crypto-micro-analysis
cd crypto-micro-analysis
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. Refresh the dataset (keyless sources only)
./.venv/bin/python strategies/build_micro_dataset.py

# 2. Read the current micro verdict (no network needed)
./.venv/bin/python strategies/micro_regime.py

# 3. Audit data integrity + signal correctness
./.venv/bin/python strategies/audit_micro.py

# 4. Reproduce every claim
./.venv/bin/python strategies/micro_backtest.py     # A/B evidence
./.venv/bin/python strategies/phase_history.py      # phase score vs real tops/bottoms
```

Or everything at once: `bash scripts/refresh.sh` (status only: `--check`).

## The dataset (data/micro_dataset/)

| Chart | Source | Span |
|-------|--------|------|
| `btcusd_daily.csv` | Bitstamp v2 ohlc | 2011+ |
| `ethusd_daily.csv` | Bitstamp v2 ohlc | 2017-08+ |
| `zecusd_yahoo.csv` | Yahoo Finance v8 chart (ZEC-USD) | 2017-11+ |
| `zecusd_bitstamp_live.csv` | Bitstamp v2 ohlc (cross-check only) | 2026-03+ |
| `fear_greed.csv` | alternative.me | 2018-06+ |

Crosses are **computed** as USD-close ratios on aligned dates - never fetched -
so every series shares one source clock. Sources, endpoints and known gaps:
[`SOURCES.md`](SOURCES.md).

## The micro engine (strategies/micro_regime.py)

Per asset - trend health (-3..+3): close vs MA50, MA50 5-day slope, RSI(14)
overbought/oversold trim bands.

Cross-pair RS: ETH/BTC, ZEC/BTC, ZEC/ETH ratio momentum (20d vs 60d MA) plus
position in its own 180d range -> who is leading, who is lagging.

Cycle phase (-3..+3), three weighted components:
| Component | Weight |
|-----------|--------|
| BTC drawdown vs trailing 180d high | 2.0 |
| Breadth: how many of the 3 assets hold their MA50 | 1.5 |
| Fear & Greed contrarian bands (<25 buy zone / >75 distribution risk) | 1.0 |

Verdict bands: >= +1.5 PHASE 1 bull expansion; +0.5..1.5 early bull;
<= -1.5 phase 2 distribution; <= -0.5 late cycle; else transition.
Unavailable components are excluded from both numerator and denominator
(same rule as the macro engine).

## What the validator says the engine can and cannot do

`strategies/phase_history.py` checks the phase score against the Dec-2017 and
Nov-2021 tops and the bear bottoms after them:

- It does NOT call tops in real time (read +1.67 at the Nov-2021 top - FAIL).
- It DOES confirm breakdowns fast (exit zone 10d after the 2021 top, 31d after 2017).
- It intentionally misses V-bottoms (stays risk-off until structure repairs).

So: use the phase score as a **hold/exit overlay**, never as an entry timer,
and let the macro repo own the cash decision.

## License

MIT - see [LICENSE](LICENSE).
