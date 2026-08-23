# AGENTS.md

Agent guide for the crypto-micro-analysis repo. Read this before changing
anything. README.md is the user-facing overview; this is the operational contract.

## What this project is

Short-horizon (days-to-weeks) research on BTC/ETH/ZEC + their cross-pairs:
trend health, relative strength, bull-cycle phase. Companion to the sibling
repo `crypto-macro-analysis` - THERE the macro engine decides whether to be in
crypto at all; HERE we decide which asset to sit in and when a trend is broken.
Public, MIT.

## Golden rules (tested facts - do not re-litigate)

1. **The MA50 close-filter beats buy-and-hold on BTC, ETH and ZEC** over
   2017-11 .. today with much lower max drawdown. Reproduce:
   `strategies/micro_backtest.py`.
2. **An MA200 hard floor helps BTC/ETH quality (Sharpe, maxDD) but HURTS ZEC**
   - it locks ZEC out of post-breakdown recovery rallies (ZEC filtered total
   drops ~1458% -> ~597%). Do not apply the 200d floor to ZEC.
3. **RS rotation is a selector, not a return driver** - it beats holding BTC
   but loses to plain MA50 filtering on each asset. Use it to pick WHICH asset;
   the MA50 filter decides WHETHER to be in.
4. **The phase score cannot call tops in real time** (+1.67 at the Nov-2021
   top = FAIL) but confirms breakdowns within ~10-31 days and deliberately
   misses V-bottoms. It is a hold/exit overlay, never an entry timer.
   Reproduce: `strategies/phase_history.py`.
5. **Every claim needs A/B evidence** (same rule as the macro repo). No signal
   weights or thresholds change without a backtest proving improvement.
6. **STATELESS: results are never saved, never read back.** Every script
   regenerates output from the committed dataset and prints to stdout. Docs
   never hardcode result numbers except as historical backtest evidence tied
   to a reproduce command.
7. **Secrets live in `.env` (gitignored), never committed.** Current sources
   are keyless; `.env` exists only if keyed sources get added later.

## Repo map

| Path | Purpose |
|------|---------|
| `strategies/micro_regime.py` | LIVE micro engine: per-asset trend health (-3..+3), cross-pair RS (ratio momentum 20d/60d + 180d band position), cycle-phase score (BTC 180d drawdown x2.0, breadth x1.5, F&G contrarian x1.0). Consumes local data only. Stateless stdout. |
| `strategies/micro_backtest.py` | A/B evidence: B&H vs MA50 vs MA50+200floor per asset; RS rotation vs holding BTC. Shifted execution (signal at close t drives t->t+1 return). |
| `strategies/phase_history.py` | Validates phase score at the Dec-2017 / Nov-2021 tops and following bottoms (T1/T2/B1/B2 checks). NaN-safe scoring: components without full lookback are excluded from BOTH sums. |
| `strategies/build_micro_dataset.py` | Fetches all charts (Bitstamp daily OHLCV, Yahoo ZEC-USD full history, Bitstamp ZEC live cross-check, alternative.me F&G). Keyless. Slow network job - only re-run to refresh data. |
| `strategies/audit_micro.py` | Data-integrity + signal-correctness audit (schema, ZEC venue cross-check on traded days only, computed ETH/BTC vs venue ETHBTC, RSI reference-vs-engine). RUN BEFORE trusting any aggregation. |
| `scripts/refresh.sh` | On-demand refresh runner: fetch latest data -> engine -> audit. `--check` = status only. Never pulls/pushes/schedules/saves results. |
| `data/micro_dataset/` | Dataset CSVs + `manifest.json`. Crosses are never stored - always computed from USD closes on aligned dates. |

## Commands

```bash
cd ~/projects/crypto-micro-analysis
./.venv/bin/python strategies/micro_regime.py          # live verdict -> stdout
./.venv/bin/python strategies/micro_backtest.py        # A/B evidence -> stdout
./.venv/bin/python strategies/phase_history.py         # phase validation -> stdout
./.venv/bin/python strategies/audit_micro.py           # audit -> stdout (exit 1 on fail)
./.venv/bin/python strategies/build_micro_dataset.py   # refresh data (network)
bash scripts/refresh.sh                                # all of the above in order
bash scripts/refresh.sh --check                        # status only
```

- Use `./.venv/bin/python` directly; do NOT `source .venv/bin/activate`.
- No test suite; non-trivial new logic leaves one runnable check (assert-based
  self-check or small /tmp script).

## Backtest conventions (CRITICAL)

- Aligned 3-asset panel: dates where ALL of BTC/ETH/ZEC have closes (2017-11+).
- True daily closes, no resampling artifacts.
- Signal computed at close t drives the t->t+1 return (`shift(1)` execution).
- No cost model (daily flips are rare after filtering; note flip counts).
- Metrics: total return, CAGR, annualized vol/Sharpe (365d), max drawdown.
- Phase-score components without full lookback stay NaN and drop out of both
  numerator and denominator - they are NEVER coerced into risk-off votes.
  (This exact bug produced a fake -3.00 "risk-off" reading at the Dec-2017 top
  before it was fixed; do not reintroduce it.)

## Data pitfalls (learned the hard way)

- **Bitstamp delisted/relisted ZEC-USD**: its history starts 2026-03. Yahoo
  `ZEC-USD` v8 chart API is the ZEC primary source (full 2017-11+ history,
  keyless).
- **Bitstamp's ZEC book is near-dead**: zero-volume days freeze the last print
  (71/156 closes unchanged; once frozen at 237.64 for 15 straight days while
  Yahoo traded 570-610). The audit therefore compares venues ONLY on traded
  days (median divergence ~0.1%) and warns when the book goes stale.
- **Binance is geo-blocked (HTTP 451)** from this network; CryptoCompare free
  endpoint needs a paid key (401); CoinGecko free tier returns 401 here too.
  Do not bother retrying them for deep ZEC history - use Yahoo.
- **F&G starts 2018-06**: any pre-2018-06 phase computation runs on available
  signals only (exclusion rule above).
- Crosses MUST be computed from aligned USD closes, never fetched separately -
  different venues have different clocks and gaps.

## Relationship to crypto-macro-analysis

- Macro verdict = context input for sizing (PHASE 1 there supports full tactical
  exposure here; LIQUIDATE there overrides everything here). Reading it is
  manual/on-demand - no automated coupling, no cron, ever.
- Same statelessness rule, same A/B discipline, same audit-before-trust flow,
  same ASCII-only docs style (no en/em dashes).
