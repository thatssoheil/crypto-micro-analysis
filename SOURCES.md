# SOURCES.md

Every data source, endpoint, span, and known gap in `data/micro_dataset/`.
Read before adding a chart. Conventions follow the sibling macro repo:
one CSV per chart, `ts,open,high,low,close,volume` for OHLCV or
`date,value[,classification]` otherwise; every fetch re-pulls full history
(charts are small), so gaps self-heal on refresh.

## Price charts

| Chart | Source | Endpoint | Span | Notes |
|-------|--------|----------|------|-------|
| btcusd_daily | Bitstamp v2 ohlc | `https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=86400&limit=1000` | 2011-08+ | Pagination: `start` is a FROM-filter; walk back with `start = oldest - 1000*86400`. |
| ethusd_daily | Bitstamp v2 ohlc | same pattern, `ethusd` | 2017-08+ | |
| zecusd_yahoo | Yahoo Finance v8 chart | `https://query1.finance.yahoo.com/v8/finance/chart/ZEC-USD?period1=1461360000&period2=now&interval=1d` | 2017-11-09+ | **ZEC primary.** Bitstamp delisted/relisted ZEC-USD (its ZEC history starts 2026-03 only). Drops no-trade days (None rows). |
| zecusd_bitstamp_live | Bitstamp v2 ohlc | same pattern, `zecusd` | 2026-03-19+ | **Cross-check only - never feed the engine from this chart.** |

## Sentiment

| Chart | Source | Endpoint | Span | Notes |
|-------|--------|----------|------|-------|
| fear_greed | alternative.me | `https://api.alternative.me/fng/?limit=3000&format=json` | 2018-06-05+ | Rows newest-first from API; saved sorted ascending. |

## Computed series (never stored)

- **ETH/BTC, ZEC/BTC, ZEC/ETH**: ratio of USD closes on ALIGNED dates.
  Never fetch cross pairs separately - venues have different clocks/gaps.
  Audit check A3 compares the computed ETH/BTC against Bitstamp's own
  ETHBTC market (<3% tolerance).

## Known gaps & pitfalls (do not rediscover these)

1. **Bitstamp ZEC book is near-dead.** Zero-volume days freeze the last print
   (71 of first 156 days had unchanged close; May 2026 froze at 237.64 for
   ~2 weeks while Yahoo printed 570-610). Venue comparison must filter to
   volume>0 days (audit does; median divergence there ~0.1%).
2. **Binance: HTTP 451** geo-block from this network. CryptoCompare free
   endpoint now returns 401 (needs key). CoinGecko free tier returns 401 here
   too. Kraken's public OHLC **ignores `since`** (returns the newest 721 daily
   bars whatever you ask for) and lists no ZEC/BTC pair, so Kraken is a live
   cross-check only, never a history source. Dead ends for deep ZEC history,
   all verified: Poloniex chart API returns 410 Gone (legacy returnChartData
   403), Coinbase ZEC-USD has only ~350 bars from 2025-09, Binance's static
   archive (data.binance.vision, reachable despite the 451 API block) starts
   2018-01 for ZECBTC.
3. **F&G starts 2018-06-05.** Phase math before that runs on available signals
   only (exclusion rule - see AGENTS.md backtest conventions).
4. Yahoo occasionally emits None rows (no-trade days); the builder drops them,
   which is why yahoo row counts can be smaller than calendar day counts.
5. All timestamps UTC. Daily candles close 00:00 UTC (Bitstamp) / market-time
   aggregates (Yahoo).

6. **Bitfinex tZECUSD / tZECBTC (probed 2026-09) - longest keyless ZEC
   history, but NOT a primary.** It reaches 2016-10-28 (Zcash launch, +376 days
   vs Yahoo) with 0 gaps, 0 duplicates, 0 zero-volume days on both series, and
   tBTCUSD/tETHUSD reach 2014-05/2016-03. But it **disagrees with the
   consensus**: Kraken and Yahoo match each other to ~0.1%, while Bitfinex runs
   one-sided BELOW both - median divergence 0.10-0.36% over 2019-2024 with
   essentially no days >5% apart, but 18.9% of overlapping days >5% apart in
   Nov-Dec 2017 and 15.3% in 2025 (max 31%). It has real volume on those days,
   so this is a thin book lagging fast moves, NOT the frozen-print failure seen
   on Bitstamp. Its own internal check agrees: ZECBTC x BTCUSD vs its ZECUSD is
   within 2.2% in 2018-2024 but 50% on the Oct-2016 launch days and 7.5% in
   2025. Kept as the audit's A5 diagnostic (WARN-only, never fails the run).
   Note the extra history is largely unusable anyway: extending ZEC alone gains
   only 85 panel days (ETH/Bitstamp binds at 2017-08-16), using it all means
   re-sourcing BTC and ETH to Bitfinex too, and it lands on the 2017 mania where
   Bitfinex is weakest. When fetching: candle field order is
   `[mts, OPEN, CLOSE, HIGH, LOW, volume]` - close is index 2, not 4, and
   `limit=10000&sort=1` returns the whole series in one request.

## Adding a new chart

1. Add a fetcher function in `build_micro_dataset.py` following the existing
   retry/backoff + save_csv pattern; register it in `main()` and manifest.
2. Extend `audit_micro.py` schema checks (A1 list) with the new chart.
3. Document endpoint/span/pitfalls here.
4. If it feeds an engine signal: A/B backtest FIRST (AGENTS.md golden rule 5),
   audit recomputation second, engine wiring last.
