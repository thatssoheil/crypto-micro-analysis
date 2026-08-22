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
   too. Keyless deep-history options for ZEC = Yahoo only.
3. **F&G starts 2018-06-05.** Phase math before that runs on available signals
   only (exclusion rule - see AGENTS.md backtest conventions).
4. Yahoo occasionally emits None rows (no-trade days); the builder drops them,
   which is why yahoo row counts can be smaller than calendar day counts.
5. All timestamps UTC. Daily candles close 00:00 UTC (Bitstamp) / market-time
   aggregates (Yahoo).

## Adding a new chart

1. Add a fetcher function in `build_micro_dataset.py` following the existing
   retry/backoff + save_csv pattern; register it in `main()` and manifest.
2. Extend `audit_micro.py` schema checks (A1 list) with the new chart.
3. Document endpoint/span/pitfalls here.
4. If it feeds an engine signal: A/B backtest FIRST (AGENTS.md golden rule 5),
   audit recomputation second, engine wiring last.
