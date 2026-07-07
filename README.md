# AlgoTrader — top-150 crypto scanner, confluence + ML signal engine, risk manager

An **honest**, paper-first decision-support and trading system for 3–5x
leveraged crypto futures. It scans the **top 150 Bybit USDT-perpetuals**,
detects a large coded library of indicators and chart patterns, scores
**confluence across independent evidence families**, gates by **market
regime**, blends an **ML meta-model** trained on backtest outcomes, and emits
fully-specified trade plans: entry, stop-loss, take-profit ladder, position
size, **liquidation price**, reward:risk, fees, and an **empirically-calibrated**
win-rate estimate with expected value. A React dashboard shows everything live.

> Read [DISCLAIMER.md](DISCLAIMER.md) first. This is not financial advice, does
> not predict the future, and does not guarantee profit. Leverage can liquidate
> you on a small adverse move.

## What it is (and isn't)

| Claim | Reality in this tool |
|-------|----------------------|
| "Reads all chart patterns" | Reads the finite, coded set in `algotrader/patterns/` (25+ candlesticks & chart patterns, S/R levels, liquidity sweeps, order blocks, FVGs) plus 30+ indicators, divergences, volume profile, funding/OI. Extendable. |
| "Smartest algorithm / godmode" | Breadth + honest calibration, not magic. Every estimate is a **historical base rate** measured by the backtester, conditioned on regime/timeframe, optionally blended with a walk-forward-validated ML meta-model. Hard-capped ≤ 78%. |
| "Picks the best trades" | Ranks by `EV × confidence × market-bias`, then prunes at the **portfolio** level (max positions, margin cap, correlation guard — ten correlated alt longs are one BTC bet). |
| "Safe signals" | No trade is safe. Strict sizing, stop-before-liquidation leverage ceiling, circuit breakers (daily loss, losing streak, kill switch), testnet-first live path. |

## Architecture

```
algotrader/
  models.py            # shared dataclasses (Side, Evidence, PatternMatch, TradePlan…)
  config.py            # YAML + .env loader, hard live-trading guard
  regime.py            # trend/range/volatile classifier + setup gating
  data/
    feed.py            # sync DataFeed + AsyncDataFeed (parquet cache, pagination)
    universe.py        # top-150 Bybit perps by 24h volume (cached, TTL)
    derivatives.py     # funding rate + open interest evidence
  indicators/          # 30+ indicators, divergences, volume profile
  patterns/            # candlesticks + chart patterns + liquidity concepts
  signals/             # confluence (family-based) + engine (orchestration)
  winrate.py           # calibrated, capped win-rate estimation + ML blend
  ml/                  # meta-model: features, walk-forward training, inference
  risk/manager.py      # sizing, leverage, liquidation, structure-aware TP ladder
  execution/           # Executor ABC, PaperExecutor (durable), BybitExecutor (guarded)
  scanner.py           # async universe scan -> ranked, portfolio-pruned picks
  backtest/engine.py   # event-driven backtest -> calibration + ML dataset
server/main.py         # FastAPI + WebSocket backend, background scan loop
web/                   # React + Tailwind dashboard (Vite; served from web/dist)
run_scan.py            # one-shot scan -> trade plans (CLI)
backtest.py            # measure win rates, write calibration.json (+ --export-dataset)
paper_trade.py         # continuous simulated trading, zero real orders
```

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 1) Prove it runs with zero network / zero keys:
python run_scan.py --offline

# 2) Measure real win rates on history, calibrate, and export the ML dataset:
python backtest.py --deep --limit 5000 --export-dataset --walkforward
python -m algotrader.ml.train

# 3) Scan the whole universe live (public OHLCV, no API keys needed):
python run_scan.py

# 4) Paper trade for weeks before you ever consider live:
python paper_trade.py

# 5) Dashboard (http://127.0.0.1:8777):
cd web && npm install && npm run build && cd ..
python server/main.py
```

## How a signal is built

1. **Indicators/patterns/liquidity/divergences** each emit `Evidence` with a
   bias, live strength, an explicit correlation **family**, and a base win rate.
2. **Confluence** nets bullish vs bearish strength (neutral evidence dampens),
   requiring agreement across `min_families` *independent* families.
3. **Regime gate** rejects setup kinds that historically fail in the current
   regime (e.g. mean-reversion longs inside a trending decline).
4. **Win rate** pools calibrated per-factor rates (regime/timeframe-conditioned
   keys first), then blends the ML meta-model in log-odds space — capped 30–78%.
5. **Risk** sizes to a fixed % of equity, picks stop-before-liquidation
   leverage, prefers pattern invalidation levels for stops, caps the TP ladder
   at measured-move targets, subtracts fees, rejects EV ≤ 0.
6. **Scanner** ranks survivors and prunes to a portfolio (positions/margin/
   correlation caps).

## Live trading (read this twice)

Live orders require ALL of: `ALLOW_LIVE_TRADING=true` (.env) → `execution.mode:
live` (config) → API keys → **testnet by default** (mainnet needs
`execution.testnet: false` AND `execution.allow_mainnet: true`) → the typed
confirmation phrase → per-entry circuit breakers (kill-switch file
`STOP_TRADING`, daily-loss halt, losing-streak halt, stale-data guard).
Paper and live share the same code path, caps, and breakers.

## Extending

- Add a pattern: return `PatternMatch` (with anchors + family) from a detector
  in `algotrader/patterns/` — it flows into confluence, calibration, and the ML
  features automatically.
- Add an indicator: add a compute + an `Evidence` rule (with family) in
  `algotrader/indicators/indicators.py`.
- Re-run `backtest.py --export-dataset` and `python -m algotrader.ml.train`
  after any change.
