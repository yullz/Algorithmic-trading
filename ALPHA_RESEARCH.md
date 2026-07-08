# Alpha research — finding a real, validation-surviving edge

## Where things stand (measured, not assumed)

The current strategy has **no edge**, proven rigorously on 10,108 real trades:
- Every gate / setup-kind / regime / timeframe / side is negative out-of-sample.
- **0 of 105 factors** are FDR-significant; model ranking AUC **0.479** (≈ random).
- Cross-sectional **momentum** looked promising on a single split (Sharpe ~1.5)
  but **failed deflated-Sharpe (0.56)** after correcting for the configs tried,
  and showed a survivorship signature (long-leg 2:1 over short-leg). Rejected.
- Short-term **reversal**: uniformly negative. Rejected.

Classic TA on liquid perps is largely priced in. The scanner correctly shows
**"0 setups"** in a choppy/risk-off market — that is capital preservation, not a
bug. The system's intelligence is that it *knows* it has no edge and refuses to
bet; the robustness math (deflated-Sharpe / PBO / walk-forward) exists to catch
exactly the false leads a naive backtest would trade.

## The one direction with real odds: order-flow / derivatives alpha

Real perp edge tends to live in **funding rate, open interest, and basis
dynamics** — data the backtest pipeline never had (there is no historical
funding/OI series; it was live-only). So we collect it forward, starting now.

### Collect the data
```
# one snapshot:
python collect_orderflow.py
# recommended — leave running (hourly) via Task Scheduler / cron / a terminal:
python collect_orderflow.py --loop --interval 3600
```
Each run appends every Bybit linear symbol's funding rate, open interest (base +
USD), basis, mark/index, and 24h volume to `data_cache/orderflow/{date}.parquet`
(one API call, ~718 symbols, de-duped by timestamp). Verified live: 718 symbols,
610 with funding, per snapshot.

### What accumulates
`algotrader/data/orderflow.load_orderflow()` concatenates all snapshots into one
time series. After a few weeks you will have the raw material to build and test:
- **Funding z-score** (crowded long/short → mean-reversion of funding)
- **OI-vs-price divergence** (price up + OI up = real; price up + OI down = short
  cover, fades)
- **Basis** term-structure signals (backwardation/contango extremes)
- **Funding-carry** harvesting (a market-neutral, non-directional return)

### Test it honestly (the gauntlet that rejected the price edges)
When enough history exists, build features → join to forward returns → run the
**same** validation used in this repo:
1. Temporal train/test + nested walk-forward (`algotrader/backtest/selection.py`)
2. FDR correction across every signal tried (multiple-testing)
3. **Deflated Sharpe** + PBO (`algotrader/backtest/robustness.py`)
4. Long/short-leg decomposition (survivorship check)

Only a signal that clears all four earns the `require_validated_edge` gate and
gets wired as tradeable. Nothing skips the gauntlet — that discipline is the
whole point.

## Honest expectation
Funding/OI edges are real but **thin and capacity-limited**, and may not survive
costs either. This is genuine research with an uncertain payoff — not a switch to
flip. But it is the highest-probability path to a real edge, and every tool to
validate it honestly is already built.
