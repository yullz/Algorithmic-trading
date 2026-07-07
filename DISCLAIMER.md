# ⚠️ Risk Disclaimer — read before using

This software is an **educational decision-support tool**, not financial advice
and not a profit guarantee.

### What it does
- Detects a defined library of technical **indicators and chart patterns**.
- Scores **confluence** (how many independent factors agree).
- Produces a fully-specified futures trade plan: entry, stop-loss, take-profits,
  position size, **liquidation price**, R:R, fees, and an **estimated** win rate.

### What it does NOT do
- It does **not** "read all possible patterns." No system can. It reads the
  finite, explicitly-coded set in `algotrader/patterns/`.
- It does **not** predict the future. The "expected win rate" is a **historical
  base rate** measured by the backtester on past data. Past performance does not
  predict future results. An uncalibrated estimate is only a rough prior.
- It does **not** guarantee "safe" signals or "good returns." There is no such
  thing as a safe leveraged trade.

### Leverage reality check
At **3–5x leverage**, a move of roughly **(20% – 33%) / leverage against you can
liquidate the position**. A 3x long is liquidated by only a ~30%+ adverse move;
a 5x long by only ~20%. The tool prints the exact liquidation price for every
plan — respect it. Only risk money you can afford to lose entirely.

### Safe operating order
1. `python backtest.py`  → calibrate win rates on history, inspect the stats.
2. `python paper_trade.py` → run for weeks against live data with fake money.
3. Only then, if ever, consider live — and `ALLOW_LIVE_TRADING` is `false` by
   default and must be deliberately enabled.

Nothing here has been reviewed by a licensed financial professional. You are
solely responsible for any trades you place.
