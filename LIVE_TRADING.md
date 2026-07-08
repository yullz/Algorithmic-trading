# Live Bybit Futures — wiring, sizing, and the go-live runbook

This document explains how the system auto-trades Bybit USDT perpetual futures,
how "open each position with 15 USDT" works, and the exact steps to go live.

---

## ⚠️ Read this first — the honest edge status

The Phase-5 robustness work measured this strategy on **10,108 real trades with
realistic fills** (next-bar-open entry, gap-through-stop slippage). The result:

| Metric | Value | Meaning |
|--------|-------|---------|
| Mean expectancy | **−0.071R** | negative edge per trade |
| Win rate | 49.7% | coin flip |
| Model ranking skill (OOS AUC) | **0.479** | worse than random |
| Reward-head rank skill (Spearman) | **0.037** | statistically zero |
| Both model heads | **untrusted** | can't find winners |

**Right now the system has no measurable edge, and the ML model cannot identify a
profitable subset of trades.** Auto-trading it on mainnet is, in expectation, a
slow bleed (negative R × fees × funding). This is exactly what the account
simulation, walk-forward, PBO, and deflated-Sharpe exist to tell you.

Because of this, a default-on **edge safety catch** (`risk.require_validated_edge:
true`) **blocks all live entries** until a positive out-of-sample edge is on
record. You can override it, but you would be trading a measured-negative edge.

**The path to a real edge is not more wiring — it is a better strategy/model
that clears the walk-forward.** Validate on testnet (mechanics) and paper (edge)
first.

---

## What "open each position with 15 USDT" does

- `risk.fixed_margin_usdt: 15.0` → every trade commits **15 USDT of margin**.
  At `default_leverage: 3`, that is a ~45 USDT position (`notional = margin ×
  leverage`). Set `fixed_margin_usdt: 0` to go back to %-of-equity sizing.
- Before each live entry, the Bybit executor checks **free USDT ≥ required
  margin** (`BybitExecutor.free_usdt()`), and refuses if not — so it only opens
  when 15 USDT is actually available.
- A new trade opens whenever free ≥ 15 **and** the safety caps still allow it:
  one position per symbol, `max_concurrent_positions` (6), the correlation guard,
  `max_total_margin_pct`, and `max_portfolio_risk_pct`. These prevent stacking
  many copies of the same bet. (You chose to keep these caps.)
- Trades are taken **best-first**: the scanner ranks by the reward-head E[R] when
  the model is trusted, else by `expected_value_r × confidence × market/breadth
  bias`, and de-duplicates correlated same-side setups.

## What the exchange does vs. what the app does (live)

- **Exchange** executes: the market **entry**, the attached **stop-loss**, and
  the reduce-only **take-profit ladder** — these are real orders on Bybit.
- **App loop** adds what the exchange won't: **move the stop to breakeven after
  TP1**, enforce the **time-stop**, sync live positions to the dashboard, and run
  the circuit breakers (kill switch, daily-loss, losing-streak). This is now
  wired for the live executor (previously only the paper executor was managed).

## The full safety stack (all must pass for a real order)

1. `ALLOW_LIVE_TRADING=true` in `.env`
2. `execution.mode: live` in `config.yaml`
3. `API_KEY` / `API_SECRET` in `.env`
4. Mainnet needs **both** `execution.testnet: false` **and**
   `execution.allow_mainnet: true` (double opt-in). Otherwise testnet is forced.
5. `LIVE_TRADING_CONFIRM=YES-I-UNDERSTAND` in `.env`
6. `risk.require_validated_edge: true` → a positive walk-forward OOS edge must be
   on record (currently it is **not**).
7. Per-entry: kill switch (`STOP_TRADING` file), daily-loss breaker, losing-streak
   breaker, portfolio caps, stale-candle fail-safe, free-margin preflight.

---

## Go-live runbook

### Step 0 — validate the edge (do not skip)
```
python backtest.py --deep --walkforward --export-dataset
python -m algotrader.ml.train
```
Look at `reports/walkforward.json` → `out_of_sample`. If `expectancy_r <= 0` or
`profit_factor <= 1`, **there is no edge to trade** and the edge catch will (and
should) block live entries. Improve the strategy/model until OOS is positive.

### Step 1 — TESTNET (validate the plumbing with fake money)
Create keys at **testnet.bybit.com**, then in `.env`:
```
ALLOW_LIVE_TRADING=true
API_KEY=<testnet key>
API_SECRET=<testnet secret>
LIVE_TRADING_CONFIRM=YES-I-UNDERSTAND
```
`config.yaml`:
```
execution:
  mode: live
  testnet: true          # keep true for testnet
  allow_mainnet: false
risk:
  fixed_margin_usdt: 15.0
  require_validated_edge: false   # only to exercise the plumbing on testnet
```
Run `python server/main.py`. Confirm on the dashboard that trades **open, move to
breakeven after TP1, respect the time-stop, and close** correctly, and that the
free-margin gate works. Watch it for a while.

### Step 2 — MAINNET (real money) — only after Steps 0–1 pass
On **bybit.com** create keys with **Derivatives/USDT-perp trade** permission and
**IP restriction** on. Ensure the account is in **one-way** position mode. Then:
```
# .env — same as above but with MAINNET keys
```
`config.yaml`:
```
execution:
  mode: live
  testnet: false
  allow_mainnet: true
risk:
  fixed_margin_usdt: 15.0
  require_validated_edge: true    # keep true; it only passes once OOS is positive
```
Start small (a few concurrent 15-USDT positions), watch the first fills by hand,
and keep the `STOP_TRADING` kill switch one `touch` away.

### Emergency stop
```
# from the repo root — halts ALL new entries immediately (paper and live)
touch STOP_TRADING       # (PowerShell: New-Item STOP_TRADING)
```
Delete the file to resume. Existing positions keep their exchange-side SL/TP.

---

## Known limitations (live)
- Live **closed-trade journaling** and the equity curve are not reconstructed in
  the dashboard; the **audit log** (`audit(...)`) is the live ledger. Open live
  positions do show on the dashboard.
- The **losing-streak breaker** increments from the app's own closes; exchange-side
  SL fills are not yet fed back into it. The **daily-loss breaker** (equity-based)
  and exchange SL remain the primary protections.
- Margin mode (isolated/cross) is left at the account default — set it on Bybit.
- Historical **funding is not modeled** in the backtest (a constant assumption,
  default 0), so funding P&L live is not pre-estimated.
