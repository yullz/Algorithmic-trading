# AlgoTrader → "Godmode" Upgrade Plan

**Status:** Approved (correctness-first · real-money-eventual · Bybit streaming + full order-flow · full 21st.dev redesign).
**Audit method:** 10 parallel subsystem readers + 1 principal-quant synthesis pass (Opus), cross-checked by hand on the core files.
**Date:** 2026-07-07

---

## Progress log

- **✅ Phase 0 — DONE** (`2d3ba4d`). C1–C5 fixed + verified; tests 106→117; walk-forward runs
  end-to-end; JSON reports valid; git + CI initialized.
- **✅ Phase 1 core — DONE** (`0e7fb8f`, `23fdc31`). The calibration circularity is broken:
  - **P1a** OOS walk-forward is now the source of truth for `calibration.json` (in-sample saved only as
    a diagnostic), factors gated on their OOS Wilson-lower bound.
  - **P1a** recency weighting switched from a per-symbol positional index to **calendar time** (fixes the
    pooled-across-symbols corruption of Kelly sizing + EV).
  - **P1b** one `label_horizon_candles` knob drives both the backtest horizon and the live time-stop, so
    the ML label matches what is actually traded.
  - **P1c** removed the relative-strength/cross-asset **train/serve skew** (was live-only, uncalibrated).
  - **P1d** ML rewritten: **purged/embargoed walk-forward CV**, **isotonic calibration**, **class_weight
    dropped**, trust earned from OOS AUC **and** Brier skill. Verified end-to-end on a 110-col real
    dataset (calibrated=True, trusted=True, loads through `MetaModel.load`). Tests → 119.
  - **Deferred (with better homes):** full cross-asset/BTC wiring through both paths → **Phase 2** (needs
    BTC streaming context); `confidence` reliability plot → **Phase 4**; cluster-adjusted effective-N
    statistics → **Phase 5** (with PBO / deflated-Sharpe robustness package).
- **🟡 Phase 2 in progress** (`7525b69`, `387808b`).
  - **P2e** indicator correctness fixes: RSI clean-rally→100 (not neutral 50); regime thresholds driven by
    stationary `atr_percentile`; TTM squeeze uses tighter 1.5× Keltner.
  - **P2a** continuous indicator VALUES fed to the ML model (`indicators.numeric_context`): ~23
    normalized `ind_*` features (distance-to-MA in ATRs, MACD-hist/ATR, BB width, cloud position, bounded
    oscillators) + the previously-unpopulated percentiles. Dataset 110→135 cols. Tests → 123.
  - **P2c** two-head EV selector (`66116bd`): E[R] reward regressor (purged-WF Spearman rank-skill gate),
    `Signal.ml_ev_r`, and `scanner._rank_of` now ranks by learned **per-trade E[R] × bias** instead of the
    global-constant `EV·confidence·bias` heuristic. First Scanner tests added. Tests → 127.
  - **P2d (breadth)** (`0303175`): `indicators/breadth.py` — universe risk-on/off (% above EMA50/EMA200,
    adv/decline) surfaced in the scan output and used as a bounded ±10% ranking tilt. Deliberately a
    selection-time tilt, NOT per-symbol evidence (avoids re-introducing the P1c skew). Tests → 131.
  - **P2f (fresh-break gating)** (`ac823b9`): 16 breakout/reversal chart-pattern detectors no longer
    re-fire every bar — a new `_fresh_break()` guard requires the prior bar on the wrong side of the level.
    Stops stale repeats from inflating confluence and polluting calibration. Tests → 133.
  - **Remaining Phase 2:** derivatives z-score / basis evidence (**P2d-deriv**) → **Phase 3** (needs
    historical funding/OI in the backtest so it calibrates, not live-only); cross-sectional rank/z-score
    features (**P2b**, needs timestamp-aligned universe layout); remaining pattern hardening (high/low
    fractal pivots, breakout volume confirmation).

- **🟡 Phase 3 in progress** — risk hardening first (real-money-critical, testable without network):
  - **P3-risk-1** (`cd97e45`): true dollar risk = `qty * stop_dist` recomputed after all size scaling, so
    the volatile-regime haircut no longer leaves `risk_amount` overstating risk by `1/regime_mult` (which
    corrupted fees_r, EV, and realized-R → calibration/Kelly).
  - **P3-risk-2** (`27bf59b`): live Bybit circuit-breaker state now **persisted + UTC-rolled**
    (`reports/live_state.json`) like the paper executor — the daily-loss anchor re-anchors at the day
    boundary and the losing-streak survives restarts (was in-memory + anchored once).
  - **P3-risk-3** (`ae65860`): `portfolio_allows` now caps **total open risk-in-R** across the book
    (`max_portfolio_risk_pct`, default 6%) — bounds worst-case loss if correlated positions all stop
    together, which the margin/count caps missed.
  - **Remaining Phase 3:** tiered maintenance-margin (per-symbol, needs live ccxt tiers) + funding in EV +
    correlation-vs-open-book (needs return series threaded); ccxt.pro **WebSocket streaming feed** (live candles, freshness guard,
    closed-candle signals); tiered maintenance-margin (flat 0.5% underestimates alt liquidation); funding
    in EV; correlation / BTC-beta / gross-notional caps enforced against the OPEN book; server hardening
    (async locks, WAL SQLite off-loop, bounded socket queues, persistent trade linkage, structured
    explainability payload, typed WS events, price ticker); close the learning loop (drift → retrain).
  - Tests → 136.

**Remote:** live at github.com/yullz/Algorithmic-trading (public); pushed after every commit.

---

---

## Executive summary

This is **not** a beginner project. It is ~11K lines of genuinely sophisticated, statistically-honest
code: log-odds confluence pooling, family de-correlation, Wilson-bounded calibration, hard win-rate caps,
liquidation-aware sizing, an event-driven backtester, a walk-forward ML meta-model, and a polished
React/Tailwind dashboard. The bones are excellent.

"Godmode" therefore means **sharpening a strong system**, not rebuilding one. There are three things to do:

1. **Fix latent crashes** that detonate the moment you train a real ML model (the flagship path).
2. **Kill one structural flaw** that quietly caps how accurate the system can ever be.
3. **Add real intelligence + real-time data + a beautiful dashboard** on top of the now-trustworthy core.

### The one structural flaw (the most important finding)

> A single **in-sample** calibration number fans out into three coupled decisions — *what* to trade
> (factor win-rates + gating), *how big* (Kelly sizing), and *what you think your edge is* (EV ranking) —
> while the honest **out-of-sample** walk-forward numbers are computed and then **thrown away**
> (`backtest.py:322`). On top of that, `confidence` is a hand-built geometric proxy used *as if it were a
> probability* in gating, ranking, **and** as an ML feature. So one over-optimistic, uncalibrated quantity
> corrupts the whole pipeline in a correlated (non-diversifying) way.

Fixing this — making an **out-of-sample-validated, calibrated, cross-sectional Expected-Value model** the
actual trade selector — is the single biggest accuracy lever in the entire project.

---

## Critical bugs (silent today, fatal on first real use)

| # | Location | Bug | Impact |
|---|----------|-----|--------|
| C1 | `algotrader/signals/engine.py:226` | `entry_time=str(sl.index[-1])` — `sl` is undefined (should be `indf`) | **NameError crash on every ML-blended signal** the moment a trusted model exists. Masked only because the shipped model is untrusted. |
| C2 | `algotrader/ml/predict.py:62` | Schema guard compares against a hardcoded 2-factor synthetic probe | A real 150-symbol model produces dozens of factor columns → hashes never match → `MetaModel.load()` returns `None` forever → **the ML model can never load in production.** |
| C3 | `backtest.py:421` | `run_walkforward` references undefined `suffix` | `python backtest.py --walkforward` **crashes at the end** after all the expensive compute — the honest OOS feature is broken end-to-end. |
| C4 | `requirements.txt:13` + stray `=2.5.0` file | Depends on junk `httpx2`; real `httpx` only transitive | A clean `pip install` can break the test suite. |
| C5 | `backtest/engine.py:344` | `profit_factor=float('inf')` → `json.dump` emits `Infinity` | **Invalid JSON**; the browser dashboard's `JSON.parse` throws. |

There are **zero end-to-end tests** for `SignalEngine.generate()` and the `Scanner` — the two functions that
literally "pick the best trades" — which is why 106 green tests hid C1.

---

## Phased plan

### Phase 0 — Stop the bleeding *(critical, small)*
- Fix C1–C5.
- Add the integration tests that would have caught them: `SignalEngine.generate()` with a stub meta-model; `Scanner` ranking + correlation pruning; backtest look-ahead / intrabar-fill; ML temporal-split assertion.
- `git init` + CI (GitHub Actions) + `pytest-cov --cov-fail-under`.

### Phase 1 — Kill the circularity & leakage *(accuracy foundation)*
- **Walk-forward OOS becomes the source of truth** for `calibration.json`; gate every factor on its OOS Wilson-lower bound; drop factors that only look good in-sample.
- ML: **purged + embargoed walk-forward CV** (respect the 48-bar label horizon); **isotonic probability calibration**; **Brier-based trust gate** (not AUC alone); remove `class_weight` probability distortion.
- **Calibrate `confidence` itself** against realized outcomes (isotonic reliability map) + a reliability-curve monitor.
- Fix train/serve skews: route `cross_asset` + relative-strength through the **same** path in scan *and* backtest; unify family classification; **unify backtest & live exit policy** (the 48-bar horizon mismatch mislabels the ML target).
- Cluster-adjust statistics for correlated trades (effective sample size ≪ raw N).

### Phase 2 — The smartest selector *(godmode intelligence)*
- Reframe roles: **confluence = de-correlated feature generator + candidate proposer**; **calibrated cross-sectional EV model = decision-maker**.
- Feed **continuous indicator values** into the ML dataset (RSI/ADX/MACD-hist/ATR%/BB-width/dist-to-MA-in-ATRs/funding/OI/basis/BTC-beta/HTF-alignment/session) — today the model only sees sparse fired-factor strengths and discards most of what the indicator layer already computed.
- Add **cross-sectional rank / z-score** features across the 150-symbol universe at each timestamp (relative strength, relative vol, relative confidence) — directly serves "pick the **best** of 150."
- **Two-head model:** isotonic-calibrated `P(win)` + `E[R]` quantile regression → rank by true `EV = P(win)·E[R|win] − P(loss)·E[loss]`; monotonic constraints; light ensemble; SHAP attributions.
- **New high-signal evidence families:** real funding-rate z-score, OI z-score, OI-vs-price divergence, perp-spot basis, **CVD + CVD divergence**, liquidation-cluster mapping; **universe breadth** (% above EMA50/200, advance/decline, BTC-dominance trend) as a risk-on/off gate.
- **Indicator correctness:** RSI zero-loss=100 (not 50), ATR%-normalized regime thresholds, high/low fractal pivots, TTM Keltner 1.5; add efficient realized-vol (Yang-Zhang) + Hurst/variance-ratio regime diagnostics.
- **Pattern hardening:** fresh-break gating (stop patterns re-firing every bar), high/low pivots, relative-volume confirmation, statistically-calibrated per-pattern confidence, harmonic/Wyckoff expansion; optional **data-driven patterns** (Matrix Profile / DTW templates) that self-prune by forward edge.

### Phase 3 — Live data & portfolio risk *(true "live charts")*
- **ccxt.pro WebSocket streaming feed** (already vendored): `watch_ohlcv`/`watch_trades`/`watch_funding_rate` → true live candles pushed over the existing broadcast; REST for backfill/fallback. Turns "refresh every 5 min" into real-time.
- Per-candle **freshness guard** (halted/delisted symbols currently serve frozen candles forever); move parquet I/O off the event loop + cap retention; **closed-candle signals** to kill intrabar repaint.
- **Risk hardening:** true dollar-risk as single source of truth (fixes the `regime_mult` R-bias that leaks into Kelly/EV); roll + persist live breakers across restarts; **tiered maintenance-margin** (flat 0.5% underestimates alt liquidation); funding in EV; **correlation / BTC-beta / gross-notional caps enforced against the OPEN book** (today correlated alts stack into one levered BTC bet); edge-aware portfolio Kelly.
- **Server hardening:** async locks around executor + persistence (real data races today), WAL SQLite off the event loop, socket-set snapshot + bounded per-client queues, persistent trade linkage (survives restart), **structured explainability payload**, typed lifecycle/alert WS events, lightweight price-ticker task for intra-scan mark-to-market.
- **Close the learning loop:** paper/live outcomes → `drift_score` → scheduled retraining.

### Phase 4 — The dashboard *(beautiful, real-time)*
- Full visual redesign using **21st.dev magic components** + `frontend-design` direction, built on the existing CSS-variable token system (keep the good foundation).
- **Genuinely live:** streaming price / uPnL / equity between scans.
- **Pro charts:** indicator overlays (EMA/VWAP/BB/ATR-stop band + RSI/MACD sub-panes), detected-pattern geometry (trendlines / necklines / breakout levels), historical trade markers, in-drawer timeframe switcher + symbol search.
- **Analytics cockpit:** real time-series equity with drawdown shading + BTC buy-&-hold benchmark, **calibration reliability plot** (predicted vs realized), ML feature-importance / SHAP, PR/ROC.
- **Microstructure panels:** funding / OI / basis, liquidation feed, N×N correlation heatmap, portfolio "heat" (open risk in R), liquidation-distance gauges.
- **Fixes:** real sparklines (or remove the fabricated ones), correct exposure-bar math, `ErrorBoundary`, TanStack Query data layer, theme-driven chart colors, drawer a11y (Esc/focus-trap).

### Phase 5 — Validation & trust
- Overfitting/robustness suite: **PBO (CSCV)**, deflated Sharpe, block-bootstrap expectancy bands, parameter-stability heatmaps; nested walk-forward + FDR for any threshold tuning (today `tune_thresholds.py` / `analyze_factors.py` grid-search on the full in-sample set — pure data-snooping).
- **Portfolio-account backtester:** shared equity, concurrent positions, real leverage + liquidation against the intrabar path, funding accrual, compounding → currency CAGR / max-DD / time-under-water / ruin probability. (Today the backtest is R-multiple only and never exercises the RiskManager's leverage/Kelly/liquidation logic.)
- Backtest realism: next-bar-open entry, gap-through-stop slippage, funding accrual, point-in-time universe (survivorship).

---

## What will measurably improve

- **Accuracy of the "best trade" pick:** cross-sectional calibrated EV ranking replaces `base_win_rate·confidence·bias` with global constants.
- **Honesty of the edge:** OOS-sourced calibration + PBO/deflated-Sharpe means the number you see is the number you get.
- **Signal breadth:** real derivatives + order-flow + breadth are where perp edge actually lives, and are absent today.
- **Latency:** seconds (streaming) instead of up-to-5-minutes (polling).
- **Trust & UX:** a real-time, explainable, institutional-grade terminal.

## Guardrails kept throughout
Paper-first defaults, the multi-gate live-trading lockout, the `STOP_TRADING` kill-switch, and the honest
win-rate cap all stay. Nothing here removes a safety control; several phases *add* to them.
