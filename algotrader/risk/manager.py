"""Risk manager: turn a directional Signal into a fully-sized, execution-ready
futures TradePlan.

Responsibilities (capital preservation first):
  * Risk-based position sizing (fixed fractional risk to the stop).
  * Leverage selection clamped to the user cap AND to a *stop-before-liquidation*
    safety ceiling, so the stop always triggers before the exchange liquidates.
  * Isolated-margin liquidation price (Binance-style linear USDT perp formula).
  * ATR/structure stop, R-multiple take-profit ladder, fees + slippage, R:R, EV.

Liquidation formulas (isolated, fees/funding excluded — a close approximation):
    LONG : liq = entry * (1 - 1/L) / (1 - mmr)
    SHORT: liq = entry * (1 + 1/L) / (1 + mmr)
"""
from __future__ import annotations

from ..explain import explain_trade
from ..models import RiskConfig, Side, Signal, TakeProfit, TradePlan
from ..utils.logging import utcnow_iso
from ..winrate import expected_value_r


class RiskManager:
    def __init__(self, cfg: RiskConfig, calibration: dict | None = None):
        self.cfg = cfg
        # Empirical payoffs measured by the backtester (calibration.json). When
        # absent, EV falls back to conservative first-target-only assumptions.
        self.calibration = calibration or {}

    # ------------------------------------------------------------------ #
    def liquidation_price(self, side: Side, entry: float, leverage: float) -> float:
        mmr = self.cfg.maintenance_margin_rate
        if side == Side.LONG:
            return entry * (1 - 1 / leverage) / (1 - mmr)
        return entry * (1 + 1 / leverage) / (1 + mmr)

    # Keep liquidation this fraction of leverage headroom BEYOND the stop, so a
    # wick/gap through the stop does not sit on top of the liquidation price.
    _LIQ_SAFETY_HAIRCUT = 0.85

    def _max_safe_leverage(self, side: Side, entry: float, stop: float) -> float:
        """Largest leverage such that liquidation is BEYOND the stop (stop fires
        first). Derived by setting liq == stop and solving for L, then a haircut
        is applied so liquidation sits strictly beyond the stop, not on it."""
        mmr = self.cfg.maintenance_margin_rate
        try:
            if side == Side.LONG:
                # want liq < stop  ->  L < 1 / (1 - stop*(1-mmr)/entry)
                denom = 1 - stop * (1 - mmr) / entry
            else:
                # want liq > stop  ->  L < 1 / (stop*(1+mmr)/entry - 1)
                denom = stop * (1 + mmr) / entry - 1
            if denom <= 0:
                return self.cfg.max_leverage
            return (1 / denom) * self._LIQ_SAFETY_HAIRCUT
        except ZeroDivisionError:
            return self.cfg.max_leverage

    # ------------------------------------------------------------------ #
    def build_plan(self, sig: Signal, equity: float | None = None,
                   leverage: float | None = None,
                   current_atr: float | None = None,
                   median_atr_lookback: float | None = None,
                   opened_at_candle: int = -1) -> TradePlan | None:
        cfg = self.cfg
        equity = equity if equity is not None else cfg.account_equity
        entry = sig.entry_ref
        stop = sig.stop_ref
        side = sig.side
        warnings: list[str] = []

        stop_dist = abs(entry - stop)
        if stop_dist <= 0 or stop_dist / entry < 0.001:
            return None  # degenerate / too-tight stop

        # ---- leverage: clamp to user cap and stop-before-liquidation ceiling
        # Reject non-positive leverage (would divide by zero / invert liq price).
        lev = leverage if (leverage is not None and leverage > 0) else cfg.default_leverage
        lev = min(max(lev, 1.0), cfg.max_leverage)
        safe_lev = self._max_safe_leverage(side, entry, stop)
        if lev > safe_lev:
            warnings.append(
                f"leverage reduced {lev:.1f}x->{max(1.0, safe_lev):.1f}x so stop "
                f"triggers before liquidation")
            lev = max(1.0, min(lev, safe_lev))

        # ---- adaptive position sizing ----
        base_risk_pct = cfg.risk_per_trade_pct
        effective_risk_pct = base_risk_pct
        sizing_note = "fixed fractional"

        if cfg.adaptive_sizing_mode == "volatility_target":
            if current_atr and median_atr_lookback and current_atr > 0:
                mult = median_atr_lookback / current_atr
                mult = max(0.5, min(mult, 2.0))
                effective_risk_pct = base_risk_pct * mult
                sizing_note = f"volatility_target (mult={mult:.2f})"
            else:
                warnings.append("volatility_target sizing requested but ATR data missing; "
                                "falling back to fixed fractional")

        elif cfg.adaptive_sizing_mode == "kelly":
            cal = self.calibration
            win_rate = float(cal.get("_overall", 0.0))
            avg_win_r = float(cal.get("_avg_win_r", 0.0))
            avg_loss_r = float(cal.get("_avg_loss_r", 1.0))
            if win_rate > 0 and avg_win_r > 0 and avg_loss_r > 0:
                # Kelly fraction: K% = W - (1-W) / (avg_win/avg_loss)
                kelly = win_rate - (1 - win_rate) / (avg_win_r / avg_loss_r)
                effective_risk_pct = max(0.0, kelly * cfg.kelly_fraction)
                # Hard ceiling: never risk more than 2% of equity per trade
                effective_risk_pct = min(effective_risk_pct, 0.02)
                sizing_note = f"kelly (raw={kelly:.2%}, quarter={effective_risk_pct:.2%})"
            else:
                warnings.append("kelly sizing requested but calibration missing; "
                                "falling back to fixed fractional")

        # Regime-dependent size scaling (applied to position, not the risk % itself)
        regime = getattr(sig, "regime", "")
        regime_mult = 1.0
        if regime == "volatile":
            regime_mult = cfg.volatile_regime_size_factor
            sizing_note += f", volatile regime factor={regime_mult:.2f}"

        # ---- position sizing --------------------------------------------------
        fixed_margin = float(getattr(cfg, "fixed_margin_usdt", 0.0) or 0.0)
        if fixed_margin > 0:
            # Fixed-margin mode: commit exactly `fixed_margin` USDT of margin, so
            # notional = margin * leverage. This is the live "open each position
            # with N USDT" policy. The %-risk and margin-cap paths are bypassed;
            # the leverage clamp + stop-before-liquidation checks above/below
            # still apply, and risk_amount is the honest qty*stop_dist so fees_r,
            # EV, and realized-R stay consistent.
            margin = fixed_margin
            notional = margin * lev
            qty = notional / entry
            risk_amount = qty * stop_dist
            sizing_note = (f"fixed margin {margin:.4g} USDT @ {lev:.0f}x "
                           f"-> {notional:.4g} notional")
        else:
            # ---- risk-% sizing (fraction of equity risked to the stop)
            risk_amount = equity * effective_risk_pct
            qty = (risk_amount / stop_dist) * regime_mult
            # True dollar risk is qty * stop_dist AFTER every size scaling (the
            # regime haircut here, the margin-cap shrink below). Recompute it so
            # risk_amount does not overstate risk by 1/regime_mult in volatile
            # regimes — that stale value would corrupt fees_r, EV, the reported
            # risk, and the realized-R that feeds calibration and Kelly sizing.
            risk_amount = qty * stop_dist
            notional = qty * entry
            margin = notional / lev

            # ---- margin allocation cap (may raise leverage or shrink size)
            cap = equity * cfg.max_margin_alloc_pct
            if margin > cap:
                needed_lev = notional / cap
                lev = min(max(needed_lev, lev), cfg.max_leverage, max(1.0, safe_lev))
                margin = notional / lev
                if margin > cap:  # still too big -> shrink position (risk drops below target)
                    notional = cap * lev
                    qty = notional / entry
                    risk_amount = qty * stop_dist
                    margin = cap
                    warnings.append(
                        f"position shrunk to respect {cfg.max_margin_alloc_pct:.0%} "
                        f"margin cap; actual risk now {risk_amount/equity:.2%} of equity")

        lev = max(1.0, lev)  # final guard: leverage must be >= 1
        liq = self.liquidation_price(side, entry, lev)

        # ---- take-profit ladder (R multiples off the stop distance)
        # A measured-move structure target caps the ladder: rungs beyond what
        # the pattern geometry supports are wishful thinking. Rungs capped to
        # the same level are merged (allocations summed).
        r_multiples = list(cfg.tp_r_multiples)
        struct_target = getattr(sig, "structure_target", None)
        if struct_target is not None:
            target_r = abs(struct_target - entry) / stop_dist
            if target_r >= 1.0 and target_r < max(r_multiples):
                r_multiples = [min(r, target_r) for r in r_multiples]
                warnings.append(
                    f"TP ladder capped at {target_r:.2f}R (pattern measured-move target)")
        merged: dict[float, float] = {}
        for r_mult, alloc in zip(r_multiples, cfg.tp_allocations):
            key = round(r_mult, 4)
            merged[key] = merged.get(key, 0.0) + alloc
        tps: list[TakeProfit] = []
        for r_mult in sorted(merged):
            price = entry + side.sign * r_mult * stop_dist
            tps.append(TakeProfit(price=price, r_multiple=r_mult,
                                  allocation=merged[r_mult]))
        ladder_best_case_r = sum(t.r_multiple * t.allocation for t in tps)

        # ---- realistic expected payoff, CONSISTENT with the calibrated win rate.
        # The win rate measures reaching the first target before the stop, so EV
        # must NOT assume every win runs the whole ladder. Prefer empirical
        # ladder-aware payoffs from the backtest; else fall back to a conservative
        # first-target-only win (avg_win = first TP R, avg_loss = full 1R stop).
        cal = self.calibration
        calibrated = "_avg_win_r" in cal
        avg_win_r = float(cal.get("_avg_win_r", cfg.tp_r_multiples[0]))
        avg_loss_r = float(cal.get("_avg_loss_r", 1.0))

        # ---- fees + slippage (round trip), expressed in quote and in R
        fee_rate = cfg.taker_fee * 2 + cfg.slippage_pct * 2
        fees_quote = notional * fee_rate
        fees_r = fees_quote / risk_amount if risk_amount > 0 else 0.0

        # Calibration now stores net-of-fee payoffs; only subtract fees for
        # uncalibrated conservative estimates to avoid double-counting.
        ev_fees_r = 0.0 if calibrated else fees_r
        ev_r = expected_value_r(sig.base_win_rate, avg_win_r, avg_loss_r, ev_fees_r)
        reward_risk = avg_win_r / max(avg_loss_r, 1e-9)

        if self._liq_before_stop(side, liq, stop):
            warnings.append("[!] liquidation is closer than the stop - DO NOT take "
                            "this trade at this leverage")

        rationale = [f"{e.name} ({e.bias.value}, str={e.strength:.2f}, "
                     f"base_wr={e.base_win_rate:.0%})" for e in sig.evidence]
        src = "calibrated" if calibrated else "uncalibrated conservative"
        rationale.append(
            f"win rate {sig.base_win_rate:.0%}, expected win {avg_win_r:.2f}R / "
            f"loss {avg_loss_r:.2f}R [{src}], ladder best-case {ladder_best_case_r:.2f}R, "
            f"fees {fees_r:.2f}R, sizing={sizing_note}")

        # time-stop bookkeeping
        time_stop_candles = cfg.max_trade_duration_candles if cfg.max_trade_duration_candles > 0 else 0

        explanation = explain_trade(
            sig, entry=entry, stop=stop, take_profits=tps, leverage=lev, liq=liq,
            win_rate=sig.base_win_rate, avg_win_r=avg_win_r, avg_loss_r=avg_loss_r,
            fees_r=fees_r, ev_r=ev_r, reward_risk=reward_risk,
            calibration=self.calibration, calibrated=calibrated, warnings=warnings)

        return TradePlan(
            symbol=sig.symbol, timeframe=sig.timeframe, side=side,
            entry=entry, stop_loss=stop, take_profits=tps, leverage=lev,
            qty=qty, notional=notional, margin=margin, risk_amount=risk_amount,
            liquidation_price=liq, reward_risk=reward_risk,
            expected_win_rate=sig.base_win_rate, expected_value_r=ev_r,
            confidence=sig.confidence, fees_estimate=fees_quote,
            rationale=rationale, explanation=explanation, warnings=warnings,
            created_at=utcnow_iso(),
            regime=getattr(sig, "regime", ""),
            families=list(getattr(sig, "families", []) or []),
            ml_prob=getattr(sig, "ml_prob", None),
            ml_weight=getattr(sig, "ml_weight", 0.0),
            ml_contribs=list(getattr(sig, "ml_contribs", []) or []),
            ml_ev_r=getattr(sig, "ml_ev_r", None),
            time_stop_candles=time_stop_candles,
            opened_at_candle=opened_at_candle,
        )

    @staticmethod
    def _liq_before_stop(side: Side, liq: float, stop: float) -> bool:
        # For a long, liq and stop are below entry; liquidation must be *lower*
        # than the stop. For a short, both above entry; liq must be *higher*.
        return (side == Side.LONG and liq >= stop) or (side == Side.SHORT and liq <= stop)
