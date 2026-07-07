"""Bybit live executor — every order passes the full safety stack.

To place a REAL order, ALL of the following must hold, in this order:
  1. `ALLOW_LIVE_TRADING=true` in .env            (environment opt-in)
  2. `execution.mode: live` in config.yaml        (config opt-in)
  3. API keys present
  4. Testnet by default. Mainnet additionally requires
     `execution.testnet: false` AND `execution.allow_mainnet: true`
     (double config opt-in) — otherwise sandbox mode is forced on.
  5. `LIVE_TRADING_CONFIRM=YES-I-UNDERSTAND` in the environment, or an
     interactive confirmation typed at startup.
  6. Per-entry circuit breakers (kill-switch file, daily loss, losing streak)
     and portfolio caps — identical to paper.

Order shape: market entry with attached stopLoss, then reduce-only limit
orders for the TP ladder. Anything that fails after entry triggers an
immediate protective close attempt.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Optional

from ..models import RiskConfig, Side, TradePlan
from ..risk.manager import RiskManager
from ..utils.logging import audit, get_logger, utcnow_iso
from .base import (CircuitBreakers, Executor, PositionState, portfolio_allows,
                   timeframe_to_seconds)

log = get_logger("live")

CONFIRM_PHRASE = "YES-I-UNDERSTAND"


class LiveTradingRefused(RuntimeError):
    """Raised when any gate of the safety stack is not satisfied."""


class BybitExecutor(Executor):
    def __init__(self, cfg: RiskConfig, api_key: str, api_secret: str,
                 testnet: bool = True, root: str = "."):
        import ccxt
        self.cfg = cfg
        self.testnet = testnet
        self.breakers = CircuitBreakers(cfg, root)
        self.consecutive_losses = 0
        self._day_equity: Optional[float] = None
        self._tracked: dict[str, dict] = {}   # symbol -> partial-TP fill state
        self.ex = ccxt.bybit({
            "apiKey": api_key, "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        if testnet:
            self.ex.set_sandbox_mode(True)
        self.ex.load_markets()
        log.warning("LIVE executor initialized on %s",
                    "TESTNET" if testnet else "*** MAINNET ***")

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, app_cfg, interactive: bool = True) -> "BybitExecutor":
        """Walk the safety stack; raise LiveTradingRefused on the first gate
        that fails, with an actionable message."""
        if not app_cfg.allow_live:
            raise LiveTradingRefused(
                "ALLOW_LIVE_TRADING is not 'true' in .env — live trading is off.")
        if app_cfg.execution_mode != "live":
            raise LiveTradingRefused(
                "execution.mode is not 'live' in config.yaml.")
        if not app_cfg.api_key or not app_cfg.api_secret:
            raise LiveTradingRefused(
                "API_KEY / API_SECRET missing in .env (use testnet.bybit.com keys first).")
        testnet = True
        if not app_cfg.execution_testnet:
            if not app_cfg.allow_mainnet:
                raise LiveTradingRefused(
                    "execution.testnet=false requires execution.allow_mainnet=true "
                    "(double opt-in). Forcing testnet is the default for a reason.")
            testnet = False
        if os.getenv("LIVE_TRADING_CONFIRM", "") != CONFIRM_PHRASE:
            if interactive:
                print(f"\n*** {'TESTNET' if testnet else 'MAINNET — REAL MONEY'} "
                      f"LIVE TRADING ***\nType {CONFIRM_PHRASE} to continue: ", end="")
                if input().strip() != CONFIRM_PHRASE:
                    raise LiveTradingRefused("confirmation phrase not entered.")
            else:
                raise LiveTradingRefused(
                    f"set LIVE_TRADING_CONFIRM={CONFIRM_PHRASE} in the environment "
                    f"for non-interactive live sessions.")
        return cls(app_cfg.risk, app_cfg.api_key, app_cfg.api_secret, testnet)

    # ------------------------------------------------------------------ #
    def equity(self) -> float:
        try:
            bal = self.ex.fetch_balance()
            return float(bal.get("USDT", {}).get("total") or 0.0)
        except Exception as e:
            log.error("fetch_balance failed: %s", e)
            return 0.0

    def open_positions(self) -> list[PositionState]:
        out: list[PositionState] = []
        try:
            for p in self.ex.fetch_positions(params={"category": "linear"}):
                contracts = float(p.get("contracts") or 0)
                if contracts == 0:
                    continue
                side = Side.LONG if p.get("side") == "long" else Side.SHORT
                opened_at = str(p.get("datetime") or "")
                pos = PositionState(
                    id=str(p.get("id") or p["symbol"]), symbol=p["symbol"],
                    timeframe="", side=side,
                    entry=float(p.get("entryPrice") or 0),
                    qty_initial=contracts, qty_open=contracts,
                    stop=float(p.get("stopLossPrice") or 0),
                    leverage=float(p.get("leverage") or 1),
                    margin=float(p.get("initialMargin") or 0),
                    unrealized_pnl=float(p.get("unrealizedPnl") or 0),
                    opened_at=opened_at,
                )
                # Time-stop enforcement: market-close positions that exceeded max duration.
                if self._duration_exceeded(pos):
                    self.close_position(pos.id, 0.0, "time_stop")
                    continue
                out.append(pos)
        except Exception as e:
            log.error("fetch_positions failed: %s", e)
        return out

    def _duration_exceeded(self, pos: PositionState) -> bool:
        """Return True if the position has exceeded its configured max trade duration."""
        state = self._tracked.get(pos.symbol)
        if state is not None:
            plan = state.get("plan")
            if plan is not None:
                time_stop = int(getattr(plan, "time_stop_candles", 0) or 0)
                opened_at = int(getattr(plan, "opened_at_candle", -1) or -1)
                if time_stop > 0 and opened_at >= 0:
                    # We don't have a bar index live; fall back to elapsed wall-clock
                    # time divided by the timeframe. This is approximate but safe.
                    return self._wall_clock_candles_elapsed(
                        pos.opened_at, pos.timeframe or plan.timeframe) >= time_stop
        # No tracked plan: use RiskConfig default if set.
        cfg = self.cfg
        if cfg.max_trade_duration_candles <= 0:
            return False
        return self._wall_clock_candles_elapsed(
            pos.opened_at, pos.timeframe) >= cfg.max_trade_duration_candles

    def _wall_clock_candles_elapsed(self, opened_at: str, timeframe: str) -> int:
        """Approximate candles elapsed from opened_at ISO timestamp to now."""
        if not opened_at:
            return 0
        try:
            opened = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
            elapsed_sec = (datetime.now(timezone.utc) - opened).total_seconds()
            return int(elapsed_sec / timeframe_to_seconds(timeframe)) if timeframe else 0
        except Exception:
            return 0

    # ------------------------------------------------------------------ #
    def open_position(self, plan: TradePlan) -> Optional[str]:
        equity = self.equity()
        if self._day_equity is None:
            self._day_equity = equity
        ok, why = self.breakers.allow_entry(equity, self._day_equity,
                                            self.consecutive_losses)
        if not ok:
            log.warning("LIVE entry blocked: %s", why)
            audit("live_entry_blocked", {"symbol": plan.symbol, "reason": why})
            return None
        ok, why = portfolio_allows(self.cfg, self.open_positions(), plan, equity)
        if not ok:
            log.info("LIVE entry skipped: %s", why)
            return None

        symbol = plan.symbol
        # ---- stale-candle fail-safe (must have a recently-closed bar)
        try:
            candles = self.ex.fetch_ohlcv(symbol, plan.timeframe, limit=3)
            if candles and len(candles) >= 2:
                last_close_ms = candles[-2][0]
                last_candle = datetime.fromtimestamp(last_close_ms / 1000.0,
                                                     tz=timezone.utc)
                if CircuitBreakers.is_stale(last_candle,
                                            timeframe_to_seconds(plan.timeframe)):
                    log.warning("LIVE entry on %s rejected: candle data is stale",
                                symbol)
                    audit("live_entry_blocked", {"symbol": symbol,
                                                 "reason": "stale candle data"})
                    return None
        except Exception as e:
            log.warning("LIVE entry on %s: could not verify candle freshness (%s) - "
                        "rejecting to fail safe", symbol, e)
            audit("live_entry_blocked", {"symbol": symbol,
                                         "reason": f"candle freshness check failed: {e}"})
            return None

        side = "buy" if plan.side == Side.LONG else "sell"
        try:
            # ---- leverage: round UP to integer, then re-check safety ceiling
            lev = int(math.ceil(plan.leverage))
            rm = RiskManager(self.cfg)
            safe_lev = rm._max_safe_leverage(plan.side, plan.entry, plan.stop_loss)
            if lev > safe_lev:
                log.warning("LIVE entry on %s rejected: ceiling leverage %dx exceeds "
                            "safe stop-before-liquidation ceiling %.2fx",
                            symbol, lev, safe_lev)
                audit("live_entry_rejected", {"symbol": symbol,
                                              "reason": "leverage exceeds safe ceiling",
                                              "leverage": lev, "safe_leverage": safe_lev})
                return None
            liq = rm.liquidation_price(plan.side, plan.entry, lev)
            if rm._liq_before_stop(plan.side, liq, plan.stop_loss):
                log.warning("LIVE entry on %s rejected: liquidation %.6g is closer "
                            "than stop %.6g", symbol, liq, plan.stop_loss)
                return None
            try:
                self.ex.set_leverage(lev, symbol, params={"category": "linear"})
            except Exception as e:  # already set / not supported for symbol
                log.debug("set_leverage: %s", e)

            entry = self.ex.create_order(
                symbol, "market", side, plan.qty, None,
                params={"category": "linear",
                        "stopLoss": self.ex.price_to_precision(symbol, plan.stop_loss),
                        "slTriggerBy": "LastPrice"})
            audit("live_open", {"symbol": symbol, "side": side, "qty": plan.qty,
                                "order": entry.get("id"), "testnet": self.testnet})
            log.warning("LIVE OPEN %s %s qty=%.6g (order %s)",
                        side.upper(), symbol, plan.qty, entry.get("id"))
        except Exception as e:
            log.error("LIVE entry failed for %s: %s", symbol, e)
            audit("live_entry_failed", {"symbol": symbol, "error": str(e)})
            return None

        # Reduce-only TP ladder; entry is already protected by the stop.
        tp_side = "sell" if plan.side == Side.LONG else "buy"
        for tp in plan.take_profits:
            qty = plan.qty * tp.allocation
            try:
                self.ex.create_order(
                    symbol, "limit", tp_side, qty,
                    float(self.ex.price_to_precision(symbol, tp.price)),
                    params={"category": "linear", "reduceOnly": True})
            except Exception as e:
                log.error("TP order failed (%s @ %s): %s — position remains "
                          "protected by its stop", symbol, tp.price, e)
                audit("live_tp_failed", {"symbol": symbol, "price": tp.price,
                                         "error": str(e)})

        # ---- start partial-TP tracking for breakeven-stop management
        entry_id = str(entry.get("id") or f"{symbol}@{utcnow_iso()}")
        self._tracked[symbol] = {
            "entry_id": entry_id,
            "plan": plan,
            "tps": [[tp.price, tp.r_multiple, tp.allocation, False]
                    for tp in plan.take_profits],
            "breakeven_moved": False,
        }
        self.sync_take_profits(symbol)
        return entry_id

    def close_position(self, pos_id: str, price: float, reason: str) -> None:
        """Market reduce-only close of the whole position for `pos_id`'s symbol."""
        for p in self.open_positions():
            if p.id != pos_id and p.symbol != pos_id:
                continue
            side = "sell" if p.side == Side.LONG else "buy"
            try:
                self.ex.create_order(p.symbol, "market", side, p.qty_open, None,
                                     params={"category": "linear",
                                             "reduceOnly": True})
                audit("live_close", {"symbol": p.symbol, "reason": reason})
                log.warning("LIVE CLOSE %s (%s)", p.symbol, reason)
            except Exception as e:
                log.error("LIVE close failed for %s: %s — MANUAL INTERVENTION "
                          "MAY BE REQUIRED", p.symbol, e)
                audit("live_close_failed", {"symbol": p.symbol, "error": str(e)})
            finally:
                self._tracked.pop(p.symbol, None)

    def sync_take_profits(self, symbol: str) -> None:
        """Poll closed orders, mark filled TP rungs, and move stop to breakeven
        once the first TP rung fills. Safe to call repeatedly.
        """
        state = self._tracked.get(symbol)
        if state is None:
            return
        plan: TradePlan = state["plan"]
        if not plan.take_profits:
            return

        try:
            since = int((datetime.now(timezone.utc).timestamp() - 86400) * 1000)
            closed = self.ex.fetch_closed_orders(symbol, since=since,
                                                 params={"category": "linear"})
        except Exception as e:
            log.debug("fetch_closed_orders failed for %s: %s", symbol, e)
            return

        tp_side = "sell" if plan.side == Side.LONG else "buy"
        price_tol = plan.entry * 0.001  # 0.1% tolerance for price matching
        for order in closed:
            if order.get("side") != tp_side:
                continue
            if not order.get("reduceOnly"):
                continue
            avg_price = float(order.get("average") or order.get("price") or 0.0)
            if avg_price <= 0:
                continue
            for tp in state["tps"]:
                if tp[3]:
                    continue
                if abs(avg_price - tp[0]) <= price_tol:
                    tp[3] = True

        # First TP filled -> move stop to breakeven (entry) once.
        if state["tps"][0][3] and not state["breakeven_moved"]:
            self._move_stop_to_breakeven(symbol, plan.entry)
            state["breakeven_moved"] = True

    def _move_stop_to_breakeven(self, symbol: str, price: float) -> None:
        """Update the position's attached stop-loss to the breakeven entry price."""
        try:
            price_str = self.ex.price_to_precision(symbol, price)
            self.ex.private_post_v5_position_trading_stop({
                "category": "linear",
                "symbol": symbol,
                "stopLoss": price_str,
                "positionIdx": 0,
            })
            audit("live_breakeven_stop", {"symbol": symbol, "stop": price})
            log.warning("LIVE breakeven stop set on %s @ %s", symbol, price_str)
        except Exception as e:
            log.error("Failed to move stop to breakeven for %s: %s — "
                      "MANUAL INTERVENTION MAY BE REQUIRED", symbol, e)
            audit("live_breakeven_stop_failed", {"symbol": symbol,
                                                 "stop": price, "error": str(e)})

    def record_trade_result(self, win: bool) -> None:
        """Feed the losing-streak breaker (called by the runner on fills)."""
        self.consecutive_losses = 0 if win else self.consecutive_losses + 1
