"""Paper executor: identical decision path to live, simulated fills, durable
state. Fill model matches the backtester exactly (stop checked before TPs on
ambiguous candles, breakeven stop after TP1) so paper results stay comparable
to calibration.

State survives restarts via reports/paper_state.json (atomic writes): equity,
open positions, closed trades, an equity curve with mark-to-market points, the
day anchor for the daily-loss breaker, and the losing streak counter.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from ..models import RiskConfig, Side, TradePlan
from ..reporting import plan_to_dict, read_json, write_json
from ..utils.logging import audit, get_logger, utcnow_iso
from .base import (CircuitBreakers, Executor, PositionState,
                    portfolio_allows, timeframe_to_seconds)

log = get_logger("paper")

_MAX_CLOSED_KEPT = 200
_MAX_CURVE_POINTS = 5000


class PaperExecutor(Executor):
    def __init__(self, cfg: RiskConfig, state_path: str = "reports/paper_state.json",
                 root: str = "."):
        self.cfg = cfg
        self.state_path = state_path
        self.breakers = CircuitBreakers(cfg, root)
        self._equity = cfg.account_equity
        self.positions: list[PositionState] = []
        self.closed: list[dict] = []
        self.equity_curve: list[list] = []       # [iso_ts, mtm_equity]
        self.consecutive_losses = 0
        self.day_anchor = {"date": "", "equity": cfg.account_equity}
        self.last_candles: dict[str, datetime] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        s = read_json(self.state_path)
        if not s or "equity" not in s:
            return
        try:
            self._equity = float(s["equity"])
            self.positions = [PositionState.from_dict(p)
                              for p in s.get("open_positions", [])]
            self.closed = list(s.get("closed_trades", []))
            self.equity_curve = list(s.get("equity_curve", []))
            self.consecutive_losses = int(s.get("consecutive_losses", 0))
            self.day_anchor = dict(s.get("day_anchor", self.day_anchor))
            log.info("restored paper state: equity=%.2f, %d open, %d closed",
                     self._equity, len(self.positions), len(self.closed))
        except (KeyError, TypeError, ValueError) as e:
            log.warning("paper state unreadable (%s) — starting fresh", e)

    def save(self) -> None:
        write_json(self.state_path, self.state_dict())

    def state_dict(self) -> dict:
        mtm = self.mtm_equity()
        return {
            "equity": self._equity,
            "mtm_equity": mtm,
            "return_pct": (mtm / self.cfg.account_equity - 1.0)
                          if self.cfg.account_equity else 0.0,
            "open_positions": [p.to_dict() for p in self.positions],
            "closed_trades": self.closed[-_MAX_CLOSED_KEPT:],
            "equity_curve": self.equity_curve[-_MAX_CURVE_POINTS:],
            "consecutive_losses": self.consecutive_losses,
            "day_anchor": self.day_anchor,
            "updated_at": utcnow_iso(),
        }

    # ------------------------------------------------------------------ #
    # Executor interface
    # ------------------------------------------------------------------ #
    def equity(self) -> float:
        return self._equity

    def mtm_equity(self) -> float:
        return self._equity + sum(p.unrealized_pnl for p in self.positions)

    def open_positions(self) -> list[PositionState]:
        return self.positions

    def open_position(self, plan: TradePlan) -> Optional[str]:
        self._roll_day_anchor()
        ok, why = self.breakers.allow_entry(self.mtm_equity(),
                                            self.day_anchor["equity"],
                                            self.consecutive_losses)
        if not ok:
            log.warning("entry blocked: %s", why)
            audit("paper_entry_blocked", {"symbol": plan.symbol, "reason": why})
            return None
        ok, why = portfolio_allows(self.cfg, self.positions, plan, self.mtm_equity())
        if not ok:
            log.info("entry skipped: %s", why)
            return None

        last_candle = self.last_candles.get(plan.symbol)
        if last_candle is None:
            log.warning("entry on %s: no recent candle timestamp - cannot check staleness",
                        plan.symbol)
        elif CircuitBreakers.is_stale(last_candle, timeframe_to_seconds(plan.timeframe)):
            log.warning("entry on %s rejected: last candle is stale", plan.symbol)
            audit("paper_entry_blocked", {"symbol": plan.symbol,
                                          "reason": "stale candle data"})
            return None

        entry_fee = plan.notional * self.cfg.taker_fee
        self._equity -= entry_fee
        pos = PositionState(
            id=uuid.uuid4().hex[:10], symbol=plan.symbol, timeframe=plan.timeframe,
            side=plan.side, entry=plan.entry, qty_initial=plan.qty,
            qty_open=plan.qty, stop=plan.stop_loss,
            take_profits=[[t.price, t.r_multiple, t.allocation, False]
                          for t in plan.take_profits],
            leverage=plan.leverage, margin=plan.margin, opened_at=utcnow_iso(),
            plan=plan_to_dict(plan), last_price=plan.entry, duration_candles=0,
        )
        self.positions.append(pos)
        audit("paper_open", {"id": pos.id, "symbol": pos.symbol,
                             "side": pos.side.value, "entry": pos.entry,
                             "qty": pos.qty_initial, "stop": pos.stop})
        log.info("OPEN %s %s @ %.6g qty=%.6g stop=%.6g",
                 pos.side.value, pos.symbol, pos.entry, pos.qty_initial, pos.stop)
        self.save()
        return pos.id

    def close_position(self, pos_id: str, price: float, reason: str) -> None:
        pos = next((p for p in self.positions if p.id == pos_id), None)
        if pos is not None:
            self._close_qty(pos, pos.qty_open, price, reason)
            self._finalize_if_flat(pos, reason)
            self.save()

    # ------------------------------------------------------------------ #
    # candle-driven simulation
    # ------------------------------------------------------------------ #
    def update_with_candle(self, symbol: str, ts: str, o: float, h: float,
                           l: float, c: float) -> None:
        """Advance every open position on `symbol` through one closed candle,
        using the backtester's conservative fill assumptions."""
        for pos in [p for p in self.positions if p.symbol == symbol]:
            sgn = pos.side.sign
            pos.duration_candles += 1
            time_stop = int(pos.plan.get("time_stop_candles") or 0)
            # 0) time stop before other checks
            if time_stop > 0 and pos.duration_candles >= time_stop:
                self._close_qty(pos, pos.qty_open, c, "time_stop")
                self._finalize_if_flat(pos, "time_stop")
                continue
            # 1) stop first on ambiguous bars (conservative)
            stopped = (l <= pos.stop) if pos.side == Side.LONG else (h >= pos.stop)
            if stopped:
                self._close_qty(pos, pos.qty_open, pos.stop, "stop")
                self._finalize_if_flat(pos, "stop")
                continue
            # 2) take-profit ladder
            for tp in pos.take_profits:
                price, _r, alloc, filled = tp
                if filled:
                    continue
                reached = (h >= price) if pos.side == Side.LONG else (l <= price)
                if reached:
                    tp[3] = True
                    self._close_qty(pos, pos.qty_initial * alloc, price, "take_profit")
                    if not pos.breakeven_moved:
                        pos.stop = pos.entry
                        pos.breakeven_moved = True
            if pos.qty_open <= pos.qty_initial * 1e-9:
                self._finalize_if_flat(pos, "ladder_complete")
                continue
            # 3) mark to market
            pos.last_price = c
            pos.unrealized_pnl = sgn * (c - pos.entry) * pos.qty_open
        try:
            self.last_candles[symbol] = datetime.fromisoformat(ts)
        except ValueError:
            pass
        self.equity_curve.append([ts, round(self.mtm_equity(), 6)])
        self.save()

    # ------------------------------------------------------------------ #
    def _close_qty(self, pos: PositionState, qty: float, price: float,
                   reason: str) -> None:
        qty = min(qty, pos.qty_open)
        if qty <= 0:
            return
        sgn = pos.side.sign
        pnl = sgn * (price - pos.entry) * qty
        fee = price * qty * self.cfg.taker_fee
        pos.qty_open -= qty
        pos.realized_pnl += pnl - fee
        self._equity += pnl - fee
        pos.unrealized_pnl = sgn * (pos.last_price - pos.entry) * pos.qty_open
        audit("paper_fill", {"id": pos.id, "symbol": pos.symbol, "qty": qty,
                             "price": price, "pnl": pnl, "fee": fee,
                             "reason": reason})

    def _finalize_if_flat(self, pos: PositionState, reason: str) -> None:
        if pos.qty_open > pos.qty_initial * 1e-9:
            return
        self.positions = [p for p in self.positions if p.id != pos.id]
        win = pos.realized_pnl > 0
        self.consecutive_losses = 0 if win else self.consecutive_losses + 1
        risk_amount = float(pos.plan.get("risk_amount") or 0.0)
        self.closed.append({
            "id": pos.id, "symbol": pos.symbol, "side": pos.side.value,
            "entry": pos.entry, "exit_reason": reason,
            "pnl": round(pos.realized_pnl, 6),
            "r": round(pos.realized_pnl / risk_amount, 3) if risk_amount else 0.0,
            "win": win, "opened_at": pos.opened_at, "closed_at": utcnow_iso(),
            "timeframe": pos.timeframe,
        })
        audit("paper_close", self.closed[-1])
        log.info("CLOSE %s %s pnl=%.4f (%s)", pos.side.value, pos.symbol,
                 pos.realized_pnl, reason)

    def _roll_day_anchor(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        if self.day_anchor.get("date") != today:
            self.day_anchor = {"date": today, "equity": self.mtm_equity()}
