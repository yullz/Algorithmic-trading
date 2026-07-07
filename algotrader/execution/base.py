"""Executor interface, position state, portfolio caps, and circuit breakers.

Everything that decides WHETHER a trade may be opened lives here so that the
paper and live paths cannot drift apart: an entry the breakers would block on
mainnet is blocked in paper mode too.
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..models import RiskConfig, Side, TradePlan

KILL_SWITCH_FILE = "STOP_TRADING"


def timeframe_to_seconds(tf: str) -> int:
    """Convert timeframe strings like '15m', '1h', '4h', '1d' to seconds."""
    m = re.match(r"^(\d+)([smhdw])$", tf.lower())
    if not m:
        raise ValueError(f"unsupported timeframe '{tf}'")
    value, unit = int(m.group(1)), m.group(2)
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    return value * multipliers[unit]


@dataclass
class PositionState:
    """A live or simulated open position (one per symbol)."""
    id: str
    symbol: str
    timeframe: str
    side: Side
    entry: float
    qty_initial: float
    qty_open: float
    stop: float
    # (price, r_multiple, allocation, filled)
    take_profits: list[list] = field(default_factory=list)
    leverage: float = 1.0
    margin: float = 0.0
    opened_at: str = ""
    plan: dict = field(default_factory=dict)   # serialized TradePlan snapshot
    realized_pnl: float = 0.0                  # quote, from partial exits
    unrealized_pnl: float = 0.0                # quote, mark-to-market
    breakeven_moved: bool = False
    last_price: float = 0.0
    duration_candles: int = 0  # candles elapsed since open (paper executor)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["side"] = self.side.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PositionState":
        d = dict(d)
        d["side"] = Side(d["side"])
        return cls(**d)


class CircuitBreakers:
    """Runtime halt conditions checked before EVERY new entry.

    * kill switch: a file named STOP_TRADING in the project root
    * daily loss: equity below day-start equity by max_daily_loss_pct
    * losing streak: max_consecutive_losses in a row
    * stale data: candle older than 2 intervals (checked via is_stale)
    """

    def __init__(self, cfg: RiskConfig, root: str = "."):
        self.cfg = cfg
        self.root = root

    def allow_entry(self, equity: float, day_start_equity: float,
                    consecutive_losses: int) -> tuple[bool, str]:
        if os.path.exists(os.path.join(self.root, KILL_SWITCH_FILE)):
            return False, f"kill switch active ({KILL_SWITCH_FILE} file present)"
        if day_start_equity > 0:
            dd = 1.0 - equity / day_start_equity
            if dd >= self.cfg.max_daily_loss_pct:
                return False, (f"daily loss breaker: -{dd:.1%} >= "
                               f"{self.cfg.max_daily_loss_pct:.1%} — no new entries today")
        if consecutive_losses >= self.cfg.max_consecutive_losses:
            return False, (f"losing-streak breaker: {consecutive_losses} straight "
                           f"losses — stop and review before continuing")
        return True, ""

    @staticmethod
    def is_stale(last_candle_ts: datetime, timeframe_sec: int) -> bool:
        """No orders on data older than 2 intervals — a dead feed must fail safe."""
        now = datetime.now(timezone.utc)
        if last_candle_ts.tzinfo is None:
            last_candle_ts = last_candle_ts.replace(tzinfo=timezone.utc)
        return (now - last_candle_ts).total_seconds() > 2 * timeframe_sec


def portfolio_allows(cfg: RiskConfig, open_positions: list[PositionState],
                     plan: TradePlan, equity: float) -> tuple[bool, str]:
    """Portfolio-level caps shared by paper and live executors."""
    if any(p.symbol == plan.symbol for p in open_positions):
        return False, f"already holding a {plan.symbol} position"
    if len(open_positions) >= cfg.max_concurrent_positions:
        return False, (f"max concurrent positions ({cfg.max_concurrent_positions}) "
                       f"reached")
    total_margin = sum(p.margin for p in open_positions) + plan.margin
    if equity > 0 and total_margin > equity * cfg.max_total_margin_pct:
        return False, (f"total margin {total_margin:.2f} would exceed "
                       f"{cfg.max_total_margin_pct:.0%} of equity")
    # Total open risk-in-R across the book. This bounds worst-case loss if
    # correlated positions all stop out together — N ~0.8-correlated alts can
    # each pass every per-trade cap yet stack into one big drawdown, which the
    # margin/count caps do not prevent.
    total_open_risk = sum(float((p.plan or {}).get("risk_amount", 0.0) or 0.0)
                          for p in open_positions) + plan.risk_amount
    if equity > 0 and total_open_risk > equity * cfg.max_portfolio_risk_pct:
        return False, (f"total open risk {total_open_risk / equity:.1%} would exceed "
                       f"portfolio cap {cfg.max_portfolio_risk_pct:.0%}")
    return True, ""


class Executor(ABC):
    """Common contract for paper and live execution."""

    @abstractmethod
    def open_position(self, plan: TradePlan) -> Optional[str]:
        """Open a position from a TradePlan; returns position id or None
        (with the refusal reason logged/audited)."""

    @abstractmethod
    def close_position(self, pos_id: str, price: float, reason: str) -> None:
        ...

    @abstractmethod
    def open_positions(self) -> list[PositionState]:
        ...

    @abstractmethod
    def equity(self) -> float:
        ...
