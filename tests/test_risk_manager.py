"""Unit tests for algotrader/risk/manager.py."""
from __future__ import annotations

import numpy as np
import pytest

from algotrader.models import Bias, Evidence, RiskConfig, SetupKind, Side, Signal
from algotrader.risk.manager import RiskManager


def _signal(side: Side, entry: float, stop: float) -> Signal:
    bias = Bias.BULLISH if side == Side.LONG else Bias.BEARISH
    return Signal(
        symbol="BTCUSDT",
        timeframe="1h",
        side=side,
        kind=SetupKind.BREAKOUT,
        entry_ref=entry,
        stop_ref=stop,
        confidence=0.60,
        score=1.0,
        evidence=[
            Evidence("trend_up", "indicator", bias, 0.7, 0.55, family="trend"),
        ],
        base_win_rate=0.55,
    )


def test_build_plan_safety_invariant():
    """Liquidation price must never be closer to entry than the stop loss."""
    cfg = RiskConfig(
        account_equity=10_000.0,
        risk_per_trade_pct=0.01,
        max_leverage=20.0,
        default_leverage=5.0,
        maintenance_margin_rate=0.005,
        max_margin_alloc_pct=0.20,
    )
    rm = RiskManager(cfg)
    rng = np.random.default_rng(13)

    for _ in range(200):
        entry = rng.uniform(100.0, 50_000.0)
        side = Side.LONG if rng.integers(2) else Side.SHORT
        stop_dist = rng.uniform(entry * 0.0015, entry * 0.12)
        stop = entry - side.sign * stop_dist
        requested_lev = rng.uniform(0.5, 25.0)

        sig = _signal(side, entry, stop)
        plan = rm.build_plan(sig, leverage=requested_lev)
        assert plan is not None

        entry_to_stop = abs(plan.entry - plan.stop_loss)
        entry_to_liq = abs(plan.entry - plan.liquidation_price)
        # Liquidation must sit strictly beyond the stop (or at worst equal due
        # to rounding) so the stop always triggers before liquidation.
        assert entry_to_liq >= entry_to_stop - 1e-9
        assert not RiskManager._liq_before_stop(
            plan.side, plan.liquidation_price, plan.stop_loss
        )


def test_build_plan_rejects_degenerate_stop():
    cfg = RiskConfig()
    rm = RiskManager(cfg)
    sig = _signal(Side.LONG, 100.0, 100.0)
    assert rm.build_plan(sig) is None
