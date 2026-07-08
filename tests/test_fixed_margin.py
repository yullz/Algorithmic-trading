"""Fixed-margin sizing: every trade commits exactly N USDT of margin.

This is the live "open each position with 15 USDT" policy. notional = margin *
leverage; the %-risk and margin-cap paths are bypassed.
"""
from __future__ import annotations

import pytest

from algotrader.models import Bias, Evidence, RiskConfig, SetupKind, Side, Signal
from algotrader.risk.manager import RiskManager


def _signal(side: Side = Side.LONG, entry: float = 100.0, stop: float = 99.0) -> Signal:
    bias = Bias.BULLISH if side == Side.LONG else Bias.BEARISH
    return Signal(
        symbol="BTC/USDT:USDT", timeframe="1h", side=side, kind=SetupKind.BREAKOUT,
        entry_ref=entry, stop_ref=stop, confidence=0.60, score=1.0,
        evidence=[Evidence("trend_up", "indicator", bias, 0.7, 0.55, family="trend")],
        base_win_rate=0.55,
    )


def test_fixed_margin_sizes_to_exact_margin():
    cfg = RiskConfig(account_equity=1000.0, fixed_margin_usdt=15.0, default_leverage=3.0)
    plan = RiskManager(cfg).build_plan(_signal(Side.LONG, 100.0, 99.0))
    assert plan is not None
    assert plan.margin == pytest.approx(15.0, abs=1e-6)
    assert plan.notional == pytest.approx(15.0 * plan.leverage, rel=1e-6)
    assert plan.qty == pytest.approx(plan.notional / plan.entry, rel=1e-6)
    stop_dist = abs(plan.entry - plan.stop_loss)
    assert plan.risk_amount == pytest.approx(plan.qty * stop_dist, rel=1e-6)


def test_fixed_margin_ignores_small_equity_no_shrink():
    # %-risk + margin cap would shrink to 30*0.20 = 6 USDT of margin; fixed mode
    # keeps the full 15 regardless of account equity.
    cfg = RiskConfig(account_equity=30.0, fixed_margin_usdt=15.0, max_margin_alloc_pct=0.20)
    plan = RiskManager(cfg).build_plan(_signal())
    assert plan is not None
    assert plan.margin == pytest.approx(15.0, abs=1e-6)


def test_fixed_margin_disabled_uses_risk_pct():
    cfg = RiskConfig(account_equity=1000.0, risk_per_trade_pct=0.01, fixed_margin_usdt=0.0)
    plan = RiskManager(cfg).build_plan(_signal(Side.LONG, 100.0, 99.0))
    assert plan is not None
    # equity * risk_pct = 10 USDT risked to a 1.0 stop distance (not fixed 15 margin).
    assert plan.risk_amount == pytest.approx(10.0, rel=1e-3)
    assert plan.margin != pytest.approx(15.0, abs=1e-6)


def test_fixed_margin_short_side():
    cfg = RiskConfig(account_equity=500.0, fixed_margin_usdt=15.0, default_leverage=3.0)
    plan = RiskManager(cfg).build_plan(_signal(Side.SHORT, 100.0, 101.0))
    assert plan is not None
    assert plan.margin == pytest.approx(15.0, abs=1e-6)
    assert plan.notional == pytest.approx(15.0 * plan.leverage, rel=1e-6)
