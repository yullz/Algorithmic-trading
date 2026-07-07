"""Regression tests for the Phase-0 critical-bug fixes (see GODMODE_PLAN.md)."""
from __future__ import annotations

import json
import math

from algotrader.backtest.engine import Backtester
from algotrader.models import RiskConfig
from algotrader.risk.manager import RiskManager
from algotrader.signals.engine import SignalEngine


def _bt() -> Backtester:
    return Backtester(SignalEngine(), RiskManager(RiskConfig()))


def _trade(r: float, win: int) -> dict:
    return {"r": r, "win": win, "factors": ["ema_stack_bull"], "kind": "breakout"}


def test_profit_factor_finite_when_no_losses():
    """C5: profit_factor must never be float('inf') — json.dump emits `Infinity`,
    which the browser dashboard's JSON.parse rejects."""
    trades = [_trade(1.5, 1) for _ in range(12)]  # all winners, zero losses
    res = _bt()._aggregate(trades)
    pf = res.summary["profit_factor"]
    assert math.isfinite(pf), "profit_factor must be finite even with no losers"
    # The whole summary must serialize to strict, valid JSON.
    text = json.dumps(res.summary)
    assert "Infinity" not in text and "NaN" not in text
    assert json.loads(text)["profit_factor"] == pf


def test_profit_factor_ratio_when_losses_present():
    trades = [_trade(2.0, 1), _trade(2.0, 1), _trade(-1.0, 0)]
    res = _bt()._aggregate(trades)
    assert res.summary["profit_factor"] == 4.0  # 4 / 1


def test_empty_trades_aggregate_is_json_safe():
    res = _bt()._aggregate([])
    json.dumps(res.summary)  # must not raise
    assert res.summary["trades"] == 0
