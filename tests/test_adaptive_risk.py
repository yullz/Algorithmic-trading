"""Tests for adaptive sizing, time stops, and regime-dependent risk scaling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrader.backtest.engine import Backtester
from algotrader.execution.paper import PaperExecutor
from algotrader.models import (
    Bias, Evidence, RiskConfig, SetupKind, Side, Signal, TakeProfit, TradePlan,
)
from algotrader.risk.manager import RiskManager
from algotrader.signals.engine import SignalEngine


def _signal(side: Side = Side.LONG, entry: float = 100.0, stop: float = 99.0,
            regime: str = "") -> Signal:
    bias = Bias.BULLISH if side == Side.LONG else Bias.BEARISH
    return Signal(
        symbol="BTC/USDT:USDT",
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
        regime=regime,
    )


def _base_plan(rm: RiskManager, sig: Signal | None = None, **kwargs) -> TradePlan:
    return rm.build_plan(sig or _signal(), **kwargs)


def test_volatility_target_increases_when_atr_low():
    cfg = RiskConfig(
        account_equity=10_000.0,
        risk_per_trade_pct=0.01,
        max_margin_alloc_pct=0.80,
        adaptive_sizing_mode="volatility_target",
    )
    rm = RiskManager(cfg)
    base = _base_plan(rm)
    # Low ATR -> multiplier > 1 -> bigger position and more capital at risk
    low_atr = _base_plan(rm, current_atr=0.5, median_atr_lookback=1.0)
    assert low_atr.qty > base.qty
    assert low_atr.risk_amount / cfg.account_equity == pytest.approx(0.02, rel=1e-9)


def test_volatility_target_decreases_when_atr_high():
    cfg = RiskConfig(
        account_equity=10_000.0,
        risk_per_trade_pct=0.01,
        max_margin_alloc_pct=0.80,
        adaptive_sizing_mode="volatility_target",
    )
    rm = RiskManager(cfg)
    base = _base_plan(rm)
    # High ATR -> multiplier < 1 -> smaller position and less capital at risk
    high_atr = _base_plan(rm, current_atr=2.0, median_atr_lookback=1.0)
    assert high_atr.qty < base.qty
    assert high_atr.risk_amount / cfg.account_equity == pytest.approx(0.005, rel=1e-9)


def test_volatility_target_clamp_limits_multiplier():
    cfg = RiskConfig(
        account_equity=10_000.0,
        risk_per_trade_pct=0.01,
        max_margin_alloc_pct=0.80,
        adaptive_sizing_mode="volatility_target",
    )
    rm = RiskManager(cfg)
    # Extreme ratios should be clamped to [0.5, 2.0]
    very_low = _base_plan(rm, current_atr=0.01, median_atr_lookback=1.0)
    very_high = _base_plan(rm, current_atr=100.0, median_atr_lookback=1.0)
    assert very_low.risk_amount / cfg.account_equity == pytest.approx(0.02, rel=1e-9)
    assert very_high.risk_amount / cfg.account_equity == pytest.approx(0.005, rel=1e-9)


def test_kelly_sizing_caps_at_two_percent():
    cfg = RiskConfig(
        account_equity=10_000.0,
        risk_per_trade_pct=0.01,
        max_margin_alloc_pct=0.80,
        adaptive_sizing_mode="kelly",
        kelly_fraction=0.25,
    )
    # Calibration implying a very high raw Kelly (>8%)
    calibration = {
        "_overall": 0.60,
        "_avg_win_r": 4.0,
        "_avg_loss_r": 1.0,
    }
    rm = RiskManager(cfg, calibration=calibration)
    plan = _base_plan(rm)
    # Raw Kelly = 0.6 - 0.4/4 = 0.5 -> quarter-Kelly = 12.5%, capped to 2%
    assert plan.risk_amount / cfg.account_equity == pytest.approx(0.02, rel=1e-9)
    assert plan.qty > 0


def test_kelly_sizing_reduces_risk_when_unfavorable():
    cfg = RiskConfig(
        account_equity=10_000.0,
        risk_per_trade_pct=0.01,
        adaptive_sizing_mode="kelly",
        kelly_fraction=0.25,
    )
    # Negative raw Kelly
    calibration = {
        "_overall": 0.40,
        "_avg_win_r": 1.0,
        "_avg_loss_r": 1.0,
    }
    rm = RiskManager(cfg, calibration=calibration)
    plan = _base_plan(rm)
    assert plan.risk_amount <= 0.0 or plan.qty <= 0


def test_volatile_regime_reduces_position_size():
    cfg = RiskConfig(
        account_equity=10_000.0,
        risk_per_trade_pct=0.01,
        volatile_regime_size_factor=0.7,
    )
    rm = RiskManager(cfg)
    normal = _base_plan(rm, _signal(regime=""))
    volatile = _base_plan(rm, _signal(regime="volatile"))
    assert volatile.qty == pytest.approx(normal.qty * 0.7, rel=1e-9)


def test_time_stop_closes_in_backtest():
    """A trade that neither stops out nor hits TP must close at the time-stop bar."""
    cfg = RiskConfig(
        account_equity=10_000.0,
        risk_per_trade_pct=0.01,
        max_trade_duration_candles=3,
    )
    rm = RiskManager(cfg)
    bt = Backtester(SignalEngine(), rm, horizon=20)

    # Flat, bounded price series: no stop/TP breach for 10 bars.
    n = 50
    close = 100.0 + np.zeros(n)
    high = close + 0.2
    low = close - 0.2
    df = pd.DataFrame({
        "open": close, "high": high, "low": low, "close": close,
        "volume": np.full(n, 1000.0),
    })

    sig = _signal(Side.LONG, entry=100.0, stop=99.0)
    plan = rm.build_plan(sig, opened_at_candle=10)
    outcome, exit_i, r = bt._simulate(df, 10, plan)
    assert exit_i == 13  # opened at 10, max 3 candles -> close at bar 13
    assert outcome == "loss"  # flat price -> small negative after fees


def test_time_stop_closes_in_paper_executor(tmp_path):
    cfg = RiskConfig(
        account_equity=10_000.0,
        risk_per_trade_pct=0.01,
        max_trade_duration_candles=2,
    )
    rm = RiskManager(cfg)
    exe = PaperExecutor(cfg, state_path=str(tmp_path / "paper_state.json"))
    plan = rm.build_plan(_signal())
    pos_id = exe.open_position(plan)
    assert pos_id is not None

    # Feed two candles without breaching stop/TP
    for i in range(3):
        exe.update_with_candle(plan.symbol, f"2024-01-0{i+1}T00:00:00Z",
                               100.0, 100.0, 99.5, 99.8)
    assert len(exe.positions) == 0
    assert any(t["exit_reason"] == "time_stop" for t in exe.closed)


def test_liquidation_safety_invariant_with_adaptive_sizing():
    """Liquidation price must remain beyond the stop for all sizing modes."""
    modes = ["none", "volatility_target", "kelly"]
    rng = np.random.default_rng(7)
    for mode in modes:
        cfg = RiskConfig(
            account_equity=10_000.0,
            risk_per_trade_pct=0.01,
            max_leverage=20.0,
            default_leverage=5.0,
            maintenance_margin_rate=0.005,
            max_margin_alloc_pct=0.20,
            adaptive_sizing_mode=mode,
        )
        calibration = {
            "_overall": 0.55,
            "_avg_win_r": 2.0,
            "_avg_loss_r": 1.0,
        }
        rm = RiskManager(cfg, calibration=calibration)
        for _ in range(200):
            entry = rng.uniform(100.0, 50_000.0)
            side = Side.LONG if rng.integers(2) else Side.SHORT
            stop_dist = rng.uniform(entry * 0.0015, entry * 0.12)
            stop = entry - side.sign * stop_dist
            requested_lev = rng.uniform(0.5, 25.0)
            current_atr = stop_dist / cfg.atr_stop_mult
            median_atr = rng.uniform(current_atr * 0.5, current_atr * 2.0)

            sig = _signal(side, entry, stop,
                          regime="volatile" if rng.random() < 0.3 else "")
            plan = rm.build_plan(
                sig,
                leverage=requested_lev,
                current_atr=current_atr,
                median_atr_lookback=median_atr,
            )
            assert plan is not None
            assert not RiskManager._liq_before_stop(
                plan.side, plan.liquidation_price, plan.stop_loss
            )
