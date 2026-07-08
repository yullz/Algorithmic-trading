"""Unit tests for execution-layer safety wiring."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from algotrader.execution.base import CircuitBreakers, timeframe_to_seconds
from algotrader.execution.bybit import BybitExecutor
from algotrader.execution.paper import PaperExecutor
from algotrader.models import RiskConfig, Side, TradePlan


def _plan(symbol: str = "BTC/USDT:USDT", leverage: float = 2.0,
          entry: float = 100.0, stop: float = 50.0,
          tps: list | None = None) -> TradePlan:
    """Build a minimal TradePlan for executor tests."""
    return TradePlan(
        symbol=symbol,
        timeframe="1h",
        side=Side.LONG,
        entry=entry,
        stop_loss=stop,
        take_profits=tps or [],
        leverage=leverage,
        qty=1.0,
        notional=entry,
        margin=entry / leverage,
        risk_amount=entry - stop,
        liquidation_price=entry * 0.4,
        reward_risk=1.0,
        expected_win_rate=0.5,
        expected_value_r=0.1,
        confidence=0.6,
        fees_estimate=0.0,
    )


def test_timeframe_to_seconds():
    assert timeframe_to_seconds("15m") == 900
    assert timeframe_to_seconds("1h") == 3600
    assert timeframe_to_seconds("4h") == 14400
    assert timeframe_to_seconds("1d") == 86400
    assert timeframe_to_seconds("1w") == 604800
    with pytest.raises(ValueError):
        timeframe_to_seconds("xyz")


def test_is_stale():
    now = datetime.now(timezone.utc)
    assert not CircuitBreakers.is_stale(now, 3600)
    assert CircuitBreakers.is_stale(now - timedelta(hours=3), 3600)


def test_paper_executor_rejects_stale_candle(tmp_path: Path):
    state = tmp_path / "paper_state.json"
    cfg = RiskConfig()
    exe = PaperExecutor(cfg, state_path=str(state))
    plan = _plan()
    # Inject a stale candle timestamp
    exe.last_candles[plan.symbol] = datetime.now(timezone.utc) - timedelta(hours=3)
    assert exe.open_position(plan) is None


def _fake_bybit(closed_orders=None, positions=None):
    """Return a fake ccxt module and exchange instance."""
    ex = MagicMock()
    ex.fetch_balance.return_value = {"USDT": {"total": 1000.0}}
    ex.fetch_positions.return_value = positions or []
    ex.set_leverage.return_value = None
    ex.price_to_precision.side_effect = lambda _s, p: f"{p:.2f}"
    # IOC entry fills fully by default; TP legs also "order1".
    ex.create_order.return_value = {"id": "order1", "filled": 1.0}
    ex.fetch_order.return_value = {"id": "order1", "filled": 1.0}
    # Order-size preflight: a symbol with a tiny step so plan.qty passes cleanly.
    ex.market.return_value = {
        "limits": {"amount": {"min": 0.001}},
        "precision": {"amount": 0.001},
        "info": {"lotSizeFilter": {"minNotionalValue": "5"}},
    }
    ex.amount_to_precision.side_effect = lambda _s, q: f"{float(q):.3f}"
    # Microstructure preflight: tight spread, benign funding.
    ex.fetch_ticker.return_value = {"bid": 99.95, "ask": 100.05,
                                    "info": {"fundingRate": "0.00001"}}

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    ex.fetch_ohlcv.return_value = [
        [int((now - timedelta(hours=2)).timestamp() * 1000),
         99.0, 101.0, 98.0, 100.0, 1000.0],
        [int((now - timedelta(hours=1)).timestamp() * 1000),
         100.0, 102.0, 99.0, 101.0, 1000.0],
        [int(now.timestamp() * 1000),
         101.0, 103.0, 100.0, 102.0, 1000.0],
    ]
    ex.fetch_closed_orders.return_value = closed_orders or []
    ex.private_post_v5_position_trading_stop.return_value = None

    fake_ccxt = MagicMock()
    fake_ccxt.bybit.return_value = ex
    return fake_ccxt, ex


def test_bybit_rejects_stale_candle(monkeypatch, tmp_path):
    fake_ccxt, ex = _fake_bybit()
    ex.fetch_ohlcv.return_value = [
        [0, 99.0, 101.0, 98.0, 100.0, 1000.0],
        [3_600_000, 100.0, 102.0, 99.0, 101.0, 1000.0],
        # Last closed bar is 25 hours old
        [(datetime.now(timezone.utc) - timedelta(hours=25)).timestamp() * 1000,
         101.0, 103.0, 100.0, 102.0, 1000.0],
    ]
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)

    exe = BybitExecutor(RiskConfig(require_validated_edge=False), "key", "secret",
                        testnet=True, root=str(tmp_path))
    plan = _plan(leverage=2.0, entry=100.0, stop=50.0)
    assert exe.open_position(plan) is None
    ex.create_order.assert_not_called()


def test_bybit_rejects_unsafe_ceiling_leverage(monkeypatch, tmp_path):
    fake_ccxt, ex = _fake_bybit()
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)

    exe = BybitExecutor(RiskConfig(require_validated_edge=False), "key", "secret",
                        testnet=True, root=str(tmp_path))
    # leverage=1.5 -> ceil=2, but safe ceiling for a 50% stop is <2
    plan = _plan(leverage=1.5, entry=100.0, stop=50.0)
    assert exe.open_position(plan) is None
    ex.set_leverage.assert_not_called()
    ex.create_order.assert_not_called()


def test_bybit_moves_stop_to_breakeven_after_tp1(monkeypatch, tmp_path):
    fake_ccxt, ex = _fake_bybit()
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)

    exe = BybitExecutor(RiskConfig(require_validated_edge=False), "key", "secret",
                        testnet=True, root=str(tmp_path))
    plan = _plan(
        leverage=2.0, entry=100.0, stop=99.0,
        tps=[MagicMock(price=101.0, r_multiple=1.0, allocation=0.4)],
    )
    assert exe.open_position(plan) == "order1"

    # Simulate TP1 filled at 101
    ex.fetch_closed_orders.return_value = [
        {"side": "sell", "reduceOnly": True, "average": 101.0, "price": 101.0},
    ]
    exe.sync_take_profits(plan.symbol)
    ex.private_post_v5_position_trading_stop.assert_called_once()
    call_args = ex.private_post_v5_position_trading_stop.call_args[0][0]
    assert call_args["symbol"] == plan.symbol
    assert float(call_args["stopLoss"]) == pytest.approx(plan.entry)


def test_bybit_tracked_state_cleared_on_close(monkeypatch, tmp_path):
    fake_ccxt, ex = _fake_bybit(positions=[{
        "symbol": "BTC/USDT:USDT",
        "contracts": 1.0,
        "side": "long",
        "entryPrice": 100.0,
        "stopLossPrice": 50.0,
        "leverage": 2.0,
        "initialMargin": 50.0,
        "unrealizedPnl": 0.0,
        "datetime": "2024-01-01T00:00:00Z",
    }])
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)

    exe = BybitExecutor(RiskConfig(require_validated_edge=False), "key", "secret",
                        testnet=True, root=str(tmp_path))
    exe._tracked["BTC/USDT:USDT"] = {"entry_id": "order1"}
    exe.close_position("BTC/USDT:USDT", 100.0, "test")
    assert "BTC/USDT:USDT" not in exe._tracked


def test_bybit_breaker_state_persists_across_restart(monkeypatch, tmp_path):
    """The losing-streak breaker must survive a process restart — otherwise it
    resets to zero exactly when a bad run should be halting new entries."""
    fake_ccxt, _ex = _fake_bybit()
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)
    root = str(tmp_path)

    exe = BybitExecutor(RiskConfig(), "key", "secret", testnet=True, root=root)
    exe.record_trade_result(win=False)
    exe.record_trade_result(win=False)
    assert exe.consecutive_losses == 2

    exe2 = BybitExecutor(RiskConfig(), "key", "secret", testnet=True, root=root)
    assert exe2.consecutive_losses == 2  # restored from disk


def test_bybit_day_anchor_rolls_at_utc_boundary(monkeypatch, tmp_path):
    """The daily-loss anchor must re-anchor at the UTC day boundary, not stay
    pinned to a stale previous-day (or first-ever) reference."""
    fake_ccxt, _ex = _fake_bybit()
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)
    exe = BybitExecutor(RiskConfig(), "key", "secret", testnet=True, root=str(tmp_path))

    exe.day_anchor = {"date": "2000-01-01", "equity": 5000.0}  # stale
    exe._roll_day_anchor(1000.0)

    today = datetime.now(timezone.utc).date().isoformat()
    assert exe.day_anchor["date"] == today
    assert exe.day_anchor["equity"] == 1000.0  # re-anchored to CURRENT equity


def test_portfolio_allows_caps_total_open_risk():
    """Total open risk-in-R across the book is capped, so correlated positions
    cannot each pass the per-trade cap and stack into one oversized drawdown."""
    from algotrader.execution.base import PositionState, portfolio_allows

    cfg = RiskConfig(account_equity=10_000.0, max_portfolio_risk_pct=0.06,
                     max_concurrent_positions=20, max_total_margin_pct=100.0)
    equity = 10_000.0

    def _pos(i):
        return PositionState(
            id=f"A{i}", symbol=f"A{i}/USDT:USDT", timeframe="1h", side=Side.LONG,
            entry=100.0, qty_initial=1.0, qty_open=1.0, stop=99.0, margin=1.0,
            plan={"risk_amount": 100.0})

    book = [_pos(i) for i in range(5)]  # 5 x 100 = 500 = 5% of 10k

    # +0.5% -> 5.5% total, under the 6% cap.
    small = _plan(symbol="NEW/USDT:USDT", entry=100.0, stop=50.0)   # risk_amount 50
    ok, _why = portfolio_allows(cfg, book, small, equity)
    assert ok

    # +1.5% -> 6.5% total, over the cap -> blocked.
    big = _plan(symbol="BIG/USDT:USDT", entry=200.0, stop=50.0)     # risk_amount 150
    ok, why = portfolio_allows(cfg, book, big, equity)
    assert not ok
    assert "open risk" in why
