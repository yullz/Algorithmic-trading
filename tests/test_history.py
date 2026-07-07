"""Unit tests for the SQLite history store."""
from __future__ import annotations

import pytest

from algotrader.history import HistoryStore, SignalHistory, TradeHistory
from algotrader.models import SetupKind, Side


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "history.db"
    return HistoryStore(db_path=str(db), root=".")


def test_record_scan_and_signals(store):
    scan_id = store.record_scan(
        timestamp="2024-01-01T00:00:00+00:00",
        btc_regime="trend_down",
        n_symbols=10,
        top_n=3,
    )
    assert scan_id > 0

    signals = [
        SignalHistory(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            side=Side.LONG,
            kind=SetupKind.BREAKOUT,
            entry=100.0,
            stop=95.0,
            confidence=0.7,
            score=2.5,
            win_rate=0.6,
            expected_value_r=0.5,
            rationale=["breakout", "volume"],
            timestamp="2024-01-01T00:00:00+00:00",
        ),
        SignalHistory(
            symbol="ETH/USDT:USDT",
            timeframe="4h",
            side=Side.SHORT,
            kind=SetupKind.MOMENTUM,
            entry=200.0,
            stop=210.0,
            confidence=0.65,
            score=-1.8,
            win_rate=0.55,
            expected_value_r=0.3,
            rationale=["momentum"],
            timestamp="2024-01-01T00:00:00+00:00",
        ),
    ]
    ids = store.record_signals(scan_id, signals)
    assert ids["BTC/USDT:USDT"] > 0
    assert ids["ETH/USDT:USDT"] > 0

    rows = store.get_signals()
    assert len(rows) == 2
    assert rows[0]["symbol"] in ("BTC/USDT:USDT", "ETH/USDT:USDT")


def test_record_trade_and_update_position(store):
    scan_id = store.record_scan(
        timestamp="2024-01-01T00:00:00+00:00",
        btc_regime="range",
        n_symbols=5,
        top_n=1,
    )
    signal_id = store.record_signals(scan_id, [
        SignalHistory(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            side=Side.LONG,
            kind=SetupKind.REVERSAL,
            entry=100.0,
            stop=95.0,
            confidence=0.7,
            score=1.2,
            win_rate=0.6,
            expected_value_r=0.4,
            rationale=["rsi oversold"],
        )
    ])["BTC/USDT:USDT"]

    trade_id = store.record_trade(
        scan_id=scan_id,
        signal_id=signal_id,
        trade=TradeHistory(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry=100.0,
            stop=95.0,
            qty=1.0,
            leverage=2.0,
            margin=50.0,
        ),
    )
    assert trade_id > 0

    trades = store.get_trades()
    assert len(trades) == 1
    assert trades[0]["outcome"] == "open"

    store.update_position(
        trade_id=trade_id,
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        opened_at="2024-01-01T00:00:00+00:00",
        closed_at="2024-01-01T01:00:00+00:00",
        status="closed",
        mtm_pnl=5.0,
        outcome="take_profit",
        realized_r=1.0,
        fees_r=0.02,
    )

    trades = store.get_trades()
    assert trades[0]["outcome"] == "take_profit"
    assert trades[0]["realized_r"] == pytest.approx(1.0)


def test_get_signals_filters(store):
    scan_id = store.record_scan("2024-01-01T00:00:00+00:00", "up", 2, 2)
    store.record_signals(scan_id, [
        SignalHistory("BTC/USDT:USDT", "1h", Side.LONG, SetupKind.BREAKOUT,
                      100, 95, 0.7, 1.0, 0.6, 0.5, ["a"], "2024-01-01T00:00:00+00:00"),
        SignalHistory("ETH/USDT:USDT", "4h", Side.SHORT, SetupKind.MOMENTUM,
                      200, 210, 0.6, -1.0, 0.5, 0.3, ["b"], "2024-01-02T00:00:00+00:00"),
    ])

    assert len(store.get_signals({"symbol": "BTC/USDT:USDT"})) == 1
    assert len(store.get_signals({"side": "SHORT"})) == 1
    assert len(store.get_signals({"from_": "2024-01-02T00:00:00+00:00"})) == 1


def test_get_win_rate_by(store):
    scan_id = store.record_scan("2024-01-01T00:00:00+00:00", "up", 2, 2)
    btc_id = store.record_signals(scan_id, [
        SignalHistory("BTC/USDT:USDT", "1h", Side.LONG, SetupKind.BREAKOUT,
                      100, 95, 0.7, 1.0, 0.6, 0.5, ["a"]),
    ])["BTC/USDT:USDT"]
    eth_id = store.record_signals(scan_id, [
        SignalHistory("ETH/USDT:USDT", "1h", Side.SHORT, SetupKind.MOMENTUM,
                      200, 210, 0.6, -1.0, 0.5, 0.3, ["b"]),
    ])["ETH/USDT:USDT"]

    t1 = store.record_trade(scan_id, btc_id, TradeHistory(
        "BTC/USDT:USDT", Side.LONG, 100, 95, 1, 2, 50))
    t2 = store.record_trade(scan_id, eth_id, TradeHistory(
        "ETH/USDT:USDT", Side.SHORT, 200, 210, 1, 2, 100))

    store.update_position(t1, "BTC/USDT:USDT", Side.LONG, "2024-01-01T00:00:00+00:00",
                          "2024-01-01T01:00:00+00:00", "closed", 5, "tp", 1.0, 0.01)
    store.update_position(t2, "ETH/USDT:USDT", Side.SHORT, "2024-01-01T00:00:00+00:00",
                          "2024-01-01T01:00:00+00:00", "closed", -3, "stop", -1.0, 0.01)

    by_symbol = store.get_win_rate_by("symbol")
    assert len(by_symbol) == 2
    btc = next(r for r in by_symbol if r["symbol"] == "BTC/USDT:USDT")
    assert btc["wins"] == 1
    assert btc["trades"] == 1

    by_side = store.get_win_rate_by("side")
    assert len(by_side) == 2


def test_get_scan_summary(store):
    scan_id = store.record_scan("2024-01-01T00:00:00+00:00", "up", 2, 1)
    sid = store.record_signals(scan_id, [
        SignalHistory("BTC/USDT:USDT", "1h", Side.LONG, SetupKind.BREAKOUT,
                      100, 95, 0.7, 1.0, 0.6, 0.5, ["a"]),
    ])["BTC/USDT:USDT"]
    tid = store.record_trade(scan_id, sid, TradeHistory(
        "BTC/USDT:USDT", Side.LONG, 100, 95, 1, 2, 50))
    store.update_position(tid, "BTC/USDT:USDT", Side.LONG, "2024-01-01T00:00:00+00:00",
                          "2024-01-01T01:00:00+00:00", "closed", 5, "tp", 1.0, 0.01)

    summary = store.get_scan_summary()
    assert summary["total_scans"] == 1
    assert summary["total_signals"] == 1
    assert summary["total_trades"] == 1
    assert summary["closed_trades"] == 1
    assert summary["win_rate"] == 1.0
    assert summary["avg_realized_r"] == pytest.approx(1.0)
    assert len(summary["by_month"]) == 1
    assert len(summary["by_regime"]) == 1
    assert len(summary["by_setup_kind"]) == 1
