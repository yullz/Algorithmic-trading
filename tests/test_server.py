"""Tests for the dashboard REST API and safety middleware."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

import server.main as main


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Yield a TestClient with the background scan_loop disabled."""
    # Prevent the lifespan from starting a real scan loop.
    monkeypatch.setattr(main, "scan_loop", AsyncMock())
    # Redirect history DB to a temp file so tests are isolated.
    monkeypatch.setattr(main.history, "db_path", str(tmp_path / "history_test.db"))
    main.history._init_db()

    from fastapi.testclient import TestClient
    # Use localhost base URL so the localhost-only middleware lets requests through.
    with TestClient(main.app, base_url="http://localhost") as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "last_scan_age_sec" in data
    assert "data_fresh" in data
    assert "calibration_stale" in data
    assert "model_trusted" in data
    assert "circuit_breakers" in data


def test_localhost_middleware_blocks_remote_host(client):
    resp = client.get("/api/health", headers={"host": "evil.com"})
    assert resp.status_code == 403
    assert resp.json()["error"] == "localhost only"


def test_pause_resume(client):
    assert client.post("/api/control/pause").json()["paused"] is True
    assert main._scan_paused.is_set() is False

    assert client.post("/api/control/resume").json()["paused"] is False
    assert main._scan_paused.is_set() is True


def test_history_endpoints_empty(client):
    assert client.get("/api/history/signals").json()["signals"] == []
    assert client.get("/api/history/trades").json()["trades"] == []
    summary = client.get("/api/history/summary").json()
    assert summary["total_scans"] == 0
    assert summary["total_signals"] == 0
    assert summary["total_trades"] == 0


def test_history_signals_filter(client):
    from algotrader.history import SignalHistory
    from algotrader.models import SetupKind, Side

    scan_id = main.history.record_scan("2024-01-01T00:00:00+00:00", "up", 1, 1)
    main.history.record_signals(scan_id, [
        SignalHistory("BTC/USDT:USDT", "1h", Side.LONG, SetupKind.BREAKOUT,
                      100, 95, 0.7, 1.0, 0.6, 0.5, ["a"]),
        SignalHistory("ETH/USDT:USDT", "4h", Side.SHORT, SetupKind.MOMENTUM,
                      200, 210, 0.6, -1.0, 0.5, 0.3, ["b"]),
    ])

    resp = client.get("/api/history/signals?symbol=BTC/USDT:USDT")
    assert resp.status_code == 200
    assert len(resp.json()["signals"]) == 1
    assert resp.json()["signals"][0]["symbol"] == "BTC/USDT:USDT"


def test_close_position_endpoint(client, monkeypatch, tmp_path):
    from algotrader.execution.paper import PaperExecutor
    from algotrader.models import RiskConfig, Side, TakeProfit, TradePlan

    # Use a fresh executor for this test.
    state_path = tmp_path / "paper_state_test.json"
    exe = PaperExecutor(RiskConfig(), state_path=str(state_path))
    monkeypatch.setattr(main, "executor", exe)

    plan = TradePlan(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        side=Side.LONG,
        entry=100.0,
        stop_loss=95.0,
        take_profits=[TakeProfit(105.0, 1.0, 1.0)],
        leverage=2.0,
        qty=1.0,
        notional=100.0,
        margin=50.0,
        risk_amount=5.0,
        liquidation_price=50.0,
        reward_risk=1.0,
        expected_win_rate=0.5,
        expected_value_r=0.1,
        confidence=0.6,
        fees_estimate=0.0,
    )
    pos_id = exe.open_position(plan)
    assert pos_id is not None

    resp = client.post("/api/positions/BTC/USDT:USDT/close")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "closed"
    assert data["symbol"] == "BTC/USDT:USDT"
    assert not exe.open_positions()


def test_close_position_not_found(client):
    resp = client.post("/api/positions/NOSUCH/USDT/close")
    assert resp.status_code == 404


def test_trade_links_persist_across_restart(tmp_path, monkeypatch):
    """The pos_id -> trade_id linkage must survive a restart, or a restored
    position's close is never journaled and its trade row stays 'open' forever."""
    monkeypatch.setattr(main, "_TRADE_LINKS_PATH", str(tmp_path / "links.json"))
    orig = (dict(main._trade_id_by_pos), dict(main._risk_amount_by_trade),
            dict(main._fees_estimate_by_trade), set(main._logged_positions),
            set(main._closed_trade_ids))
    try:
        main._trade_id_by_pos.clear(); main._trade_id_by_pos["posABC"] = 42
        main._risk_amount_by_trade.clear(); main._risk_amount_by_trade[42] = 100.0
        main._fees_estimate_by_trade.clear(); main._fees_estimate_by_trade[42] = 1.5
        main._logged_positions.clear(); main._logged_positions.add("posABC")
        main._save_trade_links()

        # Simulate a restart: wipe the in-memory maps, reload from disk.
        main._trade_id_by_pos.clear(); main._risk_amount_by_trade.clear()
        main._fees_estimate_by_trade.clear(); main._logged_positions.clear()
        main._load_trade_links()

        assert main._trade_id_by_pos == {"posABC": 42}
        assert main._risk_amount_by_trade == {42: 100.0}   # int key restored (JSON quirk)
        assert main._fees_estimate_by_trade == {42: 1.5}
        assert "posABC" in main._logged_positions
    finally:
        for target, saved in zip(
                (main._trade_id_by_pos, main._risk_amount_by_trade,
                 main._fees_estimate_by_trade, main._logged_positions,
                 main._closed_trade_ids), orig):
            target.clear()
            target.update(saved)
