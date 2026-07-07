"""Tests for portfolio exposure analysis."""
from __future__ import annotations

import pytest

from algotrader.execution.base import PositionState
from algotrader.models import Side
from algotrader.portfolio import ExposureAnalyzer


def _pos(symbol: str, side: Side, qty: float, entry: float, margin: float,
         last: float = 0.0) -> PositionState:
    return PositionState(
        id="id1", symbol=symbol, timeframe="1h", side=side,
        entry=entry, qty_initial=qty, qty_open=qty, stop=entry * 0.95,
        leverage=entry * qty / margin if margin else 1.0,
        margin=margin, last_price=last if last else entry,
    )


def test_by_sector_groups_known_tokens():
    positions = [
        _pos("BTC/USDT:USDT", Side.LONG, 1.0, 100_000.0, 20_000.0),
        _pos("ETH/USDT:USDT", Side.LONG, 2.0, 5_000.0, 2_000.0),
        _pos("SOL/USDT:USDT", Side.SHORT, 10.0, 200.0, 400.0),
        _pos("AVAX/USDT:USDT", Side.SHORT, 5.0, 100.0, 100.0),
    ]
    sectors = {s["name"]: s for s in ExposureAnalyzer.by_sector(positions)}
    assert sectors["Layer1"]["count"] == 2
    assert sectors["Layer1"]["long_notional"] == 110_000.0
    assert sectors["Alt-L1"]["count"] == 2
    assert sectors["Alt-L1"]["short_notional"] == 2_500.0


def test_by_sector_unknown_becomes_other():
    positions = [_pos("DOGE/USDT:USDT", Side.LONG, 1_000.0, 0.2, 40.0)]
    sectors = ExposureAnalyzer.by_sector(positions)
    assert len(sectors) == 1
    assert sectors[0]["name"] == "Meme"


def test_by_sector_metadata_override():
    positions = [_pos("XYZ/USDT:USDT", Side.LONG, 1.0, 100.0, 20.0)]
    meta = {"XYZ/USDT:USDT": {"sector": "Custom"}}
    sectors = ExposureAnalyzer.by_sector(positions, meta)
    assert sectors[0]["name"] == "Custom"


def test_by_correlation_bucket_default_low_when_missing():
    positions = [
        _pos("BTC/USDT:USDT", Side.LONG, 1.0, 100_000.0, 20_000.0),
        _pos("SOL/USDT:USDT", Side.SHORT, 1.0, 200.0, 40.0),
    ]
    buckets = {b["name"]: b for b in ExposureAnalyzer.by_correlation_bucket(positions)}
    assert buckets["low"]["count"] == 2
    assert buckets["medium"]["count"] == 0
    assert buckets["high"]["count"] == 0


def test_by_correlation_bucket_respects_matrix():
    positions = [
        _pos("BTC/USDT:USDT", Side.LONG, 1.0, 100_000.0, 20_000.0),
        _pos("ETH/USDT:USDT", Side.LONG, 1.0, 5_000.0, 1_000.0),
        _pos("SOL/USDT:USDT", Side.SHORT, 1.0, 200.0, 40.0),
        _pos("DOGE/USDT:USDT", Side.SHORT, 1.0, 0.2, 0.04),
    ]
    corr = {
        "BTC/USDT:USDT": {"BTC/USDT:USDT": 1.0},
        "ETH/USDT:USDT": {"BTC/USDT:USDT": 0.85},
        "SOL/USDT:USDT": {"BTC/USDT:USDT": 0.55},
        "DOGE/USDT:USDT": {"BTC/USDT:USDT": 0.25},
    }
    buckets = {b["name"]: b for b in ExposureAnalyzer.by_correlation_bucket(positions, corr)}
    assert buckets["high"]["count"] == 2  # BTC + ETH
    assert buckets["medium"]["count"] == 1  # SOL
    assert buckets["low"]["count"] == 1  # DOGE
    assert buckets["high"]["long_notional"] == 105_000.0
    assert buckets["medium"]["short_notional"] == 200.0


def test_by_correlation_bucket_symmetric_lookup():
    """Matrix may be indexed by BTC -> symbol instead of symbol -> BTC."""
    positions = [_pos("ETH/USDT:USDT", Side.LONG, 1.0, 5_000.0, 1_000.0)]
    corr = {"BTC/USDT:USDT": {"ETH/USDT:USDT": 0.9}}
    buckets = {b["name"]: b for b in ExposureAnalyzer.by_correlation_bucket(positions, corr)}
    assert buckets["high"]["count"] == 1


def test_by_side_sums_long_and_short():
    positions = [
        _pos("BTC/USDT:USDT", Side.LONG, 1.0, 100_000.0, 20_000.0),
        _pos("ETH/USDT:USDT", Side.LONG, 1.0, 5_000.0, 1_000.0),
        _pos("SOL/USDT:USDT", Side.SHORT, 10.0, 200.0, 400.0),
    ]
    sides = ExposureAnalyzer.by_side(positions)
    assert sides["long"]["notional"] == 105_000.0
    assert sides["long"]["margin"] == 21_000.0
    assert sides["long"]["count"] == 2
    assert sides["short"]["notional"] == 2_000.0
    assert sides["short"]["margin"] == 400.0
    assert sides["short"]["count"] == 1
    assert sides["net"] == 103_000.0
    assert sides["gross"] == 107_000.0


def test_by_side_uses_last_price_when_available():
    positions = [
        _pos("BTC/USDT:USDT", Side.LONG, 1.0, 100_000.0, 20_000.0, last=110_000.0),
    ]
    sides = ExposureAnalyzer.by_side(positions)
    assert sides["long"]["notional"] == 110_000.0


def test_analyze_returns_full_snapshot():
    positions = [
        _pos("BTC/USDT:USDT", Side.LONG, 1.0, 100_000.0, 20_000.0),
        _pos("SOL/USDT:USDT", Side.SHORT, 10.0, 200.0, 400.0),
    ]
    snapshot = ExposureAnalyzer.analyze(positions)
    assert "sectors" in snapshot
    assert "correlation_buckets" in snapshot
    assert "sides" in snapshot
    assert len(snapshot["correlation_buckets"]) == 3


def test_empty_positions_returns_zeroed_shape():
    snapshot = ExposureAnalyzer.analyze([])
    assert snapshot["sides"]["long"]["notional"] == 0.0
    assert snapshot["sides"]["short"]["notional"] == 0.0
    assert len(snapshot["sectors"]) == 0
    assert len(snapshot["correlation_buckets"]) == 3
    for b in snapshot["correlation_buckets"]:
        assert b["count"] == 0
        assert b["net_notional"] == 0.0
