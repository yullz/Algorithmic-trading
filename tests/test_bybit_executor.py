"""Bybit live executor: edge safety catch + free-margin preflight.

The exchange client is mocked, so these run offline and never touch the network.
They lock two money-critical rules:
  1. Never open a live position unless a positive OUT-OF-SAMPLE edge is on record
     (the edge safety catch), and
  2. never open one unless enough FREE USDT is available to fund its margin.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from algotrader.models import Bias, Evidence, RiskConfig, SetupKind, Side, Signal
from algotrader.risk.manager import RiskManager


def _plan(margin_usdt: float = 15.0):
    cfg = RiskConfig(fixed_margin_usdt=margin_usdt, default_leverage=3.0)
    sig = Signal(
        symbol="BTC/USDT:USDT", timeframe="1h", side=Side.LONG, kind=SetupKind.BREAKOUT,
        entry_ref=100.0, stop_ref=99.0, confidence=0.6, score=1.0,
        evidence=[Evidence("trend_up", "indicator", Bias.BULLISH, 0.7, 0.55, family="trend")],
        base_win_rate=0.55)
    return RiskManager(cfg).build_plan(sig)


def _mk_exec(tmp_path, free, total=None, positions=None, require_edge=False):
    total = free if total is None else total
    with patch("ccxt.bybit") as mk:
        ex = MagicMock()
        ex.fetch_balance.return_value = {"USDT": {"free": free, "total": total}}
        ex.fetch_positions.return_value = positions or []
        mk.return_value = ex
        from algotrader.execution.bybit import BybitExecutor
        exe = BybitExecutor(
            RiskConfig(fixed_margin_usdt=15.0, require_validated_edge=require_edge),
            "k", "s", testnet=True, root=str(tmp_path))
    return exe, ex


def _write_wf(tmp_path, exp, pf):
    os.makedirs(os.path.join(tmp_path, "reports"), exist_ok=True)
    with open(os.path.join(tmp_path, "reports", "walkforward.json"), "w") as f:
        json.dump({"out_of_sample": {"expectancy_r": exp, "profit_factor": pf}}, f)


def _stale_ohlcv(ex):
    old_ms = int((datetime.now(timezone.utc).timestamp() - 10 * 86400) * 1000)
    ex.fetch_ohlcv.return_value = [[old_ms - 3600_000, 1, 1, 1, 1, 1],
                                   [old_ms, 1, 1, 1, 1, 1]]


# --------------------------------------------------------------------------- #
# free-margin preflight (edge catch disabled)
# --------------------------------------------------------------------------- #
def test_free_usdt_reads_free_not_total(tmp_path):
    exe, _ = _mk_exec(tmp_path, free=42.5, total=99.0)
    assert exe.free_usdt() == pytest.approx(42.5)
    assert exe.equity() == pytest.approx(99.0)   # equity still reads total


def test_open_position_skips_when_free_below_margin(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=10.0, total=1000.0, positions=[])
    assert exe.open_position(_plan(15.0)) is None
    ex.create_order.assert_not_called()          # no order ever placed


def test_open_position_passes_preflight_when_free_sufficient(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0, total=1000.0, positions=[])
    _stale_ohlcv(ex)                             # stale -> fail safe AFTER preflight
    assert exe.open_position(_plan(15.0)) is None
    ex.fetch_ohlcv.assert_called()               # preflight was passed
    ex.create_order.assert_not_called()


# --------------------------------------------------------------------------- #
# edge safety catch (live-only, default ON)
# --------------------------------------------------------------------------- #
def test_edge_gate_blocks_when_no_walkforward(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=1000.0, total=1000.0, require_edge=True)
    assert exe.open_position(_plan(15.0)) is None
    ex.create_order.assert_not_called()
    ex.fetch_ohlcv.assert_not_called()           # blocked before the freshness check


def test_edge_gate_blocks_on_negative_oos(tmp_path):
    _write_wf(tmp_path, exp=-0.07, pf=0.9)       # the real deep-data picture
    exe, ex = _mk_exec(tmp_path, free=1000.0, total=1000.0, require_edge=True)
    assert exe.open_position(_plan(15.0)) is None
    ex.create_order.assert_not_called()


def test_edge_gate_passes_on_positive_oos(tmp_path):
    _write_wf(tmp_path, exp=0.2, pf=1.5)         # a validated positive OOS edge
    exe, ex = _mk_exec(tmp_path, free=1000.0, total=1000.0, require_edge=True)
    _stale_ohlcv(ex)
    assert exe.open_position(_plan(15.0)) is None  # stale -> fail safe
    ex.fetch_ohlcv.assert_called()               # but it cleared the edge gate + preflight
