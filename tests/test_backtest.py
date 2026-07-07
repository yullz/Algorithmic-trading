"""Tests for the backtest runner and engine aggregation."""
from __future__ import annotations

import pytest

from backtest import _build_parser, _resolve_symbols, _resolve_timeframes
from algotrader.backtest.engine import Backtester
from algotrader.config import AppConfig
from algotrader.risk.manager import RiskManager
from algotrader.signals.engine import SignalEngine


def _make_cfg(symbols=None, timeframes=None) -> AppConfig:
    return AppConfig(
        raw={},
        risk=type("RiskConfig", (), {})(),  # minimal stub; not used by resolver
        exchange_id="bybit",
        market_type="swap",
        symbols=symbols or ["BTC/USDT:USDT"],
        timeframes=timeframes or ["1h"],
        context_timeframe="4h",
        lookback_candles=500,
        min_confidence=0.6,
        min_confluence=3,
        validity_candles=3,
        calibration_file="calibration.json",
    )


def test_parser_defaults():
    ap = _build_parser()
    args = ap.parse_args([])
    assert args.offline is False
    assert args.symbol is None
    assert args.symbols is None
    assert args.tf == "1h"
    assert args.timeframes is None
    assert args.limit == 3000
    assert args.walkforward is False
    assert args.folds == 4
    assert args.export_dataset is False
    assert args.deep is False
    assert args.workers == 4


def test_parser_deep_and_multi_symbol():
    ap = _build_parser()
    args = ap.parse_args([
        "--offline", "--deep", "--workers", "8",
        "--symbols", "BTC/USDT:USDT,ETH/USDT:USDT",
        "--timeframes", "15m,1h,4h",
        "--export-dataset",
    ])
    assert args.offline is True
    assert args.deep is True
    assert args.workers == 8
    assert args.symbols == "BTC/USDT:USDT,ETH/USDT:USDT"
    assert args.timeframes == "15m,1h,4h"
    assert args.export_dataset is True


def test_parser_legacy_symbol_tf():
    ap = _build_parser()
    args = ap.parse_args(["--symbol", "ETH/USDT:USDT", "--tf", "4h"])
    assert args.symbol == "ETH/USDT:USDT"
    assert args.tf == "4h"


def test_resolve_timeframes():
    cfg = _make_cfg(timeframes=["15m", "1h", "4h"])
    ap = _build_parser()

    args = ap.parse_args(["--timeframes", "1h,4h"])
    assert _resolve_timeframes(cfg, args) == ["1h", "4h"]

    args = ap.parse_args(["--deep"])
    assert _resolve_timeframes(cfg, args) == ["15m", "1h", "4h"]

    args = ap.parse_args(["--tf", "1d"])
    assert _resolve_timeframes(cfg, args) == ["1d"]


def test_resolve_symbols_offline():
    cfg = _make_cfg(symbols=["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"])
    ap = _build_parser()
    engine = SignalEngine()
    risk = RiskManager(cfg.risk)

    args = ap.parse_args(["--offline", "--symbols", "BTC/USDT:USDT,ETH/USDT:USDT"])
    assert _resolve_symbols(cfg, args, engine, risk) == ["BTC/USDT:USDT", "ETH/USDT:USDT"]

    args = ap.parse_args(["--offline", "--symbol", "SOL/USDT:USDT"])
    assert _resolve_symbols(cfg, args, engine, risk) == ["SOL/USDT:USDT"]

    args = ap.parse_args(["--offline", "--symbols", "all"])
    assert _resolve_symbols(cfg, args, engine, risk) == cfg.symbols


def test_aggregate_empty_trades():
    bt = Backtester(SignalEngine(), RiskManager(type("RiskConfig", (), {})()))
    res = bt._aggregate([])
    assert res.trades == []
    assert res.summary == {"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0}


def test_aggregate_computes_summary():
    bt = Backtester(SignalEngine(), RiskManager(type("RiskConfig", (), {})()))
    trades = [
        {"r": 1.0, "win": True, "factors": ["a"], "kind": "reversal",
         "symbol": "BTC/USDT:USDT", "tf": "1h"},
        {"r": 1.0, "win": True, "factors": ["a", "b"], "kind": "reversal",
         "symbol": "BTC/USDT:USDT", "tf": "1h"},
        {"r": -1.0, "win": False, "factors": ["b"], "kind": "breakout",
         "symbol": "ETH/USDT:USDT", "tf": "1h"},
        {"r": 2.0, "win": True, "factors": ["c"], "kind": "breakout",
         "symbol": "ETH/USDT:USDT", "tf": "1h"},
    ]
    res = bt._aggregate(trades)
    assert res.summary["trades"] == 4
    assert res.summary["win_rate"] == 0.75
    assert res.summary["expectancy_r"] == 0.75
    assert res.factor_win_rate["a"] == 1.0
    assert res.factor_win_rate["b"] == 0.5
    assert res.kind_win_rate["reversal"] == 1.0
    assert res.kind_win_rate["breakout"] == 0.5


def test_aggregate_equity_and_drawdown():
    bt = Backtester(SignalEngine(), RiskManager(type("RiskConfig", (), {})()))
    trades = [
        {"r": 1.0, "win": True, "factors": ["a"], "kind": "reversal",
         "symbol": "BTC/USDT:USDT", "tf": "1h"},
        {"r": 1.0, "win": True, "factors": ["a"], "kind": "reversal",
         "symbol": "BTC/USDT:USDT", "tf": "1h"},
        {"r": -2.0, "win": False, "factors": ["a"], "kind": "reversal",
         "symbol": "BTC/USDT:USDT", "tf": "1h"},
    ]
    res = bt._aggregate(trades)
    assert res.equity_curve == [0.0, 1.0, 2.0, 0.0]
    assert res.summary["max_drawdown_r"] == 2.0


def test_to_dataset_has_symbol_and_tf_columns():
    bt = Backtester(SignalEngine(), RiskManager(type("RiskConfig", (), {})()))
    trades = [
        {"r": 1.0, "win": True, "factors": ["a"], "kind": "reversal",
         "symbol": "BTC/USDT:USDT", "tf": "1h", "side": "LONG",
         "entry_time": "2024-01-01", "confidence": 0.7, "score": 1.2,
         "n_families": 2, "rule_win_rate": 0.6, "stop_pct": 0.01,
         "factor_strengths": {"a": 0.8}, "regime": "trend"},
        {"r": -1.0, "win": False, "factors": ["b"], "kind": "breakout",
         "symbol": "ETH/USDT:USDT", "tf": "4h", "side": "SHORT",
         "entry_time": "2024-01-02", "confidence": 0.65, "score": 1.0,
         "n_families": 1, "rule_win_rate": 0.5, "stop_pct": 0.02,
         "factor_strengths": {"b": 0.6}, "regime": "range"},
    ]
    res = bt._aggregate(trades)
    ds = res.to_dataset()
    assert list(ds["symbol"]) == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert list(ds["tf"]) == ["1h", "4h"]
    assert "factor__a" in ds.columns
    assert "factor__b" in ds.columns
