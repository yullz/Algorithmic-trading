"""Configuration loading: YAML + environment, with a hard live-trading guard."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import yaml
from dotenv import load_dotenv

from .models import RiskConfig
from .utils.logging import get_logger

load_dotenv()

log = get_logger(__name__)


@dataclass
class AppConfig:
    raw: dict[str, Any]
    risk: RiskConfig
    exchange_id: str
    market_type: str
    symbols: list[str]
    timeframes: list[str]
    context_timeframe: str
    lookback_candles: int
    min_confidence: float
    min_confluence: int
    validity_candles: int
    calibration_file: str
    # signal quality gates
    min_families: int = 3
    htf_veto: bool = False
    regime_gating: bool = True
    # calibration governance (see backtest.py / winrate.py)
    calibration_half_life_days: float = 45.0    # recency decay half-life (calendar days)
    calibration_min_wilson_lower: float = 0.35  # drop factors whose OOS Wilson-lower is below this
    label_horizon_candles: int = 48             # bars a trade is simulated/held; also the live time-stop
    # universe selection
    universe_mode: str = "static"        # "static" (symbols list) | "top_volume"
    universe_size: int = 150
    # scanner
    scan_interval_sec: int = 300
    scan_concurrency: int = 8
    cache_dir: str = "data_cache"
    # ML meta-model
    ml_enabled: bool = True
    ml_model_path: str = "models/meta_model.pkl"
    ml_min_training_trades: int = 300
    # streaming (ccxt.pro live ticks for the dashboard; default OFF — the REST
    # feed remains the source of truth for signals regardless)
    streaming_enabled: bool = False
    # execution
    execution_mode: str = "paper"        # "paper" | "live"
    execution_testnet: bool = True       # live orders go to testnet unless BOTH
    allow_mainnet: bool = False          # testnet=false AND allow_mainnet=true
    # secrets / switches (never printed)
    api_key: str = field(default="", repr=False)
    api_secret: str = field(default="", repr=False)
    allow_live: bool = False

    _calibration_cache: dict | None = field(default=None, repr=False)
    _calibration_mtime: float = field(default=-1.0, repr=False)

    @property
    def calibration(self) -> dict[str, float]:
        """calibration.json, cached by mtime (this is read in hot loops)."""
        try:
            mtime = os.path.getmtime(self.calibration_file)
            if self._calibration_cache is None or mtime != self._calibration_mtime:
                with open(self.calibration_file, encoding="utf-8") as f:
                    object.__setattr__(self, "_calibration_cache", json.load(f))
                object.__setattr__(self, "_calibration_mtime", mtime)
            return self._calibration_cache or {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}


def _risk_from(d: dict[str, Any]) -> RiskConfig:
    r = RiskConfig()
    for k, v in (d or {}).items():
        if hasattr(r, k):
            # tuples for TP schedules
            if k in ("tp_r_multiples", "tp_allocations") and isinstance(v, list):
                v = tuple(v)
            setattr(r, k, v)
        else:
            # A typoed risk parameter silently falling back to defaults is
            # dangerous — say so loudly.
            log.warning("Unknown risk config key %r ignored (typo?)", k)
    return r


def load_config(path: str = "config.yaml") -> AppConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    exch = raw.get("exchange", {})
    uni = raw.get("universe", {})
    sig = raw.get("signals", {})
    cal = raw.get("calibration", {})
    scan = raw.get("scanner", {})
    ml = raw.get("ml", {})
    execu = raw.get("execution", {})

    allow_live = os.getenv("ALLOW_LIVE_TRADING", "false").strip().lower() == "true"

    risk_cfg = _risk_from(raw.get("risk", {}))
    label_horizon = int(cal.get("label_horizon_candles", 48))
    # Single knob for the holding/label horizon: the backtest simulates and the
    # live executor time-stops a trade after this many bars, so the calibrated
    # label ("profitable within N bars") matches what is actually traded. An
    # explicit risk.max_trade_duration_candles > 0 still wins.
    if risk_cfg.max_trade_duration_candles <= 0:
        risk_cfg.max_trade_duration_candles = label_horizon

    return AppConfig(
        raw=raw,
        risk=risk_cfg,
        exchange_id=os.getenv("EXCHANGE", exch.get("id", "bybit")),
        market_type=exch.get("market_type", "swap"),
        symbols=uni.get("symbols", ["BTC/USDT:USDT"]),
        timeframes=uni.get("timeframes", ["1h"]),
        context_timeframe=uni.get("context_timeframe", "4h"),
        lookback_candles=int(uni.get("lookback_candles", 500)),
        min_confidence=float(sig.get("min_confidence", 0.55)),
        min_confluence=int(sig.get("min_confluence", 3)),
        validity_candles=int(sig.get("validity_candles", 3)),
        calibration_file=cal.get("file", "calibration.json"),
        min_families=int(sig.get("min_families", 3)),
        htf_veto=bool(sig.get("htf_veto", False)),
        regime_gating=bool(sig.get("regime_gating", True)),
        calibration_half_life_days=float(cal.get("half_life_days", 45.0)),
        calibration_min_wilson_lower=float(cal.get("min_wilson_lower", 0.35)),
        label_horizon_candles=label_horizon,
        universe_mode=uni.get("mode", "static"),
        universe_size=int(uni.get("size", 150)),
        scan_interval_sec=int(scan.get("interval_sec", 300)),
        scan_concurrency=int(scan.get("concurrency", 8)),
        cache_dir=scan.get("cache_dir", "data_cache"),
        ml_enabled=bool(ml.get("enabled", True)),
        ml_model_path=ml.get("model_path", "models/meta_model.pkl"),
        ml_min_training_trades=int(ml.get("min_training_trades", 300)),
        streaming_enabled=bool(raw.get("streaming", {}).get("enabled", False)),
        execution_mode=execu.get("mode", "paper"),
        execution_testnet=bool(execu.get("testnet", True)),
        allow_mainnet=bool(execu.get("allow_mainnet", False)),
        api_key=os.getenv("API_KEY", ""),
        api_secret=os.getenv("API_SECRET", ""),
        allow_live=allow_live,
    )
