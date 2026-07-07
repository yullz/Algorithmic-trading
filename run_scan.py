#!/usr/bin/env python3
"""Live signal scan: fetch data for every configured symbol/timeframe, generate
trade plans, and print the actionable ones.

Usage:
    python run_scan.py                 # scan live data (needs internet, no keys)
    python run_scan.py --offline       # use synthetic data (fully offline demo)
    python run_scan.py --symbol BTC/USDT:USDT --tf 1h
"""
from __future__ import annotations

import argparse
import zlib

from algotrader.backtest.engine import Backtester  # noqa: F401 (re-export convenience)
from algotrader.config import load_config
from algotrader.data.feed import DataFeed
from algotrader.risk.manager import RiskManager
from algotrader.signals.engine import SignalEngine
from algotrader.utils.logging import audit, get_logger
from algotrader.utils.render import render_plan

log = get_logger("scan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use synthetic data")
    ap.add_argument("--symbol", help="scan a single symbol")
    ap.add_argument("--tf", help="scan a single timeframe")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    feed = DataFeed(cfg.exchange_id, cfg.market_type, cfg.api_key, cfg.api_secret)
    meta_model = None
    if cfg.ml_enabled:
        from algotrader.ml import MetaModel
        meta_model = MetaModel.load(cfg.ml_model_path, cfg.ml_min_training_trades)
    engine = SignalEngine(cfg.min_confidence, cfg.min_confluence, cfg.calibration,
                          min_families=cfg.min_families, htf_veto=cfg.htf_veto,
                          regime_gating=cfg.regime_gating,
                          max_stop_atr_mult=cfg.risk.max_stop_atr_mult,
                          meta_model=meta_model,
                          calibration_path=cfg.calibration_file)
    risk = RiskManager(cfg.risk, cfg.calibration)

    symbols = [args.symbol] if args.symbol else cfg.symbols
    tfs = [args.tf] if args.tf else cfg.timeframes

    if not cfg.calibration:
        log.warning("no calibration.json found - win rates are UNCALIBRATED priors. "
                    "Run `python backtest.py` first for measured rates.")

    found = 0
    for symbol in symbols:
        for tf in tfs:
            try:
                if args.offline:
                    seed = zlib.crc32((symbol + tf).encode()) % 1000
                    df = DataFeed.synthetic(cfg.lookback_candles, seed=seed)
                    htf = None
                else:
                    df = feed.fetch_ohlcv(symbol, tf, cfg.lookback_candles)
                    htf = feed.fetch_ohlcv(symbol, cfg.context_timeframe, cfg.lookback_candles)
            except Exception as e:
                log.error("data fetch failed for %s %s: %s", symbol, tf, e)
                continue

            sig = engine.generate(df, symbol, tf, htf=htf)
            if sig is None:
                continue
            plan = risk.build_plan(sig)
            if plan is None or not plan.is_actionable(cfg.risk):
                continue
            render_plan(plan)
            audit("signal", {"symbol": symbol, "tf": tf, "side": plan.side.value,
                             "entry": plan.entry, "stop": plan.stop_loss,
                             "wr": plan.expected_win_rate, "ev_r": plan.expected_value_r})
            found += 1

    if found == 0:
        log.info("No actionable setups right now (this is normal and healthy - "
                 "most bars have no edge).")


if __name__ == "__main__":
    main()
