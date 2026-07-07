#!/usr/bin/env python3
"""Paper trading loop: continuously scans the whole universe, opens simulated
positions through the same executor interface live trading uses, tracks them
candle-by-candle with mark-to-market, and persists durable state — with ZERO
real orders.

This is the mandatory step between backtesting and any live consideration.

Usage:
    python paper_trade.py                    # scan on the configured cadence
    python paper_trade.py --interval 60      # override cadence (seconds)
    python paper_trade.py --offline          # replay synthetic data quickly
"""
from __future__ import annotations

import argparse
import asyncio
import zlib

from algotrader.config import load_config
from algotrader.data.feed import DataFeed
from algotrader.execution.paper import PaperExecutor
from algotrader.reporting import write_json
from algotrader.risk.manager import RiskManager
from algotrader.scanner import Scanner
from algotrader.signals.engine import SignalEngine
from algotrader.utils.logging import get_logger
from algotrader.utils.render import render_plan

log = get_logger("paper")


def build_engine(cfg) -> SignalEngine:
    meta_model = None
    if cfg.ml_enabled:
        from algotrader.ml import MetaModel
        meta_model = MetaModel.load(cfg.ml_model_path, cfg.ml_min_training_trades)
    return SignalEngine(cfg.min_confidence, cfg.min_confluence, cfg.calibration,
                        min_families=cfg.min_families, htf_veto=cfg.htf_veto,
                        regime_gating=cfg.regime_gating,
                        max_stop_atr_mult=cfg.risk.max_stop_atr_mult,
                        meta_model=meta_model,
                        calibration_path=cfg.calibration_file)


# --------------------------------------------------------------------------- #
async def live_loop(cfg, interval: int) -> None:
    from algotrader.data.feed import AsyncDataFeed
    feed = AsyncDataFeed(cfg.exchange_id, cfg.market_type, cfg.api_key,
                         cfg.api_secret, cache_dir=cfg.cache_dir,
                         concurrency=cfg.scan_concurrency)
    engine = build_engine(cfg)
    risk = RiskManager(cfg.risk, cfg.calibration)
    scanner = Scanner(cfg, engine, risk, feed)
    executor = PaperExecutor(cfg.risk)
    seen_candle: dict[str, str] = {}   # position id -> last processed candle ts

    log.info("paper trading started: universe=%s equity=%.2f interval=%ds "
             "(Ctrl-C to stop)", cfg.universe_mode, executor.mtm_equity(), interval)
    try:
        while True:
            result = await scanner.scan()
            plans = result.pop("plan_objects", [])
            write_json("reports/last_scan.json", result)

            # 1) advance open positions on their newest CLOSED candle
            for pos in list(executor.open_positions()):
                try:
                    df = await feed.fetch_ohlcv(pos.symbol, pos.timeframe, 3)
                except Exception as e:
                    log.error("candle refresh failed %s: %s", pos.symbol, e)
                    continue
                if df is None or len(df) < 2:
                    continue
                bar, ts = df.iloc[-2], str(df.index[-2])
                if seen_candle.get(pos.id) == ts:
                    continue
                seen_candle[pos.id] = ts
                executor.update_with_candle(pos.symbol, ts, float(bar["open"]),
                                            float(bar["high"]), float(bar["low"]),
                                            float(bar["close"]))

            # 2) open the scanner's ranked picks (executor re-checks all caps)
            for plan in plans:
                if executor.open_position(plan):
                    render_plan(plan)

            executor.save()
            await asyncio.sleep(interval)
    finally:
        await feed.close()


# --------------------------------------------------------------------------- #
def offline_replay(cfg) -> None:
    """Bar-stepped replay over synthetic series — fast, deterministic, zero
    network. Exercises the entire scan->size->execute->track pipeline."""
    engine = build_engine(cfg)
    risk = RiskManager(cfg.risk, cfg.calibration)
    executor = PaperExecutor(cfg.risk, state_path="reports/paper_state.json")
    symbols = cfg.symbols
    series = {s: DataFeed.synthetic(600, seed=zlib.crc32(s.encode()) % 1000)
              for s in symbols}
    warmup = 250
    for step in range(warmup, 600):
        for symbol in symbols:
            df = series[symbol].iloc[: step + 1]
            bar = df.iloc[-1]
            executor.update_with_candle(symbol, str(df.index[-1]),
                                        float(bar["open"]), float(bar["high"]),
                                        float(bar["low"]), float(bar["close"]))
            if any(p.symbol == symbol for p in executor.open_positions()):
                continue
            sig = engine.generate(df, symbol, cfg.timeframes[0])
            if sig is None:
                continue
            plan = risk.build_plan(sig)
            if plan is None or not plan.is_actionable(cfg.risk):
                continue
            executor.open_position(plan)
    executor.save()
    log.info("offline replay done: equity=%.2f (mtm %.2f), %d closed trades",
             executor.equity(), executor.mtm_equity(), len(executor.closed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=0,
                    help="scan cadence seconds (default: scanner.interval_sec)")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.offline:
        offline_replay(cfg)
        return
    interval = args.interval or cfg.scan_interval_sec
    try:
        asyncio.run(live_loop(cfg, interval))
    except KeyboardInterrupt:
        log.info("paper trading stopped.")


if __name__ == "__main__":
    main()
