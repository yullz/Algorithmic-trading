#!/usr/bin/env python3
"""Collect forward order-flow snapshots (funding / OI / basis) for later alpha
research. Builds the historical derivatives dataset the backtest pipeline lacks.

    python collect_orderflow.py                        # one snapshot, then exit
    python collect_orderflow.py --loop --interval 3600 # every hour (leave running)

Schedule the --loop form (Task Scheduler / cron / a spare terminal). Over weeks
this accumulates data_cache/orderflow/*.parquet, which you then test for a real,
validation-surviving edge (funding z-score, OI-vs-price divergence, basis).
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from algotrader.config import load_config
from algotrader.data.orderflow import append_snapshot, fetch_orderflow_snapshot
from algotrader.utils.logging import get_logger

log = get_logger("orderflow")


def _exchange():
    import ccxt
    return ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})


def collect_once(ex, cache_dir: str = "data_cache") -> int:
    ts = datetime.now(timezone.utc).isoformat()
    rows = fetch_orderflow_snapshot(ex, ts)
    path = append_snapshot(rows, cache_dir)
    n_funded = sum(1 for r in rows if r.get("funding_rate"))
    log.info("collected %d symbols (%d with funding) -> %s", len(rows), n_funded, path)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--loop", action="store_true", help="collect forever")
    ap.add_argument("--interval", type=int, default=3600,
                    help="seconds between snapshots in --loop mode (default 3600)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ex = _exchange()
    if not args.loop:
        collect_once(ex, cfg.cache_dir)
        return
    log.info("order-flow collector running every %ds (Ctrl-C to stop)", args.interval)
    while True:
        try:
            collect_once(ex, cfg.cache_dir)
        except Exception as e:  # keep the collector alive across transient errors
            log.error("collect failed: %s", e)
        time.sleep(max(60, args.interval))


if __name__ == "__main__":
    main()
