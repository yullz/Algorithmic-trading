#!/usr/bin/env python3
"""Backtest + calibrate. Measures the historical win rate of the strategy and of
each contributing factor, then writes calibration.json so live scans report
measured rates instead of priors.

Usage:
    python backtest.py                 # backtest configured symbols/timeframes
    python backtest.py --offline       # synthetic data (offline smoke test)
    python backtest.py --symbols BTC/USDT:USDT,ETH/USDT:USDT --timeframes 1h,4h
    python backtest.py --symbols all   # top-volume universe from the scanner
    python backtest.py --deep          # 5000 candles, all TFs, top-25 symbols
    python backtest.py --walkforward   # add anchored out-of-sample validation
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import Semaphore
from typing import Optional

from algotrader.backtest.engine import Backtester, walk_forward
from algotrader.config import AppConfig, load_config
from algotrader.data.feed import DataFeed
from algotrader.models import RiskConfig
from algotrader.risk.manager import RiskManager
from algotrader.scanner import Scanner
from algotrader.signals.engine import SignalEngine
from algotrader.utils.logging import get_logger

log = get_logger("backtest")

_DEFAULT_LIMIT = 3000
_DEEP_LIMIT = 5000
_DEEP_SYMBOLS = 25
_PAGE_CAP = 1000
_PROGRESS_EVERY = 5

_TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
}


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="use deterministic synthetic data (no network)")
    ap.add_argument("--symbol", help="(legacy) single symbol to backtest")
    ap.add_argument("--symbols",
                    help="comma-separated symbols or 'all' for top-volume universe")
    ap.add_argument("--tf", default="1h",
                    help="(legacy) single timeframe")
    ap.add_argument("--timeframes",
                    help="comma-separated timeframes (default: config.timeframes)")
    ap.add_argument("--limit", type=int, default=_DEFAULT_LIMIT,
                    help=f"candles per symbol/timeframe (default: {_DEFAULT_LIMIT})")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--walkforward", action="store_true",
                    help="also run anchored out-of-sample walk-forward validation")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--export-dataset", action="store_true",
                    help="write per-trade ML training rows to reports/dataset.parquet")
    ap.add_argument("--deep", action="store_true",
                    help=f"deep mode: {_DEEP_LIMIT} candles, all configured timeframes, "
                         f"top {_DEEP_SYMBOLS} volume symbols")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel backtest workers (default: 4)")
    ap.add_argument("--output-suffix", default="",
                    help="suffix for reports/backtest{suffix}.json and "
                         "reports/dataset{suffix}.parquet (for batching)")
    return ap


def parse_backtest_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _tf_minutes(timeframe: str) -> int:
    return _TF_MINUTES.get(timeframe, 60)


def _resolve_symbols(cfg: AppConfig, args: argparse.Namespace,
                     engine: SignalEngine, risk: RiskManager) -> list[str]:
    """Return the symbol list for this run, honouring --deep and --symbols all."""
    if args.deep:
        base = _load_universe(cfg, engine, risk) if not args.offline else list(cfg.symbols)
        return base[:_DEEP_SYMBOLS]

    if args.symbols:
        if args.symbols.strip().lower() == "all":
            if args.offline:
                return list(cfg.symbols)
            return _load_universe(cfg, engine, risk)
        return [s.strip() for s in args.symbols.split(",") if s.strip()]

    if args.symbol:
        return [args.symbol]

    return list(cfg.symbols)


def _resolve_timeframes(cfg: AppConfig, args: argparse.Namespace) -> list[str]:
    if args.deep:
        return list(cfg.timeframes)
    if args.timeframes:
        return [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    return [args.tf]


def _load_universe(cfg: AppConfig, engine: SignalEngine, risk: RiskManager) -> list[str]:
    """Fetch the top-volume universe via the scanner (sync wrapper)."""
    scanner = Scanner(cfg, engine, risk, None)
    try:
        return asyncio.run(scanner.universe())
    except RuntimeError as e:
        log.error("could not load scanner universe: %s; falling back to config symbols", e)
        return list(cfg.symbols)


def _fetch_data(feed: DataFeed, symbol: str, timeframe: str, limit: int,
                use_history: bool, sem: Optional[Semaphore]):
    """Fetch OHLCV with optional deep pagination and rate-limit semaphore."""
    def _do_fetch():
        if use_history and hasattr(feed, "fetch_history"):
            return feed.fetch_history(symbol, timeframe, limit)
        return feed.fetch_ohlcv(symbol, timeframe, limit)

    if sem is not None:
        with sem:
            return _do_fetch()
    return _do_fetch()


def _fetch_all(pairs: list[tuple[str, str]], cfg: AppConfig,
               args: argparse.Namespace) -> dict[tuple[str, str], pd.DataFrame]:
    """Fetch all OHLCV upfront so CPU-bound workers are not blocked on I/O."""
    if args.offline:
        return {}
    feed = DataFeed(cfg.exchange_id, cfg.market_type, cfg.api_key, cfg.api_secret)
    fetch_sem = Semaphore(max(1, cfg.scan_concurrency))
    limit = _DEEP_LIMIT if args.deep else args.limit
    use_history = (limit > _PAGE_CAP) or args.deep
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol, timeframe in pairs:
        try:
            df = _fetch_data(feed, symbol, timeframe, limit, use_history, fetch_sem)
            if df is not None and len(df) >= 60:
                frames[(symbol, timeframe)] = df
        except Exception as e:
            log.warning("%s %s fetch failed: %s", symbol, timeframe, e)
    return frames


def _run_one(args_tuple) -> dict:
    """Run a single symbol/timeframe backtest in a worker process.

    Accepts a tuple so the function is picklable for ProcessPoolExecutor.
    """
    pair, cfg_dict, args_dict = args_tuple
    symbol, timeframe = pair
    df = args_dict.pop("_df")
    try:
        htf_full = DataFeed.resample(df, cfg_dict["context_timeframe"])
        engine = SignalEngine(
            cfg_dict["min_confidence"], cfg_dict["min_confluence"], calibration={},
            min_families=cfg_dict["min_families"], htf_veto=cfg_dict["htf_veto"],
            regime_gating=cfg_dict["regime_gating"],
            max_stop_atr_mult=cfg_dict["risk"]["max_stop_atr_mult"])
        risk = RiskManager(RiskConfig(**cfg_dict["risk"]))
        bt = Backtester(engine, risk)
        res = bt.run(df, symbol, timeframe, htf_full=htf_full)
        log.info("%s %s -> %s", symbol, timeframe, json.dumps(res.summary))
        return {"symbol": symbol, "tf": timeframe, "result": res,
                "error": None, "n_trades": len(res.trades)}
    except Exception as e:
        log.error("%s %s backtest failed: %s", symbol, timeframe, e)
        return {"symbol": symbol, "tf": timeframe, "result": None,
                "error": str(e), "n_trades": 0}


def _run_one_offline(args_tuple) -> dict:
    """Offline variant: generate synthetic data in the worker thread."""
    pair, cfg_dict, args = args_tuple
    symbol, timeframe = pair
    try:
        limit = _DEEP_LIMIT if args.deep else args.limit
        seed = zlib.crc32(f"{symbol}:{timeframe}".encode()) % 1000
        df = DataFeed.synthetic(limit, seed=seed,
                                timeframe_minutes=_tf_minutes(timeframe))
        htf_full = DataFeed.resample(df, cfg_dict["context_timeframe"])
        engine = SignalEngine(
            cfg_dict["min_confidence"], cfg_dict["min_confluence"], calibration={},
            min_families=cfg_dict["min_families"], htf_veto=cfg_dict["htf_veto"],
            regime_gating=cfg_dict["regime_gating"],
            max_stop_atr_mult=cfg_dict["risk"]["max_stop_atr_mult"])
        risk = RiskManager(RiskConfig(**cfg_dict["risk"]))
        bt = Backtester(engine, risk)
        res = bt.run(df, symbol, timeframe, htf_full=htf_full)
        log.info("%s %s -> %s", symbol, timeframe, json.dumps(res.summary))
        return {"symbol": symbol, "tf": timeframe, "result": res,
                "error": None, "n_trades": len(res.trades)}
    except Exception as e:
        log.error("%s %s backtest failed: %s", symbol, timeframe, e)
        return {"symbol": symbol, "tf": timeframe, "result": None,
                "error": str(e), "n_trades": 0}


def _print_progress(done: int, total: int) -> None:
    if total <= 0:
        return
    pct = 100 * done // total
    log.info("progress: %d/%d pairs completed (%d%%)", done, total, pct)


def main() -> None:
    args = parse_backtest_args()
    cfg = load_config(args.config)

    # Uncalibrated engine (priors) so we measure honestly.
    engine = SignalEngine(
        cfg.min_confidence, cfg.min_confluence, calibration={},
        min_families=cfg.min_families, htf_veto=cfg.htf_veto,
        regime_gating=cfg.regime_gating,
        max_stop_atr_mult=cfg.risk.max_stop_atr_mult)
    risk = RiskManager(cfg.risk)

    symbols = _resolve_symbols(cfg, args, engine, risk)
    timeframes = _resolve_timeframes(cfg, args)
    pairs = [(s, tf) for s in symbols for tf in timeframes]

    if not pairs:
        log.error("no symbol/timeframe pairs to backtest")
        sys.exit(1)

    log.info("backtesting %d symbols x %d timeframes = %d pairs (limit=%s, offline=%s)",
             len(symbols), len(timeframes), len(pairs),
             _DEEP_LIMIT if args.deep else args.limit, args.offline)

    # Fetch all data upfront so CPU-bound backtests can run in parallel processes
    # without fighting the GIL or blocking on network I/O.
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    if not args.offline:
        log.info("fetching OHLCV for %d pairs ...", len(pairs))
        frames = _fetch_all(pairs, cfg, args)
        log.info("fetched %d/%d pairs", len(frames), len(pairs))

    # Serialize config into plain dicts for picklable worker args.
    cfg_dict = {
        "min_confidence": cfg.min_confidence,
        "min_confluence": cfg.min_confluence,
        "min_families": cfg.min_families,
        "htf_veto": cfg.htf_veto,
        "regime_gating": cfg.regime_gating,
        "context_timeframe": cfg.context_timeframe,
        "risk": cfg.risk.__dict__,
    }

    results: list[dict] = []
    workers = max(1, args.workers)

    if args.offline:
        # Offline uses synthetic data, cheap enough for threads.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_one_offline, (pair, cfg_dict, args)): pair
                for pair in pairs
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % _PROGRESS_EVERY == 0 or completed == len(pairs):
                    _print_progress(completed, len(pairs))
                results.append(future.result())
    else:
        work = []
        for pair in pairs:
            if pair not in frames:
                continue
            args_dict = {"_df": frames[pair]}
            work.append((pair, cfg_dict, args_dict))

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, w): w[0] for w in work}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % _PROGRESS_EVERY == 0 or completed == len(work):
                    _print_progress(completed, len(work))
                results.append(future.result())

    # Collect successful results for aggregate calibration + reports.
    all_trades: list[dict] = []
    detail: list[dict] = []
    for r in results:
        if r["result"] is not None:
            res = r["result"]
            all_trades.extend(res.trades)
            detail.append({
                "symbol": r["symbol"], "timeframe": r["tf"],
                "n_trades": len(res.trades),
                "summary": res.summary,
                "factor_win_rate": {k: round(v, 4)
                                    for k, v in res.factor_win_rate.items()},
                "kind_win_rate": {k: round(v, 4)
                                  for k, v in res.kind_win_rate.items()},
            })

    bt = Backtester(engine, risk)
    combined = bt._aggregate(all_trades)
    print("\n===== AGGREGATE =====")
    print(json.dumps(combined.summary, indent=2))
    print("\nPer-factor win rates (samples>=25 written to calibration):")
    for f, wr in sorted(combined.factor_win_rate.items(), key=lambda x: -x[1]):
        print(f"  {f:28s} {wr:.0%}")
    suffix = args.output_suffix
    calib_path = cfg.calibration_file.replace(".json", f"{suffix}.json") if suffix else cfg.calibration_file
    calib = combined.write_calibration(calib_path)
    n_factor_keys = sum(1 for k in calib if not k.startswith("_"))
    log.info("wrote %d calibrated factors -> %s", n_factor_keys, calib_path)

    # Persist reports.
    os.makedirs("reports", exist_ok=True)
    detail_path = f"reports/backtest_detail{suffix}.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump({
            "pairs": detail,
            "n_pairs": len(pairs),
            "n_successful": len(detail),
            "symbols": symbols,
            "timeframes": timeframes,
            "source": "synthetic" if args.offline else cfg.exchange_id,
            "limit": _DEEP_LIMIT if args.deep else args.limit,
        }, f, indent=2)

    report = combined.to_report()
    report["symbols"] = symbols
    report["timeframes"] = timeframes
    report["source"] = "synthetic" if args.offline else cfg.exchange_id
    report["limit"] = _DEEP_LIMIT if args.deep else args.limit
    report["n_pairs"] = len(pairs)
    report_path = f"reports/backtest{suffix}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("wrote %s (%d trades across %d pairs)",
             report_path, report["n_trades"], len(pairs))

    if args.export_dataset:
        ds = combined.to_dataset()
        ds_path = f"reports/dataset{suffix}.parquet"
        ds.to_parquet(ds_path, index=False)
        log.info("wrote %s (%d rows, %d cols) — train the "
                 "meta-model with: python -m algotrader.ml.train", ds_path, *ds.shape)

    if args.walkforward:
        run_walkforward(cfg, symbols, timeframes, args, combined)

    print("\nReminder: results above are IN-SAMPLE. The walk-forward (--walkforward) "
          "OOS numbers are the honest test. Past performance != future results.")


def run_walkforward(cfg: AppConfig, symbols: list[str], timeframes: list[str],
                    args: argparse.Namespace, in_sample) -> None:
    """Anchored out-of-sample validation across all symbol/timeframe pairs."""
    suffix = args.output_suffix
    log.info("running walk-forward (%d folds)...", args.folds)
    oos_trades, all_folds = [], []
    feed = DataFeed(cfg.exchange_id, cfg.market_type, cfg.api_key, cfg.api_secret)
    for symbol in symbols:
        for tf in timeframes:
            try:
                wf_limit = _DEEP_LIMIT if args.deep else args.limit
                if args.offline:
                    df = DataFeed.synthetic(
                        wf_limit, seed=zlib.crc32(f"{symbol}:{tf}".encode()) % 1000,
                        timeframe_minutes=_tf_minutes(tf))
                elif args.deep and hasattr(feed, "fetch_history"):
                    df = feed.fetch_history(symbol, tf, _DEEP_LIMIT)
                else:
                    df = feed.fetch_ohlcv(symbol, tf, args.limit)
            except Exception as e:
                log.warning("%s %s: walk-forward fetch failed: %s", symbol, tf, e)
                continue
            if df is None or len(df) < 60:
                continue
            htf_full = DataFeed.resample(df, cfg.context_timeframe)
            wf = walk_forward(df, symbol, tf, cfg, htf_full=htf_full, folds=args.folds)
            if wf is None:
                log.warning("%s %s: not enough data for walk-forward", symbol, tf)
                continue
            oos_trades.extend(wf["oos"].trades)
            for row in wf["folds"]:
                row["symbol"] = symbol
                row["timeframe"] = tf
            all_folds.extend(wf["folds"])

    oos = Backtester(SignalEngine(), RiskManager(cfg.risk))._aggregate(oos_trades)
    is_s, oos_s = in_sample.summary, oos.summary

    def line(label, key, suffix=""):
        return f"  {label:16s} IS {is_s.get(key,0):>8}{suffix}   |   OOS {oos_s.get(key,0):>8}{suffix}"

    print("\n===== WALK-FORWARD (in-sample vs OUT-OF-SAMPLE) =====")
    print(line("trades", "trades"))
    print(line("win rate", "win_rate"))
    print(line("expectancy R", "expectancy_r"))
    print(line("profit factor", "profit_factor"))
    print(line("max drawdown R", "max_drawdown_r"))
    verdict = ("OOS edge holds" if oos_s.get("expectancy_r", 0) > 0 and oos_s.get("profit_factor", 0) > 1
               else "OOS edge does NOT hold — likely in-sample overfit")
    print(f"\n  Verdict: {verdict}. OOS trades={oos_s.get('trades',0)} across {len(all_folds)} folds.")

    wf_report = {
        "in_sample": is_s, "out_of_sample": oos_s,
        "folds": all_folds, "oos_equity_curve": oos.equity_curve,
        "verdict": verdict, "n_folds": args.folds,
    }
    wf_path = f"reports/walkforward{suffix}.json"
    with open(wf_path, "w", encoding="utf-8") as f:
        json.dump(wf_report, f, indent=2)
    log.info("wrote %s", wf_path)


if __name__ == "__main__":
    main()
