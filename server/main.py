"""AlgoTrader dashboard server.

    python server/main.py            # http://127.0.0.1:8777

FastAPI + uvicorn. A background task runs the scan → execute → track loop on
the configured cadence and pushes every update over /ws, so the dashboard is
event-driven instead of poll-and-pray. Execution defaults to the durable
PaperExecutor; the Bybit live executor is only attached when the entire
safety stack in algotrader/execution/bybit.py passes.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager

# Run from anywhere; resolve project root = parent of this file's directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from datetime import datetime, timezone                              # noqa: E402

import pandas as pd                                                    # noqa: E402
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect # noqa: E402
from fastapi.middleware.cors import CORSMiddleware                   # noqa: E402
from fastapi.responses import FileResponse, JSONResponse             # noqa: E402
from fastapi.staticfiles import StaticFiles                          # noqa: E402

from algotrader.config import load_config                            # noqa: E402
from algotrader.execution.base import PositionState                  # noqa: E402
from algotrader.execution.paper import PaperExecutor                 # noqa: E402
from algotrader.history import HistoryStore, SignalHistory, TradeHistory  # noqa: E402
from algotrader.portfolio import ExposureAnalyzer                    # noqa: E402
from algotrader.reporting import read_json, write_json               # noqa: E402
from algotrader.risk.manager import RiskManager                      # noqa: E402
from algotrader.scanner import Scanner                               # noqa: E402
from algotrader.utils.logging import audit, get_logger, utcnow_iso   # noqa: E402

log = get_logger("server")

cfg = load_config()
executor = PaperExecutor(cfg.risk)
live_executor = None
scanner: Scanner | None = None
last_scan: dict = read_json("reports/last_scan.json")
_seen_candle: dict[str, str] = {}
_sockets: set[WebSocket] = set()

# Historical persistence
history = HistoryStore()

# Pause/resume control for scan_loop
_scan_paused = asyncio.Event()
_scan_paused.set()  # running by default

# Serializes all executor + persistence mutations. The scan loop (event-loop
# thread) and the control endpoints must not interleave writes to the shared
# PaperExecutor, whose position list and state file are not otherwise guarded.
_exec_lock = asyncio.Lock()

# Rolling recent liquidations per symbol — filled by the streaming watcher when
# streaming is enabled, drained by /api/liquidations for the detail drawer.
_liquidations: dict[str, list] = {}
_LIQ_KEEP = 40

# Track which positions have been journaled so we only record opens once.
_logged_positions: set[str] = set()
_closed_trade_ids: set[str] = set()
_trade_id_by_pos: dict[str, int] = {}
_risk_amount_by_trade: dict[int, float] = {}
_fees_estimate_by_trade: dict[int, float] = {}

_TRADE_LINKS_PATH = "reports/trade_links.json"


def _save_trade_links() -> None:
    """Persist the pos_id -> trade_id linkage. Without this, a restart starts
    these maps empty while PaperExecutor restores its open positions from disk,
    so when a restored position later closes _record_closed_positions can't find
    its trade row and it stays 'open' forever — silently corrupting the ledger
    the dashboard and win-rate summaries read from."""
    write_json(_TRADE_LINKS_PATH, {
        "trade_id_by_pos": _trade_id_by_pos,
        "risk_amount_by_trade": {str(k): v for k, v in _risk_amount_by_trade.items()},
        "fees_estimate_by_trade": {str(k): v for k, v in _fees_estimate_by_trade.items()},
        "logged_positions": list(_logged_positions),
        "closed_trade_ids": list(_closed_trade_ids),
    })


def _load_trade_links() -> None:
    s = read_json(_TRADE_LINKS_PATH)
    if not s:
        return
    try:
        _trade_id_by_pos.update(
            {k: int(v) for k, v in s.get("trade_id_by_pos", {}).items()})
        _risk_amount_by_trade.update(
            {int(k): float(v) for k, v in s.get("risk_amount_by_trade", {}).items()})
        _fees_estimate_by_trade.update(
            {int(k): float(v) for k, v in s.get("fees_estimate_by_trade", {}).items()})
        _logged_positions.update(s.get("logged_positions", []))
        _closed_trade_ids.update(s.get("closed_trade_ids", []))
    except (TypeError, ValueError):
        pass


_load_trade_links()

_BTC = "BTC/USDT:USDT"


async def _correlation_matrix_for(positions: list[PositionState]) -> dict:
    """Best-effort 30d hourly return correlation of open positions vs BTC."""
    if scanner is None or not hasattr(scanner, "feed"):
        return {}
    symbols = {p.symbol for p in positions}
    if not symbols:
        return {}
    tf = cfg.timeframes[0] if cfg.timeframes else "1h"
    try:
        btc_df = await scanner.feed.fetch_ohlcv(_BTC, tf, 500)
    except Exception:
        return {}
    if btc_df is None or len(btc_df) < 100:
        return {}
    btc_rets = btc_df["close"].pct_change().dropna()
    matrix: dict[str, dict] = {}
    for sym in symbols:
        if sym == _BTC:
            matrix[sym] = {_BTC: 1.0}
            continue
        try:
            df = await scanner.feed.fetch_ohlcv(sym, tf, 500)
        except Exception:
            continue
        if df is None or len(df) < 100:
            continue
        rets = df["close"].pct_change().dropna()
        joined = pd.concat([btc_rets, rets], axis=1, join="inner").dropna()
        if len(joined) > 50:
            matrix[sym] = {_BTC: float(joined.corr().iloc[0, 1])}
    return matrix


# --------------------------------------------------------------------------- #
# WebSocket fan-out
# --------------------------------------------------------------------------- #
async def broadcast(msg_type: str, data) -> None:
    dead = []
    payload = json.dumps({"type": msg_type, "data": data, "ts": utcnow_iso()},
                         default=str)
    # Snapshot the set: ws_endpoint may add/discard sockets during the awaits
    # below, and mutating a set mid-iteration raises RuntimeError and aborts the
    # whole broadcast. A per-send timeout stops one slow/half-open client from
    # head-of-line-blocking the fan-out (and the scan loop that awaits it).
    for ws in list(_sockets):
        try:
            await asyncio.wait_for(ws.send_text(payload), timeout=5.0)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _sockets.discard(ws)


# --------------------------------------------------------------------------- #
# Background scan/track loop
# --------------------------------------------------------------------------- #
def _signals_to_history(signal_objs: list, timestamp: str) -> list[SignalHistory]:
    """Convert scanner Signal objects to flat history records."""
    out: list[SignalHistory] = []
    for sig in signal_objs:
        rationale = []
        if hasattr(sig, "rationale"):
            rationale = sig.rationale or []
        elif hasattr(sig, "evidence") and sig.evidence:
            rationale = [getattr(e, "name", str(e)) for e in sig.evidence]
        out.append(SignalHistory(
            symbol=sig.symbol,
            timeframe=sig.timeframe,
            side=sig.side,
            kind=sig.kind,
            entry=float(sig.entry_ref),
            stop=float(sig.stop_ref),
            confidence=float(sig.confidence),
            score=float(sig.score),
            win_rate=float(sig.base_win_rate),
            expected_value_r=float(getattr(sig, "expected_value_r", 0.0)),
            rationale=rationale,
            timestamp=timestamp,
        ))
    return out


def _record_closed_positions() -> None:
    """Detect any newly closed paper positions and update their history rows."""
    global _closed_trade_ids
    for closed in executor.closed:
        pos_id = closed.get("id")
        if not pos_id or pos_id in _closed_trade_ids:
            continue
        _closed_trade_ids.add(pos_id)
        trade_id = _trade_id_by_pos.get(pos_id)
        if trade_id is None:
            continue
        realized_r = closed.get("r", 0.0)
        risk_amount = _risk_amount_by_trade.get(trade_id, 0.0)
        fees_abs = _fees_estimate_by_trade.get(trade_id, 0.0)
        fees_r = round(fees_abs / risk_amount, 4) if risk_amount else 0.0
        history.update_position(
            trade_id=trade_id,
            symbol=closed.get("symbol", ""),
            side=closed.get("side", ""),
            opened_at=closed.get("opened_at", ""),
            closed_at=closed.get("closed_at"),
            status="closed",
            mtm_pnl=closed.get("pnl", 0.0),
            outcome=closed.get("exit_reason", "closed"),
            realized_r=realized_r,
            fees_r=fees_r,
        )


async def scan_loop() -> None:
    global last_scan, scanner
    from algotrader.data.feed import AsyncDataFeed
    from paper_trade import build_engine

    feed = AsyncDataFeed(cfg.exchange_id, cfg.market_type, cfg.api_key,
                         cfg.api_secret, cache_dir=cfg.cache_dir,
                         concurrency=cfg.scan_concurrency)
    engine = build_engine(cfg)
    risk = RiskManager(cfg.risk, cfg.calibration)
    scanner = Scanner(cfg, engine, risk, feed)
    trade_exec = live_executor or executor

    log.info("scan loop started (every %ds)", cfg.scan_interval_sec)
    try:
        while True:
            await _scan_paused.wait()
            scan_id: int | None = None
            try:
                result = await scanner.scan()
                signal_objs = result.pop("signal_objects", [])
                plans = result.pop("plan_objects", [])
                last_scan = result
                write_json("reports/last_scan.json", result)

                # ---- journal scan + signals ----------------------------------
                scan_id = history.record_scan(
                    timestamp=result.get("scanned_at") or utcnow_iso(),
                    btc_regime=result.get("btc_regime", ""),
                    n_symbols=result.get("universe_size", 0),
                    top_n=result.get("candidates", 0),
                )
                signal_ids = history.record_signals(
                    scan_id, _signals_to_history(signal_objs, result.get("scanned_at")))

                await broadcast("scan", result)

                # All executor mutations run under the lock so a concurrent
                # /close or pause/resume cannot interleave with the scan loop's
                # position advance / open / save sequence.
                async with _exec_lock:
                    # advance open positions on their newest CLOSED candle
                    for pos in list(executor.open_positions()):
                        try:
                            df = await feed.fetch_ohlcv(pos.symbol, pos.timeframe, 3)
                        except Exception:
                            continue
                        if df is None or len(df) < 2:
                            continue
                        bar, ts = df.iloc[-2], str(df.index[-2])
                        if _seen_candle.get(pos.id) == ts:
                            continue
                        _seen_candle[pos.id] = ts
                        executor.update_with_candle(
                            pos.symbol, ts, float(bar["open"]), float(bar["high"]),
                            float(bar["low"]), float(bar["close"]))

                    # journal any closes that happened during this candle step
                    _record_closed_positions()

                    # ---- open new plans -------------------------------------
                    for plan in plans:
                        pos_id = trade_exec.open_position(plan)
                        if pos_id and pos_id not in _logged_positions:
                            _logged_positions.add(pos_id)
                            signal_id = signal_ids.get(plan.symbol)
                            trade_id = history.record_trade(
                                scan_id=scan_id,
                                signal_id=signal_id,
                                trade=TradeHistory(
                                    symbol=plan.symbol,
                                    side=plan.side,
                                    entry=float(plan.entry),
                                    stop=float(plan.stop_loss),
                                    qty=float(plan.qty),
                                    leverage=float(plan.leverage),
                                    margin=float(plan.margin),
                                    timestamp=result.get("scanned_at"),
                                ),
                            )
                            _trade_id_by_pos[pos_id] = trade_id
                            _risk_amount_by_trade[trade_id] = float(plan.risk_amount)
                            _fees_estimate_by_trade[trade_id] = float(plan.fees_estimate)

                    # journal closes triggered by open_position rejections (e.g.
                    # circuit breakers closing positions) or ladder finalization.
                    _record_closed_positions()

                    executor.save()
                    _save_trade_links()  # durable pos_id -> trade_id linkage
                await broadcast("positions", executor.state_dict())
                exposure_corr = await _correlation_matrix_for(executor.open_positions())
                await broadcast("exposure", ExposureAnalyzer.analyze(
                    executor.open_positions(), correlation_matrix=exposure_corr))
            except Exception as e:
                log.error("scan loop error: %s", e)
                await broadcast("error", {"message": str(e)})
                if scan_id is None:
                    history.record_scan(
                        timestamp=utcnow_iso(),
                        btc_regime=last_scan.get("btc_regime", ""),
                        n_symbols=0,
                        top_n=0,
                        error_msg=str(e),
                    )
            await asyncio.sleep(cfg.scan_interval_sec)
    finally:
        await feed.close()


async def price_ticker_loop() -> None:
    """Stream live prices for open-position symbols (ccxt.pro) and push price_tick
    events so the dashboard's prices/PnL move between scans. No-op unless
    streaming is enabled and ccxt.pro is importable — the REST feed always
    remains the source of truth for signals."""
    from algotrader.data.stream import StreamingFeed
    if not cfg.streaming_enabled or not StreamingFeed.available():
        return
    feed = StreamingFeed(cfg.exchange_id, cfg.market_type)
    log.info("price ticker started (ccxt.pro streaming enabled)")
    try:
        while True:
            symbols = sorted({p.symbol for p in executor.open_positions()})
            if not symbols:
                await asyncio.sleep(5)
                continue
            feed._stop = asyncio.Event()

            async def on_tick(sym: str, price: float, ts: str) -> None:
                async with _exec_lock:
                    for pos in executor.open_positions():
                        if pos.symbol == sym:
                            pos.last_price = price
                            pos.unrealized_pnl = (pos.side.sign
                                                  * (price - pos.entry) * pos.qty_open)
                await broadcast("price_tick", {"symbol": sym, "price": price, "ts": ts})

            async def on_liq(sym: str, ev: dict) -> None:
                item = {
                    "symbol": sym, "side": ev.get("side"),
                    "price": float(ev.get("price") or 0.0),
                    "value": float(ev.get("quoteValue")
                                   or (ev.get("amount") or 0) * (ev.get("price") or 0)
                                   or 0.0),
                    "ts": utcnow_iso(),
                }
                buf = _liquidations.setdefault(sym, [])
                buf.append(item)
                del buf[:-_LIQ_KEEP]
                await broadcast("liquidation", item)

            async def resubscribe_when_positions_change() -> None:
                while not feed._stop.is_set():
                    await asyncio.sleep(5)
                    if sorted({p.symbol for p in executor.open_positions()}) != symbols:
                        feed.stop()  # break the watch; the outer loop re-subscribes

            await asyncio.gather(feed.watch_prices(symbols, on_tick),
                                 feed.watch_liquidations(symbols, on_liq),
                                 resubscribe_when_positions_change())
    except asyncio.CancelledError:
        pass
    finally:
        await feed.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global live_executor
    if cfg.execution_mode == "live":
        from algotrader.execution.bybit import BybitExecutor, LiveTradingRefused
        try:
            live_executor = BybitExecutor.from_config(cfg, interactive=False)
            log.warning("LIVE execution attached (%s)",
                        "testnet" if live_executor.testnet else "MAINNET")
        except LiveTradingRefused as e:
            log.warning("live execution refused (%s) — running paper-only", e)
    task = asyncio.create_task(scan_loop())
    ticker_task = asyncio.create_task(price_ticker_loop())
    yield
    task.cancel()
    ticker_task.cancel()


app = FastAPI(title="AlgoTrader", lifespan=lifespan)

# Restrict cross-origin access to localhost development origins only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1",
        "http://127.0.0.1:5173",
        "http://localhost",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def localhost_and_audit(request: Request, call_next):
    """Reject non-localhost requests and audit every state-changing call."""
    host = request.headers.get("host", "").split(":")[0].lower()
    if host not in ("127.0.0.1", "localhost"):
        return JSONResponse({"error": "localhost only"}, status_code=403)
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        audit("api_state_change", {
            "method": request.method,
            "path": request.url.path,
            "client_host": request.client.host if request.client else None,
        })
    return await call_next(request)


# --------------------------------------------------------------------------- #
# REST API
# --------------------------------------------------------------------------- #
@app.get("/api/config")
def api_config():
    return {
        "exchange": cfg.exchange_id,
        "universe_mode": cfg.universe_mode,
        "universe_size": cfg.universe_size,
        "timeframes": cfg.timeframes,
        "context_timeframe": cfg.context_timeframe,
        "equity": cfg.risk.account_equity,
        "risk_per_trade_pct": cfg.risk.risk_per_trade_pct,
        "max_leverage": cfg.risk.max_leverage,
        "default_leverage": cfg.risk.default_leverage,
        "max_concurrent_positions": cfg.risk.max_concurrent_positions,
        "max_daily_loss_pct": cfg.risk.max_daily_loss_pct,
        "scan_interval_sec": cfg.scan_interval_sec,
        "calibrated": bool(cfg.calibration),
        "ml_enabled": cfg.ml_enabled,
        "execution_mode": cfg.execution_mode,
        "testnet": cfg.execution_testnet,
        "live_enabled": cfg.allow_live,
        "live_attached": live_executor is not None,
    }


_EMPTY_SCAN = {
    "plans": [], "market": [], "suppressed_by_correlation": [],
    "universe_size": 0, "pairs_scanned": 0, "fetch_errors": 0,
    "candidates": 0, "btc_regime": "", "scanned_at": None, "duration_sec": 0.0,
}


@app.get("/api/signals")
def api_signals():
    # Merge over the empty shape so a partial/stale file can't ship an
    # incomplete object to the frontend.
    return {**_EMPTY_SCAN, **(last_scan or {})}


@app.get("/api/positions")
def api_positions():
    return executor.state_dict()


@app.get("/api/exposure")
async def api_exposure():
    corr = await _correlation_matrix_for(executor.open_positions())
    return ExposureAnalyzer.analyze(
        executor.open_positions(), correlation_matrix=corr)


@app.get("/api/backtest")
def api_backtest():
    return read_json("reports/backtest.json", {"summary": {}, "equity_curve": []})


@app.get("/api/walkforward")
def api_walkforward():
    return read_json("reports/walkforward.json")


@app.get("/api/calibration")
def api_calibration():
    return cfg.calibration


@app.get("/api/mlmodel")
def api_mlmodel():
    import pickle
    path = cfg.ml_model_path
    if not os.path.exists(path):
        return {"present": False}
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        meta = blob["meta"]
        from algotrader.ml.predict import MetaModel
        mm = MetaModel(blob["model"], meta)
        # Feature importance for the dashboard (before stripping the raw arrays).
        top_features = [{"name": n, "importance": round(float(v), 5)}
                        for n, v in mm.top_features(12)]
        meta["drift_score"] = mm.drift_score([], [])
        # Strip large arrays from the API response.
        for k in ("feature_columns", "feature_names", "feature_importances_"):
            meta.pop(k, None)
        return {"present": True, "top_features": top_features, **meta}
    except Exception as e:
        return {"present": False, "error": str(e)}


@app.get("/api/analytics/reliability")
def api_reliability():
    """Calibration reliability: bucket the rule win-rate prediction and compare
    it to the REALIZED win rate, from the exported ML dataset. Points on the
    diagonal mean the predicted probability matches reality."""
    ds_path = "reports/dataset.parquet"
    if not os.path.exists(ds_path):
        return {"present": False}
    try:
        df = pd.read_parquet(ds_path, columns=["rule_win_rate", "win"]).dropna()
        if len(df) < 20:
            return {"present": False}
        edges = [i / 10 for i in range(11)]
        df = df.assign(bucket=pd.cut(df["rule_win_rate"], bins=edges,
                                     include_lowest=True))
        buckets = []
        for _b, g in df.groupby("bucket", observed=True):
            if len(g) == 0:
                continue
            buckets.append({
                "predicted": round(float(g["rule_win_rate"].mean()), 4),
                "realized": round(float(g["win"].mean()), 4),
                "n": int(len(g)),
            })
        buckets.sort(key=lambda x: x["predicted"])
        return {"present": True, "buckets": buckets, "n": int(len(df))}
    except Exception as e:
        return {"present": False, "error": str(e)}


@app.get("/api/analytics/robustness")
def api_robustness():
    """Deflated Sharpe, PBO, bootstrap CI, and the account-level simulation from
    the last backtest (reports/robustness.json)."""
    rep = read_json("reports/robustness.json")
    if not rep:
        return {"present": False}
    return {"present": True, **rep}


@app.get("/api/history/signals")
def api_history_signals(symbol: str = "", timeframe: str = "", side: str = "",
                        regime: str = "", from_: str = "", to: str = "",
                        limit: int = 100):
    filters = {
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "regime": regime,
        "from_": from_,
        "to": to,
        "limit": limit,
    }
    return {"signals": history.get_signals(filters)}


@app.get("/api/history/trades")
def api_history_trades(symbol: str = "", outcome: str = "", from_: str = "",
                       to: str = "", limit: int = 100):
    filters = {
        "symbol": symbol,
        "outcome": outcome,
        "from_": from_,
        "to": to,
        "limit": limit,
    }
    return {"trades": history.get_trades(filters)}


@app.get("/api/history/summary")
def api_history_summary():
    return history.get_scan_summary()


@app.get("/api/health")
def api_health():
    now = datetime.now(timezone.utc)
    last_scan_age_sec: int | None = None
    data_fresh = False
    scan_ts = last_scan.get("scanned_at") if last_scan else None
    if scan_ts:
        try:
            last_dt = datetime.fromisoformat(scan_ts)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            last_scan_age_sec = int((now - last_dt).total_seconds())
            data_fresh = last_scan_age_sec < cfg.scan_interval_sec * 2
        except Exception:
            pass

    # Calibration freshness: stale if missing or older than 7 days.
    calibration_stale = True
    try:
        mtime = os.path.getmtime(cfg.calibration_file)
        calibration_stale = (now.timestamp() - mtime) > 7 * 86400
    except OSError:
        pass

    # Model trust: present, readable, and not obviously ancient.
    model_trusted = False
    try:
        model_mtime = os.path.getmtime(cfg.ml_model_path)
        model_age_days = (now.timestamp() - model_mtime) / 86400
        model_trusted = cfg.ml_enabled and model_age_days < 30
    except OSError:
        pass

    # Circuit breakers.
    day_start = executor.day_anchor.get("equity", executor.cfg.account_equity)
    mtm = executor.mtm_equity()
    daily_dd = 1.0 - mtm / day_start if day_start > 0 else 0.0
    circuit_breakers = {
        "kill_switch": os.path.exists(os.path.join(ROOT, "STOP_TRADING")),
        "daily_loss_pct": round(daily_dd * 100, 2),
        "daily_loss_triggered": daily_dd >= cfg.risk.max_daily_loss_pct,
        "consecutive_losses": executor.consecutive_losses,
        "losing_streak_triggered": executor.consecutive_losses >= cfg.risk.max_consecutive_losses,
    }

    return {
        "last_scan_age_sec": last_scan_age_sec,
        "data_fresh": data_fresh,
        "calibration_stale": calibration_stale,
        "model_trusted": model_trusted,
        "circuit_breakers": circuit_breakers,
        "paused": not _scan_paused.is_set(),
    }


# These are async (not sync `def`) so they run on the event-loop thread rather
# than a worker thread: an asyncio.Event and the PaperExecutor are not
# thread-safe, and mutating them from the threadpool races the loop.
@app.post("/api/control/pause")
async def api_control_pause():
    _scan_paused.clear()
    return {"paused": True}


@app.post("/api/control/resume")
async def api_control_resume():
    _scan_paused.set()
    return {"paused": False}


@app.post("/api/positions/{symbol:path}/close")
async def api_close_position(symbol: str):
    async with _exec_lock:
        pos = next((p for p in executor.open_positions() if p.symbol == symbol), None)
        if pos is None:
            return JSONResponse({"error": f"no open position for {symbol}"},
                                status_code=404)
        price = pos.last_price if pos.last_price else pos.entry
        executor.close_position(pos.id, float(price), "api_close")
        executor.save()
    return {"status": "closed", "symbol": symbol, "price": price}


@app.get("/api/candles")
async def api_candles(symbol: str, tf: str = "1h", limit: int = 300):
    """Candles + chart annotations for the signal-detail view."""
    if scanner is None:
        return JSONResponse({"error": "scanner not ready"}, status_code=503)
    try:
        df = await scanner.feed.fetch_ohlcv(symbol, tf, min(limit, 1000))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    if df is None or df.empty:
        return JSONResponse({"error": "no data"}, status_code=404)

    candles = [{"time": int(ts.timestamp()), "open": float(r["open"]),
                "high": float(r["high"]), "low": float(r["low"]),
                "close": float(r["close"]), "volume": float(r["volume"])}
               for ts, r in df.iterrows()]
    levels = []
    try:
        from algotrader.patterns.chart_patterns import get_sr_levels
        levels = [{"price": float(p), "touches": int(t)}
                  for p, t in get_sr_levels(df)]
    except Exception:
        pass

    # Indicator overlays: the same series that produced the signal, so the chart
    # visually justifies the thesis (EMA ribbon, VWAP, Bollinger band).
    overlays: dict[str, list] = {}
    try:
        from algotrader.indicators.indicators import compute_all
        indf = compute_all(df)
        times = [int(ts.timestamp()) for ts in df.index]

        def _series(col: str) -> list:
            if col not in indf:
                return []
            out = []
            for t, v in zip(times, indf[col]):
                if v == v:  # skip NaN warmup
                    out.append({"time": t, "value": float(v)})
            return out

        overlays = {k: _series(k) for k in
                    ("ema20", "ema50", "ema200", "vwap", "bb_up", "bb_low",
                     "rsi", "macd_hist")}
    except Exception as e:
        log.debug("candle overlays failed for %s: %s", symbol, e)

    # Detected chart patterns -> labeled markers at their location, so "recognizes
    # patterns" is visible on the chart, not just narrated.
    patterns: list = []
    try:
        from algotrader.patterns import chart_patterns as cp
        for pm in cp.detect(df):
            if 0 <= pm.end_idx < len(df):
                patterns.append({
                    "name": pm.name, "bias": pm.bias.value,
                    "time": int(df.index[pm.end_idx].timestamp()),
                    "breakout_level": pm.breakout_level,
                    "target_level": pm.target_level,
                })
    except Exception as e:
        log.debug("candle patterns failed for %s: %s", symbol, e)

    return {"symbol": symbol, "tf": tf, "candles": candles,
            "sr_levels": levels, "overlays": overlays, "patterns": patterns}


@app.get("/api/derivatives")
async def api_derivatives(symbol: str):
    """Current funding rate + open-interest trend for the detail view —
    the crowd-positioning read that OHLCV can't show."""
    if scanner is None or not hasattr(scanner, "feed"):
        return {"present": False}
    ex = getattr(scanner.feed, "exchange", None)
    if ex is None:
        return {"present": False}
    out = {"present": True, "symbol": symbol, "funding_rate": None,
           "oi": None, "oi_change_pct": None, "basis_pct": None}
    try:
        # Perp-spot basis: (perp - spot) / spot. Positive = perp trades rich
        # (contango, leveraged-long lean); negative = backwardation. The swap
        # client doesn't load spot markets, so hit Bybit's raw spot-ticker
        # endpoint directly (no market load needed) for the spot price.
        base = symbol.replace("/", "").replace(":USDT", "")  # OP/USDT:USDT -> OPUSDT
        perp_last = float((await ex.fetch_ticker(symbol)).get("last") or 0.0)
        spot_resp = await ex.public_get_v5_market_tickers(
            {"category": "spot", "symbol": base})
        lst = (spot_resp.get("result") or {}).get("list") or []
        spot_last = float(lst[0]["lastPrice"]) if lst else 0.0
        if perp_last > 0 and spot_last > 0:
            out["basis_pct"] = round((perp_last - spot_last) / spot_last * 100, 4)
    except Exception as e:
        log.debug("basis fetch failed for %s: %s", symbol, e)
    try:
        fr = await ex.fetch_funding_rate(symbol)
        rate = fr.get("fundingRate") if isinstance(fr, dict) else None
        if rate is None and isinstance(fr, dict):
            rate = (fr.get("info") or {}).get("fundingRate")
        out["funding_rate"] = float(rate) if rate is not None else None
    except Exception as e:
        log.debug("funding fetch failed for %s: %s", symbol, e)
    try:
        from algotrader.data.derivatives import _extract_oi_values
        hist = await ex.fetch_open_interest_history(symbol, timeframe="1h", limit=14)
        vals = _extract_oi_values(hist)
        if len(vals) >= 3 and vals[0] > 0:
            out["oi"] = round(float(vals[-1]), 2)
            out["oi_change_pct"] = round((vals[-1] - vals[0]) / vals[0] * 100, 2)
    except Exception as e:
        log.debug("OI fetch failed for %s: %s", symbol, e)
    return out


@app.get("/api/liquidations")
def api_liquidations(symbol: str, limit: int = 20):
    """Recent liquidations for a symbol (populated live when streaming is on)."""
    buf = _liquidations.get(symbol, [])
    return {"symbol": symbol, "streaming": cfg.streaming_enabled,
            "liquidations": list(reversed(buf[-limit:]))}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _sockets.add(ws)
    try:
        await ws.send_text(json.dumps(
            {"type": "hello",
             "data": {"signals": {**_EMPTY_SCAN, **(last_scan or {})},
                      "positions": executor.state_dict(),
                      "exposure": ExposureAnalyzer.analyze(
                          executor.open_positions())},
             "ts": utcnow_iso()}, default=str))
        while True:
            await ws.receive_text()   # keepalive pings from the client
    except WebSocketDisconnect:
        pass
    finally:
        _sockets.discard(ws)


# --------------------------------------------------------------------------- #
# Static frontend (built React app)
# --------------------------------------------------------------------------- #
DIST = os.path.join(ROOT, "web", "dist")
if os.path.isdir(DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")),
              name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        # Confine to DIST: a non-normalizing client (e.g. curl --path-as-is) could
        # otherwise request ../../ to read arbitrary files off a process that
        # holds exchange API keys. Serve index.html for anything outside DIST.
        target = os.path.realpath(os.path.join(DIST, path))
        dist_real = os.path.realpath(DIST)
        if (path and (target == dist_real or target.startswith(dist_real + os.sep))
                and os.path.isfile(target)):
            return FileResponse(target)
        return FileResponse(os.path.join(DIST, "index.html"))
else:
    @app.get("/")
    def no_frontend():
        return {"status": "API up. Frontend not built — run: cd web && npm run build"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8777")))
