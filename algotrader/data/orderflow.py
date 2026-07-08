"""Forward order-flow / derivatives collection.

Snapshots per-symbol funding rate, open interest, basis, and price from Bybit's
v5 linear tickers endpoint (ONE call covers every symbol) and appends to a
growing, dated parquet time series under data_cache/orderflow/.

This exists because the backtest pipeline has NO historical funding/OI/basis
series — the one data class with a real shot at perp alpha (funding z-score,
OI-vs-price divergence, basis, crowded-side pressure). Run it on a schedule now
so that in a few weeks there is a real dataset to test those signals on, using
the same walk-forward + deflated-Sharpe gauntlet that (correctly) rejected the
price-only edges.

`parse_tickers` is pure and unit-tested; the network call is a thin wrapper.
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import pandas as pd


def _f(v, default=None):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_tickers(rows: list[dict], ts: str, symbols=None) -> list[dict]:
    """Parse Bybit v5 linear-ticker rows into flat snapshot dicts. `ts` is the
    ISO capture time; `symbols` optionally restricts to a set (raw 'BTCUSDT')."""
    out = []
    allow = set(symbols) if symbols else None
    for r in rows:
        sym = r.get("symbol")
        if not sym or (allow is not None and sym not in allow):
            continue
        mark, index = _f(r.get("markPrice")), _f(r.get("indexPrice"))
        basis = _f(r.get("basisRate"))
        if basis is None and mark and index and index > 0:
            basis = (mark - index) / index
        out.append({
            "ts": ts, "symbol": sym,
            "last": _f(r.get("lastPrice")),
            "mark": mark, "index": index,
            "funding_rate": _f(r.get("fundingRate")),
            "funding_interval_h": _f(r.get("fundingIntervalHour")),
            "next_funding_ms": _f(r.get("nextFundingTime")),
            "open_interest": _f(r.get("openInterest")),
            "oi_value": _f(r.get("openInterestValue")),
            "basis_rate": basis,
            "volume_24h": _f(r.get("volume24h")),
            "turnover_24h": _f(r.get("turnover24h")),
            "price_24h_pcnt": _f(r.get("price24hPcnt")),
            "prev_price_1h": _f(r.get("prevPrice1h")),
        })
    return out


def fetch_orderflow_snapshot(exchange, ts: str, symbols=None,
                             category: str = "linear") -> list[dict]:
    """One raw v5 tickers call -> parsed snapshot rows for every linear symbol."""
    resp = exchange.public_get_v5_market_tickers({"category": category})
    rows = ((resp or {}).get("result") or {}).get("list") or []
    return parse_tickers(rows, ts, symbols)


def append_snapshot(rows: list[dict], cache_dir: str = "data_cache") -> Optional[str]:
    """Append a snapshot to data_cache/orderflow/{YYYY-MM-DD}.parquet, de-duped on
    (ts, symbol). Dated files keep each write cheap as history grows."""
    if not rows:
        return None
    day = str(rows[0]["ts"])[:10]
    d = os.path.join(cache_dir, "orderflow")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{day}.parquet")
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        try:
            df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
            df = df.drop_duplicates(subset=["ts", "symbol"], keep="last")
        except Exception:  # pragma: no cover - corrupt file, start fresh
            pass
    df.to_parquet(path, index=False)
    return path


def load_orderflow(cache_dir: str = "data_cache") -> pd.DataFrame:
    """Concatenate all collected order-flow snapshots into one time series
    (ts parsed, sorted). Empty frame if nothing collected yet."""
    files = sorted(glob.glob(os.path.join(cache_dir, "orderflow", "*.parquet")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    return df.dropna(subset=["ts"]).sort_values(["symbol", "ts"]).reset_index(drop=True)
