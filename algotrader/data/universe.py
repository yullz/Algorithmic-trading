"""Tradeable universe selection: top-N USDT perpetuals by 24h quote volume.

The scanner should hunt where the liquidity is. Illiquid perps have wide
spreads, thin books and manipulated candles — patterns "work" there in a
backtest and then cost you the spread in reality. Ranking by 24h quote volume
(turnover) keeps the universe honest and self-refreshing as attention rotates.

Design intent:
  * Mainnet public data only — no keys, no testnet wiring in the data layer.
  * Stablecoin bases and leveraged/index products are excluded: a USDC/USDT
    perp has no directional edge to find, and 3L/3S-style tokens decay by
    construction.
  * Results are cached to {cache_dir}/universe.json with a fetched_at
    timestamp. A fresh cache (younger than ttl_sec) short-circuits the network
    call entirely; a stale cache is still served when the exchange call fails,
    because a slightly old universe beats a crashed scanner.
  * `static_symbols` (the config majors) always come first and are never
    dropped by the volume ranking.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional, Sequence

from ..utils.logging import get_logger

log = get_logger(__name__)

# Bases with no directional edge: pegged assets trading against USDT.
STABLECOIN_BASES = frozenset({
    "USDC", "USDE", "FDUSD", "DAI", "TUSD", "BUSD", "USDD", "PYUSD",
})

# Leveraged-token style bases (3L/3S/2S...) decay structurally; BULL/BEAR
# suffixed products likewise. Plain 2-4 letter tickers that merely end in
# "UP" (e.g. JUP) must NOT be caught, so no bare UP/DOWN suffix rule here.
_LEVERAGED_RE = re.compile(r"\d+[LS]$")


def _is_excluded_base(base: str) -> bool:
    b = (base or "").upper()
    if not b:
        return True
    if b in STABLECOIN_BASES:
        return True
    if _LEVERAGED_RE.search(b):
        return True
    if len(b) > 4 and b.endswith(("BULL", "BEAR")):
        return True
    return False


def _quote_volume(ticker: dict) -> float:
    """24h quote turnover from a unified ccxt ticker (with a raw fallback)."""
    if not isinstance(ticker, dict):
        return 0.0
    v = ticker.get("quoteVolume")
    if v is None:
        v = (ticker.get("info") or {}).get("turnover24h")
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _fetch_ranked(exchange_id: str) -> list[str]:
    """All eligible linear USDT swaps on `exchange_id`, ranked by 24h quote
    volume descending. Raises on network/exchange failure — the caller decides
    whether a stale cache can cover for it."""
    import ccxt  # lazy: offline paths must not require ccxt

    klass = getattr(ccxt, exchange_id)
    ex = klass({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    markets = ex.load_markets()

    eligible: list[str] = []
    for sym, m in markets.items():
        if not m.get("swap") or not m.get("linear"):
            continue
        if m.get("quote") != "USDT" or m.get("settle") != "USDT":
            continue
        if m.get("active") is False:
            continue
        if m.get("expiry"):  # dated futures masquerading as swaps
            continue
        if _is_excluded_base(m.get("base", "")):
            continue
        eligible.append(sym)

    # One bulk call: per-symbol fetch_ticker at 150+ symbols would hammer the
    # rate limiter. Symbols without a ticker rank as zero volume and drop out.
    tickers = ex.fetch_tickers()
    vols = {s: _quote_volume(tickers.get(s, {})) for s in eligible}
    ranked = sorted((s for s in eligible if vols[s] > 0),
                    key=lambda s: vols[s], reverse=True)
    return ranked


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "universe.json")


def _read_cache(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data.get("symbols"), list):
            return None
        float(data["fetched_at"])  # must be a usable timestamp
        return data
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError,
            ValueError, OSError):
        return None


def _write_cache(path: str, exchange_id: str, symbols: list[str]) -> None:
    """Best-effort persist — the cache is an optimization, never fatal."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"fetched_at": time.time(), "exchange": exchange_id,
                       "count": len(symbols), "symbols": symbols}, f)
        os.replace(tmp, path)
    except OSError as e:
        log.warning("could not write universe cache %s: %s", path, e)


def _merge(static_symbols: Sequence[str], ranked: Sequence[str],
           size: int) -> list[str]:
    """Static majors first, then volume-ranked fill, deduped, capped at size
    (statics are never dropped even if the static list itself exceeds size)."""
    seen: set[str] = set()
    out: list[str] = []
    for s in static_symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    for s in ranked:
        if len(out) >= size:
            break
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_universe(exchange_id: str = "bybit", size: int = 150,
                 cache_dir: str = "data_cache", ttl_sec: int = 3600,
                 static_symbols: Optional[Sequence[str]] = None) -> list[str]:
    """Top-`size` USDT perpetuals by 24h volume, as ccxt unified symbols
    (e.g. "BTC/USDT:USDT").

    Serving order of truth:
      1. cache younger than `ttl_sec`;
      2. a live exchange ranking (which refreshes the cache);
      3. the stale cache, if the exchange call fails;
      4. `static_symbols` alone, if there is no cache at all.
    """
    static = list(static_symbols or [])
    path = _cache_path(cache_dir)
    cached = _read_cache(path)
    now = time.time()

    if (cached is not None and cached.get("exchange") == exchange_id
            and now - float(cached["fetched_at"]) < ttl_sec):
        return _merge(static, cached["symbols"], size)

    try:
        ranked = _fetch_ranked(exchange_id)
    except Exception as e:
        log.warning("universe fetch from %s failed: %s", exchange_id, e)
        if cached is not None:
            age = now - float(cached["fetched_at"])
            log.warning("serving stale universe cache (age %.0f min, %d symbols)",
                        age / 60.0, len(cached["symbols"]))
            return _merge(static, cached["symbols"], size)
        log.error("no universe cache available; falling back to %d static symbols",
                  len(static))
        return _merge(static, [], size)

    _write_cache(path, exchange_id, ranked)
    log.info("universe refreshed from %s: %d eligible perps, using top %d",
             exchange_id, len(ranked), min(size, len(ranked) + len(static)))
    return _merge(static, ranked, size)
