"""Market data feed via CCXT with rate limiting and robust error handling.

OHLCV for public data needs no API keys. A synthetic generator is provided so
the whole pipeline (indicators/patterns/signals/backtest) can be smoke-tested
fully offline.

Two feeds live here:
  * DataFeed      — simple sync feed (single-symbol tools, backtest history).
  * AsyncDataFeed — concurrent feed for the 150-symbol scanner, with an
    incremental parquet candle cache so each scan cycle fetches only the
    candles that are actually new. Mainnet public data only.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)

_OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

# Exchanges page OHLCV at ~1000 candles per request (Bybit: 1000).
_PAGE_CAP = 1000


class DataFeed:
    def __init__(self, exchange_id: str = "binanceusdm", market_type: str = "swap",
                 api_key: str = "", api_secret: str = ""):
        self.exchange_id = exchange_id
        self.market_type = market_type
        self._ex = None
        self._api_key = api_key
        self._api_secret = api_secret

    def _exchange(self):
        if self._ex is not None:
            return self._ex
        import ccxt  # imported lazily so offline/synthetic use needs no ccxt
        klass = getattr(ccxt, self.exchange_id)
        params: dict = {"enableRateLimit": True, "options": {"defaultType": self.market_type}}
        if self._api_key:
            params["apiKey"] = self._api_key
            params["secret"] = self._api_secret
        self._ex = klass(params)
        return self._ex

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 500,
                    retries: int = 3) -> pd.DataFrame:
        """Fetch OHLCV as a DataFrame indexed by UTC datetime.

        Retries with exponential backoff on network/exchange errors.
        """
        import ccxt
        ex = self._exchange()
        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                return self._to_df(raw)
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:  # type: ignore[attr-defined]
                last_err = e
                wait = min(2 ** attempt, 30)
                log.warning("fetch_ohlcv %s %s failed (attempt %d/%d): %s; retry in %ds",
                            symbol, timeframe, attempt, retries, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"fetch_ohlcv failed for {symbol} {timeframe}: {last_err}")

    def fetch_history(self, symbol: str, timeframe: str = "1h",
                      total_limit: int = 3000, retries: int = 3) -> pd.DataFrame:
        """Deep history via since-based pagination (for backtests).

        A single fetch_ohlcv call is capped at ~1000 candles by the exchange;
        this walks forward page by page from `now - total_limit` bars, then
        dedupes and sorts. Returns at most `total_limit` rows (fewer if the
        market is younger than the requested window).
        """
        ex = self._exchange()
        tf_ms = int(ex.parse_timeframe(timeframe)) * 1000
        now_ms = int(ex.milliseconds())
        since = now_ms - total_limit * tf_ms
        rows: list = []
        while True:
            page = self._fetch_page(symbol, timeframe, since, _PAGE_CAP, retries)
            if not page:
                break
            rows.extend(page)
            last_ts = int(page[-1][0])
            next_since = last_ts + tf_ms
            # Stop on no forward progress (exchange re-serving the same
            # candles) or once the still-forming candle has been reached.
            if next_since <= since or last_ts >= now_ms - tf_ms:
                break
            since = next_since
            if len(rows) >= total_limit + _PAGE_CAP:  # safety cap
                break
        df = self._to_df(rows)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df.iloc[-total_limit:]

    def _fetch_page(self, symbol: str, timeframe: str, since: int,
                    limit: int, retries: int) -> list:
        """One paginated OHLCV request with the same retry policy as
        fetch_ohlcv."""
        import ccxt
        ex = self._exchange()
        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                return ex.fetch_ohlcv(symbol, timeframe=timeframe,
                                      since=since, limit=limit)
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:  # type: ignore[attr-defined]
                last_err = e
                wait = min(2 ** attempt, 30)
                log.warning("fetch_history page %s %s since=%d failed "
                            "(attempt %d/%d): %s; retry in %ds",
                            symbol, timeframe, since, attempt, retries, e, wait)
                time.sleep(wait)
        raise RuntimeError(
            f"fetch_history page failed for {symbol} {timeframe}: {last_err}")

    _TF_RULE = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
                "30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h",
                "6h": "6h", "12h": "12h", "1d": "1D"}

    @staticmethod
    def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Aggregate a lower-timeframe OHLCV frame up to `timeframe`. Used to
        derive higher-timeframe context from base data (e.g. in the backtester)
        so HTF factors are exercised on the same inputs as live."""
        rule = DataFeed._TF_RULE.get(timeframe, timeframe)
        agg = {"open": "first", "high": "max", "low": "min",
               "close": "last", "volume": "sum"}
        return df.resample(rule).agg(agg).dropna()

    @staticmethod
    def _to_df(raw: list) -> pd.DataFrame:
        df = pd.DataFrame(raw, columns=_OHLCV_COLS)
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("dt")
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna()

    # ------------------------------------------------------------------ #
    # Offline synthetic data — deterministic given a seed. Lets the whole
    # engine be exercised without a network connection.
    # ------------------------------------------------------------------ #
    @staticmethod
    def synthetic(n: int = 500, seed: int = 7, start: float = 30000.0,
                  timeframe_minutes: int = 60) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        # geometric random walk with mild trend + volatility clustering
        rets = rng.normal(0.0002, 0.01, n) + 0.003 * np.sin(np.linspace(0, 12, n))
        vol = 0.008 * (1 + 0.5 * np.abs(np.sin(np.linspace(0, 20, n))))
        rets = rets * (vol / vol.mean())
        close = start * np.exp(np.cumsum(rets))
        opens = np.concatenate([[start], close[:-1]])
        highs = np.maximum(opens, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
        lows = np.minimum(opens, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
        volume = rng.uniform(100, 1000, n) * (1 + 3 * vol)
        idx = pd.date_range("2024-01-01", periods=n, freq=f"{timeframe_minutes}min", tz="UTC")
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": close, "volume": volume},
            index=idx,
        )


# --------------------------------------------------------------------------- #
# Async feed with incremental parquet cache (the 150-symbol scanner path)
# --------------------------------------------------------------------------- #
class AsyncDataFeed:
    """Concurrent OHLCV feed on ccxt.async_support with an incremental cache.

    One exchange instance, one aiohttp session, an asyncio.Semaphore capping
    in-flight requests, and ccxt's own rate limiter enabled — 150 symbols x 3
    timeframes must not turn into a request storm.

    Cache design: {cache_dir}/ohlcv/{SAFE_SYMBOL}_{tf}.parquet holds every
    candle ever fetched for that pair. On each call we fetch only candles
    since the last cached timestamp minus one bar (the last cached candle was
    still forming when stored, so it is always refreshed), then concat, dedupe
    keeping the newest reading, and persist. A corrupt or unreadable parquet
    file is deleted and refetched from scratch — the cache is an optimization,
    never a source of truth.

    Mainnet public data only; no testnet wiring in the data layer.

    Usage:
        async with AsyncDataFeed("bybit", concurrency=8) as feed:
            frames = await feed.fetch_many([("BTC/USDT:USDT", "1h", 500), ...])
    """

    def __init__(self, exchange_id: str = "bybit", market_type: str = "swap",
                 api_key: str = "", api_secret: str = "",
                 cache_dir: str = "data_cache", concurrency: int = 8):
        self.exchange_id = exchange_id
        self.market_type = market_type
        self.cache_dir = cache_dir
        self._api_key = api_key
        self._api_secret = api_secret
        self._ex = None
        self._sem = asyncio.Semaphore(max(1, int(concurrency)))

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def _exchange(self):
        if self._ex is not None:
            return self._ex
        import ccxt.async_support as ccxt_async  # lazy, like the sync feed
        klass = getattr(ccxt_async, self.exchange_id)
        params: dict = {"enableRateLimit": True, "timeout": 30000,
                        "options": {"defaultType": self.market_type}}
        if self.exchange_id == "bybit" and self.market_type == "swap":
            # We only trade linear USDT perps — skip the huge spot/option
            # instrument dumps when loading markets.
            params["options"]["fetchMarkets"] = ["linear"]
        if self._api_key:
            params["apiKey"] = self._api_key
            params["secret"] = self._api_secret
        self._ex = klass(params)
        self._attach_dns_safe_session(self._ex)
        return self._ex

    @staticmethod
    def _attach_dns_safe_session(ex) -> None:
        """aiohttp silently switches to a c-ares (aiodns) resolver when aiodns
        is installed; on many Windows setups (VPNs, virtual adapters) c-ares
        cannot read the DNS configuration and every request dies with 'Could
        not contact DNS servers'. Force the threaded getaddrinfo resolver —
        the same resolution path the sync feed's `requests` uses."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop yet; ccxt will lazily build its default session
        try:
            import aiohttp
            connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
            ex.session = aiohttp.ClientSession(connector=connector, trust_env=True)
        except Exception as e:  # pragma: no cover
            log.warning("could not attach DNS-safe session: %s", e)

    @property
    def exchange(self):
        """The underlying async ccxt exchange (e.g. for derivatives calls)."""
        return self._exchange()

    async def close(self) -> None:
        """Release the aiohttp session. Always call this (or use async with)."""
        if self._ex is not None:
            try:
                await self._ex.close()
            finally:
                self._ex = None

    async def __aenter__(self) -> "AsyncDataFeed":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        await self.close()
        return False

    # ------------------------------------------------------------------ #
    # Cache plumbing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        """'BTC/USDT:USDT' -> 'BTC_USDT_USDT' (filesystem-safe)."""
        return re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_")

    def _cache_path(self, symbol: str, timeframe: str) -> str:
        return os.path.join(self.cache_dir, "ohlcv",
                            f"{self._safe_symbol(symbol)}_{timeframe}.parquet")

    def _load_cached(self, path: str) -> Optional[pd.DataFrame]:
        """Cached frame, or None. Corrupt files are deleted, not tolerated."""
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_parquet(path)
            if df.empty or not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("empty frame or non-datetime index")
            missing = [c for c in ("open", "high", "low", "close", "volume")
                       if c not in df.columns]
            if missing:
                raise ValueError(f"missing columns {missing}")
            return df
        except Exception as e:
            log.warning("corrupt ohlcv cache %s (%s); deleting and refetching",
                        path, e)
            try:
                os.remove(path)
            except OSError:
                pass
            return None

    def _persist(self, path: str, df: pd.DataFrame) -> None:
        """Best-effort atomic-ish write (tmp + replace); failure is non-fatal."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            df.to_parquet(tmp)
            os.replace(tmp, path)
        except Exception as e:
            log.warning("could not persist ohlcv cache %s: %s", path, e)

    # ------------------------------------------------------------------ #
    # Fetching
    # ------------------------------------------------------------------ #
    async def _fetch_raw(self, symbol: str, timeframe: str,
                         since: Optional[int] = None, limit: int = _PAGE_CAP,
                         retries: int = 2) -> list:
        """One request with `retries` retries and short exponential backoff on
        transient ccxt errors."""
        import ccxt
        ex = self._exchange()
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                return await ex.fetch_ohlcv(symbol, timeframe=timeframe,
                                            since=since, limit=limit)
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:  # type: ignore[attr-defined]
                last_err = e
                if attempt < retries:
                    await asyncio.sleep(min(0.5 * (2 ** attempt), 4.0))
        raise RuntimeError(
            f"async fetch_ohlcv failed for {symbol} {timeframe}: {last_err}")

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h",
                          limit: int = 500) -> pd.DataFrame:
        """The trailing `limit` candles, served incrementally from cache.

        Cache hit: fetch only the gap since the last cached bar (minus one
        bar so the previously still-open candle gets its final values).
        Cache miss or unbridgeable gap: full fetch of `limit` (page-capped).
        """
        path = self._cache_path(symbol, timeframe)
        page = min(int(limit), _PAGE_CAP)
        async with self._sem:
            cached = self._load_cached(path)
            ex = self._exchange()
            tf_ms = int(ex.parse_timeframe(timeframe)) * 1000
            if cached is not None:
                last_ms = int(cached.index[-1].value // 1_000_000)
                gap_bars = max(0, (int(ex.milliseconds()) - last_ms) // tf_ms)
                if gap_bars <= _PAGE_CAP - 2:
                    raw = await self._fetch_raw(
                        symbol, timeframe, since=last_ms - tf_ms,
                        limit=min(_PAGE_CAP, int(gap_bars) + 3))
                else:
                    # Too stale to bridge in one page: refetch the window and
                    # accept the (historical, closed-candle) hole in the file.
                    raw = await self._fetch_raw(symbol, timeframe, limit=page)
                df = pd.concat([cached, self._to_df_static(raw)])
            else:
                df = self._to_df_static(
                    await self._fetch_raw(symbol, timeframe, limit=page))
            df = df[~df.index.duplicated(keep="last")].sort_index()
            if not df.empty:
                self._persist(path, df)
        return df.iloc[-limit:].copy()

    @staticmethod
    def _to_df_static(raw: list) -> pd.DataFrame:
        return DataFeed._to_df(raw)

    async def fetch_many(self, pairs: list[tuple[str, str, int]],
                         ) -> dict[tuple[str, str], Optional[pd.DataFrame]]:
        """Fetch many (symbol, timeframe, limit) tuples concurrently.

        Concurrency is bounded by the semaphore inside fetch_ohlcv. A failing
        pair maps to None — one delisted symbol must never kill a 450-pair
        scan batch.
        """
        async def one(symbol: str, timeframe: str, limit: int):
            try:
                return (symbol, timeframe), await self.fetch_ohlcv(
                    symbol, timeframe, limit)
            except Exception as e:
                log.warning("fetch_many: %s %s failed: %s", symbol, timeframe, e)
                return (symbol, timeframe), None

        results = await asyncio.gather(*(one(s, tf, lim) for s, tf, lim in pairs))
        return dict(results)
