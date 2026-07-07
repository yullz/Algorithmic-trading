"""Live WebSocket streaming via ccxt.pro.

The REST `AsyncDataFeed` polls on the scan interval (minutes); this pushes live
ticks so the dashboard and open-position mark-to-market move continuously between
scans. Each symbol has its own resilient watch loop (catch, back off, retry) so
one dead subscription can't take down the rest, and the whole thing degrades
gracefully to nothing when ccxt.pro is unavailable — the REST path is always the
source of truth for signals.

Design notes:
  * `exchange` can be injected (tests pass a fake); otherwise a ccxt.pro client
    is created lazily so importing this module never requires a network stack.
  * `on_tick(symbol, price, iso_ts)` may be sync or async.
  * Prices are cached (`self.prices`) so a consumer can read the latest without
    subscribing.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional, Sequence, Union

from ..utils.logging import get_logger, utcnow_iso

log = get_logger("stream")

TickCallback = Callable[[str, float, str], Union[None, Awaitable[None]]]

_MAX_BACKOFF = 30.0


class StreamingFeed:
    def __init__(self, exchange_id: str = "bybit", market_type: str = "swap",
                 api_key: str = "", api_secret: str = "", exchange=None):
        self._exchange_id = exchange_id
        self._market_type = market_type
        self._api_key = api_key
        self._api_secret = api_secret
        self._ex = exchange
        self.prices: dict[str, float] = {}
        self.price_ts: dict[str, str] = {}
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------ #
    @staticmethod
    def available() -> bool:
        """True if ccxt.pro can be imported (streaming is possible)."""
        try:
            import ccxt.pro  # noqa: F401
            return True
        except Exception:
            return False

    def _exchange(self):
        if self._ex is None:
            import ccxt.pro as ccxtpro
            klass = getattr(ccxtpro, self._exchange_id)
            self._ex = klass({
                "apiKey": self._api_key, "secret": self._api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": self._market_type},
            })
        return self._ex

    # ------------------------------------------------------------------ #
    async def watch_prices(self, symbols: Sequence[str],
                           on_tick: TickCallback) -> None:
        """Stream tickers for `symbols`, calling on_tick(symbol, price, iso_ts)
        on every update, until stop() is called. Returns when all loops end."""
        if not symbols:
            return
        await asyncio.gather(*(self._watch_one(s, on_tick) for s in symbols),
                             return_exceptions=True)

    async def _watch_one(self, symbol: str, on_tick: TickCallback) -> None:
        ex = self._exchange()
        backoff = 1.0
        while not self._stop.is_set():
            try:
                ticker = await ex.watch_ticker(symbol)
                price = float(ticker.get("last") or ticker.get("close") or 0.0)
                if price > 0:
                    ts = utcnow_iso()
                    self.prices[symbol] = price
                    self.price_ts[symbol] = ts
                    res = on_tick(symbol, price, ts)
                    if asyncio.iscoroutine(res):
                        await res
                backoff = 1.0
            except asyncio.CancelledError:
                break
            except Exception as e:  # dead sub, rate limit, transient network...
                log.debug("watch_ticker %s failed: %s", symbol, e)
                try:
                    await asyncio.wait_for(self._stop.wait(),
                                           timeout=min(backoff, _MAX_BACKOFF))
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, _MAX_BACKOFF)

    def latest_price(self, symbol: str) -> Optional[float]:
        return self.prices.get(symbol)

    def stop(self) -> None:
        self._stop.set()

    async def close(self) -> None:
        self.stop()
        if self._ex is not None:
            try:
                await self._ex.close()
            except Exception:
                pass
