"""Tests for the ccxt.pro streaming feed (mocked exchange — no network)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from algotrader.data.stream import StreamingFeed


def test_streaming_feed_pushes_ticks_and_caches_price():
    ex = MagicMock()
    ticks = [{"last": 100.0}, {"last": 101.0}, {"last": 102.0}]
    state = {"i": 0}

    async def fake_watch_ticker(symbol):
        i = state["i"]
        state["i"] += 1
        if i < len(ticks):
            return ticks[i]
        await asyncio.sleep(0.01)
        raise asyncio.CancelledError()

    ex.watch_ticker = fake_watch_ticker
    ex.close = AsyncMock()

    feed = StreamingFeed(exchange=ex)
    received = []

    def on_tick(sym, price, ts):
        received.append((sym, price, ts))
        if len(received) >= 3:
            feed.stop()

    asyncio.run(feed.watch_prices(["BTC/USDT:USDT"], on_tick))
    assert [p for _, p, _ in received] == [100.0, 101.0, 102.0]
    assert feed.latest_price("BTC/USDT:USDT") == 102.0
    assert feed.price_ts["BTC/USDT:USDT"]


def test_streaming_feed_recovers_from_errors():
    """A watch loop that raises keeps retrying instead of dying."""
    ex = MagicMock()
    state = {"i": 0}

    async def flaky(symbol):
        state["i"] += 1
        if state["i"] == 1:
            raise RuntimeError("transient socket error")
        return {"last": 50.0}

    ex.watch_ticker = flaky
    feed = StreamingFeed(exchange=ex)
    got = []

    def on_tick(sym, price, ts):
        got.append(price)
        feed.stop()

    # Shrink the backoff wait so the retry is fast.
    import algotrader.data.stream as stream_mod
    orig = stream_mod._MAX_BACKOFF
    stream_mod._MAX_BACKOFF = 0.05
    try:
        asyncio.run(feed.watch_prices(["X/USDT:USDT"], on_tick))
    finally:
        stream_mod._MAX_BACKOFF = orig
    assert got == [50.0]  # recovered after the first error


def test_streaming_feed_empty_symbols_is_noop():
    feed = StreamingFeed(exchange=MagicMock())
    asyncio.run(feed.watch_prices([], lambda *a: None))  # must not raise


def test_available_true_when_ccxt_pro_present():
    assert StreamingFeed.available() is True
