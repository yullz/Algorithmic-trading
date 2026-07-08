"""Tests for forward order-flow collection (funding/OI/basis snapshots)."""
from __future__ import annotations

from algotrader.data.orderflow import (
    append_snapshot, load_orderflow, parse_tickers)

_ROWS = [
    {"symbol": "BTCUSDT", "lastPrice": "63000", "markPrice": "62980",
     "indexPrice": "63008", "fundingRate": "0.00005", "fundingIntervalHour": "8",
     "nextFundingTime": "1783497600000", "openInterest": "54547.6",
     "openInterestValue": "3435407620", "basisRate": "-0.0004",
     "volume24h": "90612", "turnover24h": "5742266948", "price24hPcnt": "0.012",
     "prevPrice1h": "62900"},
    {"symbol": "ETHUSDT", "lastPrice": "3000", "markPrice": "3001",
     "indexPrice": "3000", "fundingRate": "-0.0001", "openInterest": "1000",
     "openInterestValue": "3000000", "volume24h": "1", "turnover24h": "1"},
    {"symbol": "JUNK", "lastPrice": None, "markPrice": "", "indexPrice": "0"},
]


def test_parse_tickers_flattens_fields():
    out = parse_tickers(_ROWS, ts="2026-07-08T05:00:00+00:00")
    btc = next(r for r in out if r["symbol"] == "BTCUSDT")
    assert btc["funding_rate"] == 0.00005
    assert btc["open_interest"] == 54547.6
    assert btc["basis_rate"] == -0.0004
    assert btc["funding_interval_h"] == 8.0
    assert btc["ts"] == "2026-07-08T05:00:00+00:00"


def test_parse_tickers_computes_basis_when_absent():
    # ETH has no basisRate -> computed from (mark - index) / index.
    out = parse_tickers(_ROWS, ts="t")
    eth = next(r for r in out if r["symbol"] == "ETHUSDT")
    assert eth["basis_rate"] == (3001 - 3000) / 3000
    # JUNK row has no usable prices -> basis stays None (no crash).
    junk = next(r for r in out if r["symbol"] == "JUNK")
    assert junk["basis_rate"] is None
    assert junk["last"] is None


def test_parse_tickers_symbol_filter():
    out = parse_tickers(_ROWS, ts="t", symbols={"BTCUSDT"})
    assert [r["symbol"] for r in out] == ["BTCUSDT"]


def test_append_and_load_roundtrip_dedupes(tmp_path):
    rows = parse_tickers(_ROWS, ts="2026-07-08T05:00:00+00:00")
    p1 = append_snapshot(rows, cache_dir=str(tmp_path))
    assert p1 and p1.endswith("2026-07-08.parquet")
    # re-append the SAME ts+symbols -> de-duped, not doubled.
    append_snapshot(rows, cache_dir=str(tmp_path))
    # a later snapshot appends new rows.
    append_snapshot(parse_tickers(_ROWS, ts="2026-07-08T06:00:00+00:00"),
                    cache_dir=str(tmp_path))
    df = load_orderflow(cache_dir=str(tmp_path))
    assert set(df["symbol"]) == {"BTCUSDT", "ETHUSDT", "JUNK"}
    # 3 symbols x 2 distinct timestamps = 6 rows (first duplicate collapsed).
    assert len(df) == 6
    assert append_snapshot([], cache_dir=str(tmp_path)) is None


def test_load_orderflow_empty(tmp_path):
    assert load_orderflow(cache_dir=str(tmp_path)).empty
