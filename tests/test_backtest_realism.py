"""Tests for backtest fill realism: next-bar-open entry, gap-through-stop
slippage, and funding accrual (GODMODE_PLAN.md Phase 5).

These build lightweight plan stubs and drive Backtester._simulate directly on
hand-crafted OHLC so the realized R is exact. Fees/slippage are zeroed so the
only thing under test is the realism mechanic itself.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from algotrader.backtest.engine import Backtester
from algotrader.models import RiskConfig, Side, TakeProfit
from algotrader.risk.manager import RiskManager
from algotrader.signals.engine import SignalEngine


def _bt(cfg: RiskConfig, horizon: int = 20) -> Backtester:
    return Backtester(SignalEngine(), RiskManager(cfg), horizon=horizon)


def _plan(side: Side, entry: float, stop: float,
          tps: list[TakeProfit], time_stop: int = 0) -> SimpleNamespace:
    return SimpleNamespace(side=side, entry=entry, stop_loss=stop,
                           take_profits=tps, time_stop_candles=time_stop)


def _df(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    o, h, l, c = zip(*rows)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": np.full(len(rows), 1000.0)})


def _nofees(**kw) -> RiskConfig:
    base = dict(taker_fee=0.0, slippage_pct=0.0, apply_funding=False)
    base.update(kw)
    return RiskConfig(**base)


# --------------------------------------------------------------------------- #
# next-bar-open entry
# --------------------------------------------------------------------------- #
def test_next_bar_open_fills_at_next_open_not_signal_close():
    """A LONG whose next bar OPENS above the signal close enters worse, so the
    same +1R target yields less realized R than a signal-close fill would."""
    # bar1 = signal bar (close 100). bar2 opens at 100.5 and reaches TP 101.
    df = _df([(100, 100, 100, 100), (100, 100, 100, 100),
              (100.5, 101.5, 100.5, 101), (101, 101, 101, 101)])
    tps = [TakeProfit(price=101.0, r_multiple=1.0, allocation=1.0)]  # stop_dist = 1

    legacy = _bt(_nofees(next_bar_open_entry=False))._simulate(
        df, 1, _plan(Side.LONG, 100.0, 99.0, tps), bar_minutes=60)
    nextopen = _bt(_nofees(next_bar_open_entry=True))._simulate(
        df, 1, _plan(Side.LONG, 100.0, 99.0, tps), bar_minutes=60)

    assert legacy[2] == pytest.approx(1.0)     # filled at 100 -> full +1R
    assert nextopen[2] == pytest.approx(0.5)   # filled at 100.5 -> only +0.5R
    assert nextopen[2] < legacy[2]


def test_next_bar_open_on_last_bar_is_unfillable():
    df = _df([(100, 100, 100, 100), (100, 100, 100, 100)])
    tps = [TakeProfit(price=101.0, r_multiple=1.0, allocation=1.0)]
    # entry_i is the final bar -> no next open to fill against -> skip.
    out = _bt(_nofees(next_bar_open_entry=True))._simulate(
        df, 1, _plan(Side.LONG, 100.0, 99.0, tps), bar_minutes=60)
    assert out is None


# --------------------------------------------------------------------------- #
# gap-through-stop slippage
# --------------------------------------------------------------------------- #
def test_gap_through_stop_fills_at_open_not_stop():
    """A bar that gaps below the stop should fill the LONG at the (worse) open,
    turning a nominal -1R stop into the realized gap loss."""
    # bar2 gaps down: opens 98 (below stop 99), low 97.
    df = _df([(100, 100, 100, 100), (100, 100, 100, 100),
              (98, 98, 97, 97.5), (97.5, 97.5, 97.5, 97.5)])
    tps = [TakeProfit(price=102.0, r_multiple=2.0, allocation=1.0)]  # never hit

    gap = _bt(_nofees(next_bar_open_entry=False, gap_fill_stops=True))._simulate(
        df, 1, _plan(Side.LONG, 100.0, 99.0, tps), bar_minutes=60)
    nogap = _bt(_nofees(next_bar_open_entry=False, gap_fill_stops=False))._simulate(
        df, 1, _plan(Side.LONG, 100.0, 99.0, tps), bar_minutes=60)

    assert nogap[2] == pytest.approx(-1.0)   # optimistic: filled exactly at 99
    assert gap[2] == pytest.approx(-2.0)     # realistic: filled at the 98 open
    assert gap[2] < nogap[2]


def test_gap_fill_never_improves_a_normal_stop():
    """When the bar does NOT gap (open above the stop), gap-fill leaves the fill
    at the stop level — it must never hand a better-than-stop price."""
    df = _df([(100, 100, 100, 100), (100, 100, 100, 100),
              (99.5, 99.5, 98.8, 99.0), (99, 99, 99, 99)])
    tps = [TakeProfit(price=102.0, r_multiple=2.0, allocation=1.0)]
    gap = _bt(_nofees(next_bar_open_entry=False, gap_fill_stops=True))._simulate(
        df, 1, _plan(Side.LONG, 100.0, 99.0, tps), bar_minutes=60)
    assert gap[2] == pytest.approx(-1.0)     # open 99.5 > stop 99 -> fill at stop


# --------------------------------------------------------------------------- #
# funding accrual
# --------------------------------------------------------------------------- #
def _funding_cfg(rate: float) -> RiskConfig:
    return RiskConfig(taker_fee=0.0, slippage_pct=0.0, next_bar_open_entry=False,
                      apply_funding=True, funding_rate_8h=rate,
                      funding_interval_hours=8.0)


def test_funding_charges_longs_and_pays_shorts():
    """With one funding interval per bar, holding 3 bars accrues 3x the per-bar
    carry. Longs pay it (R drops); shorts receive it (R rises)."""
    flat = _df([(100, 100, 100, 100)] * 6)  # perfectly flat -> no stop/TP, time-stop closes
    tp_long = [TakeProfit(price=110.0, r_multiple=10.0, allocation=1.0)]   # never hit
    tp_short = [TakeProfit(price=90.0, r_multiple=10.0, allocation=1.0)]   # never hit
    # bar_minutes == interval (480 = 8h) so per-bar carry == funding_rate_8h.
    # stop_dist = 1, entry = 100 -> funding in R = rate * 100 per bar.
    rate = 0.001  # -> 0.1 R per bar; 3 bars held -> 0.3 R

    long_none = _bt(_funding_cfg(0.0))._simulate(
        flat, 1, _plan(Side.LONG, 100.0, 99.0, tp_long, time_stop=3), bar_minutes=480)
    long_fund = _bt(_funding_cfg(rate))._simulate(
        flat, 1, _plan(Side.LONG, 100.0, 99.0, tp_long, time_stop=3), bar_minutes=480)
    short_none = _bt(_funding_cfg(0.0))._simulate(
        flat, 1, _plan(Side.SHORT, 100.0, 101.0, tp_short, time_stop=3), bar_minutes=480)
    short_fund = _bt(_funding_cfg(rate))._simulate(
        flat, 1, _plan(Side.SHORT, 100.0, 101.0, tp_short, time_stop=3), bar_minutes=480)

    assert long_none[2] == pytest.approx(0.0, abs=1e-9)
    assert long_fund[2] == pytest.approx(-0.3, abs=1e-6)   # long pays 3x0.1R
    assert short_none[2] == pytest.approx(0.0, abs=1e-9)
    assert short_fund[2] == pytest.approx(0.3, abs=1e-6)   # short receives it


def test_funding_scales_with_bar_duration():
    """A 1h bar accrues 1/8 the carry of an 8h bar over the same bar count."""
    flat = _df([(100, 100, 100, 100)] * 6)
    tps = [TakeProfit(price=110.0, r_multiple=10.0, allocation=1.0)]
    r8 = _bt(_funding_cfg(0.001))._simulate(
        flat, 1, _plan(Side.LONG, 100.0, 99.0, tps, time_stop=3), bar_minutes=480)
    r1 = _bt(_funding_cfg(0.001))._simulate(
        flat, 1, _plan(Side.LONG, 100.0, 99.0, tps, time_stop=3), bar_minutes=60)
    assert r1[2] == pytest.approx(r8[2] / 8.0, rel=1e-6)
