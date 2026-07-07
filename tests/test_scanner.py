"""Tests for the Scanner's ranking objective (the 'pick the best trades' core)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from algotrader import regime as regime_mod
from algotrader.models import Side
from algotrader.scanner import Scanner


def _candidate(ml_ev_r, ev_r=0.5, confidence=0.8, side=Side.LONG):
    plan = SimpleNamespace(side=side, expected_value_r=ev_r, confidence=confidence)
    signal = SimpleNamespace(ml_ev_r=ml_ev_r)
    return {"plan": plan, "signal": signal, "symbol": "X/USDT:USDT", "tf": "1h"}


def test_rank_uses_ml_reward_head_when_available():
    """When the reward head is trusted, rank = predicted per-trade E[R] * bias,
    replacing the old base_win_rate*confidence*bias heuristic."""
    bias = regime_mod.market_bias_factor(Side.LONG, "range")
    c = _candidate(ml_ev_r=1.4, ev_r=0.5, confidence=0.8)
    assert Scanner._rank_of(c, "range") == pytest.approx(1.4 * bias)


def test_rank_falls_back_to_heuristic_without_reward_head():
    bias = regime_mod.market_bias_factor(Side.LONG, "range")
    c = _candidate(ml_ev_r=None, ev_r=0.5, confidence=0.8)
    assert Scanner._rank_of(c, "range") == pytest.approx(0.5 * 0.8 * bias)


def test_rank_prefers_higher_predicted_ev():
    """A candidate with higher predicted E[R] outranks a lower one (same side)."""
    hi = _candidate(ml_ev_r=2.0)
    lo = _candidate(ml_ev_r=0.3)
    assert Scanner._rank_of(hi, "range") > Scanner._rank_of(lo, "range")
