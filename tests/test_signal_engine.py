"""End-to-end tests for SignalEngine.generate().

This closes the biggest coverage hole found in the audit: the central
evidence->Signal function had ZERO tests, which hid a guaranteed NameError on
the ML-blend path (engine.py referenced an undefined `sl`). The
`test_generate_ml_blend_*` cases exercise that exact branch.
"""
from __future__ import annotations

import pytest

from algotrader.data.feed import DataFeed
from algotrader.models import Bias, Evidence, Side, SetupKind
from algotrader.signals import confluence
from algotrader.signals.engine import SignalEngine


def _frame(n: int = 160, seed: int = 1):
    return DataFeed.synthetic(n, seed=seed, timeframe_minutes=60)


def _long_agg() -> dict:
    """A forced confluence result that passes every gate for a LONG."""
    agreeing = [
        Evidence("ema_stack_bull", "indicator", Bias.BULLISH, 0.8, 0.55, family="trend"),
        Evidence("rsi_oversold", "indicator", Bias.BULLISH, 0.6, 0.55, family="mean_reversion"),
        Evidence("macd_bull_cross", "indicator", Bias.BULLISH, 0.7, 0.55, family="momentum"),
    ]
    return {
        "side": Side.LONG,
        "agreeing": agreeing,
        "n_families": 3,
        "confidence": 0.9,
        "kind": SetupKind.BREAKOUT,
        "score": 2.5,
        "families": ["trend", "mean_reversion", "momentum"],
    }


class _StubMeta:
    """Minimal stand-in for MetaModel, returning a fixed (prob, weight, contribs)."""

    def __init__(self):
        self.seen_entry_time = None

    def predict_for_signal(self, **kwargs):
        # C1 regression: the engine builds entry_time=str(indf.index[-1]).
        # If that referenced an undefined variable this call never happens.
        self.seen_entry_time = kwargs.get("entry_time")
        return (0.62, 0.3, ["ema_stack_bull (+0.10)"])


def test_generate_ml_blend_branch_does_not_crash(monkeypatch):
    """The flagship ML path must run without NameError and must blend the prob."""
    monkeypatch.setattr(confluence, "score", lambda *a, **k: _long_agg())
    stub = _StubMeta()
    eng = SignalEngine(min_confidence=0.5, min_confluence=3, min_families=2,
                       regime_gating=False, meta_model=stub)

    sig = eng.generate(_frame(), "BTC/USDT:USDT", "1h")

    assert sig is not None, "forced LONG agg should produce a Signal"
    assert sig.side == Side.LONG
    # The ML branch executed and blended.
    assert sig.ml_prob == pytest.approx(0.62)
    assert sig.ml_weight == pytest.approx(0.3)
    assert sig.ml_contribs == ["ema_stack_bull (+0.10)"]
    # entry_time was actually passed as a non-empty string (the C1 fix).
    assert isinstance(stub.seen_entry_time, str) and stub.seen_entry_time


def test_generate_without_meta_model_is_rules_only(monkeypatch):
    monkeypatch.setattr(confluence, "score", lambda *a, **k: _long_agg())
    eng = SignalEngine(min_confidence=0.5, min_confluence=3, min_families=2,
                       regime_gating=False, meta_model=None)
    sig = eng.generate(_frame(), "BTC/USDT:USDT", "1h")
    assert sig is not None
    assert sig.ml_prob is None
    assert sig.ml_weight == 0.0


def test_generate_returns_none_when_flat(monkeypatch):
    flat = {**_long_agg(), "side": Side.FLAT}
    monkeypatch.setattr(confluence, "score", lambda *a, **k: flat)
    eng = SignalEngine(regime_gating=False)
    assert eng.generate(_frame(), "BTC/USDT:USDT", "1h") is None


def test_generate_family_gate_blocks_single_family(monkeypatch):
    """A signal with fewer independent families than min_families is rejected."""
    one_family = {**_long_agg(), "n_families": 1}
    monkeypatch.setattr(confluence, "score", lambda *a, **k: one_family)
    eng = SignalEngine(min_confidence=0.5, min_confluence=3, min_families=2,
                       regime_gating=False)
    assert eng.generate(_frame(), "BTC/USDT:USDT", "1h") is None


def test_generate_confidence_gate(monkeypatch):
    low_conf = {**_long_agg(), "confidence": 0.2}
    monkeypatch.setattr(confluence, "score", lambda *a, **k: low_conf)
    eng = SignalEngine(min_confidence=0.55, min_confluence=3, min_families=2,
                       regime_gating=False)
    assert eng.generate(_frame(), "BTC/USDT:USDT", "1h") is None


def test_generate_too_few_bars_returns_none():
    eng = SignalEngine(regime_gating=False)
    assert eng.generate(_frame(n=40), "BTC/USDT:USDT", "1h") is None
