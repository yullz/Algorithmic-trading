"""Unit tests for algotrader/signals/confluence.py."""
from __future__ import annotations

import math
import warnings

import pytest

from algotrader.models import Bias, Evidence, Side, SetupKind
from algotrader.signals import confluence


def _ev(name: str, bias: Bias, strength: float, family: str,
        source: str = "test", candle_index: int | None = None) -> Evidence:
    return Evidence(name, source, bias, strength, 0.52, family=family,
                    candle_index=candle_index).clamp()


def test_family_based_scoring_boosts_independence():
    """Three correlated trend readings should count as one family; three
    independent families should produce higher confidence and n_families."""
    correlated = [
        _ev("ema_stack_bull", Bias.BULLISH, 0.7, "trend"),
        _ev("supertrend_bull", Bias.BULLISH, 0.7, "trend"),
        _ev("adx_trend_bull", Bias.BULLISH, 0.7, "trend"),
    ]
    independent = [
        _ev("ema_stack_bull", Bias.BULLISH, 0.7, "trend"),
        _ev("volume_spike", Bias.BULLISH, 0.5, "volume"),
        _ev("rsi_oversold", Bias.BULLISH, 0.5, "mean_reversion"),
    ]
    s_corr = confluence.score(correlated)
    s_ind = confluence.score(independent)

    assert s_corr["side"] == Side.LONG
    assert s_ind["side"] == Side.LONG
    assert s_corr["n_families"] == 1
    assert s_ind["n_families"] == 3
    # More independent agreement -> higher confidence (all else bullish).
    assert s_ind["confidence"] > s_corr["confidence"]


def test_confidence_bounds():
    """Confidence is always clamped to [0, 1], even with recalibration."""
    extreme = [
        _ev("e1", Bias.BULLISH, 999.0, "trend"),
        _ev("e2", Bias.BEARISH, 999.0, "volume"),
    ]
    result = confluence.score(extreme)
    assert 0.0 <= result["confidence"] <= 1.0

    empty = confluence.score([])
    assert empty["side"] == Side.FLAT
    assert empty["confidence"] == 0.0

    # Recalibration should also stay bounded.
    recal = confluence.recalibrate_confidence(0.95, extreme, calibration_error=1.0)
    assert 0.0 <= recal <= 1.0
    recal_low = confluence.recalibrate_confidence(0.05, extreme, calibration_error=1.0)
    assert 0.0 <= recal_low <= 1.0


def test_min_families_count_is_exposed():
    """score() exposes n_families so a downstream gate can reject singletons."""
    single_family = [
        _ev("ema_stack_bull", Bias.BULLISH, 0.7, "trend", source="indicator"),
        _ev("adx_trend_bull", Bias.BULLISH, 0.6, "trend", source="indicator"),
    ]
    result = confluence.score(single_family)
    assert result["n_families"] == 1
    # A hypothetical min_families=2 gate would therefore reject this score.


def test_registry_classifies_known_patterns():
    """The setup-kind registry should map known evidence to expected kinds."""
    cases = [
        (Evidence("rsi_oversold", "indicator", Bias.BULLISH, 0.5, 0.52,
                  family="mean_reversion"), SetupKind.MEAN_REVERSION),
        (Evidence("macd_bull_cross", "indicator", Bias.BULLISH, 0.6, 0.53,
                  family="momentum"), SetupKind.MOMENTUM),
        (Evidence("volume_spike", "indicator", Bias.BULLISH, 0.4, 0.51,
                  family="volume"), SetupKind.BREAKOUT),
        (Evidence("double_top", "chart", Bias.BEARISH, 0.62, 0.52,
                  family="chart"), SetupKind.REVERSAL),
        (Evidence("bull_flag", "chart", Bias.BULLISH, 0.6, 0.52,
                  family="chart"), SetupKind.CONTINUATION),
        (Evidence("hammer", "candlestick", Bias.BULLISH, 0.55, 0.52,
                  family="candle_reversal"), SetupKind.REVERSAL),
        (Evidence("htf_uptrend", "structure", Bias.BULLISH, 0.6, 0.57,
                  family="trend"), SetupKind.MOMENTUM),
        (Evidence("fvg_fill_bull", "chart", Bias.BULLISH, 0.5, 0.52,
                  family="liquidity"), SetupKind.CONTINUATION),
    ]
    for ev, expected in cases:
        result = confluence.score([ev])
        assert result["kind"] == expected, f"{ev.name} expected {expected}, got {result['kind']}"


def test_registry_classifies_incoming_modules():
    """Substring registry entries should classify patterns from incoming
    subagent modules without hard-coding every name."""
    cases = [
        ("market_structure_bull", "indicator", SetupKind.BREAKOUT),
        ("volume_breakout_up", "indicator", SetupKind.BREAKOUT),
        ("harmonic_bull", "chart", SetupKind.REVERSAL),
        ("wyckoff_spring", "chart", SetupKind.REVERSAL),
        ("continuation_bull", "chart", SetupKind.CONTINUATION),
    ]
    for name, source, expected in cases:
        ev = Evidence(name, source, Bias.BULLISH, 0.5, 0.52, family="trend")
        result = confluence.score([ev])
        assert result["kind"] == expected, f"{name} expected {expected}, got {result['kind']}"


def test_unknown_family_emits_warning():
    """Evidence with an unknown or missing family should trigger a warning."""
    with pytest.warns(RuntimeWarning, match="no explicit family|unknown family"):
        confluence.score([Evidence("unknown_thing", "test", Bias.BULLISH, 0.5, 0.52)])

    with pytest.warns(RuntimeWarning, match="unknown family"):
        confluence.score([Evidence("bad", "test", Bias.BULLISH, 0.5, 0.52,
                                   family="not_a_family")])


def test_unclassified_name_emits_warning():
    """A name that misses both the registry and the heuristic fallback warns."""
    ev = Evidence("xyz_nonsense", "test", Bias.BULLISH, 0.5, 0.52, family="trend")
    with pytest.warns(RuntimeWarning, match="Cannot classify"):
        confluence.score([ev])


def test_time_decay_weights_reduce_old_evidence():
    """Older evidence contributes less strength when current_index is supplied."""
    current = 10
    fresh = _ev("volume_spike", Bias.BULLISH, 1.0, "volume",
                source="indicator", candle_index=current)
    stale = _ev("volume_spike", Bias.BULLISH, 1.0, "volume",
                source="indicator", candle_index=current - 3)

    fresh_only = confluence.score([fresh], current_index=current)
    stale_only = confluence.score([stale], current_index=current)

    # Both are bullish, same nominal strength, so direction and family count match.
    assert fresh_only["side"] == stale_only["side"] == Side.LONG
    assert fresh_only["n_families"] == stale_only["n_families"] == 1

    # Stale evidence is weighted down, lowering its effective score/confidence.
    assert stale_only["score"] < fresh_only["score"]
    assert stale_only["confidence"] < fresh_only["confidence"]

    # Minimum decay weight is respected: 3-candle gap at 0.2 -> exp(-0.6) ~ 0.55.
    expected_stale_weight = max(math.exp(-0.2 * 3), confluence.MIN_DECAY_WEIGHT)
    assert expected_stale_weight == confluence.MIN_DECAY_WEIGHT or math.isclose(
        stale_only["score"] / fresh_only["score"], expected_stale_weight, rel_tol=1e-9
    )


def test_time_decay_only_when_candle_index_set():
    """When candle_index is missing, decay is not applied."""
    ev = _ev("volume_spike", Bias.BULLISH, 1.0, "volume", source="indicator")
    assert ev.candle_index is None
    result = confluence.score([ev], current_index=99)
    # Without candle_index the weight is 1.0, so score equals raw strength.
    assert result["score"] == pytest.approx(1.0)


def test_recalibration_is_conservative():
    """recalibrate_confidence nudges toward 0.5 but does not override raw."""
    raw = 0.8
    recal = confluence.recalibrate_confidence(raw, [_ev("x", Bias.BULLISH, 0.5, "trend")],
                                              calibration_error=1.0)
    assert recal < raw
    assert recal > 0.5

    raw_low = 0.2
    recal_low = confluence.recalibrate_confidence(
        raw_low, [_ev("x", Bias.BULLISH, 0.5, "trend")], calibration_error=1.0)
    assert recal_low > raw_low
    assert recal_low < 0.5

    # Zero error leaves confidence untouched.
    assert confluence.recalibrate_confidence(0.8, [_ev("x", Bias.BULLISH, 0.5, "trend")],
                                             calibration_error=0.0) == pytest.approx(0.8)
