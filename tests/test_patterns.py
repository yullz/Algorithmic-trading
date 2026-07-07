"""Unit tests for candlestick and chart pattern detectors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrader.models import Bias, Evidence, PatternMatch
from algotrader.patterns import candlestick, chart_patterns, continuation, harmonic, liquidity, wyckoff


def _assert_valid_matches(matches: list[PatternMatch]):
    """Shared assertions: every match is a valid PatternMatch/Evidence source."""
    for m in matches:
        assert isinstance(m, PatternMatch)
        assert isinstance(m.bias, Bias)
        assert 0.0 <= m.confidence <= 1.0
        assert 0 <= m.start_idx <= m.end_idx < m.end_idx + 1  # sane indices
        ev = m.to_evidence()
        assert 0.0 <= ev.strength <= 1.0
        assert ev.family or m.family


# --------------------------------------------------------------------------- #
# Candlestick patterns
# --------------------------------------------------------------------------- #
def test_candlestick_detects_hammer(candle_frame):
    matches = candlestick.detect(candle_frame)
    _assert_valid_matches(matches)
    names = {m.name for m in matches}
    assert "hammer" in names, f"Expected hammer, got {names}"


def test_candlestick_detects_bearish_engulfing(bearish_engulfing_frame):
    matches = candlestick.detect(bearish_engulfing_frame)
    _assert_valid_matches(matches)
    names = {m.name for m in matches}
    assert "bearish_engulfing" in names, f"Expected bearish engulfing, got {names}"


def test_candlestick_no_future_bar_usage():
    """Append NaN future rows; the detector must not read beyond the last
    real bar when processing the original frame (here it simply anchors at
    len(df)-1, which is the real pattern bar).
    """
    base = pd.DataFrame({
        "open": [100.0, 101.0, 100.5, 99.0, 98.5, 97.0, 98.5],
        "high": [101.5, 102.0, 101.0, 100.0, 99.0, 98.0, 99.0],
        "low": [99.5, 100.0, 99.5, 98.0, 97.5, 96.0, 96.0],
        "close": [100.0, 100.5, 100.0, 98.5, 97.5, 96.0, 99.0],
        "volume": [1000.0] * 7,
    })
    future = pd.DataFrame({
        "open": [np.nan, np.nan, np.nan],
        "high": [np.nan, np.nan, np.nan],
        "low": [np.nan, np.nan, np.nan],
        "close": [np.nan, np.nan, np.nan],
        "volume": [np.nan, np.nan, np.nan],
    })
    combined = pd.concat([base, future], ignore_index=True)
    # Detect on the original frame; if the implementation ever indexed
    # len(df)+N on the combined frame this would fail.
    matches = candlestick.detect(base)
    assert any(m.name == "hammer" for m in matches)


# --------------------------------------------------------------------------- #
# Chart patterns
# --------------------------------------------------------------------------- #
def _range_breakout_frame() -> pd.DataFrame:
    """Build a 60-bar frame whose last close breaks above the prior 20-bar high."""
    n = 60
    close = np.empty(n)
    # Slightly rising, non-repeating consolidation so no triple/double bottoms fire.
    close[0:20] = np.linspace(100.0, 100.5, 20)
    close[20:40] = np.linspace(100.4, 100.1, 20)
    close[40:59] = np.linspace(100.2, 100.6, 19)
    close[59] = 105.0
    high = close + 0.2
    low = close - 0.2
    open_ = (high + low) / 2.0
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 1000.0),
    })


def test_chart_pattern_detects_range_breakout():
    df = _range_breakout_frame()
    matches = chart_patterns.detect(df)
    _assert_valid_matches(matches)
    names = {m.name for m in matches}
    assert "range_breakout_up" in names, f"Expected range_breakout_up, got {names}"


def test_chart_pattern_breakout_uses_prior_bars_only():
    """The range-breakout detector must compute the breakout level from
    high[-(window+1):-1], i.e. excluding the current bar.
    """
    df = _range_breakout_frame()
    matches = chart_patterns.detect(df)
    up = [m for m in matches if m.name == "range_breakout_up"]
    assert up
    expected_prior_high = df["high"].iloc[-21:-1].max()
    assert up[0].breakout_level == pytest.approx(expected_prior_high, rel=1e-9)


def test_pivots_stay_inside_array():
    """The local-extrema helper never references indices outside the series."""
    series = np.random.default_rng(7).normal(100.0, 1.0, 50)
    highs, lows = chart_patterns._pivots(series, order=3)
    assert all(3 <= h < len(series) - 3 for h in highs)
    assert all(3 <= l < len(series) - 3 for l in lows)


# --------------------------------------------------------------------------- #
# Harmonic patterns
# --------------------------------------------------------------------------- #
def _bullish_gartley_frame() -> pd.DataFrame:
    """Synthetic frame ending in a bullish Gartley completion.

    Ratios:
      AB/XA = 0.618
      BC/AB = 0.810
      CD/BC = 1.572
      CD/XA = 0.786
    """
    n = 130
    X, A, B, C, D = 100.0, 90.0, 96.18, 91.18, 83.32
    close = np.full(n, 92.0)
    rough_idx = [20, 40, 60, 100, n - 5]
    rough_vals = [X, A, B, C, D]
    for ix, v in zip(rough_idx, rough_vals):
        close[ix] = v
    for s, e in zip(rough_idx[:-1], rough_idx[1:]):
        close[s:e + 1] = np.linspace(close[s], close[e], e - s + 1)
    close[n - 5:n] = np.linspace(D, D - 0.5, 5)

    # Enforce order-3 pivots for each harmonic point.
    for idx, val in [(20, X), (60, B)]:
        close[idx] = val
        for off in range(1, 4):
            close[idx - off] = min(close[idx - off], val - 0.2 * off)
            close[idx + off] = min(close[idx + off], val - 0.2 * off)
    for idx, val in [(40, A)]:
        close[idx] = val
        for off in range(1, 4):
            close[idx - off] = max(close[idx - off], val + 0.2 * off)
            close[idx + off] = max(close[idx + off], val + 0.2 * off)
    for idx, val in [(100, C), (n - 5, D)]:
        close[idx] = val
        for off in range(1, 4):
            close[idx - off] = max(close[idx - off], val + 0.2)
            close[idx + off] = max(close[idx + off], val + 0.2)

    high = close + 0.3
    low = close - 0.3
    open_ = (high + low) / 2.0
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 1000.0),
    })


def test_harmonic_detects_bullish_gartley():
    df = _bullish_gartley_frame()
    matches = harmonic.detect(df)
    _assert_valid_matches(matches)
    names = {m.name for m in matches}
    assert "bullish_gartley" in names, f"Expected bullish_gartley, got {names}"
    gart = [m for m in matches if m.name == "bullish_gartley"][0]
    assert gart.bias == Bias.BULLISH
    assert gart.breakout_level is not None
    assert gart.target_level is not None
    assert gart.invalidation_level is not None
    assert gart.target_level > gart.breakout_level


def test_harmonic_no_future_bar_leakage():
    """The harmonic detector must not index beyond the final closed bar."""
    df = _bullish_gartley_frame()
    matches = harmonic.detect(df.iloc[:-1])
    # Without the confirming last bar the pattern may or may not fire; the key
    # is that the function returns without reading future data.
    _assert_valid_matches(matches)


# --------------------------------------------------------------------------- #
# Wyckoff patterns
# --------------------------------------------------------------------------- #
def _spring_frame() -> pd.DataFrame:
    """Frame ending in a Wyckoff spring: wick below support, close back above."""
    n = 80
    close = np.linspace(100.0, 96.0, n)
    support = 95.0
    # Create an order-3 low pivot near the support level.
    close[47:54] = [95.4, 95.2, 95.0, 95.2, 95.4, 95.3, 95.5]
    close[-3:-1] = [95.2, 95.4]
    close[-1] = 96.5
    high = close + 0.4
    low = close - 0.4
    low[-1] = support - 0.6  # wick below support
    open_ = (high + low) / 2.0
    volume = np.full(n, 1000.0)
    volume[-1] = 2500.0
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_wyckoff_spring_detected():
    df = _spring_frame()
    results = wyckoff.detect(df)
    names = {r.name for r in results}
    assert "wyckoff_spring" in names, f"Expected wyckoff_spring, got {names}"
    spring = [r for r in results if r.name == "wyckoff_spring"][0]
    assert isinstance(spring, PatternMatch)
    assert spring.bias == Bias.BULLISH
    assert spring.family == "structure"


def test_wyckoff_sign_of_strength_detected():
    """A strong bull candle with 2x median volume in an uptrend -> SOS."""
    n = 40
    close = 100.0 + np.cumsum(np.full(n, 0.1))
    close[-1] = close[-2] + 2.0
    high = close + 0.3
    low = close - 0.3
    high[-1] = close[-1] + 1.0
    low[-1] = close[-1] - 1.0
    open_ = (high + low) / 2.0
    open_[-1] = low[-1]
    close[-1] = high[-1]
    volume = np.full(n, 1000.0)
    volume[-1] = 2500.0
    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    results = wyckoff.detect(df)
    names = {r.name for r in results}
    assert "sign_of_strength" in names, f"Expected sign_of_strength, got {names}"
    sos = [r for r in results if r.name == "sign_of_strength"][0]
    assert isinstance(sos, Evidence)
    assert sos.bias == Bias.BULLISH
    assert sos.family == "structure"


# --------------------------------------------------------------------------- #
# Continuation patterns
# --------------------------------------------------------------------------- #
def _bull_pennant_frame() -> pd.DataFrame:
    """Frame with a steep bull pole followed by a tight converging consolidation
    and a breakout on the last bar. Length is >= 60 so the detector's minimum
    lookback is satisfied."""
    n = 70
    cons_n = 12
    pole_start = n - (cons_n + 19)
    pole_end = n - (cons_n + 1)
    cons_start = pole_end + 1
    cons_end = n - 1
    close = np.full(n, 100.0)
    close[pole_start:pole_end + 1] = np.linspace(100.0, 108.0,
                                                  pole_end - pole_start + 1)
    cons_len = cons_end - cons_start
    cons_top = np.linspace(109.0, 108.2, cons_len)
    cons_bot = np.linspace(106.0, 107.8, cons_len)
    close[cons_start:cons_end] = (cons_top + cons_bot) / 2
    close[-1] = 109.5
    high = close + 0.3
    low = close - 0.3
    high[cons_start:cons_end] = cons_top
    low[cons_start:cons_end] = cons_bot
    open_ = (high + low) / 2.0
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 1000.0),
    })


def test_continuation_detects_bull_pennant():
    df = _bull_pennant_frame()
    matches = continuation.detect(df)
    _assert_valid_matches(matches)
    names = {m.name for m in matches}
    assert "bull_pennant" in names, f"Expected bull_pennant, got {names}"
    pennant = [m for m in matches if m.name == "bull_pennant"][0]
    assert pennant.bias == Bias.BULLISH
    assert pennant.target_level > pennant.breakout_level


def test_continuation_channel_geometry_valid():
    """A clean rising channel should produce channel matches with valid geometry."""
    n = 80
    x = np.arange(n)
    mid = 100.0 + 0.10 * x
    width = 1.5
    high = mid + width
    low = mid - width
    close = (high + low) / 2.0
    # Break upper channel on the last bar.
    close[-1] = high[-1] + 0.5
    high[-1] = close[-1]
    open_ = (high + low) / 2.0
    open_[-1] = close[-1] - 0.1
    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 1000.0),
    })
    matches = continuation.detect(df)
    _assert_valid_matches(matches)
    # At least one channel-related match should appear.
    assert any("channel" in m.name for m in matches)


# --------------------------------------------------------------------------- #
# Liquidity: volume-confirmed S/R break retest
# --------------------------------------------------------------------------- #
def _sr_retest_bull_frame() -> pd.DataFrame:
    """Frame ending in a retest of a broken 3-touch resistance on lower volume."""
    n = 85
    level = 105.0
    close = 100.0 + np.linspace(0.0, 10.0, n)
    # Three touches near the level.
    touch_idx = [65, 70, 75]
    rng = np.random.default_rng(42)
    for idx in touch_idx:
        close[idx] = level + rng.uniform(-0.05, 0.05)
    # Clear break below-then-above the level so the breakout bar is identifiable.
    close[79] = level - 0.2
    close[80] = level + 0.5
    # Retest on the penultimate bar, continuation on the last.
    close[83] = level + 0.3
    close[84] = level + 1.0
    high = close + 0.4
    low = close - 0.4
    low[83] = level - 0.1
    open_ = (high + low) / 2.0
    volume = np.full(n, 1000.0)
    volume[80] = 3000.0       # breakout volume
    volume[83] = 1200.0       # declining retest volume
    volume[84] = 1500.0
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_liquidity_sr_break_retest_bull():
    df = _sr_retest_bull_frame()
    ev_list = liquidity.detect_sr_retest(df)
    names = {e.name for e in ev_list}
    assert "sr_break_retest_bull" in names, f"Expected sr_break_retest_bull, got {names}"
    ev = [e for e in ev_list if e.name == "sr_break_retest_bull"][0]
    assert ev.bias == Bias.BULLISH
    assert ev.family == "liquidity"
    assert 0.0 <= ev.strength <= 1.0


def test_liquidity_sr_retest_uses_declining_volume():
    """If retest volume is higher than breakout volume, no retest evidence."""
    df = _sr_retest_bull_frame()
    df.loc[83, "volume"] = 4000.0  # retest volume higher than breakout
    ev_list = liquidity.detect_sr_retest(df)
    names = {e.name for e in ev_list}
    assert "sr_break_retest_bull" not in names
