"""Market-structure detector: swing pivots and trend-of-swing progression.

Detects order-3 fractal swing highs/lows and compares the most recent swing
to the previous one to classify higher highs, higher lows, lower highs, and
lower lows. Raw swing-high/swing-low evidence is also emitted whenever a fresh
pivot is present at the current bar.

Design notes:
  * Order-3 fractal pivots require 3 confirming bars on each side, so the
    current bar can never be a pivot — no lookahead.
  * Only the most recent two swings of each type are compared; this is a
    simple, robust trend-of-structure read rather than a full Elliott-wave
    parser.
  * base_win_rate values are conservative priors; calibration overwrites them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Bias, Evidence

_ORDER = 3
_FRESH_BARS = 10  # pivot must be within this many bars to be considered fresh


def _pivots(series: np.ndarray, order: int = _ORDER):
    """Return (high_idx, low_idx) local extrema with `order` bars each side."""
    highs, lows = [], []
    n = len(series)
    for i in range(order, n - order):
        window = series[i - order:i + order + 1]
        if series[i] == window.max() and window.argmax() == order:
            highs.append(i)
        if series[i] == window.min() and window.argmin() == order:
            lows.append(i)
    return highs, lows


def read_evidence(df: pd.DataFrame) -> list[Evidence]:
    """Emit swing-pivot and trend-of-structure Evidence.

    Operates on the raw OHLCV frame (structural pivots are price-based, not
    indicator-based). Returns [] when the frame is too short for order-3
    fractals.
    """
    if len(df) < 2 * _ORDER + 4 or "close" not in df.columns:
        return []
    close = df["close"].to_numpy(dtype=float)
    last = len(close) - 1
    hi_idx, lo_idx = _pivots(close, _ORDER)
    out: list[Evidence] = []

    def _add(name: str, bias: Bias, strength: float, base: float, note: str):
        out.append(Evidence(name, "structure", bias, strength, base, note,
                            family="structure").clamp())

    # Raw swing high / swing low if the most recent confirmed pivot is fresh.
    if hi_idx and (last - hi_idx[-1] <= _FRESH_BARS):
        _add("swing_high", Bias.BEARISH, 0.35, 0.51,
             f"fractal swing high at {close[hi_idx[-1]]:.6g}")
    if lo_idx and (last - lo_idx[-1] <= _FRESH_BARS):
        _add("swing_low", Bias.BULLISH, 0.35, 0.51,
             f"fractal swing low at {close[lo_idx[-1]]:.6g}")

    # Trend-of-structure: compare last two swing highs and last two swing lows.
    if len(hi_idx) >= 2:
        h1, h2 = hi_idx[-2], hi_idx[-1]
        if last - h2 <= _FRESH_BARS:
            if close[h2] > close[h1]:
                _add("higher_high", Bias.BULLISH, 0.55, 0.54,
                     f"HH {close[h1]:.6g} -> {close[h2]:.6g}")
            elif close[h2] < close[h1]:
                _add("lower_high", Bias.BEARISH, 0.55, 0.54,
                     f"LH {close[h1]:.6g} -> {close[h2]:.6g}")

    if len(lo_idx) >= 2:
        l1, l2 = lo_idx[-2], lo_idx[-1]
        if last - l2 <= _FRESH_BARS:
            if close[l2] > close[l1]:
                _add("higher_low", Bias.BULLISH, 0.55, 0.54,
                     f"HL {close[l1]:.6g} -> {close[l2]:.6g}")
            elif close[l2] < close[l1]:
                _add("lower_low", Bias.BEARISH, 0.55, 0.54,
                     f"LL {close[l1]:.6g} -> {close[l2]:.6g}")

    return out
