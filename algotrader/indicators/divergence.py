"""Oscillator divergence detection (RSI, MACD histogram, OBV).

A divergence is a disagreement between price structure and an oscillator at
matching swing pivots:

  * Regular bullish:  lower price low  + higher oscillator low  -> reversal up
  * Regular bearish:  higher price high + lower oscillator high -> reversal down
  * Hidden bullish:   HIGHER price low + LOWER oscillator low   -> uptrend continuation
  * Hidden bearish:   lower price high + higher oscillator high -> downtrend continuation

Honesty/design notes:
  * Pivots are order-3 fractals on CLOSE (same discipline as
    patterns/chart_patterns.py), so a pivot only exists once 3 bars have
    printed after it — detection never peeks forward.
  * The most recent pivot must be FRESH (within the last 10 bars) and not yet
    invalidated: if the current close has already traded beyond the pivot
    against the divergence direction, the structure is broken and the signal
    is dead, not "still forming".
  * At most ONE divergence is emitted per oscillator (the strongest, regular
    beating hidden, ties broken by recency) — three flavours of the same RSI
    disagreement are one observation, not three. Cross-oscillator correlation
    is then handled by the shared "divergence" family in confluence.
  * base_win_rate values are conservative priors, overwritten by calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Bias, Evidence

_LOOKBACK = 60          # bars scanned for pivots
_ORDER = 3              # fractal half-width (pivot confirmed 3 bars later)
_FRESH_BARS = 10        # most recent pivot must be at most this old

# evidence label prefix -> indicator-frame column
_OSCILLATORS = (("rsi", "rsi"), ("macd", "macd_hist"), ("obv", "obv"))


def _pivots(series: np.ndarray, order: int = _ORDER):
    """(high_idx, low_idx) local extrema with `order` bars on each side."""
    highs, lows = [], []
    n = len(series)
    for i in range(order, n - order):
        window = series[i - order:i + order + 1]
        if series[i] == window.max() and (window.argmax() == order):
            highs.append(i)
        if series[i] == window.min() and (window.argmin() == order):
            lows.append(i)
    return highs, lows


def read_evidence(indf: pd.DataFrame) -> list[Evidence]:
    """Divergence Evidence from the indicator frame (compute_all output)."""
    if len(indf) < 2 * _ORDER + 4 or "close" not in indf.columns:
        return []
    d = indf.iloc[-_LOOKBACK:]
    close = d["close"].to_numpy(dtype=float)
    last = len(d) - 1
    hi_idx, lo_idx = _pivots(close, _ORDER)

    out: list[Evidence] = []
    for label, col in _OSCILLATORS:
        if col not in d.columns:
            continue
        osc = d[col].to_numpy(dtype=float)
        # (strength, recency, name, bias, base, note) — best one wins
        candidates: list[tuple] = []

        if len(lo_idx) >= 2:
            p1, p2 = lo_idx[-2], lo_idx[-1]
            if (last - p2 <= _FRESH_BARS
                    and not (np.isnan(osc[p1]) or np.isnan(osc[p2]))
                    and close[last] >= close[p2]):   # not already invalidated
                if close[p2] < close[p1] and osc[p2] > osc[p1]:
                    candidates.append((
                        0.6, p2, f"{label}_bullish_divergence", Bias.BULLISH, 0.55,
                        f"lower price low, higher {label} low"))
                elif close[p2] > close[p1] and osc[p2] < osc[p1]:
                    candidates.append((
                        0.5, p2, f"{label}_hidden_bull_divergence", Bias.BULLISH, 0.53,
                        f"higher price low, lower {label} low (continuation)"))

        if len(hi_idx) >= 2:
            h1, h2 = hi_idx[-2], hi_idx[-1]
            if (last - h2 <= _FRESH_BARS
                    and not (np.isnan(osc[h1]) or np.isnan(osc[h2]))
                    and close[last] <= close[h2]):   # not already invalidated
                if close[h2] > close[h1] and osc[h2] < osc[h1]:
                    candidates.append((
                        0.6, h2, f"{label}_bearish_divergence", Bias.BEARISH, 0.55,
                        f"higher price high, lower {label} high"))
                elif close[h2] < close[h1] and osc[h2] > osc[h1]:
                    candidates.append((
                        0.5, h2, f"{label}_hidden_bear_divergence", Bias.BEARISH, 0.53,
                        f"lower price high, higher {label} high (continuation)"))

        if candidates:
            strength, _, name, bias, base, note = max(
                candidates, key=lambda c: (c[0], c[1]))
            out.append(Evidence(name, "indicator", bias, strength, base, note,
                                family="divergence").clamp())
    return out
