"""Wyckoff structural pattern detection.

Detects:
  * spring          — sharp below-support sweep that reclaims on volume
  * upthrust        — above-resistance sweep that fails
  * sign_of_strength — climax bull bar on high volume
  * sign_of_weakness — climax bear bar on high volume

All readings are anchored on the latest closed bar and use structure strictly
from the past.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Bias, Evidence, PatternMatch, SetupKind
from .chart_patterns import _atr, _pivots


def detect(df: pd.DataFrame, lookback: int = 80) -> list[PatternMatch | Evidence]:
    """Return Wyckoff Evidence/PatternMatch objects for the current bar."""
    if len(df) < 40:
        return []
    d = df.iloc[-lookback:] if len(df) > lookback else df
    o = d["open"].to_numpy()
    h = d["high"].to_numpy()
    l = d["low"].to_numpy()
    c = d["close"].to_numpy()
    vol = d["volume"].to_numpy()
    base = len(df) - len(d)
    last = len(d) - 1
    atr = _atr(h, l, c)
    mean_price = max(float(c.mean()), 1e-9)

    hi_idx, lo_idx = _pivots(c, 3)
    out: list[PatternMatch | Evidence] = []

    # Recent volume context: today's volume vs the prior 20-bar median.
    vol_median = float(np.median(vol[-21:-1])) if len(vol) > 21 else float(vol.mean())
    vol_mult = vol[-1] / max(vol_median, 1e-9)
    body = abs(c[-1] - o[-1])
    rng = max(h[-1] - l[-1], 1e-9)
    large_bar = body >= 0.6 * rng and rng >= 1.0 * atr

    # ----------------------------------------------------------------------- #
    # Spring / Upthrust: stop-run beyond a recent structural level with close
    # back on the right side.
    # ----------------------------------------------------------------------- #
    # Use recent swing lows/highs from order-3 pivots.
    recent_lo = [p for p in lo_idx if last - 40 <= p < last - 1]
    recent_hi = [p for p in hi_idx if last - 40 <= p < last - 1]

    if recent_lo:
        support = float(max(c[p] for p in recent_lo[-3:])) if len(recent_lo) >= 3 else float(c[recent_lo[-1]])
        # Wick clearly below support, close back above it, volume confirms.
        if l[-1] < support - 0.3 * atr and c[-1] > support and vol_mult >= 1.3:
            out.append(PatternMatch(
                "wyckoff_spring", SetupKind.REVERSAL, Bias.BULLISH,
                min(0.55 + 0.15 * vol_mult, 1.0),
                base + recent_lo[-1], base + last,
                breakout_level=float(support),
                target_level=float(c[-1] + 2 * atr),
                invalidation_level=float(min(l[-1], support - 0.5 * atr)),
                note="spring: stop hunt below support reclaimed on volume",
                family="structure",
            ))

    if recent_hi:
        resistance = float(min(c[p] for p in recent_hi[-3:])) if len(recent_hi) >= 3 else float(c[recent_hi[-1]])
        if h[-1] > resistance + 0.3 * atr and c[-1] < resistance and vol_mult >= 1.3:
            out.append(PatternMatch(
                "wyckoff_upthrust", SetupKind.REVERSAL, Bias.BEARISH,
                min(0.55 + 0.15 * vol_mult, 1.0),
                base + recent_hi[-1], base + last,
                breakout_level=float(resistance),
                target_level=float(c[-1] - 2 * atr),
                invalidation_level=float(max(h[-1], resistance + 0.5 * atr)),
                note="upthrust: stop hunt above resistance failed on volume",
                family="structure",
            ))

    # ----------------------------------------------------------------------- #
    # Sign of Strength / Weakness: climax bar on high volume in the direction
    # of the local trend.
    # ----------------------------------------------------------------------- #
    # Local trend: use 10-bar slope.
    if len(c) >= 12:
        trend_slope = np.polyfit(np.arange(10), c[-10:], 1)[0]
    else:
        trend_slope = 0.0

    if large_bar and vol_mult >= 1.5:
        if c[-1] > o[-1] and trend_slope >= 0:
            out.append(Evidence(
                name="sign_of_strength",
                source="structure",
                bias=Bias.BULLISH,
                strength=min(0.5 + 0.1 * vol_mult, 1.0),
                base_win_rate=0.55,
                note=f"climax bull bar, {vol_mult:.1f}x volume",
                family="structure",
            ).clamp())
        elif c[-1] < o[-1] and trend_slope <= 0:
            out.append(Evidence(
                name="sign_of_weakness",
                source="structure",
                bias=Bias.BEARISH,
                strength=min(0.5 + 0.1 * vol_mult, 1.0),
                base_win_rate=0.55,
                note=f"climax bear bar, {vol_mult:.1f}x volume",
                family="structure",
            ).clamp())

    return out
