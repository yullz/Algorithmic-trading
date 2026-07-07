"""Continuation pattern detection: pennants and channels.

Pennant: a tight converging consolidation after a steep flagpole; the pole height
projects the measured move. Channel: parallel upper/lower trendlines that bound
a trend, with bounce/break variants.

All geometry uses closed bars only; the current bar confirms the pattern.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Bias, PatternMatch, SetupKind
from .chart_patterns import _atr


def detect(df: pd.DataFrame, lookback: int = 120) -> list[PatternMatch]:
    """Detect pennant and channel patterns on the most recent closed bar."""
    if len(df) < 60:
        return []
    d = df.iloc[-lookback:] if len(df) > lookback else df
    high = d["high"].to_numpy()
    low = d["low"].to_numpy()
    close = d["close"].to_numpy()
    open_ = d["open"].to_numpy()
    base = len(df) - len(d)
    last = len(d) - 1
    last_close = close[-1]
    prev_close = close[-2]
    atr = _atr(high, low, close)
    mean_price = max(float(close.mean()), 1e-9)

    out: list[PatternMatch] = []

    # ----------------------------------------------------------------------- #
    # Pennant: pole + converging consolidation + fresh breakout.
    # ----------------------------------------------------------------------- #
    for cons_n in (5, 8, 12):
        if len(d) < cons_n + 20:
            continue
        pole_start = -cons_n - 19
        pole_end = -cons_n - 1
        pole_move = float(close[pole_end] - close[pole_start])
        if abs(pole_move) < 3 * atr:
            continue

        cons_h = high[pole_end + 1:-1]
        cons_l = low[pole_end + 1:-1]
        if len(cons_h) < 4:
            continue

        x = np.arange(len(cons_h))
        h_coef = np.polyfit(x, cons_h, 1)
        l_coef = np.polyfit(x, cons_l, 1)
        # Converging: slopes have opposite signs and gap narrows.
        h_slope = h_coef[0] / mean_price
        l_slope = l_coef[0] / mean_price
        start_gap = float(np.polyval(h_coef, 0) - np.polyval(l_coef, 0))
        end_gap = float(np.polyval(h_coef, x[-1]) - np.polyval(l_coef, x[-1]))
        if not (start_gap > 0 and end_gap < 0.7 * start_gap):
            continue
        if abs(h_slope) < 1e-6 or abs(l_slope) < 1e-6:
            continue
        if h_slope * l_slope >= 0:
            continue

        # Fresh breakout beyond the consolidation range.
        if pole_move > 0:
            cons_top = float(cons_h.max())
            if prev_close <= cons_top and last_close > cons_top:
                target = cons_top + abs(pole_move)
                out.append(PatternMatch(
                    "bull_pennant", SetupKind.CONTINUATION, Bias.BULLISH, 0.6,
                    base + len(d) + pole_start, base + last,
                    breakout_level=float(cons_top),
                    target_level=float(target),
                    invalidation_level=float(cons_l.min()),
                    note=f"bull pennant after {abs(pole_move)/atr:.1f} ATR pole",
                    family="chart",
                ))
                break
        else:
            cons_bot = float(cons_l.min())
            if prev_close >= cons_bot and last_close < cons_bot:
                target = cons_bot - abs(pole_move)
                out.append(PatternMatch(
                    "bear_pennant", SetupKind.CONTINUATION, Bias.BEARISH, 0.6,
                    base + len(d) + pole_start, base + last,
                    breakout_level=float(cons_bot),
                    target_level=float(target),
                    invalidation_level=float(cons_h.max()),
                    note=f"bear pennant after {abs(pole_move)/atr:.1f} ATR pole",
                    family="chart",
                ))
                break

    # ----------------------------------------------------------------------- #
    # Channel: parallel upper/lower trendlines with >=3 touches each.
    # ----------------------------------------------------------------------- #
    if len(d) >= 30:
        window = min(80, len(d) - 1)
        xs = np.arange(window)
        # Upper trendline through highs.
        h_coef = np.polyfit(xs, high[-window - 1:-1], 1)
        # Lower trendline through lows.
        l_coef = np.polyfit(xs, low[-window - 1:-1], 1)

        h_slope = h_coef[0] / mean_price
        l_slope = l_coef[0] / mean_price
        # Parallel: slopes within 10% of each other.
        if abs(h_slope) > 1e-6 and abs((h_slope - l_slope) / h_slope) <= 0.10:
            up_line_last = float(np.polyval(h_coef, window))
            lo_line_last = float(np.polyval(l_coef, window))
            gap = up_line_last - lo_line_last
            if gap <= 0:
                pass
            elif h_slope > 0.0005 and l_slope > 0.0005:
                # Rising channel.
                if high[-1] >= up_line_last and prev_close <= up_line_last:
                    out.append(PatternMatch(
                        "rising_channel_break_up", SetupKind.BREAKOUT,
                        Bias.BULLISH, 0.55,
                        base + len(d) - window, base + last,
                        breakout_level=float(up_line_last),
                        target_level=float(up_line_last + gap),
                        invalidation_level=float(lo_line_last),
                        note="rising channel upper boundary broken",
                        family="chart",
                    ))
                elif low[-1] <= lo_line_last and prev_close >= lo_line_last and close[-1] > open_[-1]:
                    out.append(PatternMatch(
                        "rising_channel_bounce", SetupKind.CONTINUATION,
                        Bias.BULLISH, 0.55,
                        base + len(d) - window, base + last,
                        breakout_level=float(lo_line_last),
                        target_level=float(up_line_last),
                        invalidation_level=float(lo_line_last - 0.3 * gap),
                        note="bounced off rising channel lower boundary",
                        family="chart",
                    ))
            elif h_slope < -0.0005 and l_slope < -0.0005:
                # Falling channel.
                if low[-1] <= lo_line_last and prev_close >= lo_line_last:
                    out.append(PatternMatch(
                        "falling_channel_break_down", SetupKind.BREAKOUT,
                        Bias.BEARISH, 0.55,
                        base + len(d) - window, base + last,
                        breakout_level=float(lo_line_last),
                        target_level=float(lo_line_last - gap),
                        invalidation_level=float(up_line_last),
                        note="falling channel lower boundary broken",
                        family="chart",
                    ))
                elif high[-1] >= up_line_last and prev_close <= up_line_last and close[-1] < open_[-1]:
                    out.append(PatternMatch(
                        "falling_channel_bounce", SetupKind.CONTINUATION,
                        Bias.BEARISH, 0.55,
                        base + len(d) - window, base + last,
                        breakout_level=float(up_line_last),
                        target_level=float(lo_line_last),
                        invalidation_level=float(up_line_last + 0.3 * gap),
                        note="rejected at falling channel upper boundary",
                        family="chart",
                    ))

    return out
