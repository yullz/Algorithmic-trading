"""Liquidity-based patterns: swing failures (stop hunts), order-block retests,
fair-value-gap fills, and volume-confirmed S/R break retests.

These "smart money" concepts overlap heavily with each other, so this module
emits AT MOST one match per direction — the strongest — instead of spamming
confluence with three names for the same wick.

No lookahead: everything is judged off the latest closed bar against structure
strictly in the past.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Bias, Evidence, PatternMatch, SetupKind

from .chart_patterns import _atr, _pivots, get_sr_levels


def detect(df: pd.DataFrame, lookback: int = 60) -> list[PatternMatch]:
    if len(df) < 40:
        return []
    d = df.iloc[-lookback:] if len(df) > lookback else df
    o = d["open"].to_numpy()
    h = d["high"].to_numpy()
    l = d["low"].to_numpy()
    c = d["close"].to_numpy()
    base = len(df) - len(d)
    last = len(d) - 1
    atr = _atr(h, l, c)

    bulls: list[PatternMatch] = []
    bears: list[PatternMatch] = []

    hi_idx, lo_idx = _pivots(c, 3)

    # ------------------------------------------------------------------ #
    # Swing failure pattern (liquidity sweep): wick through a prior pivot,
    # close back on the right side of it.
    # ------------------------------------------------------------------ #
    for j in (last, last - 1):                     # sweep on this bar or the previous
        swept_lows = [p for p in lo_idx if p < j - 1 and p >= last - 40]
        for p in reversed(swept_lows):
            pivot_low = float(l[p])
            if l[j] < pivot_low and c[j] > pivot_low and c[last] > pivot_low:
                opposing = [float(h[q]) for q in hi_idx if q > p]
                bulls.append(PatternMatch(
                    "sfp_long", SetupKind.REVERSAL, Bias.BULLISH, 0.6,
                    base + p, base + j,
                    breakout_level=pivot_low,
                    target_level=max(opposing) if opposing else None,
                    invalidation_level=float(l[j]),
                    note="swept a prior low, closed back above (stop hunt)",
                    family="liquidity"))
                break
        swept_highs = [p for p in hi_idx if p < j - 1 and p >= last - 40]
        for p in reversed(swept_highs):
            pivot_high = float(h[p])
            if h[j] > pivot_high and c[j] < pivot_high and c[last] < pivot_high:
                opposing = [float(l[q]) for q in lo_idx if q > p]
                bears.append(PatternMatch(
                    "sfp_short", SetupKind.REVERSAL, Bias.BEARISH, 0.6,
                    base + p, base + j,
                    breakout_level=pivot_high,
                    target_level=min(opposing) if opposing else None,
                    invalidation_level=float(h[j]),
                    note="swept a prior high, closed back below (stop hunt)",
                    family="liquidity"))
                break
        if bulls or bears:
            break

    # ------------------------------------------------------------------ #
    # Order block retest: the last opposing candle before an impulsive move
    # defines a zone; a return to that zone that holds is an entry.
    # ------------------------------------------------------------------ #
    for j in range(last - 5, max(last - 40, 5), -1):
        impulse_up = float(c[min(j + 5, last)] - c[j]) >= 2.5 * atr
        if impulse_up and c[j] < o[j]:             # bearish candle before up-impulse
            zone_hi, zone_lo = float(o[j]), float(l[j])
            touched = any(l[k] <= zone_hi for k in (last - 1, last))
            if touched and c[last] > zone_hi and c[last] > o[last]:
                bulls.append(PatternMatch(
                    "bullish_ob_retest", SetupKind.CONTINUATION, Bias.BULLISH, 0.55,
                    base + j, base + last,
                    breakout_level=zone_hi, target_level=None,
                    invalidation_level=zone_lo,
                    note="retested the order block that launched the impulse",
                    family="liquidity"))
            break
    for j in range(last - 5, max(last - 40, 5), -1):
        impulse_dn = float(c[j] - c[min(j + 5, last)]) >= 2.5 * atr
        if impulse_dn and c[j] > o[j]:             # bullish candle before down-impulse
            zone_lo, zone_hi = float(o[j]), float(h[j])
            touched = any(h[k] >= zone_lo for k in (last - 1, last))
            if touched and c[last] < zone_lo and c[last] < o[last]:
                bears.append(PatternMatch(
                    "bearish_ob_retest", SetupKind.CONTINUATION, Bias.BEARISH, 0.55,
                    base + j, base + last,
                    breakout_level=zone_lo, target_level=None,
                    invalidation_level=zone_hi,
                    note="retested the order block that launched the impulse",
                    family="liquidity"))
            break

    # ------------------------------------------------------------------ #
    # Fair value gap fill: 3-bar gap, price returns to fill >=50% of it and
    # closes back in the gap's direction.
    # ------------------------------------------------------------------ #
    for j in range(last - 2, max(last - 30, 2), -1):
        gap_lo, gap_hi = float(h[j - 2]), float(l[j])
        if gap_hi > gap_lo + 0.2 * atr:            # bullish FVG (up-imbalance)
            mid = (gap_lo + gap_hi) / 2
            if l[last] <= mid and c[last] > o[last] and c[last] > mid:
                bulls.append(PatternMatch(
                    "fvg_fill_bull", SetupKind.CONTINUATION, Bias.BULLISH, 0.5,
                    base + j - 2, base + last,
                    breakout_level=gap_hi, target_level=None,
                    invalidation_level=gap_lo,
                    note="bullish fair value gap filled and defended",
                    family="liquidity"))
            break
    for j in range(last - 2, max(last - 30, 2), -1):
        gap_hi, gap_lo = float(l[j - 2]), float(h[j])
        if gap_hi > gap_lo + 0.2 * atr:            # bearish FVG (down-imbalance)
            mid = (gap_lo + gap_hi) / 2
            if h[last] >= mid and c[last] < o[last] and c[last] < mid:
                bears.append(PatternMatch(
                    "fvg_fill_bear", SetupKind.CONTINUATION, Bias.BEARISH, 0.5,
                    base + j - 2, base + last,
                    breakout_level=gap_lo, target_level=None,
                    invalidation_level=gap_hi,
                    note="bearish fair value gap filled and defended",
                    family="liquidity"))
            break

    out: list[PatternMatch] = []
    if bulls:
        out.append(max(bulls, key=lambda m: m.confidence))
    if bears:
        out.append(max(bears, key=lambda m: m.confidence))
    return out


def detect_sr_retest(df: pd.DataFrame, lookback: int = 120) -> list[Evidence]:
    """Volume-confirmed S/R break retest.

    After a horizontal S/R level (as found by chart_patterns.get_sr_levels) is
    broken, look for a retest of the broken level on declining volume followed
    by continuation. This emits an Evidence object rather than a PatternMatch
    because the original break is already the structural anchor.
    """
    if len(df) < 40:
        return []
    d = df.iloc[-lookback:] if len(df) > lookback else df
    h = d["high"].to_numpy()
    l = d["low"].to_numpy()
    c = d["close"].to_numpy()
    vol = d["volume"].to_numpy()
    base = len(df) - len(d)
    last = len(d) - 1
    atr = _atr(h, l, c)

    levels = get_sr_levels(df, lookback=lookback, max_levels=12)
    if not levels:
        return []

    out: list[Evidence] = []
    vol_median = float(np.median(vol[-21:-1])) if len(vol) > 21 else float(vol.mean())

    for level, touches in levels:
        # We need evidence of a prior break (price crossed the level in the last
        # 15 bars), then a retest in the last 1-2 bars on lower volume, and the
        # current bar continuing in the break direction.
        start = max(0, last - 15)
        cross_window = c[start:last + 1]
        crossed = bool(np.any(cross_window <= level) and np.any(cross_window >= level))
        if not crossed:
            continue

        # Determine break direction: where is the current close vs the level?
        broke_bull = c[last] > level
        broke_bear = c[last] < level

        # Retest: low/high kissed the level on the previous bar but close held
        # the right side.
        retest_bull = (l[last - 1] <= level <= h[last - 1]
                       and c[last - 1] > level
                       and c[last] > c[last - 1])
        retest_bear = (l[last - 1] <= level <= h[last - 1]
                       and c[last - 1] < level
                       and c[last] < c[last - 1])

        # Declining volume on retest vs the breakout bar. The breakout bar is
        # the first close in the cross window that crosses the level (previous
        # close on the opposite side).
        abs_break_idx: int | None = None
        for k in range(start + 1, last + 1):
            if broke_bull and c[k - 1] <= level < c[k]:
                abs_break_idx = k
                break
            if broke_bear and c[k - 1] >= level > c[k]:
                abs_break_idx = k
                break
        if abs_break_idx is None:
            continue
        break_vol = float(vol[abs_break_idx])
        retest_vol = float(vol[last - 1])
        declining = break_vol > 0 and retest_vol < 0.8 * break_vol

        conf = 0.5 + 0.04 * min(touches, 5)
        if broke_bull and retest_bull and declining:
            out.append(Evidence(
                name="sr_break_retest_bull",
                source="liquidity",
                bias=Bias.BULLISH,
                strength=min(conf, 1.0),
                base_win_rate=0.55,
                note=f"retest of broken {touches}-touch S/R on declining volume, continuation up",
                family="liquidity",
            ).clamp())
        elif broke_bear and retest_bear and declining:
            out.append(Evidence(
                name="sr_break_retest_bear",
                source="liquidity",
                bias=Bias.BEARISH,
                strength=min(conf, 1.0),
                base_win_rate=0.55,
                note=f"retest of broken {touches}-touch S/R on declining volume, continuation down",
                family="liquidity",
            ).clamp())

    # If multiple levels fit, keep only the strongest per direction.
    best_bull = max((e for e in out if e.bias == Bias.BULLISH),
                    key=lambda e: e.strength, default=None)
    best_bear = max((e for e in out if e.bias == Bias.BEARISH),
                    key=lambda e: e.strength, default=None)
    result: list[Evidence] = []
    if best_bull is not None:
        result.append(best_bull)
    if best_bear is not None:
        result.append(best_bear)
    return result
