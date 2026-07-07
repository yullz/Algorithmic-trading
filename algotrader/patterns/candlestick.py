"""Candlestick pattern detection.

Each detector inspects the last 1-5 candles and returns a PatternMatch if the
pattern is present on the most recent closed bar. Confidence is derived from the
geometry (body/range ratios, gap sizes) so it is a real number, not a constant.

Reversal candles are trend-gated where the textbook demands it: a hammer means
nothing unless something was falling; a shooting star means nothing unless
something was rising. Every match carries an explicit correlation family
(candle_reversal, or momentum for the continuation/thrust patterns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Bias, PatternMatch, SetupKind


def _body(o, c):
    return abs(c - o)


def _range(h, l):
    return max(h - l, 1e-12)


def detect(df: pd.DataFrame) -> list[PatternMatch]:
    """Return candlestick patterns present on the final bar of df."""
    if len(df) < 3:
        return []
    out: list[PatternMatch] = []
    i = len(df) - 1
    o, h, l, c = (df["open"].to_numpy(), df["high"].to_numpy(),
                  df["low"].to_numpy(), df["close"].to_numpy())

    rng = _range(h[i], l[i])
    body = _body(o[i], c[i])
    upper = h[i] - max(o[i], c[i])
    lower = min(o[i], c[i]) - l[i]
    bull = c[i] > o[i]
    # Trend context for the gated reversal candles.
    downswing = i >= 5 and c[i] < c[i - 5]
    upswing = i >= 5 and c[i] > c[i - 5]

    def add(name, kind, bias, conf, note="", family="candle_reversal"):
        out.append(PatternMatch(name, kind, bias, float(min(max(conf, 0), 1)),
                                i, i, note=note, family=family))

    # Doji — indecision
    if body <= 0.1 * rng:
        add("doji", SetupKind.REVERSAL, Bias.NEUTRAL, 0.4, "small body")

    # Hammer (long lower wick) — only meaningful after a downswing
    if lower >= 2 * body and upper <= 0.4 * body and body > 0 and downswing:
        add("hammer", SetupKind.REVERSAL, Bias.BULLISH, 0.55 + 0.2 * (lower / rng),
            "long lower wick after downswing")

    # Shooting star (long upper wick) — only meaningful after an upswing
    if upper >= 2 * body and lower <= 0.4 * body and body > 0 and upswing:
        add("shooting_star", SetupKind.REVERSAL, Bias.BEARISH, 0.55 + 0.2 * (upper / rng),
            "long upper wick after upswing")

    # Marubozu — strong momentum
    if body >= 0.9 * rng:
        add("marubozu", SetupKind.MOMENTUM, Bias.BULLISH if bull else Bias.BEARISH,
            0.6, "full body", family="momentum")

    # Engulfing (2-bar)
    prev_body = _body(o[i - 1], c[i - 1])
    prev_bull = c[i - 1] > o[i - 1]
    if bull and not prev_bull and c[i] >= o[i - 1] and o[i] <= c[i - 1] and body > prev_body:
        add("bullish_engulfing", SetupKind.REVERSAL, Bias.BULLISH,
            0.6 + 0.2 * min(body / max(prev_body, 1e-9) - 1, 1), "engulfs prior bear")
    if not bull and prev_bull and o[i] >= c[i - 1] and c[i] <= o[i - 1] and body > prev_body:
        add("bearish_engulfing", SetupKind.REVERSAL, Bias.BEARISH,
            0.6 + 0.2 * min(body / max(prev_body, 1e-9) - 1, 1), "engulfs prior bull")

    # Harami (2-bar): small body fully inside the prior big body
    inside_prior_body = (max(o[i], c[i]) <= max(o[i - 1], c[i - 1]) and
                         min(o[i], c[i]) >= min(o[i - 1], c[i - 1]))
    if inside_prior_body and prev_body > 0 and body <= 0.6 * prev_body:
        cross = body <= 0.1 * rng
        if not prev_bull:      # bear candle then contained bar -> bullish turn
            add("bullish_harami_cross" if cross else "bullish_harami",
                SetupKind.REVERSAL, Bias.BULLISH, 0.52,
                "small body inside prior bear candle")
        elif prev_bull:
            add("bearish_harami_cross" if cross else "bearish_harami",
                SetupKind.REVERSAL, Bias.BEARISH, 0.52,
                "small body inside prior bull candle")

    # Inside bar (2-bar range compression) — direction unknown, information only
    if h[i] <= h[i - 1] and l[i] >= l[i - 1]:
        add("inside_bar", SetupKind.CONTINUATION, Bias.NEUTRAL, 0.35,
            "range compression", family="momentum")

    # Outside bar closing in its extreme third (2-bar)
    if h[i] > h[i - 1] and l[i] < l[i - 1]:
        if c[i] >= h[i] - rng / 3:
            add("outside_bar_bull", SetupKind.REVERSAL, Bias.BULLISH, 0.5,
                "engulfing range, strong close")
        elif c[i] <= l[i] + rng / 3:
            add("outside_bar_bear", SetupKind.REVERSAL, Bias.BEARISH, 0.5,
                "engulfing range, weak close")

    # Piercing / dark cloud (2-bar)
    mid_prev = (o[i - 1] + c[i - 1]) / 2
    if bull and not prev_bull and o[i] < l[i - 1] and c[i] > mid_prev and c[i] < o[i - 1]:
        add("piercing_line", SetupKind.REVERSAL, Bias.BULLISH, 0.58)
    if not bull and prev_bull and o[i] > h[i - 1] and c[i] < mid_prev and c[i] > o[i - 1]:
        add("dark_cloud_cover", SetupKind.REVERSAL, Bias.BEARISH, 0.58)

    # Tweezer (2-bar)
    if abs(l[i] - l[i - 1]) <= 0.001 * l[i] and bull and not prev_bull:
        add("tweezer_bottom", SetupKind.REVERSAL, Bias.BULLISH, 0.5)
    if abs(h[i] - h[i - 1]) <= 0.001 * h[i] and not bull and prev_bull:
        add("tweezer_top", SetupKind.REVERSAL, Bias.BEARISH, 0.5)

    # Morning / evening star (3-bar)
    small_mid = _body(o[i - 1], c[i - 1]) <= 0.5 * _body(o[i - 2], c[i - 2])
    if (c[i - 2] < o[i - 2]) and small_mid and bull and c[i] > (o[i - 2] + c[i - 2]) / 2:
        add("morning_star", SetupKind.REVERSAL, Bias.BULLISH, 0.66)
    if (c[i - 2] > o[i - 2]) and small_mid and not bull and c[i] < (o[i - 2] + c[i - 2]) / 2:
        add("evening_star", SetupKind.REVERSAL, Bias.BEARISH, 0.66)

    # Three soldiers / crows (3-bar)
    if all(c[j] > o[j] for j in (i - 2, i - 1, i)) and c[i] > c[i - 1] > c[i - 2]:
        add("three_white_soldiers", SetupKind.CONTINUATION, Bias.BULLISH, 0.62,
            family="momentum")
    if all(c[j] < o[j] for j in (i - 2, i - 1, i)) and c[i] < c[i - 1] < c[i - 2]:
        add("three_black_crows", SetupKind.CONTINUATION, Bias.BEARISH, 0.62,
            family="momentum")

    # Rising / falling three methods (5-bar continuation)
    if i >= 4:
        big0 = _body(o[i - 4], c[i - 4])
        rng0 = _range(h[i - 4], l[i - 4])
        mids_inside_up = all(
            _body(o[j], c[j]) < big0 and l[j] >= l[i - 4] and h[j] <= h[i - 4]
            for j in (i - 3, i - 2, i - 1))
        if (c[i - 4] > o[i - 4] and big0 >= 0.6 * rng0 and mids_inside_up
                and bull and c[i] > c[i - 4]):
            add("rising_three_methods", SetupKind.CONTINUATION, Bias.BULLISH, 0.6,
                "pause inside a strong bull bar, then new high close",
                family="momentum")
        if (c[i - 4] < o[i - 4] and big0 >= 0.6 * rng0 and mids_inside_up
                and not bull and c[i] < c[i - 4]):
            add("falling_three_methods", SetupKind.CONTINUATION, Bias.BEARISH, 0.6,
                "pause inside a strong bear bar, then new low close",
                family="momentum")

    return out
