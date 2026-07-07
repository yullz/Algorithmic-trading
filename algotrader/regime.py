"""Market regime detection and setup gating.

Classifies a symbol/timeframe into one of four regimes using ADX, the
Choppiness Index, an EMA-200 slope, and a realized-volatility percentile:

    trend_up   — directional advance; continuation/breakout longs favored
    trend_down — directional decline; continuation/breakout shorts favored
    range      — sideways; mean reversion + range breakouts favored
    volatile   — high-vol chop; the regime where leveraged accounts die

The regime does two jobs:
  1. Gates setup kinds (`allows_setup`) — e.g. no mean-reversion longs inside
     a trending decline, no continuation trades inside a range.
  2. Conditions calibration keys ("factor|regime|timeframe") so a factor is
     credited only in the regimes where it actually worked historically.

Everything here is computed from a plain OHLCV frame (or an indicator frame
that already carries adx/ema200/atr columns) with no external calls, so the
backtester sees exactly what the live scanner sees.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .models import Side, SetupKind

REGIMES = ("trend_up", "trend_down", "range", "volatile")

_CHOP_PERIOD = 14
_VOL_WINDOW = 100


def choppiness(df: pd.DataFrame, n: int = _CHOP_PERIOD) -> float:
    """Choppiness Index of the last `n` bars. 100 = pure chop, 0 = pure trend.
    >61.8 conventionally = choppy, <38.2 = trending."""
    if len(df) < n + 1:
        return 50.0
    h, l, c = df["high"].iloc[-n:], df["low"].iloc[-n:], df["close"].iloc[-(n + 1):]
    prev_close = c.shift(1).iloc[1:]
    tr = np.maximum(h.values - l.values,
                    np.maximum(abs(h.values - prev_close.values),
                               abs(l.values - prev_close.values)))
    tr_sum = float(tr.sum())
    rng = float(h.max() - l.min())
    if rng <= 0 or tr_sum <= 0:
        return 50.0
    return 100.0 * math.log10(tr_sum / rng) / math.log10(n)


def vol_percentile(df: pd.DataFrame, window: int = _VOL_WINDOW) -> float:
    """Percentile (0..1) of current ATR-proportional volatility vs. recent history."""
    if len(df) < 30:
        return 0.5
    ret = df["close"].pct_change()
    vol = ret.rolling(14).std()
    recent = vol.iloc[-window:].dropna()
    if len(recent) < 20:
        return 0.5
    cur = float(recent.iloc[-1])
    return float((recent < cur).mean())


def classify(df: pd.DataFrame, indf: pd.DataFrame | None = None) -> str:
    """Classify the regime of the most recent bar.

    `indf` may be the output of indicators.compute_all(df) to reuse its
    adx/ema200 columns; otherwise the needed pieces are computed here.
    """
    if len(df) < 60:
        return "range"

    if indf is not None and "adx" in indf.columns and "ema200" in indf.columns:
        adx_val = float(indf["adx"].iloc[-1]) if not np.isnan(indf["adx"].iloc[-1]) else 15.0
        ema200 = indf["ema200"]
    else:
        adx_val = 15.0
        ema200 = df["close"].ewm(span=200, min_periods=50).mean()

    close = float(df["close"].iloc[-1])
    e200_now = float(ema200.iloc[-1])
    # slope over the last 10 bars, in fractions of price
    lookback = min(10, len(ema200) - 1)
    e200_then = float(ema200.iloc[-1 - lookback])
    slope = (e200_now - e200_then) / max(abs(e200_then), 1e-9)

    chop = choppiness(df)
    volp = vol_percentile(df)

    # High volatility without directional conviction = the dangerous chop.
    if volp > 0.90 and adx_val < 22:
        return "volatile"
    if adx_val >= 25 and slope > 0 and close > e200_now:
        return "trend_up"
    if adx_val >= 25 and slope < 0 and close < e200_now:
        return "trend_down"
    if chop > 61.8 or adx_val < 20:
        return "range"
    # Weak/forming trend: lean on price vs. long EMA.
    if close > e200_now and slope > 0:
        return "trend_up"
    if close < e200_now and slope < 0:
        return "trend_down"
    return "range"


# --------------------------------------------------------------------------- #
# Setup gating
# --------------------------------------------------------------------------- #
# For each regime: setup kinds that are REJECTED per side. Everything not
# listed is allowed. The philosophy: never fight a trending regime with
# counter-trend continuation/mean-reversion; ranges reject continuation;
# volatile chop rejects the styles most punished by whipsaw.
_REJECT: dict[str, dict[Side, tuple[SetupKind, ...]]] = {
    "trend_up": {
        Side.SHORT: (SetupKind.MEAN_REVERSION, SetupKind.MOMENTUM, SetupKind.CONTINUATION),
        Side.LONG: (),
    },
    "trend_down": {
        Side.LONG: (SetupKind.MEAN_REVERSION, SetupKind.MOMENTUM, SetupKind.CONTINUATION),
        Side.SHORT: (),
    },
    "range": {
        Side.LONG: (SetupKind.CONTINUATION,),
        Side.SHORT: (SetupKind.CONTINUATION,),
    },
    "volatile": {
        Side.LONG: (SetupKind.MEAN_REVERSION, SetupKind.CONTINUATION),
        Side.SHORT: (SetupKind.MEAN_REVERSION, SetupKind.CONTINUATION),
    },
}


def allows_setup(kind: SetupKind, side: Side, regime: str) -> bool:
    """True if this setup kind/side combination is tradeable in this regime."""
    rejected = _REJECT.get(regime, {}).get(side, ())
    return kind not in rejected


def market_bias_factor(side: Side, btc_regime: str) -> float:
    """Ranking multiplier from the global (BTC) regime: trading against the
    whole market is not vetoed, but it must out-score with-market setups."""
    if btc_regime == "trend_down" and side == Side.LONG:
        return 0.75
    if btc_regime == "trend_up" and side == Side.SHORT:
        return 0.75
    if btc_regime == "volatile":
        return 0.85
    return 1.0
