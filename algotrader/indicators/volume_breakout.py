"""Volume-confirmed breakout detector.

A breakout is credible when:
  1. Price closes above/below the trailing 20-bar Donchian channel, and
  2. Volume is materially above average (>1.5x the 20-bar SMA), and
  3. OBV is moving in the same direction as price (accumulation on break up,
     distribution on break down).

Design notes:
  * Uses only lookback windows; the current bar is included in the Donchian
    and volume-average windows, so there is no lookahead.
  * OBV confirmation is judged by the sign of the 5-bar linear slope to avoid
    one-bar noise.
  * Evidence is emitted only on the breakout bar; subsequent bars resting
    beyond the channel do not repeat the signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Bias, Evidence
from .indicators import donchian, obv

_WINDOW = 20
_OBV_CONFIRM_BARS = 5


def read_evidence(df: pd.DataFrame) -> list[Evidence]:
    """Return volume-confirmed breakout Evidence for the latest bar."""
    if len(df) < _WINDOW + 2 or "close" not in df.columns:
        return []

    d = df.copy()
    d["obv"] = obv(d)
    dc_up, dc_low, _ = donchian(d, _WINDOW)
    vol_sma = d["volume"].rolling(_WINDOW, min_periods=_WINDOW).mean()

    prev_close = d["close"].iloc[-2]
    close = d["close"].iloc[-1]
    prev_dc_up = dc_up.iloc[-2]
    prev_dc_low = dc_low.iloc[-2]
    cur_dc_up = dc_up.iloc[-1]
    cur_dc_low = dc_low.iloc[-1]
    volume = d["volume"].iloc[-1]
    avg_vol = vol_sma.iloc[-1]

    out: list[Evidence] = []

    def _obv_confirms(bull: bool) -> bool:
        obv_series = d["obv"].iloc[-_OBV_CONFIRM_BARS:].to_numpy(dtype=float)
        if len(obv_series) < 2 or np.isnan(obv_series).any():
            return False
        slope = np.polyfit(np.arange(len(obv_series)), obv_series, 1)[0]
        return slope > 0 if bull else slope < 0

    # Breakout above the 20-bar high ending at the previous bar.
    if (
        close > prev_dc_up
        and prev_close <= prev_dc_up
        and volume > 1.5 * avg_vol
        and _obv_confirms(bull=True)
    ):
        out.append(Evidence(
            "volume_confirmed_breakout_up", "indicator", Bias.BULLISH,
            0.65, 0.56,
            f"close {close:.6g} broke 20-bar high on {volume:.0f} vol "
            f"({volume / avg_vol:.1f}x avg)",
            family="volume",
        ).clamp())

    # Breakout below the 20-bar low ending at the previous bar.
    if (
        close < prev_dc_low
        and prev_close >= prev_dc_low
        and volume > 1.5 * avg_vol
        and _obv_confirms(bull=False)
    ):
        out.append(Evidence(
            "volume_confirmed_breakout_down", "indicator", Bias.BEARISH,
            0.65, 0.56,
            f"close {close:.6g} broke 20-bar low on {volume:.0f} vol "
            f"({volume / avg_vol:.1f}x avg)",
            family="volume",
        ).clamp())

    return out
