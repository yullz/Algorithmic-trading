"""Tests for universe breadth (market-wide risk-on/off context)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from algotrader.indicators.breadth import breadth_bias, compute_breadth


def _frame(trend: float, n: int = 250) -> pd.DataFrame:
    close = 100 + np.cumsum(np.full(n, trend, dtype=float))
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": np.full(n, 100.0)})


def test_compute_breadth_risk_on():
    frames = {f"S{i}": _frame(0.2) for i in range(10)}  # all rising
    b = compute_breadth(frames)
    assert b["n"] == 10
    assert b["pct_above_ema50"] > 0.9
    assert b["pct_above_ema200"] > 0.9
    assert b["risk_state"] == "risk_on"
    assert b["advancers"] == 10 and b["decliners"] == 0


def test_compute_breadth_risk_off():
    frames = {f"S{i}": _frame(-0.2) for i in range(10)}  # all falling
    b = compute_breadth(frames)
    assert b["pct_above_ema50"] < 0.1
    assert b["risk_state"] == "risk_off"


def test_compute_breadth_skips_short_frames_and_handles_empty():
    frames = {"A": _frame(0.2, n=250), "B": _frame(0.2, n=50), "C": None}
    b = compute_breadth(frames, min_bars=200)
    assert b["n"] == 1
    assert compute_breadth({})["risk_state"] == "neutral"


def test_breadth_bias_tilts_with_regime():
    # risk-on favors longs, discounts shorts; risk-off mirrors; neutral is a no-op.
    assert breadth_bias(1, "risk_on") > 1.0
    assert breadth_bias(-1, "risk_on") < 1.0
    assert breadth_bias(1, "risk_off") < 1.0
    assert breadth_bias(-1, "risk_off") > 1.0
    assert breadth_bias(1, "neutral") == 1.0
    assert breadth_bias(-1, "neutral") == 1.0
