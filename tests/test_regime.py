"""Unit tests for algotrader/regime.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrader import regime
from algotrader.indicators import indicators as ind


def test_classify_returns_known_regime(ohlcv_frame):
    indf = ind.compute_all(ohlcv_frame)
    label = regime.classify(ohlcv_frame, indf)
    assert label in regime.REGIMES


def test_classify_short_frame_defaults_to_range():
    df = pd.DataFrame({
        "open": [1.0, 1.1],
        "high": [1.2, 1.3],
        "low": [0.9, 1.0],
        "close": [1.1, 1.2],
        "volume": [10.0, 10.0],
    })
    assert regime.classify(df) == "range"


def test_classify_uptrend_label():
    """A strong directional advance should land in a trending regime."""
    n = 300
    x = np.arange(n)
    close = 100.0 + 0.3 * x + np.random.default_rng(1).normal(0.0, 0.1, n)
    df = pd.DataFrame({
        "open": close - 0.05,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": np.full(n, 1000.0),
    })
    indf = ind.compute_all(df)
    label = regime.classify(df, indf)
    assert label in ("trend_up", "volatile")
