"""Shared test configuration and helpers."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when pytest collects from tests/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv_frame() -> pd.DataFrame:
    """A deterministic 300-bar OHLCV frame with enough history for all windows."""
    n = 300
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.8, n))
    high = close + rng.uniform(0.0, 1.2, n)
    low = close - rng.uniform(0.0, 1.2, n)
    open_ = low + rng.uniform(0.0, high - low)
    volume = rng.uniform(1000.0, 5000.0, n)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def candle_frame() -> pd.DataFrame:
    """Minimal frame ending in a bullish hammer after a short downswing."""
    return pd.DataFrame({
        "open": [100.0, 101.0, 100.5, 99.0, 98.5, 97.0, 98.5],
        "high": [101.5, 102.0, 101.0, 100.0, 99.0, 98.0, 99.0],
        "low": [99.5, 100.0, 99.5, 98.0, 97.5, 96.0, 96.0],
        "close": [100.0, 100.5, 100.0, 98.5, 97.5, 96.0, 99.0],
        "volume": [1000.0] * 7,
    })


@pytest.fixture
def bearish_engulfing_frame() -> pd.DataFrame:
    """Two-bar bearish engulfing at the end of a short upswing."""
    return pd.DataFrame({
        "open": [100.0, 101.0, 100.5, 101.5, 102.0, 103.0, 104.0],
        "high": [101.5, 102.0, 101.5, 102.5, 103.0, 104.0, 104.0],
        "low": [99.5, 100.0, 100.0, 100.5, 101.0, 102.0, 100.0],
        "close": [100.5, 101.0, 101.0, 102.0, 102.5, 103.5, 100.5],
        "volume": [1000.0] * 7,
    })
