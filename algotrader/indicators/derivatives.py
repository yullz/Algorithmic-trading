"""Funding-rate and open-interest rank evidence (derivatives enrichment).

This module is designed for futures-specific context. When real funding/OI
series are available the helper ranks them against the trailing 30-day window
and emits extreme readings. Until the scanner wires real derivatives data, the
module falls back to synthetic proxies derived from OHLCV volume percentiles —
this keeps the signal-engine integration stable and testable without requiring
external market-data calls in this scope.

Design notes:
  * Percentile ranks are computed over the last `lookback` bars of the supplied
    series; no lookahead is possible because the rank uses only historical values.
  * "Extreme" is defined as above the 90th or below the 10th percentile.
  * base_win_rate values are conservative priors; calibration overwrites them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Bias, Evidence

_LOOKBACK = 30
_EXTREME_PCT = 0.90


def _percentile_rank(series: pd.Series, lookback: int = _LOOKBACK) -> float:
    """Return the percentile rank (0..1) of the latest value vs trailing window."""
    window = series.iloc[-lookback:].dropna()
    if len(window) < 10:
        return 0.5
    cur = float(window.iloc[-1])
    return float((window < cur).mean())


def funding_oi_rank_evidence(
    funding_series: pd.Series,
    oi_series: pd.Series,
) -> list[Evidence]:
    """Emit derivatives-family Evidence from funding-rate and OI rank.

    Args:
        funding_series: Funding-rate series (annualized or periodic). When None
            or too short, a neutral placeholder is used.
        oi_series: Open-interest series. When None or too short, a neutral
            placeholder is used.

    Returns:
        A list of Evidence objects with family="derivatives".
    """
    out: list[Evidence] = []

    if funding_series is not None and len(funding_series) >= 10:
        f_rank = _percentile_rank(funding_series, _LOOKBACK)
        if f_rank > _EXTREME_PCT:
            out.append(Evidence(
                "funding_extreme_long", "indicator", Bias.BEARISH, 0.45, 0.52,
                f"funding at {f_rank:.0%} percentile — longs pay heavily",
                family="derivatives",
            ).clamp())
        elif f_rank < 1 - _EXTREME_PCT:
            out.append(Evidence(
                "funding_extreme_short", "indicator", Bias.BULLISH, 0.45, 0.52,
                f"funding at {f_rank:.0%} percentile — shorts pay heavily",
                family="derivatives",
            ).clamp())

    if oi_series is not None and len(oi_series) >= 10:
        oi_rank = _percentile_rank(oi_series, _LOOKBACK)
        if oi_rank > _EXTREME_PCT:
            out.append(Evidence(
                "oi_extreme_long", "indicator", Bias.BULLISH, 0.4, 0.52,
                f"OI at {oi_rank:.0%} percentile — fresh long interest",
                family="derivatives",
            ).clamp())
        elif oi_rank < 1 - _EXTREME_PCT:
            out.append(Evidence(
                "oi_extreme_short", "indicator", Bias.BEARISH, 0.4, 0.52,
                f"OI at {oi_rank:.0%} percentile — fresh short interest",
                family="derivatives",
            ).clamp())

    return out


def synthetic_from_ohlcv(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Build synthetic funding/OI proxies from OHLCV volume percentiles.

    The proxies are deliberately simple: funding proxy = volume percentile
    centered around zero; OI proxy = cumulative volume delta. They preserve the
    right shape for ranking without claiming to be real derivatives data.
    """
    if df is None or len(df) < 30 or "volume" not in df.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    vol_pct = df["volume"].rolling(30, min_periods=10).rank(pct=True)
    # Funding proxy: volume percentile mapped to [-1, 1]. High volume days tend
    # to coincide with stressed funding in this synthetic model.
    funding = (vol_pct - 0.5) * 2.0

    # OI proxy: signed volume accumulation (close up = +vol, down = -vol).
    sign = np.sign(df["close"].diff().fillna(0.0))
    oi = (sign * df["volume"]).cumsum()

    return funding, oi
