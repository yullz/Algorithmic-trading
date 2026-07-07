"""Cross-asset context: BTC regime, ETH/BTC spread, and sector relative strength.

Provides Evidence that compares an asset against BTC and against its own
recent history. All functions are pure pandas/numpy and use only lookback data.

Design notes:
  * Alignment is performed on the shared datetime index, so gappy or partially
    listed symbols cannot fabricate a spread.
  * Normalized spread deviations are expressed in standard deviations of the
    lookback window — a familiar, regime-robust scale.
  * base_win_rate values are conservative priors; calibration overwrites them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import regime as regime_mod
from ..models import Bias, Evidence


def btc_regime_evidence(btc_df: pd.DataFrame) -> Evidence:
    """Wrap `regime.classify` into an Evidence object.

    Uses the existing regime module's logic on the BTC frame. The evidence
    family is "trend" so the confluence engine treats it as directional context.
    """
    label = regime_mod.classify(btc_df)
    if label == "trend_up":
        return Evidence(
            "btc_regime_uptrend", "structure", Bias.BULLISH, 0.55, 0.56,
            "BTC in uptrend", family="trend",
        ).clamp()
    if label == "trend_down":
        return Evidence(
            "btc_regime_downtrend", "structure", Bias.BEARISH, 0.55, 0.56,
            "BTC in downtrend", family="trend",
        ).clamp()
    if label == "volatile":
        return Evidence(
            "btc_regime_volatile", "structure", Bias.NEUTRAL, 0.45, 0.51,
            "BTC volatile chop", family="trend",
        ).clamp()
    return Evidence(
        "btc_regime_range", "structure", Bias.NEUTRAL, 0.35, 0.5,
        "BTC ranging", family="trend",
    ).clamp()


def eth_btc_spread_evidence(
    eth_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    lookback: int = 50,
    threshold: float = 1.5,
) -> list[Evidence]:
    """Normalized ETH/BTC spread deviation from its `lookback`-period mean.

    The spread is log(eth_close) - log(btc_close). A z-score above `threshold`
    suggests ETH is over-extended vs BTC (bearish for ETH relative value),
    below -threshold suggests under-extended (bullish relative value).
    """
    if eth_df is None or btc_df is None:
        return []
    idx = eth_df.index.intersection(btc_df.index)
    if len(idx) < max(30, lookback + 1):
        return []
    eth = eth_df.loc[idx, "close"]
    btc = btc_df.loc[idx, "close"]
    spread = np.log(eth) - np.log(btc)
    recent = spread.iloc[-lookback:]
    if len(recent) < lookback:
        return []
    mean = float(recent.mean())
    std = float(recent.std(ddof=0))
    z = float((spread.iloc[-1] - mean) / (std + 1e-9))

    out: list[Evidence] = []
    if z > threshold:
        out.append(Evidence(
            "eth_btc_bearish", "indicator", Bias.BEARISH,
            min(abs(z) / 3.0, 1.0), 0.53,
            f"ETH/BTC spread +{z:.2f}σ above {lookback}-bar mean",
            family="relative_strength",
        ).clamp())
    elif z < -threshold:
        out.append(Evidence(
            "eth_btc_bullish", "indicator", Bias.BULLISH,
            min(abs(z) / 3.0, 1.0), 0.53,
            f"ETH/BTC spread {z:.2f}σ below {lookback}-bar mean",
            family="relative_strength",
        ).clamp())
    return out


def sector_strength_evidence(
    df: pd.DataFrame,
    btc_df: pd.DataFrame,
    lookback: int = 20,
    threshold: float = 0.05,
) -> list[Evidence]:
    """Compare symbol's ROC to BTC's ROC over the shared lookback.

    Emits relative-strength evidence when the symbol out- or under-performs BTC
    by more than `threshold` (fractional return).
    """
    if df is None or btc_df is None:
        return []
    idx = df.index.intersection(btc_df.index)
    if len(idx) < max(30, lookback + 1):
        return []
    s = df.loc[idx, "close"]
    b = btc_df.loc[idx, "close"]
    s0, b0 = float(s.iloc[-1 - lookback]), float(b.iloc[-1 - lookback])
    if s0 <= 0 or b0 <= 0:
        return []
    excess = (float(s.iloc[-1]) / s0 - 1) - (float(b.iloc[-1]) / b0 - 1)

    out: list[Evidence] = []
    if excess > threshold:
        out.append(Evidence(
            "sector_outperform", "indicator", Bias.BULLISH,
            min(excess / 0.15, 1.0), 0.54,
            f"+{excess:.1%} vs BTC over {lookback} bars",
            family="relative_strength",
        ).clamp())
    elif excess < -threshold:
        out.append(Evidence(
            "sector_underperform", "indicator", Bias.BEARISH,
            min(-excess / 0.15, 1.0), 0.54,
            f"{excess:.1%} vs BTC over {lookback} bars",
            family="relative_strength",
        ).clamp())
    return out
