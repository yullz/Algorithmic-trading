"""Universe breadth — a market-wide risk-on/risk-off read across the scanned set.

Per-symbol indicators judge each coin in isolation; breadth asks "is the WHOLE
market participating?" — the context a discretionary trader reads before sizing
up. It is computed from the OHLCV frames the scanner already loads (percent of
symbols above their EMA50/EMA200, advancers vs decliners), so it costs nothing
extra and — crucially — introduces NO train/serve skew: it is used only as a
selection-time ranking tilt and dashboard context, never as a per-symbol
evidence factor that the backtester (which has no universe view) couldn't
calibrate.
"""
from __future__ import annotations

import pandas as pd

from .indicators import ema

# Fraction thresholds that define the risk regime.
RISK_ON_EMA50 = 0.60
RISK_ON_EMA200 = 0.55
RISK_OFF_EMA50 = 0.40
RISK_OFF_EMA200 = 0.45

# Bounded ranking tilt applied to trades aligned / opposed to the breadth regime.
_TILT = 0.10


def compute_breadth(frames: dict[str, pd.DataFrame], min_bars: int = 200) -> dict:
    """Aggregate breadth across `frames` ({symbol: ohlcv_df}) on one timeframe.

    Returns percent-above-EMA50/EMA200, advancers/decliners on the latest bar,
    the advance/decline ratio, and a `risk_state` in {risk_on, neutral, risk_off}.
    Symbols with too little history are skipped; `n` reports how many counted.
    """
    above50 = above200 = adv = dec = total = 0
    for df in frames.values():
        if df is None or len(df) < min_bars:
            continue
        close = df["close"]
        e50 = ema(close, 50).iloc[-1]
        e200 = ema(close, 200).iloc[-1]
        if pd.isna(e50) or pd.isna(e200):
            continue
        total += 1
        last = float(close.iloc[-1])
        if last > e50:
            above50 += 1
        if last > e200:
            above200 += 1
        if len(close) >= 2:
            prev = float(close.iloc[-2])
            if last > prev:
                adv += 1
            elif last < prev:
                dec += 1

    if total == 0:
        return {"n": 0, "pct_above_ema50": None, "pct_above_ema200": None,
                "advancers": 0, "decliners": 0, "ad_ratio": None,
                "risk_state": "neutral"}

    p50 = above50 / total
    p200 = above200 / total
    if p50 >= RISK_ON_EMA50 and p200 >= RISK_ON_EMA200:
        risk_state = "risk_on"
    elif p50 <= RISK_OFF_EMA50 and p200 <= RISK_OFF_EMA200:
        risk_state = "risk_off"
    else:
        risk_state = "neutral"

    return {
        "n": total,
        "pct_above_ema50": round(p50, 3),
        "pct_above_ema200": round(p200, 3),
        "advancers": adv, "decliners": dec,
        "ad_ratio": round(adv / max(dec, 1), 2),
        "risk_state": risk_state,
    }


def breadth_bias(side_sign: int, risk_state: str) -> float:
    """Bounded (±10%) ranking multiplier for a trade's side vs the breadth regime.

    `side_sign` is +1 for long, -1 for short. In a risk-on tape longs are
    favored and shorts discounted; risk-off mirrors it; neutral is a no-op.
    """
    if risk_state == "risk_on":
        return 1.0 + _TILT * side_sign
    if risk_state == "risk_off":
        return 1.0 - _TILT * side_sign
    return 1.0
