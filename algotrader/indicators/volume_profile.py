"""Rolling volume profile: Point of Control (POC) and 70% Value Area.

A volume profile histograms traded volume by PRICE instead of by time. The
POC (the price bin with the most volume) is where the market found the most
agreement — it behaves like a magnet/support-resistance level. The value area
(the ~70% of volume around the POC) separates "accepted" prices from
excursions.

Design choices (documented deliberately):
  * Each bar's volume is distributed UNIFORMLY across its high-low range,
    pro-rated by overlap with each price bin (fully vectorized as one
    bars x bins overlap matrix — no per-bar Python loop). This is smoother
    and less arbitrary than dumping a whole bar's volume into the close bin;
    zero-range bars fall back to their close bin.
  * The profile is ROLLING over the last 120 bars with 40 bins — enough
    resolution to separate levels without turning noise into structure.
  * Value area expands greedily from the POC toward the heavier neighboring
    bin until 70% of volume is covered (the standard construction).
  * Evidence is deliberately soft: POC touch evidence is directional but
    modest (0.45); being outside the value area is NEUTRAL context — it says
    "price is accepted away from fair value", not "buy" or "sell".
  * No lookahead: the profile uses only the trailing window ending at the
    current bar. base_win_rate values are priors; calibration overwrites them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Bias, Evidence
from .indicators import atr

_LOOKBACK = 120
_BINS = 40
_VALUE_AREA = 0.70
_MIN_BARS = 60


def _profile(df: pd.DataFrame, lookback: int = _LOOKBACK, bins: int = _BINS):
    """Histogram of volume by price over the trailing window.

    Returns dict(poc, vah, val, hist, edges) or None when the data cannot
    support a profile (too short, degenerate range, no volume).
    """
    if len(df) < _MIN_BARS:
        return None
    d = df.iloc[-lookback:]
    lo = d["low"].to_numpy(dtype=float)
    hi = d["high"].to_numpy(dtype=float)
    cl = d["close"].to_numpy(dtype=float)
    vol = d["volume"].to_numpy(dtype=float)
    if np.isnan(lo).any() or np.isnan(hi).any() or np.isnan(vol).any():
        return None
    p_min, p_max = float(lo.min()), float(hi.max())
    if not np.isfinite(p_min) or not np.isfinite(p_max) or p_max <= p_min:
        return None

    edges = np.linspace(p_min, p_max, bins + 1)
    # bars x bins overlap of [low, high] with each bin, pro-rated by bar range
    left = np.maximum(lo[:, None], edges[None, :-1])
    right = np.minimum(hi[:, None], edges[None, 1:])
    overlap = np.clip(right - left, 0.0, None)
    rng = hi - lo
    safe_rng = np.where(rng > 0, rng, 1.0)
    frac = overlap / safe_rng[:, None]
    frac[rng <= 0] = 0.0
    hist = (frac * vol[:, None]).sum(axis=0)
    # zero-range bars: all volume into the close's bin
    zero = rng <= 0
    if zero.any():
        idx = np.clip(np.searchsorted(edges, cl[zero], side="right") - 1, 0, bins - 1)
        np.add.at(hist, idx, vol[zero])

    total = float(hist.sum())
    if total <= 0:
        return None

    poc_bin = int(np.argmax(hist))
    # expand from POC toward the heavier neighbor until 70% covered
    l = u = poc_bin
    covered = float(hist[poc_bin])
    while covered < _VALUE_AREA * total and (l > 0 or u < bins - 1):
        below = hist[l - 1] if l > 0 else -np.inf
        above = hist[u + 1] if u < bins - 1 else -np.inf
        if above >= below:
            u += 1
            covered += float(hist[u])
        else:
            l -= 1
            covered += float(hist[l])

    centers = (edges[:-1] + edges[1:]) / 2
    return {
        "poc": float(centers[poc_bin]),
        "vah": float(edges[u + 1]),
        "val": float(edges[l]),
        "hist": hist,
        "edges": edges,
    }


def profile_levels(df: pd.DataFrame) -> dict:
    """POC / value-area levels for the dashboard & API.

    Returns {"poc": float, "vah": float, "val": float}, or {} when the frame
    cannot support a profile.
    """
    p = _profile(df)
    if p is None:
        return {}
    return {"poc": p["poc"], "vah": p["vah"], "val": p["val"]}


def read_evidence(df: pd.DataFrame) -> list[Evidence]:
    """Structural evidence from the rolling profile at the current bar.

    * price_at_poc_support / price_at_poc_resist: close within 0.25 ATR of
      the POC, reached by drifting INTO it (from above -> support test, from
      below -> resistance test).
    * price_above_value / price_below_value: NEUTRAL context — acceptance
      outside the value area; it dilutes confidence rather than voting.
    """
    p = _profile(df)
    if p is None:
        return []
    a = atr(df, 14).iloc[-1]
    close = float(df["close"].iloc[-1])
    if pd.isna(a) or a <= 0:
        a = max(close * 0.01, 1e-9)     # degenerate-ATR fallback
    poc, vah, val = p["poc"], p["vah"], p["val"]
    band = 0.25 * float(a)
    drift = float(df["close"].iloc[-6:-1].mean())   # where price came from

    out: list[Evidence] = []
    if poc <= close <= poc + band and drift > close:
        out.append(Evidence(
            "price_at_poc_support", "structure", Bias.BULLISH, 0.45, 0.53,
            f"testing POC {poc:.6g} from above", family="structure").clamp())
    elif poc - band <= close <= poc and drift < close:
        out.append(Evidence(
            "price_at_poc_resist", "structure", Bias.BEARISH, 0.45, 0.53,
            f"testing POC {poc:.6g} from below", family="structure").clamp())

    if close > vah:
        out.append(Evidence(
            "price_above_value", "structure", Bias.NEUTRAL, 0.2, 0.5,
            f"acceptance above value area (VAH {vah:.6g})",
            family="structure").clamp())
    elif close < val:
        out.append(Evidence(
            "price_below_value", "structure", Bias.NEUTRAL, 0.2, 0.5,
            f"acceptance below value area (VAL {val:.6g})",
            family="structure").clamp())
    return out
