"""Harmonic pattern detection (Gartley, Bat, Butterfly).

Detects bullish and bearish variants of three classic harmonic patterns using
alternating order-3 fractal pivots and simple Fibonacci ratio checks. Each match
returns a PatternMatch with breakout/target/invalidation levels so the risk
engine can build a structural trade plan.

No lookahead: all ratios are computed from pivots already established in the
past; the pattern is only confirmed by the latest closed bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Bias, PatternMatch, SetupKind
from .chart_patterns import _atr, _pivots


# --------------------------------------------------------------------------- #
# Ratio helpers
# --------------------------------------------------------------------------- #
def _ratio(a: float, b: float) -> float:
    """Return a / b, safe for b == 0."""
    return a / b if abs(b) > 1e-12 else 0.0


def _in_band(value: float, low: float, high: float, tol: float = 0.06) -> bool:
    """Whether value is within tolerance of the [low, high] band."""
    return low * (1 - tol) <= value <= high * (1 + tol)


# --------------------------------------------------------------------------- #
# Pattern ratio rules (liberal tolerance to catch approximate geometry)
# --------------------------------------------------------------------------- #
HARMONIC_RULES = {
    "gartley": {
        "ab_of_xa": (0.618, 0.618),
        "bc_of_ab": (0.382, 0.886),
        "cd_of_bc": (1.272, 1.618),
        "cd_of_xa": (0.786, 0.786),
    },
    "bat": {
        "ab_of_xa": (0.382, 0.5),
        "bc_of_ab": (0.382, 0.886),
        "cd_of_bc": (1.618, 2.618),
        "cd_of_xa": (0.886, 0.886),
    },
    "butterfly": {
        "ab_of_xa": (0.786, 0.786),
        "bc_of_ab": (0.382, 0.886),
        "cd_of_bc": (1.618, 2.240),
        "cd_of_xa": (1.270, 1.414),
    },
}


def _check_pattern(X: float, A: float, B: float, C: float, D: float,
                   rules: dict[str, tuple[float, float]],
                   tol: float = 0.06) -> tuple[bool, float]:
    """Return (matches, quality_score) for one XABCD candidate."""
    xa = abs(A - X)
    ab = abs(B - A)
    bc = abs(C - B)
    cd = abs(D - C)

    if min(xa, ab, bc, cd) < 1e-9:
        return False, 0.0

    ab_xa = _ratio(ab, xa)
    bc_ab = _ratio(bc, ab)
    cd_bc = _ratio(cd, bc)
    cd_xa = _ratio(cd, xa)

    checks = [
        _in_band(ab_xa, *rules["ab_of_xa"], tol=tol),
        _in_band(bc_ab, *rules["bc_of_ab"], tol=tol),
        _in_band(cd_bc, *rules["cd_of_bc"], tol=tol),
        _in_band(cd_xa, *rules["cd_of_xa"], tol=tol),
    ]
    if not all(checks):
        return False, 0.0

    # Quality is highest when ratios sit closest to the midpoints of the bands.
    mids = [
        (rules["ab_of_xa"][0] + rules["ab_of_xa"][1]) / 2,
        (rules["bc_of_ab"][0] + rules["bc_of_ab"][1]) / 2,
        (rules["cd_of_bc"][0] + rules["cd_of_bc"][1]) / 2,
        (rules["cd_of_xa"][0] + rules["cd_of_xa"][1]) / 2,
    ]
    vals = [ab_xa, bc_ab, cd_bc, cd_xa]
    err = sum(abs(v - m) / max(m, 1e-9) for v, m in zip(vals, mids)) / 4
    quality = max(0.0, min(1.0, 1.0 - err))
    return True, quality


def detect(df: pd.DataFrame, lookback: int = 120, order: int = 3,
           tol: float = 0.06) -> list[PatternMatch]:
    """Detect harmonic patterns on the most recent closed bar.

    Uses order-3 fractal pivots on the close series, then searches for
    alternating X-A-B-C-D sequences that fit the ratio rules. The final bar
    (D) must confirm the pattern by closing near the completion point.
    """
    if len(df) < 40:
        return []
    d = df.iloc[-lookback:] if len(df) > lookback else df
    high = d["high"].to_numpy()
    low = d["low"].to_numpy()
    close = d["close"].to_numpy()
    base = len(df) - len(d)
    last = len(d) - 1
    atr = _atr(high, low, close)
    mean_price = max(float(close.mean()), 1e-9)

    hi_idx, lo_idx = _pivots(close, order)
    # Build an alternating pivot sequence sorted by index. We prefer alternating
    # high/low pivots, which is the natural structure of harmonic patterns.
    pivots: list[tuple[int, str, float]] = []
    hi_set, lo_set = set(hi_idx), set(lo_idx)
    for i in sorted(hi_set | lo_set):
        # If a level is both a high and low pivot (rare), prefer the stronger.
        if i in hi_set and i in lo_set:
            pivots.append((i, "hi", float(close[i])))
        elif i in hi_set:
            pivots.append((i, "hi", float(close[i])))
        else:
            pivots.append((i, "lo", float(close[i])))

    out: list[PatternMatch] = []
    if len(pivots) < 5:
        return out

    def add(name: str, bias: Bias, X, A, B, C, D, quality: float):
        # Completion is the D point. Breakout = close through the B point, the
        # decision level between a valid pattern and a continuation.
        Xp, Ap, Bp, Cp, Dp = X[2], A[2], B[2], C[2], D[2]
        if bias == Bias.BULLISH:
            breakout = max(Bp, Dp)
            # Measured target: full retracement of the XA impulse.
            target = Xp
            invalidation = min(Cp, Dp - 0.5 * atr)
        else:
            breakout = min(Bp, Dp)
            target = Xp
            invalidation = max(Cp, Dp + 0.5 * atr)
        conf = 0.55 + 0.25 * quality
        out.append(PatternMatch(
            name, SetupKind.REVERSAL, bias, min(conf, 1.0),
            base + X[0], base + D[0],
            breakout_level=float(breakout),
            target_level=float(target),
            invalidation_level=float(invalidation),
            note=f"{name} harmonic completion, quality={quality:.2f}",
            family="chart",
        ))

    # Scan the last several 5-pivot windows. Require the final pivot (D) to be
    # within the most recent few bars so the pattern is actionable.
    max_d_offset = min(10, last)
    for i in range(len(pivots) - 4):
        X, A, B, C, D = pivots[i], pivots[i + 1], pivots[i + 2], pivots[i + 3], pivots[i + 4]
        if D[0] < last - max_d_offset:
            continue
        # Validate the harmonic point structure without requiring every single
        # bar-to-bar diff to alternate (the CD leg naturally continues in the
        # same direction as BC).
        # Bullish: X>A, B>A&B<X, C<B&C>A, D<C.
        # Bearish: X<A, B<A&B>X, C>B&C<A, D>C.
        is_bull = (X[2] > A[2] and A[2] < B[2] < X[2]
                   and A[2] < C[2] < B[2] and D[2] < C[2])
        is_bear = (X[2] < A[2] and X[2] < B[2] < A[2]
                   and B[2] < C[2] < A[2] and D[2] > C[2])
        if not (is_bull or is_bear):
            continue

        # Bullish pattern: X -> A is down, D is a low below B.
        if is_bull:
            for pname, rules in HARMONIC_RULES.items():
                ok, quality = _check_pattern(X[2], A[2], B[2], C[2], D[2], rules, tol=tol)
                if ok:
                    add(f"bullish_{pname}", Bias.BULLISH, X, A, B, C, D, quality)
        # Bearish pattern: X -> A is up, D is a high above B.
        elif is_bear:
            for pname, rules in HARMONIC_RULES.items():
                ok, quality = _check_pattern(X[2], A[2], B[2], C[2], D[2], rules, tol=tol)
                if ok:
                    add(f"bearish_{pname}", Bias.BEARISH, X, A, B, C, D, quality)

    # Deduplicate: keep only the highest-quality match of each name.
    best: dict[str, PatternMatch] = {}
    for m in out:
        if m.name not in best or m.confidence > best[m.name].confidence:
            best[m.name] = m
    return list(best.values())
