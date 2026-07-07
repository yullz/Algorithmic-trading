"""Honest model-selection helpers: nested walk-forward + multiple-comparison
correction.

Two data-snooping traps live in the post-backtest analysis scripts:

  1. Grid-searching a strength cutoff and reporting the winning cutoff's edge on
     the SAME rows it was chosen to maximize (in-sample-optimistic thresholds).
  2. Testing dozens of factors for "edge" without correcting for the number of
     hypotheses (inflated false-positive discoveries).

This module fixes both with pure, testable functions:

  * ``anchored_folds`` / ``time_order`` — time-ordered expanding-window splits
    that mirror the backtester's anchored walk-forward.
  * ``tune_threshold_cv`` — choose the cutoff on TRAIN folds, report the edge on
    the pooled held-out TEST folds only.
  * ``binom_p_greater`` + ``benjamini_hochberg`` — one-sided binomial p-values
    against the dataset base rate, corrected for false-discovery rate.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .robustness import _norm_cdf


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson-score interval for a binomial proportion (the canonical copy;
    tune_thresholds.py / analyze_factors.py import this instead of duplicating)."""
    if n <= 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def time_order(times) -> np.ndarray | None:
    """Return positional order (argsort) of rows by parsed timestamp, or None if
    the timestamps are unusable (missing/constant) — in which case the caller
    falls back to the existing row order."""
    if times is None:
        return None
    ts = pd.to_datetime(pd.Series(list(times)), errors="coerce", utc=True)
    if ts.notna().sum() < 2 or ts.nunique() < 2:
        return None
    return np.argsort(ts.values, kind="stable")


def anchored_folds(order: np.ndarray, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Anchored (expanding-window) splits: fold k trains on everything BEFORE
    test segment k and tests on segment k. Mirrors engine.walk_forward so the
    threshold CV is validated the same honest way the calibration is."""
    order = np.asarray(order)
    n = len(order)
    if n < (n_folds + 1) * 2:
        return []
    chunks = np.array_split(order, n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        train = np.concatenate(chunks[:k]) if k > 0 else np.array([], dtype=int)
        test = chunks[k]
        if len(train) and len(test):
            folds.append((train, test))
    return folds


def tune_threshold_cv(strength, r, times=None, grid=None, n_folds: int = 5,
                      min_train: int = 20, min_test: int = 10) -> dict | None:
    """Nested walk-forward threshold tuning.

    For each anchored fold, pick the strength cutoff that maximizes a penalized
    expectancy score on the TRAIN slice, then evaluate that cutoff ONLY on the
    held-out TEST slice. The reported edge is the pooled out-of-sample number
    (never the train-max), plus the per-fold chosen thresholds and their spread
    (stability). Returns None when there is not enough data to split honestly.
    """
    strength = np.abs(np.asarray(strength, dtype=float))
    r = np.asarray(r, dtype=float)
    keep = ~(np.isnan(strength) | np.isnan(r))
    strength, r = strength[keep], r[keep]
    t = None if times is None else np.asarray(list(times))[keep]
    n = len(r)
    if n < 40:
        return None
    if grid is None:
        grid = np.linspace(0.1, 0.9, 9)

    order = time_order(t)
    if order is None:
        order = np.arange(n)
    folds = anchored_folds(order, n_folds)
    if not folds:
        return None

    oos_r: list[float] = []
    chosen: list[float] = []
    for train, test in folds:
        best_cut, best_score = None, None
        for cut in grid:
            a = train[strength[train] >= cut]
            if len(a) < min_train:
                continue
            exp = float(r[a].mean())
            wr = float((r[a] > 0).mean())
            wl, _ = wilson_interval(wr, len(a))
            score = exp if wl > 0.5 else exp - 0.5  # penalize sub-50% lower CI
            if best_score is None or score > best_score:
                best_score, best_cut = score, float(cut)
        if best_cut is None:
            continue
        chosen.append(best_cut)
        te = test[strength[test] >= best_cut]
        if len(te) >= min_test:
            oos_r.extend(r[te].tolist())

    if len(oos_r) < min_test or not chosen:
        return None
    oos = np.asarray(oos_r, dtype=float)
    wr = float((oos > 0).mean())
    wl, wu = wilson_interval(wr, len(oos))
    return {
        "threshold_median": round(float(np.median(chosen)), 3),
        "threshold_std": round(float(np.std(chosen)), 3),
        "n_folds_used": len(chosen),
        "oos_n": len(oos),
        "oos_win_rate": round(wr, 4),
        "oos_wilson_lower": round(wl, 4),
        "oos_expectancy_r": round(float(oos.mean()), 4),
    }


def binom_p_greater(wins: int, n: int, p0: float) -> float:
    """One-sided P(X >= wins | X ~ Binomial(n, p0)): the probability of seeing at
    least this many wins by chance if the factor had only the base win rate p0.
    Exact via scipy when available; normal approximation (continuity-corrected)
    otherwise."""
    if n <= 0:
        return 1.0
    p0 = min(max(float(p0), 1e-9), 1.0 - 1e-9)
    wins = max(0, min(int(wins), int(n)))
    try:
        from scipy.stats import binom
        return float(binom.sf(wins - 1, int(n), p0))  # P(X > wins-1) = P(X >= wins)
    except Exception:  # pragma: no cover - scipy is a hard dep of the ML path
        mu = n * p0
        sd = math.sqrt(n * p0 * (1.0 - p0))
        if sd == 0:
            return 1.0 if wins <= mu else 0.0
        z = (wins - 0.5 - mu) / sd  # continuity correction
        return float(1.0 - _norm_cdf(z))


def benjamini_hochberg(pvals: dict[str, float], q: float = 0.10) -> dict[str, dict]:
    """Benjamini-Hochberg false-discovery-rate correction across many hypotheses.

    Returns per-key {p_value, p_adjusted, significant, rank}. `significant` marks
    the keys that survive FDR control at level q — the factors whose edge is
    unlikely to be a multiple-comparisons artifact.
    """
    items = [(k, float(v)) for k, v in pvals.items() if v is not None]
    m = len(items)
    if m == 0:
        return {}
    ordered = sorted(items, key=lambda kv: kv[1])
    # BH-adjusted p-values: monotone non-decreasing from the smallest p up.
    adj = [0.0] * m
    running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(running, ordered[i][1] * m / (i + 1))
        adj[i] = min(1.0, running)
    # Largest rank i with p_(i) <= (i/m)*q -> everything up to it is significant.
    crit = 0
    for i in range(m):
        if ordered[i][1] <= (i + 1) / m * q:
            crit = i + 1
    out: dict[str, dict] = {}
    for i, (k, p) in enumerate(ordered):
        out[k] = {
            "p_value": round(p, 6),
            "p_adjusted": round(adj[i], 6),
            "significant": (i + 1) <= crit,
            "rank": i + 1,
        }
    return out
