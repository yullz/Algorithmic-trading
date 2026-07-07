"""Statistical robustness tests that separate a real edge from an overfit one.

A high in-sample expectancy proves nothing on its own — with enough parameter
tries, a random strategy will look great. These are the standard defenses:

  * Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): the probability the
    true Sharpe exceeds a benchmark AFTER correcting for the number of trials,
    the track-record length, and non-normal returns (skew/kurtosis). A raw
    Sharpe of 2 from 200 tries is not a Sharpe of 2.
  * Probability of Backtest Overfitting (PBO) via CSCV (Bailey et al. 2015):
    across many candidate configurations, how often does the in-sample-best
    config land in the WORSE half out-of-sample? PBO near 0.5 == pure luck.
  * Block bootstrap: a confidence band on expectancy that respects serial
    dependence (i.i.d. resampling understates the interval for trade streams).

All functions are pure and operate on plain return arrays, so they can validate
either per-trade R multiples or an account equity's period returns.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np

_EULER_GAMMA = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def sharpe_ratio(returns) -> float:
    """Per-period Sharpe (mean/std). Not annualized — the deflation is unit-free."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def deflated_sharpe_ratio(returns, n_trials: int,
                          sr_trials_std: float | None = None,
                          benchmark_sr: float = 0.0) -> dict:
    """Deflated Sharpe Ratio.

    Returns a dict with the observed Sharpe, the null expected-maximum Sharpe
    across `n_trials`, and `dsr` = P(true SR > benchmark) in [0, 1]. A DSR below
    ~0.95 means the observed Sharpe is not convincingly better than what the
    search itself would produce by chance.
    """
    r = np.asarray(returns, dtype=float)
    T = len(r)
    if T < 3:
        return {"sharpe": 0.0, "sr0": 0.0, "dsr": 0.0, "n_trials": n_trials, "n": T}
    sr = sharpe_ratio(r)
    # Higher moments of the returns (skew, kurtosis) — non-normality inflates the
    # variance of the Sharpe estimator.
    mu, sd = r.mean(), r.std(ddof=1)
    if sd <= 0:
        return {"sharpe": 0.0, "sr0": 0.0, "dsr": 0.0, "n_trials": n_trials, "n": T}
    z = (r - mu) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())  # non-excess (normal == 3)

    # Expected maximum Sharpe under the null across n_trials independent tries.
    # Trial-Sharpe std defaults to the estimator std under the null (~1/sqrt(T)).
    if sr_trials_std is None or sr_trials_std <= 0:
        sr_trials_std = 1.0 / math.sqrt(max(T - 1, 1))
    n = max(int(n_trials), 1)
    if n <= 1:
        sr0 = benchmark_sr
    else:
        emc = ((1 - _EULER_GAMMA) * _norm_ppf(1 - 1.0 / n)
               + _EULER_GAMMA * _norm_ppf(1 - 1.0 / (n * math.e)))
        sr0 = benchmark_sr + sr_trials_std * emc

    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2))
    dsr = _norm_cdf(((sr - sr0) * math.sqrt(max(T - 1, 1))) / denom)
    return {"sharpe": round(sr, 4), "sr0": round(sr0, 4), "dsr": round(dsr, 4),
            "skew": round(skew, 3), "kurtosis": round(kurt, 3),
            "n_trials": n, "n": T}


def probability_backtest_overfitting(returns_matrix, n_splits: int = 10) -> dict:
    """PBO via Combinatorially Symmetric Cross-Validation.

    `returns_matrix` is T periods x N configurations. Time is cut into `n_splits`
    contiguous blocks; for every way to choose half the blocks as in-sample, the
    config that maximizes IS performance is checked for its OUT-of-sample rank.
    PBO is the fraction of splits where that IS-best config lands below the OOS
    median. ~0.5 == the ranking is noise; near 0 == robust.
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2 or M.shape[0] < n_splits:
        return {"pbo": None, "n_configs": int(M.shape[1]) if M.ndim == 2 else 0,
                "n_splits": 0, "note": "need >=2 configs and >=n_splits periods"}
    T, N = M.shape
    if n_splits % 2 == 1:
        n_splits -= 1
    bounds = np.linspace(0, T, n_splits + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_splits)]

    logits = []
    for is_ids in combinations(range(n_splits), n_splits // 2):
        is_rows = np.concatenate([blocks[i] for i in is_ids])
        oos_rows = np.concatenate([blocks[i] for i in range(n_splits)
                                   if i not in is_ids])
        is_perf = M[is_rows].mean(axis=0)
        oos_perf = M[oos_rows].mean(axis=0)
        best = int(np.argmax(is_perf))
        # Relative OOS rank of the IS-best (1 = worst .. N = best).
        rank = float((oos_perf <= oos_perf[best]).sum())
        omega = rank / (N + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(math.log(omega / (1 - omega)))

    logits = np.asarray(logits)
    pbo = float((logits <= 0).mean())
    return {"pbo": round(pbo, 4), "n_configs": N, "n_splits": n_splits,
            "n_combinations": len(logits)}


def block_bootstrap_expectancy_ci(returns, block: int = 10, n_boot: int = 1000,
                                  alpha: float = 0.05, seed: int = 7) -> dict:
    """Confidence interval on mean return via circular block bootstrap, which
    preserves short-range serial dependence that i.i.d. resampling ignores."""
    r = np.asarray(returns, dtype=float)
    T = len(r)
    if T < max(block, 5):
        return {"mean": float(r.mean()) if T else 0.0, "lo": None, "hi": None, "n": T}
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(T / block))
    means = np.empty(n_boot)
    ext = np.concatenate([r, r[:block]])  # circular
    for i in range(n_boot):
        starts = rng.integers(0, T, n_blocks)
        sample = np.concatenate([ext[s:s + block] for s in starts])[:T]
        means[i] = sample.mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": round(float(r.mean()), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "n": T, "positive_frac": round(float((means > 0).mean()), 4)}


_DEFAULT_PARAM_GRID = {
    "confidence": [0.55, 0.60, 0.65, 0.70, 0.75],
    "n_families": [2, 3, 4, 5],
}


def parameter_stability(trades, params: dict | None = None, n_periods: int = 8,
                        min_cell: int = 5) -> dict:
    """Parameter-stability heatmap: how the realized edge holds as each tunable
    gate is tightened, across time periods.

    A robust edge stays positive across a RANGE of a gate's values and across
    periods — it should not hinge on one lucky knob setting. For a pure
    post-filter gate on a recorded feature (confidence, n_families), keeping only
    trades with feature >= threshold IS the honest counterfactual: those trades
    and their realized R already happened, so no re-simulation is needed.

    Returns per-param {thresholds, matrix (mean-R per threshold x period, null
    where a cell has < min_cell trades), counts, positive_frac} plus an overall
    positive-cell fraction — a single "is the edge knob-robust" number.
    """
    if not trades:
        return {"present": False}
    specs = params if params is not None else _DEFAULT_PARAM_GRID
    order = sorted(range(len(trades)),
                   key=lambda i: str(trades[i].get("entry_time", "")))
    chunks = [c for c in np.array_split(order, max(1, n_periods)) if len(c)]
    if not chunks:
        return {"present": False}

    out_params: dict = {}
    all_finite: list[float] = []
    for pname, thresholds in specs.items():
        if not any(pname in t for t in trades):
            continue
        matrix, counts = [], []
        for thr in thresholds:
            row, crow = [], []
            for chunk in chunks:
                rs = [trades[i]["r"] for i in chunk
                      if _as_float(trades[i].get(pname)) >= thr]
                if len(rs) >= min_cell:
                    row.append(round(float(np.mean(rs)), 4))
                else:
                    row.append(None)
                crow.append(len(rs))
            matrix.append(row)
            counts.append(crow)
        finite = [v for r in matrix for v in r if v is not None]
        all_finite.extend(finite)
        out_params[pname] = {
            "thresholds": list(thresholds),
            "matrix": matrix,
            "counts": counts,
            "n_cells": len(finite),
            "positive_frac": round(sum(v > 0 for v in finite) / len(finite), 4)
            if finite else None,
        }

    return {
        "present": bool(out_params),
        "n_periods": len(chunks),
        "params": out_params,
        "overall_positive_frac": round(sum(v > 0 for v in all_finite) / len(all_finite), 4)
        if all_finite else None,
    }


def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("-inf")
