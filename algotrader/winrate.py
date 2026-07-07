"""Honest win-rate estimation.

The "expected win rate" is NOT a prediction of the future. It is a blend of the
historical hit rates of each contributing factor (measured by the backtester and
stored in calibration.json; a conservative prior is used until calibrated),
combined via log-odds pooling with a *bounded, diminishing* confluence bonus,
optionally blended with an ML meta-model probability (also in log-odds space).

Design choices that keep it honest:
  * Individual base rates are clamped to [0.05, 0.95].
  * The confluence bonus grows with log(n) and saturates — 8 agreeing signals
    are not meaningfully better than 5.
  * The final estimate is hard-capped to [0.30, 0.78]. If a system claims a 90%
    win rate on leveraged crypto, it is overfit or lying. The ML blend cannot
    escape this cap either.
  * Calibration lookups prefer regime/timeframe-conditioned keys
    ("factor|regime|tf" -> "factor|regime" -> "factor") so a factor that only
    works in trends is not credited in chop — but always fall back to the
    global rate when the conditioned sample is missing.
  * Calibrated factors with a Wilson lower bound below 0.35 have their
    confluence weight penalized, so low-confidence historical edges do not
    dominate the pooled estimate.
"""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone

from .models import Evidence
from .signals.confluence import family_of

WIN_RATE_FLOOR = 0.30
WIN_RATE_CAP = 0.78

log = logging.getLogger("algotrader.winrate")

# Module-level caches so calibration warnings don't spam hot loops.
_FALLBACK_WARNED: set[str] = set()
_STALENESS_WARNED: set[str] = set()


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _family_representatives(agreeing: list[Evidence]) -> list[Evidence]:
    """Collapse correlated evidence: keep only the single strongest reading per
    family, so e.g. six co-firing trend indicators count once, not six times."""
    best: dict[str, Evidence] = {}
    for e in agreeing:
        fam = family_of(e)
        if fam not in best or e.strength > best[fam].strength:
            best[fam] = e
    return list(best.values())


def _warn_staleness(calibration_path: str | None, staleness_days: int) -> None:
    """Warn once per path if the calibration file is older than staleness_days."""
    if not calibration_path or calibration_path in _STALENESS_WARNED:
        return
    try:
        mtime = os.path.getmtime(calibration_path)
        age_days = (
            datetime.now(timezone.utc)
            - datetime.fromtimestamp(mtime, tz=timezone.utc)
        ).total_seconds() / 86400
        if age_days > staleness_days:
            log.warning(
                "Calibration file %s is %.1f days old (older than %d days); "
                "consider regenerating it with backtest.py",
                calibration_path, age_days, staleness_days,
            )
    except OSError:
        pass
    _STALENESS_WARNED.add(calibration_path)


def calibrated_rate(name: str, default: float, calibration: dict,
                    regime: str = "", timeframe: str = "",
                    calibration_path: str | None = None,
                    staleness_days: int = 30) -> float:
    """Factor hit-rate lookup with a regime/timeframe fallback chain.

    Values in ``calibration`` may be either plain floats (legacy files) or
    dicts produced by the new backtest export (in which case the ``rate`` key
    is used). Logs a warning when falling all the way back to the default
    prior, and optionally warns if ``calibration_path`` is stale.
    """
    rate, _ = _lookup_calibrated(
        name, default, calibration, regime, timeframe,
        calibration_path, staleness_days,
    )
    return rate


def _lookup_calibrated(name: str, default: float, calibration: dict,
                       regime: str, timeframe: str,
                       calibration_path: str | None,
                       staleness_days: int) -> tuple[float, dict | None]:
    """Return (rate, metadata) for the best matching calibration key."""
    _warn_staleness(calibration_path, staleness_days)

    keys = []
    if regime and timeframe:
        keys.append(f"{name}|{regime}|{timeframe}")
    if regime:
        keys.append(f"{name}|{regime}")
    keys.append(name)

    for key in keys:
        v = calibration.get(key)
        if v is None:
            continue
        if isinstance(v, dict):
            return float(v["rate"]), v
        return float(v), None

    # Fall all the way back to the default prior. Only warn when a calibration
    # dictionary is actually present; an empty calibration (e.g. during an
    # uncalibrated backtest run) is expected to fall back to priors.
    if calibration:
        warn_key = name
        if warn_key not in _FALLBACK_WARNED:
            _FALLBACK_WARNED.add(warn_key)
            log.warning(
                "calibrated_rate fell back to default prior for factor %r "
                "(regime=%r, timeframe=%r)",
                name, regime, timeframe,
            )
    return float(default), None


def estimate_win_rate(agreeing: list[Evidence],
                      calibration: dict[str, float] | None = None,
                      regime: str = "", timeframe: str = "",
                      calibration_path: str | None = None,
                      staleness_days: int = 30) -> float:
    calibration = calibration or {}
    if not agreeing:
        return 0.5

    reps = _family_representatives(agreeing)  # de-correlated evidence

    logits, weights = [], []
    for e in reps:
        base, meta = _lookup_calibrated(
            e.name, e.base_win_rate, calibration, regime, timeframe,
            calibration_path, staleness_days,
        )
        base = min(max(base, 0.05), 0.95)

        # Penalize factors whose historical edge is not confidently above
        # randomness (Wilson lower bound < 0.35).
        penalty = 1.0
        if isinstance(meta, dict) and meta.get("wilson_lower", 1.0) < 0.35:
            penalty = 0.5

        logits.append(_logit(base))
        weights.append(max(e.strength, 0.05) * penalty)

    wmean = sum(w * l for w, l in zip(weights, logits)) / sum(weights)

    # Confluence bonus from the number of INDEPENDENT families (not raw factors).
    n_families = len(reps)
    bonus = 0.18 * math.log(1 + n_families)      # diminishing, in log-odds
    bonus = min(bonus, 0.5)                        # saturate

    p = _sigmoid(wmean + bonus)
    return min(max(p, WIN_RATE_FLOOR), WIN_RATE_CAP)


def blend_win_rate(rule_p: float, ml_p: float | None, ml_weight: float) -> float:
    """Blend the rule-based pooled estimate with an ML meta-model probability
    in log-odds space. `ml_weight` in [0, 1] reflects how much the model has
    earned trust (training size, OOS AUC); 0 means rules-only. The honesty cap
    always applies."""
    if ml_p is None or ml_weight <= 0:
        return min(max(rule_p, WIN_RATE_FLOOR), WIN_RATE_CAP)
    w = min(max(ml_weight, 0.0), 1.0)
    blended = _sigmoid((1 - w) * _logit(rule_p) + w * _logit(ml_p))
    return min(max(blended, WIN_RATE_FLOOR), WIN_RATE_CAP)


def expected_value_r(win_rate: float, avg_win_r: float, avg_loss_r: float = 1.0,
                     fees_r: float = 0.0) -> float:
    """Expected value per trade in R units, net of fees (also in R)."""
    return win_rate * avg_win_r - (1 - win_rate) * avg_loss_r - fees_r
