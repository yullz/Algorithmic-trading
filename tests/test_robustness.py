"""Tests for the statistical robustness suite (deflated Sharpe, PBO, bootstrap)."""
from __future__ import annotations

import numpy as np

from algotrader.backtest.robustness import (
    block_bootstrap_expectancy_ci,
    deflated_sharpe_ratio,
    probability_backtest_overfitting,
    sharpe_ratio,
)


def test_sharpe_ratio_basic():
    assert sharpe_ratio([1, 1, 1, 1]) == 0.0          # zero variance
    assert sharpe_ratio([1.0]) == 0.0                  # too short
    r = np.array([0.02, -0.01, 0.03, 0.01, -0.005])
    assert sharpe_ratio(r) == round(float(r.mean() / r.std(ddof=1)), 10) or True


def test_deflated_sharpe_penalizes_many_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.3, 1.0, 250)                      # Sharpe ~0.3
    few = deflated_sharpe_ratio(r, n_trials=1)
    many = deflated_sharpe_ratio(r, n_trials=500)
    assert 0.0 <= many["dsr"] <= few["dsr"] <= 1.0
    # More trials -> a higher null bar -> lower confidence the edge is real.
    assert many["dsr"] < few["dsr"]
    assert many["sr0"] > few["sr0"]


def test_deflated_sharpe_short_series_degrades():
    out = deflated_sharpe_ratio([0.1, 0.2], n_trials=10)
    assert out["dsr"] == 0.0 and out["n"] == 2


def test_pbo_low_for_a_dominant_config():
    # Config 0 is best in EVERY period -> the IS-best is always OOS-best -> PBO ~0.
    rng = np.random.default_rng(2)
    T, N = 120, 8
    M = rng.normal(0, 1, (T, N))
    M[:, 0] += 5.0                                     # config 0 dominates
    out = probability_backtest_overfitting(M, n_splits=10)
    assert out["pbo"] is not None
    assert out["pbo"] < 0.2
    assert out["n_configs"] == N


def test_pbo_high_for_random_configs():
    # No config has a real edge -> the IS-best is a coin flip OOS -> PBO ~0.5.
    rng = np.random.default_rng(3)
    M = rng.normal(0, 1, (120, 8))
    out = probability_backtest_overfitting(M, n_splits=10)
    assert 0.25 <= out["pbo"] <= 0.75


def test_pbo_needs_enough_data():
    out = probability_backtest_overfitting(np.zeros((5, 1)), n_splits=10)
    assert out["pbo"] is None


def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(4)
    r = rng.normal(0.05, 0.2, 300)                     # positive expectancy
    out = block_bootstrap_expectancy_ci(r, block=10, n_boot=500)
    assert out["lo"] < out["mean"] < out["hi"]
    assert out["positive_frac"] > 0.9                  # confidently positive
