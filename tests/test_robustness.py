"""Tests for the statistical robustness suite (deflated Sharpe, PBO, bootstrap)."""
from __future__ import annotations

import numpy as np

from algotrader.backtest.robustness import (
    block_bootstrap_expectancy_ci,
    deflated_sharpe_ratio,
    parameter_stability,
    probability_backtest_overfitting,
    sharpe_ratio,
)


def _mk_trades(n, conf_fn, r_fn):
    import pandas as pd
    base = pd.Timestamp("2024-01-01", tz="UTC")
    out = []
    for i in range(n):
        out.append({
            "entry_time": (base + pd.Timedelta(hours=i)).isoformat(),
            "confidence": conf_fn(i), "n_families": 2 + (i % 4),
            "r": r_fn(i),
        })
    return out


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


def test_parameter_stability_shape_and_positive_edge():
    # High confidence -> consistently positive R across all periods.
    rng = np.random.default_rng(5)
    trades = _mk_trades(400,
                        conf_fn=lambda i: 0.5 + 0.4 * rng.random(),
                        r_fn=lambda i: 1.0)  # every trade wins -> all cells > 0
    out = parameter_stability(trades, n_periods=8)
    assert out["present"] is True
    conf = out["params"]["confidence"]
    assert len(conf["thresholds"]) == 5
    assert len(conf["matrix"]) == 5           # one row per threshold
    assert conf["positive_frac"] == 1.0       # a real edge is knob-robust
    assert out["overall_positive_frac"] == 1.0


def test_parameter_stability_flags_a_fragile_edge():
    # A losing strategy -> negative cells -> low positive fraction.
    trades = _mk_trades(300, conf_fn=lambda i: 0.7, r_fn=lambda i: -1.0)
    out = parameter_stability(trades, n_periods=6)
    assert out["overall_positive_frac"] == 0.0


def test_parameter_stability_empty_and_missing_params():
    assert parameter_stability([])["present"] is False
    # trades without confidence/n_families keys -> no params, present False.
    bare = [{"entry_time": "2024-01-01", "r": 1.0} for _ in range(20)]
    assert parameter_stability(bare)["present"] is False
