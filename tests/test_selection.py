"""Tests for nested walk-forward + FDR selection helpers (GODMODE_PLAN.md P5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrader.backtest.selection import (
    anchored_folds, benjamini_hochberg, binom_p_greater, time_order,
    tune_threshold_cv, wilson_interval)


def test_wilson_interval_brackets_p():
    lo, hi = wilson_interval(0.7, 100)
    assert 0.0 <= lo <= 0.7 <= hi <= 1.0
    assert wilson_interval(0.5, 0) == (0.0, 1.0)


def test_time_order_sorts_and_degrades():
    order = time_order(["2024-01-03", "2024-01-01", "2024-01-02"])
    assert list(order) == [1, 2, 0]
    assert time_order(None) is None
    assert time_order(["x", "x", "x"]) is None          # constant -> unusable
    assert time_order(["2024-01-01"]) is None            # <2 usable -> None


def test_anchored_folds_are_expanding_and_disjoint():
    order = np.arange(60)
    folds = anchored_folds(order, n_folds=5)
    assert len(folds) == 5
    prev_train = -1
    for train, test in folds:
        # train strictly precedes test in time order (anchored / no leakage)
        assert train.max() < test.min()
        assert len(train) > prev_train          # expanding window
        prev_train = len(train)
    assert anchored_folds(np.arange(3), n_folds=5) == []  # too few rows


def test_tune_threshold_cv_finds_oos_edge():
    """High-strength trades win; the CV should surface a positive OOS edge."""
    rng = np.random.default_rng(0)
    n = 400
    strength = rng.uniform(0, 1, n)
    # r is +1 when strength is high, -1 otherwise (a genuine, learnable cutoff).
    r = np.where(strength >= 0.6, 1.0, -1.0)
    times = pd.date_range("2024-01-01", periods=n, freq="h").astype(str)
    out = tune_threshold_cv(strength, r, times)
    assert out is not None
    assert out["oos_expectancy_r"] > 0
    assert 0.4 <= out["threshold_median"] <= 0.9
    assert out["oos_n"] >= 10


def test_tune_threshold_cv_needs_enough_data():
    assert tune_threshold_cv(np.linspace(0, 1, 20), np.ones(20)) is None


def test_binom_p_greater_monotone_and_bounded():
    assert binom_p_greater(90, 100, 0.5) < 1e-6      # way above chance
    assert binom_p_greater(50, 100, 0.5) == pytest.approx(0.5, abs=0.1)
    assert binom_p_greater(10, 100, 0.5) > 0.99      # below chance -> not rare
    # monotonically decreasing in wins
    ps = [binom_p_greater(w, 100, 0.5) for w in (40, 50, 60, 70)]
    assert ps == sorted(ps, reverse=True)
    assert binom_p_greater(5, 0, 0.5) == 1.0         # n=0 guard


def test_benjamini_hochberg_flags_true_signal():
    pvals = {"real": 0.0001, "n1": 0.62, "n2": 0.71, "n3": 0.55, "n4": 0.9}
    bh = benjamini_hochberg(pvals, q=0.10)
    assert bh["real"]["significant"] is True
    assert all(not bh[k]["significant"] for k in ("n1", "n2", "n3", "n4"))
    # adjusted p is never below raw p, and ranks are 1..m
    for k, info in bh.items():
        assert info["p_adjusted"] >= info["p_value"] - 1e-9
    assert sorted(i["rank"] for i in bh.values()) == [1, 2, 3, 4, 5]


def test_benjamini_hochberg_all_null_flags_nothing():
    rng = np.random.default_rng(1)
    pvals = {f"f{i}": float(rng.uniform(0.2, 1.0)) for i in range(30)}
    bh = benjamini_hochberg(pvals, q=0.10)
    assert sum(v["significant"] for v in bh.values()) == 0
    assert benjamini_hochberg({}, q=0.1) == {}
