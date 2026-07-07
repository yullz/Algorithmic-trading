"""Unit tests for algotrader/winrate.py and calibration robustness."""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from algotrader.backtest.engine import BacktestResult
from algotrader.models import Bias, Evidence
from algotrader import winrate


def test_calibrated_rate_fallback_chain():
    cal = {
        "ema_stack_bull|trend_up|1h": 0.70,
        "ema_stack_bull|trend_up": 0.65,
        "ema_stack_bull": 0.60,
    }
    assert winrate.calibrated_rate(
        "ema_stack_bull", 0.50, cal, regime="trend_up", timeframe="1h"
    ) == pytest.approx(0.70)
    assert winrate.calibrated_rate(
        "ema_stack_bull", 0.50, cal, regime="trend_up", timeframe="4h"
    ) == pytest.approx(0.65)
    assert winrate.calibrated_rate(
        "ema_stack_bull", 0.50, cal, regime="range", timeframe="1h"
    ) == pytest.approx(0.60)
    assert winrate.calibrated_rate(
        "missing_factor", 0.50, cal, regime="trend_up", timeframe="1h"
    ) == pytest.approx(0.50)


def test_calibrated_rate_fallback_logs_warning(caplog):
    """Falling all the way back to the default prior should emit one warning."""
    # Reset the module-level warning cache so the test sees the log.
    winrate._FALLBACK_WARNED.discard("warned_factor")
    with caplog.at_level("WARNING", logger="algotrader.winrate"):
        rate = winrate.calibrated_rate(
            "warned_factor", 0.45, {"other_factor": 0.6},
            regime="trend_up", timeframe="1h"
        )
    assert rate == pytest.approx(0.45)
    assert any(
        "fell back to default prior" in r.message and "warned_factor" in r.message
        for r in caplog.records
    )


def test_calibrated_rate_staleness_warning(tmp_path, caplog):
    """A calibration file older than staleness_days should warn once."""
    cal_path = tmp_path / "stale.json"
    cal_path.write_text(json.dumps({"some_factor": 0.6}))
    old_mtime = time.mktime((datetime.now(timezone.utc) - timedelta(days=60)).timetuple())
    os.utime(cal_path, (old_mtime, old_mtime))

    winrate._STALENESS_WARNED.discard(str(cal_path))
    with caplog.at_level("WARNING", logger="algotrader.winrate"):
        rate = winrate.calibrated_rate(
            "some_factor", 0.50, {"some_factor": 0.6},
            calibration_path=str(cal_path), staleness_days=30,
        )
    assert rate == pytest.approx(0.6)
    assert any("days old" in r.message and "60." in r.message for r in caplog.records)


def test_calibrated_rate_handles_dict_and_float_values():
    """New dict-style calibration entries should return the 'rate' key."""
    cal = {
        "legacy_factor": 0.72,
        "new_factor": {"rate": 0.68, "raw": 0.75, "wilson_lower": 0.55},
    }
    assert winrate.calibrated_rate("legacy_factor", 0.5, cal) == pytest.approx(0.72)
    assert winrate.calibrated_rate("new_factor", 0.5, cal) == pytest.approx(0.68)


def test_estimate_win_rate_clamped():
    """The final win-rate estimate must respect the [0.30, 0.78] honesty cap."""
    strong_independent = [
        Evidence(f"e{i}", "indicator", Bias.BULLISH, 1.0, 0.95, family=f"fam{i}")
        for i in range(5)
    ]
    high = winrate.estimate_win_rate(strong_independent, {})
    assert high <= 0.78

    weak = [Evidence("loser", "indicator", Bias.BEARISH, 1.0, 0.05, family="x")]
    low = winrate.estimate_win_rate(weak, {})
    assert low >= 0.30


def test_estimate_win_rate_empty():
    assert winrate.estimate_win_rate([], {}) == 0.5


def test_calibration_shrinkage():
    """Raw empirical rate should be stored, but the live rate is shrunk toward 0.5."""
    trades = [
        {"entry_idx": i, "win": i < 20, "r": 1.0 if i < 20 else -1.0,
         "factors": ["f"], "regime": "", "tf": ""}
        for i in range(25)
    ]
    res = BacktestResult(trades=trades, summary={"win_rate": 0.8})
    # These trades carry no entry_time, so recency weights are uniform and the
    # shrinkage formula can be verified in closed form.
    cal = res.calibration_dict(min_samples=25, half_life_days=100000)

    assert cal["f"]["raw"] == pytest.approx(0.80)
    assert cal["f"]["weighted"] == pytest.approx(0.80, abs=1e-3)
    # shrunk = (n * empirical + prior_strength * prior) / (n + prior_strength)
    assert cal["f"]["rate"] == pytest.approx((25 * 0.8 + 10 * 0.5) / 35, rel=1e-3)
    assert cal["f"]["rate"] < cal["f"]["raw"]
    assert cal["f"]["wilson_lower"] < cal["f"]["wilson_upper"]
    assert cal["f"]["n"] == 25


def test_calibration_recency_weighting():
    """With a short half-life, recent (calendar-time) outcomes dominate."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trades = []
    # 20 old losing trades (days 0..19).
    for i in range(20):
        trades.append({
            "entry_idx": i, "win": False, "r": -1.0,
            "factors": ["f"], "regime": "", "tf": "",
            "entry_time": (base + timedelta(days=i)).isoformat(),
        })
    # 20 new winning trades (~3 months later, days 90..109).
    for i in range(20):
        trades.append({
            "entry_idx": 20 + i, "win": True, "r": 1.0,
            "factors": ["f"], "regime": "", "tf": "",
            "entry_time": (base + timedelta(days=90 + i)).isoformat(),
        })

    res = BacktestResult(trades=trades, summary={"win_rate": 0.5})
    cal = res.calibration_dict(min_samples=25, half_life_days=10)

    assert cal["f"]["raw"] == pytest.approx(0.50)
    # Weighted rate should be pulled above 0.5 because recent trades are wins.
    assert cal["f"]["weighted"] > 0.55
    # Shrunk rate should sit between prior and weighted rate.
    assert 0.5 < cal["f"]["rate"] < cal["f"]["weighted"]


def test_calibration_recency_weighted_payoffs():
    """_avg_win_r / _avg_loss_r should reflect recency weighting."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trades = [
        {"entry_idx": 0, "win": True, "r": 0.5,
         "factors": ["f"], "regime": "", "tf": "",
         "entry_time": base.isoformat()},                          # old 0.5R win
        {"entry_idx": 100, "win": True, "r": 3.0,
         "factors": ["f"], "regime": "", "tf": "",
         "entry_time": (base + timedelta(days=100)).isoformat()},  # recent 3R win
        {"entry_idx": 101, "win": False, "r": -0.5,
         "factors": ["f"], "regime": "", "tf": "",
         "entry_time": (base + timedelta(days=101)).isoformat()},  # recent 0.5R loss
    ]
    res = BacktestResult(trades=trades, summary={"avg_win_r": 1.0, "avg_loss_r": -1.0})
    cal = res.calibration_dict(min_samples=1, half_life_days=10)

    # With a very short half-life, the old 0.5R win contributes almost nothing.
    assert cal["_avg_win_r"] > 1.5
    assert cal["_avg_loss_r"] == pytest.approx(0.5)


def test_confidence_interval_reduces_weight():
    """A factor with Wilson lower bound < 0.35 should have less pooling weight."""
    def make_evidence(name: str) -> Evidence:
        return Evidence(
            name, "indicator", Bias.BULLISH, 1.0, 0.8, family=name
        )

    neutral = make_evidence("neutral")
    high_ci = make_evidence("high_ci")
    low_ci = make_evidence("low_ci")

    cal = {
        "neutral": {"rate": 0.5, "wilson_lower": 0.45},
        "high_ci": {"rate": 0.8, "wilson_lower": 0.60},
        "low_ci": {"rate": 0.8, "wilson_lower": 0.20},
    }

    high_signal = winrate.estimate_win_rate([high_ci, neutral], cal)
    low_signal = winrate.estimate_win_rate([low_ci, neutral], cal)

    # Both strong factors have the same calibrated rate, but the low-CI one
    # is penalized, so the pooled estimate is pulled toward the neutral 0.5.
    assert low_signal < high_signal


def test_calibration_dict_min_samples_raised():
    """Default min_samples should now be 25; fewer samples drops the factor."""
    trades = [
        {"entry_idx": i, "win": True, "r": 1.0,
         "factors": ["thin"], "regime": "", "tf": ""}
        for i in range(20)
    ]
    res = BacktestResult(trades=trades, summary={"win_rate": 1.0})
    cal = res.calibration_dict()
    assert "thin" not in cal
    cal_loose = res.calibration_dict(min_samples=10)
    assert "thin" in cal_loose


def test_calibration_wilson_lower_gating():
    """min_wilson_lower drops factors whose edge is not confidently above it —
    the mechanism that keeps in-sample-lucky factors out of live calibration."""
    trades = []
    # 'weak': 26 trades at 50% -> Wilson lower ~0.32 (< 0.35) -> dropped.
    for i in range(26):
        win = i % 2 == 0
        trades.append({"entry_idx": i, "win": win, "r": 1.0 if win else -1.0,
                       "factors": ["weak"], "regime": "", "tf": ""})
    # 'strong': 30 trades at 80% -> Wilson lower ~0.63 -> kept.
    for i in range(30):
        win = i % 5 != 0
        trades.append({"entry_idx": 100 + i, "win": win, "r": 1.0 if win else -1.0,
                       "factors": ["strong"], "regime": "", "tf": ""})
    res = BacktestResult(trades=trades, summary={"win_rate": 0.6})

    gated = res.calibration_dict(min_samples=25, min_wilson_lower=0.35)
    assert "strong" in gated
    assert "weak" not in gated
    # Aggregate payoff keys always survive gating.
    assert "_overall" in gated

    ungated = res.calibration_dict(min_samples=25, min_wilson_lower=0.0)
    assert "weak" in ungated and "strong" in ungated
