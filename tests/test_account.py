"""Tests for the account-level backtester (compounding, drawdown, ruin)."""
from __future__ import annotations

from algotrader.backtest.account import simulate_account
from algotrader.models import RiskConfig


def _trades(rs, tf="1h", start="2024-01-01"):
    import pandas as pd
    base = pd.Timestamp(start, tz="UTC")
    out = []
    for i, r in enumerate(rs):
        out.append({
            "entry_time": (base + pd.Timedelta(hours=i * 5)).isoformat(),
            "tf": tf, "entry_idx": i * 5, "exit_idx": i * 5 + 3, "r": float(r),
        })
    return out


def test_account_grows_on_a_winning_edge():
    cfg = RiskConfig(account_equity=1000.0, risk_per_trade_pct=0.02,
                     max_concurrent_positions=3)
    # A positive-expectancy stream (2R wins outnumber 1R losses).
    rs = ([2.0, 2.0, -1.0] * 40)
    out = simulate_account(_trades(rs), cfg, mc_runs=200)
    assert out["present"] is True
    assert out["final_equity"] > out["start_equity"]
    assert out["cagr_pct"] > 0
    assert 0.0 <= out["max_drawdown_pct"] <= 100.0
    assert out["ruin_prob"] < 0.5


def test_account_ruin_on_a_losing_edge():
    cfg = RiskConfig(account_equity=1000.0, risk_per_trade_pct=0.10,
                     max_concurrent_positions=3)
    rs = ([-1.0, -1.0, 1.0] * 40)   # negative expectancy, big risk -> ruin likely
    out = simulate_account(_trades(rs), cfg, mc_runs=300)
    assert out["final_equity"] < out["start_equity"]
    assert out["ruin_prob"] > 0.5
    assert out["max_drawdown_pct"] > 0


def test_account_empty_or_too_few_trades():
    cfg = RiskConfig()
    assert simulate_account([], cfg)["present"] is False
    assert simulate_account(_trades([1.0, 2.0]), cfg)["present"] is False


def test_account_respects_concurrency_cap():
    cfg = RiskConfig(account_equity=1000.0, risk_per_trade_pct=0.01,
                     max_concurrent_positions=1)
    # Overlapping trades (long holds, close entries) with cap 1 -> some skipped.
    import pandas as pd
    base = pd.Timestamp("2024-01-01", tz="UTC")
    trades = [{
        "entry_time": (base + pd.Timedelta(minutes=i * 5)).isoformat(),
        "tf": "1h", "entry_idx": 0, "exit_idx": 10, "r": 1.0,  # 10h hold, 5m apart
    } for i in range(20)]
    out = simulate_account(trades, cfg, mc_runs=50)
    assert out["skipped_full_book"] > 0
