"""Bybit live executor: edge safety catch + free-margin preflight.

The exchange client is mocked, so these run offline and never touch the network.
They lock two money-critical rules:
  1. Never open a live position unless a positive OUT-OF-SAMPLE edge is on record
     (the edge safety catch), and
  2. never open one unless enough FREE USDT is available to fund its margin.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from algotrader.models import Bias, Evidence, RiskConfig, SetupKind, Side, Signal
from algotrader.risk.manager import RiskManager


def _plan(margin_usdt: float = 15.0):
    cfg = RiskConfig(fixed_margin_usdt=margin_usdt, default_leverage=3.0)
    sig = Signal(
        symbol="BTC/USDT:USDT", timeframe="1h", side=Side.LONG, kind=SetupKind.BREAKOUT,
        entry_ref=100.0, stop_ref=99.0, confidence=0.6, score=1.0,
        evidence=[Evidence("trend_up", "indicator", Bias.BULLISH, 0.7, 0.55, family="trend")],
        base_win_rate=0.55)
    return RiskManager(cfg).build_plan(sig)


def _mk_exec(tmp_path, free, total=None, positions=None, require_edge=False):
    total = free if total is None else total
    with patch("ccxt.bybit") as mk:
        ex = MagicMock()
        ex.fetch_balance.return_value = {"USDT": {"free": free, "total": total}}
        ex.fetch_positions.return_value = positions or []
        # order-size preflight + microstructure preflight mocks (tiny step so any
        # plan.qty clears; tight spread; benign funding; full IOC fill).
        ex.market.return_value = {
            "limits": {"amount": {"min": 0.001}},
            "precision": {"amount": 0.001},
            "info": {"lotSizeFilter": {"minNotionalValue": "5"}},
        }
        ex.amount_to_precision.side_effect = lambda _s, q: f"{float(q):.6f}"
        ex.price_to_precision.side_effect = lambda _s, p: f"{float(p):.6f}"
        ex.fetch_ticker.return_value = {"bid": 99.9, "ask": 100.1,
                                        "info": {"fundingRate": "0.00001"}}
        ex.create_order.return_value = {"id": "o", "filled": 0.45}
        ex.fetch_order.return_value = {"id": "o", "filled": 0.45}
        mk.return_value = ex
        from algotrader.execution.bybit import BybitExecutor
        exe = BybitExecutor(
            RiskConfig(fixed_margin_usdt=15.0, require_validated_edge=require_edge),
            "k", "s", testnet=True, root=str(tmp_path))
    return exe, ex


def _write_wf(tmp_path, exp, pf):
    os.makedirs(os.path.join(tmp_path, "reports"), exist_ok=True)
    with open(os.path.join(tmp_path, "reports", "walkforward.json"), "w") as f:
        json.dump({"out_of_sample": {"expectancy_r": exp, "profit_factor": pf}}, f)


def _stale_ohlcv(ex):
    old_ms = int((datetime.now(timezone.utc).timestamp() - 10 * 86400) * 1000)
    ex.fetch_ohlcv.return_value = [[old_ms - 3600_000, 1, 1, 1, 1, 1],
                                   [old_ms, 1, 1, 1, 1, 1]]


# --------------------------------------------------------------------------- #
# free-margin preflight (edge catch disabled)
# --------------------------------------------------------------------------- #
def test_free_usdt_reads_free_not_total(tmp_path):
    exe, _ = _mk_exec(tmp_path, free=42.5, total=99.0)
    assert exe.free_usdt() == pytest.approx(42.5)
    assert exe.equity() == pytest.approx(99.0)   # equity still reads total


def test_open_position_skips_when_free_below_margin(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=10.0, total=1000.0, positions=[])
    assert exe.open_position(_plan(15.0)) is None
    ex.create_order.assert_not_called()          # no order ever placed


def test_open_position_passes_preflight_when_free_sufficient(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0, total=1000.0, positions=[])
    _stale_ohlcv(ex)                             # stale -> fail safe AFTER preflight
    assert exe.open_position(_plan(15.0)) is None
    ex.fetch_ohlcv.assert_called()               # preflight was passed
    ex.create_order.assert_not_called()


# --------------------------------------------------------------------------- #
# edge safety catch (live-only, default ON)
# --------------------------------------------------------------------------- #
def test_edge_gate_blocks_when_no_walkforward(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=1000.0, total=1000.0, require_edge=True)
    assert exe.open_position(_plan(15.0)) is None
    ex.create_order.assert_not_called()
    ex.fetch_ohlcv.assert_not_called()           # blocked before the freshness check


def test_edge_gate_blocks_on_negative_oos(tmp_path):
    _write_wf(tmp_path, exp=-0.07, pf=0.9)       # the real deep-data picture
    exe, ex = _mk_exec(tmp_path, free=1000.0, total=1000.0, require_edge=True)
    assert exe.open_position(_plan(15.0)) is None
    ex.create_order.assert_not_called()


def test_edge_gate_passes_on_positive_oos(tmp_path):
    _write_wf(tmp_path, exp=0.2, pf=1.5)         # a validated positive OOS edge
    exe, ex = _mk_exec(tmp_path, free=1000.0, total=1000.0, require_edge=True)
    _stale_ohlcv(ex)
    assert exe.open_position(_plan(15.0)) is None  # stale -> fail safe
    ex.fetch_ohlcv.assert_called()               # but it cleared the edge gate + preflight


# --------------------------------------------------------------------------- #
# free==0.0 must not be backfilled from total (review fix)
# --------------------------------------------------------------------------- #
def test_free_usdt_genuine_zero_is_not_backfilled(tmp_path):
    exe, _ = _mk_exec(tmp_path, free=0.0, total=100.0)   # margin fully deployed
    assert exe.free_usdt() == 0.0                        # NOT 100.0


def test_free_usdt_missing_key_falls_back_to_total(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=0.0, total=50.0)
    ex.fetch_balance.return_value = {"USDT": {"total": 50.0}}  # no 'free' key
    assert exe.free_usdt() == 50.0


def test_free_usdt_uses_unified_available_balance(tmp_path):
    # Unified account: per-coin USDT.free reads ~0 while margin is deployed, but
    # totalAvailableBalance is the real openable amount -> must use the latter.
    exe, ex = _mk_exec(tmp_path, free=0.24, total=119.8)
    ex.fetch_balance.return_value = {
        "USDT": {"free": 0.24, "total": 119.8},
        "info": {"result": {"list": [
            {"accountType": "UNIFIED", "totalAvailableBalance": "33.40"}]}}}
    assert exe.free_usdt() == pytest.approx(33.40)


def test_zero_free_blocks_open(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=0.0, total=1000.0, positions=[])
    assert exe.open_position(_plan(15.0)) is None
    ex.create_order.assert_not_called()          # genuine 0 free -> blocked


# --------------------------------------------------------------------------- #
# live losing-streak breaker: reconcile_closed feeds it (review fix)
# --------------------------------------------------------------------------- #
def _closed_pnl(ex, pnl):
    ex.private_get_v5_position_closed_pnl.return_value = {
        "result": {"list": [{"closedPnl": str(pnl)}]}}


def test_reconcile_closed_advances_losing_streak(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    exe._tracked = {"BTC/USDT:USDT": {}}
    _closed_pnl(ex, -5.0)                          # the position closed at a loss
    exe.reconcile_closed({"BTC/USDT:USDT"}, set())
    assert exe.consecutive_losses == 1
    assert "BTC/USDT:USDT" not in exe._tracked


def test_reconcile_closed_resets_streak_on_win(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    exe.consecutive_losses = 3
    exe._tracked = {"ETH/USDT:USDT": {}}
    _closed_pnl(ex, 12.0)                          # closed at a profit
    exe.reconcile_closed({"ETH/USDT:USDT"}, set())
    assert exe.consecutive_losses == 0


def test_reconcile_closed_keeps_streak_when_pnl_unknown(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    exe.consecutive_losses = 2
    exe._tracked = {"SOL/USDT:USDT": {}}
    ex.private_get_v5_position_closed_pnl.side_effect = Exception("api down")
    exe.reconcile_closed({"SOL/USDT:USDT"}, set())
    assert exe.consecutive_losses == 2            # not guessed
    assert "SOL/USDT:USDT" not in exe._tracked    # still untracked


def _foreign_pos(symbol, margin=30.0):
    """A position NOT opened by the bot (e.g. a manual trade on the account)."""
    return {"symbol": symbol, "contracts": 1.0, "side": "short",
            "entryPrice": 100.0, "leverage": 3, "initialMargin": margin,
            "unrealizedPnl": 0.0, "datetime": "2026-07-01T00:00:00Z", "id": symbol}


def test_manual_positions_do_not_starve_the_bot(tmp_path):
    # 4 manual shorts (~120 margin) exist; the bot has 33 free. A NEW symbol must
    # still clear the portfolio caps (the bot's OWN book is empty) and reach the
    # freshness check -> the manual positions no longer block the bot's trade.
    foreign = [_foreign_pos(s) for s in
               ("MU/USDT:USDT", "XLM/USDT:USDT", "HYPE/USDT:USDT", "INTC/USDT:USDT")]
    exe, ex = _mk_exec(tmp_path, free=33.0, total=120.0, positions=foreign)
    _stale_ohlcv(ex)                     # fail-safe AFTER caps + free gate pass
    assert exe.open_position(_plan(15.0)) is None
    ex.fetch_ohlcv.assert_called()       # got past portfolio caps + free preflight
    ex.create_order.assert_not_called()


def test_bot_never_stacks_on_an_existing_symbol(tmp_path):
    # A position already exists on the bot's target symbol (BTC) -> one-per-symbol
    # guard blocks BEFORE the freshness check, regardless of whose position it is.
    exe, ex = _mk_exec(tmp_path, free=100.0, total=120.0,
                       positions=[_foreign_pos("BTC/USDT:USDT")])
    assert exe.open_position(_plan(15.0)) is None
    ex.fetch_ohlcv.assert_not_called()   # blocked at one-per-symbol
    ex.create_order.assert_not_called()


def test_restart_restores_tracked_positions_for_caps(tmp_path):
    # Simulate a restart: the bot previously opened ETH; the persisted state must
    # restore it so the concurrency cap still counts the bot's own position and
    # does NOT reset to an empty book (the money-safety bug the review caught).
    os.makedirs(os.path.join(tmp_path, "reports"), exist_ok=True)
    with open(os.path.join(tmp_path, "reports", "live_state.json"), "w") as f:
        json.dump({"consecutive_losses": 0, "day_anchor": {"date": "", "equity": 1000},
                   "tracked_symbols": ["ETH/USDT:USDT"]}, f)
    with patch("ccxt.bybit") as mk:
        ex = MagicMock()
        ex.fetch_balance.return_value = {"USDT": {"free": 100.0, "total": 100.0}}
        ex.fetch_positions.return_value = [_foreign_pos("ETH/USDT:USDT")]  # still open
        mk.return_value = ex
        from algotrader.execution.bybit import BybitExecutor
        exe = BybitExecutor(
            RiskConfig(fixed_margin_usdt=15.0, require_validated_edge=False,
                       max_concurrent_positions=1),
            "k", "s", testnet=True, root=str(tmp_path))
    assert "ETH/USDT:USDT" in exe._tracked         # restored across the "restart"
    # cap is 1 and the restored ETH counts -> a new BTC trade must be blocked.
    assert exe.open_position(_plan(15.0)) is None   # _plan is on BTC/USDT:USDT
    ex.create_order.assert_not_called()


def test_open_persists_tracked_symbols(tmp_path):
    # _save_state must write tracked_symbols so a crash right after opening still
    # remembers the position for the caps.
    exe, _ = _mk_exec(tmp_path, free=100.0)
    exe._tracked["SOL/USDT:USDT"] = {"plan": None}
    exe._save_state()
    with open(os.path.join(tmp_path, "reports", "live_state.json")) as f:
        assert "SOL/USDT:USDT" in json.load(f)["tracked_symbols"]


def test_reconcile_closed_ignores_still_open(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    exe._tracked = {"BTC/USDT:USDT": {}}
    exe.reconcile_closed({"BTC/USDT:USDT"}, {"BTC/USDT:USDT"})  # still open
    assert exe.consecutive_losses == 0
    ex.private_get_v5_position_closed_pnl.assert_not_called()
    assert "BTC/USDT:USDT" in exe._tracked


# --------------------------------------------------------------------------- #
# Tier 0-2: order-size preflight, TP ladder from real fill, filters, time-stop,
# day-anchor poison discard.
# --------------------------------------------------------------------------- #
def _tp(price, r, alloc):
    return MagicMock(price=price, r_multiple=r, allocation=alloc)


def test_preflight_qty_rounds_and_passes(tmp_path):
    exe, _ = _mk_exec(tmp_path, free=100.0)
    assert exe._preflight_qty("BTC/USDT:USDT", _plan(15.0)) == pytest.approx(0.45, abs=1e-6)


def test_preflight_qty_rejects_below_min(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    ex.market.return_value = {"limits": {"amount": {"min": 1.0}},
                              "precision": {"amount": 0.001}, "info": {}}
    assert exe._preflight_qty("BTC/USDT:USDT", _plan(15.0)) is None   # 0.45 < 1.0


def test_preflight_qty_rejects_excess_truncation(tmp_path):
    import math
    exe, ex = _mk_exec(tmp_path, free=100.0)
    ex.market.return_value = {"limits": {"amount": {"min": 0.1}},
                              "precision": {"amount": 0.1}, "info": {}}
    ex.amount_to_precision.side_effect = lambda _s, q: f"{math.floor(float(q) / 0.1) * 0.1:.4f}"
    assert exe._preflight_qty("BTC/USDT:USDT", _plan(15.0)) is None   # 0.45 -> 0.4 = 11%


def test_preflight_qty_invalidorder_is_skip(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    ex.amount_to_precision.side_effect = Exception("below minimum amount precision")
    assert exe._preflight_qty("BTC/USDT:USDT", _plan(15.0)) is None


def test_build_tp_legs_sums_to_fill(tmp_path):
    exe, _ = _mk_exec(tmp_path, free=100.0)
    legs = exe._build_tp_legs("BTC/USDT:USDT", 1.0,
                              [_tp(101, 1, 0.4), _tp(102, 2, 0.35), _tp(103, 3, 0.25)])
    assert len(legs) == 3
    assert sum(q for _, _, q in legs) == pytest.approx(1.0, abs=1e-9)


def test_build_tp_legs_merges_submin_rung(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    ex.market.return_value = {"limits": {"amount": {"min": 0.2}},
                              "precision": {"amount": 0.1}, "info": {}}
    # 0.5 fill, step 0.1 -> 5 steps, min 2 steps; the 1-step middle rung merges.
    legs = exe._build_tp_legs("BTC/USDT:USDT", 0.5,
                              [_tp(101, 1, 0.4), _tp(102, 2, 0.35), _tp(103, 3, 0.25)])
    assert sum(q for _, _, q in legs) == pytest.approx(0.5, abs=1e-9)
    assert all(q >= 0.2 - 1e-9 for _, _, q in legs)   # every placed leg clears min


def test_build_tp_legs_tiny_fill_returns_empty(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    ex.market.return_value = {"limits": {"amount": {"min": 0.1}},
                              "precision": {"amount": 0.1}, "info": {}}
    assert exe._build_tp_legs("BTC/USDT:USDT", 0.05, [_tp(101, 1, 1.0)]) == []


def test_open_position_happy_path_is_ioc(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0, total=1000.0)
    pid = exe.open_position(_plan(15.0))
    assert pid == "o"
    assert "BTC/USDT:USDT" in exe._tracked
    entry = ex.create_order.call_args_list[0]
    assert entry.args[1] == "limit"
    assert entry.kwargs["params"]["timeInForce"] == "IOC"


def test_open_skips_blocked_regime(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    plan = _plan(15.0); plan.regime = "volatile"
    assert exe.open_position(plan) is None
    ex.create_order.assert_not_called()


def test_open_skips_blocked_setup(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    plan = _plan(15.0); plan.kind = "reversal"
    assert exe.open_position(plan) is None
    ex.create_order.assert_not_called()


def test_open_skips_when_tf_budget_full(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    exe.cfg.live_tf_budget = {"15m": 0.2}   # max_concurrent 6 -> 1 slot
    exe._tracked["ETH/USDT:USDT"] = {"tf": "15m"}
    plan = _plan(15.0); plan.timeframe = "15m"
    assert exe.open_position(plan) is None
    ex.create_order.assert_not_called()


def test_open_skips_on_cooldown(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    exe._cooldown["BTC/USDT:USDT"] = datetime.now(timezone.utc).timestamp() + 3600
    assert exe.open_position(_plan(15.0)) is None
    ex.create_order.assert_not_called()


def test_open_skips_wide_spread(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    ex.fetch_ticker.return_value = {"bid": 99.0, "ask": 101.0, "info": {"fundingRate": "0"}}
    assert exe.open_position(_plan(15.0)) is None
    ex.create_order.assert_not_called()


def test_open_skips_paying_extreme_funding(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    ex.fetch_ticker.return_value = {"bid": 99.95, "ask": 100.05, "info": {"fundingRate": "0.01"}}
    assert exe.open_position(_plan(15.0)) is None   # LONG paying +1% > 0.05% cap
    ex.create_order.assert_not_called()


def test_open_skips_ioc_unfilled(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    ex.create_order.return_value = {"id": "o", "filled": 0.0}
    ex.fetch_order.return_value = {"id": "o", "filled": 0.0}
    assert exe.open_position(_plan(15.0)) is None   # nothing filled -> skip


def test_time_stop_closes_tracked_expired_once(tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    pos = _foreign_pos("BTC/USDT:USDT"); pos["datetime"] = old
    exe, ex = _mk_exec(tmp_path, free=100.0, positions=[pos])
    exe._tracked["BTC/USDT:USDT"] = {"tf": "1h", "time_stop": 1, "opened_at": old}
    out = exe.open_positions()
    assert ex.create_order.called                         # closed
    assert "BTC/USDT:USDT" not in [p.symbol for p in out]


def test_time_stop_ignores_manual_position(tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
    pos = _foreign_pos("XLM/USDT:USDT"); pos["datetime"] = old
    exe, ex = _mk_exec(tmp_path, free=100.0, positions=[pos])   # NOT tracked -> manual
    out = exe.open_positions()
    assert "XLM/USDT:USDT" in [p.symbol for p in out]
    ex.create_order.assert_not_called()


def test_foreign_stamped_state_discards_poison_keeps_tracked(tmp_path):
    # A DIFFERENT account's stamp -> its anchor/streak are foreign poison and are
    # dropped; tracked positions are still restored (they self-correct).
    os.makedirs(os.path.join(tmp_path, "reports"), exist_ok=True)
    with open(os.path.join(tmp_path, "reports", "live_state.json"), "w") as f:
        json.dump({"account": "deadbeef0000", "consecutive_losses": 4,
                   "day_anchor": {"date": "2026-07-08", "equity": 1000.0},
                   "tracked_symbols": ["BTC/USDT:USDT"]}, f)
    exe, _ = _mk_exec(tmp_path, free=120.0, total=120.0)
    assert exe.day_anchor["date"] == ""            # foreign anchor discarded
    assert exe.consecutive_losses == 0             # foreign streak discarded
    assert "BTC/USDT:USDT" in exe._tracked          # tracked restored (self-corrects)


def test_legacy_unstamped_state_is_trusted(tmp_path):
    # A present-but-UNSTAMPED file is OUR OWN legacy state (prior version wrote no
    # stamp) — trust it, or the upgrade would disarm both live breakers.
    os.makedirs(os.path.join(tmp_path, "reports"), exist_ok=True)
    with open(os.path.join(tmp_path, "reports", "live_state.json"), "w") as f:
        json.dump({"consecutive_losses": 4,
                   "day_anchor": {"date": "2026-07-08", "equity": 117.0},
                   "tracked_symbols": ["BTC/USDT:USDT"]}, f)     # no 'account'
    exe, _ = _mk_exec(tmp_path, free=120.0, total=120.0)
    assert exe.consecutive_losses == 4             # streak PRESERVED across upgrade
    assert exe.day_anchor["equity"] == 117.0       # anchor PRESERVED
    assert "BTC/USDT:USDT" in exe._tracked


def test_stamped_state_restores_anchor_and_streak(tmp_path):
    fp = hashlib.sha256(b"k|testnet").hexdigest()[:12]
    os.makedirs(os.path.join(tmp_path, "reports"), exist_ok=True)
    with open(os.path.join(tmp_path, "reports", "live_state.json"), "w") as f:
        json.dump({"account": fp, "consecutive_losses": 3,
                   "day_anchor": {"date": "2026-07-08", "equity": 117.0},
                   "tracked": {"BTC/USDT:USDT": {"tf": "1h"}}}, f)
    exe, _ = _mk_exec(tmp_path, free=120.0)
    assert exe.consecutive_losses == 3             # trusted (stamp matches)
    assert exe.day_anchor["equity"] == 117.0
    assert exe._tracked["BTC/USDT:USDT"]["tf"] == "1h"


def test_failed_time_stop_close_keeps_position(tmp_path):
    # A transient close failure must NOT orphan the position: it stays in the
    # book (so reconcile won't phantom-close it) and stays tracked (so the
    # time-stop retries next loop).
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    pos = _foreign_pos("BTC/USDT:USDT"); pos["datetime"] = old
    exe, ex = _mk_exec(tmp_path, free=100.0, positions=[pos])
    exe._tracked["BTC/USDT:USDT"] = {"tf": "1h", "time_stop": 1, "opened_at": old}
    ex.create_order.side_effect = Exception("rate limit")     # close fails
    out = exe.open_positions()
    assert "BTC/USDT:USDT" in [p.symbol for p in out]          # still in the book
    assert "BTC/USDT:USDT" in exe._tracked                     # still tracked -> retries


def test_live_position_carries_risk_snapshot(tmp_path):
    # portfolio_allows' open-risk cap needs a per-position risk; live positions
    # must expose qty*|entry-stop| via .plan (else every live pos counts 0).
    pos = _foreign_pos("BTC/USDT:USDT")
    pos["entryPrice"] = 100.0; pos["stopLossPrice"] = 98.0; pos["contracts"] = 2.0
    exe, _ = _mk_exec(tmp_path, free=100.0, positions=[pos])
    raw = exe._fetch_positions_raw()
    assert raw[0].plan["risk_amount"] == pytest.approx(4.0)    # 2 * |100-98|


def test_open_skips_below_confidence_floor(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    exe.cfg.min_live_confidence = 0.65        # _plan confidence is 0.60
    assert exe.open_position(_plan(15.0)) is None
    ex.create_order.assert_not_called()


def test_open_skips_below_ev_floor(tmp_path):
    exe, ex = _mk_exec(tmp_path, free=100.0)
    exe.cfg.min_live_ev_r = 5.0               # unreachably high EV bar
    assert exe.open_position(_plan(15.0)) is None
    ex.create_order.assert_not_called()
