"""Bybit live executor — every order passes the full safety stack.

To place a REAL order, ALL of the following must hold, in this order:
  1. `ALLOW_LIVE_TRADING=true` in .env            (environment opt-in)
  2. `execution.mode: live` in config.yaml        (config opt-in)
  3. API keys present
  4. Testnet by default. Mainnet additionally requires
     `execution.testnet: false` AND `execution.allow_mainnet: true`
     (double config opt-in) — otherwise sandbox mode is forced on.
  5. `LIVE_TRADING_CONFIRM=YES-I-UNDERSTAND` in the environment, or an
     interactive confirmation typed at startup.
  6. Per-entry circuit breakers (kill-switch file, daily loss, losing streak)
     and portfolio caps — identical to paper.

Order shape: market entry with attached stopLoss, then reduce-only limit
orders for the TP ladder. Anything that fails after entry triggers an
immediate protective close attempt.
"""
from __future__ import annotations

import hashlib
import math
import os
from datetime import datetime, timezone
from typing import Optional

from ..models import RiskConfig, Side, TradePlan
from ..reporting import read_json, write_json
from ..risk.manager import RiskManager
from ..utils.logging import audit, get_logger, utcnow_iso
from .base import (CircuitBreakers, Executor, PositionState, portfolio_allows,
                   timeframe_to_seconds, validated_edge)

log = get_logger("live")

CONFIRM_PHRASE = "YES-I-UNDERSTAND"


class LiveTradingRefused(RuntimeError):
    """Raised when any gate of the safety stack is not satisfied."""


class BybitExecutor(Executor):
    def __init__(self, cfg: RiskConfig, api_key: str, api_secret: str,
                 testnet: bool = True, root: str = "."):
        import ccxt
        self.cfg = cfg
        self.testnet = testnet
        self.breakers = CircuitBreakers(cfg, root)
        self.state_path = os.path.join(root, "reports", "live_state.json")
        self.consecutive_losses = 0
        # Daily-loss breaker anchor: persisted and rolled at the UTC day boundary
        # (mirrors the paper executor) so it survives restarts and behaves as a
        # true INTRADAY guard instead of anchoring once to first-ever equity.
        self.day_anchor = {"date": "", "equity": cfg.account_equity}
        self._tracked: dict[str, dict] = {}   # symbol -> partial-TP fill state
        self._cooldown: dict[str, float] = {}  # symbol -> re-entry-allowed epoch ts
        self._close_pending: dict[str, float] = {}  # symbol -> close-issued epoch ts
        # Persisted state is only valid for THIS account+environment. A stamp of
        # the key + testnet flag lets _load_state discard foreign/poisoned state
        # (e.g. written by a test or another account) instead of trusting it.
        self._account_fp = hashlib.sha256(
            f"{api_key}|{'testnet' if testnet else 'mainnet'}".encode()
        ).hexdigest()[:12]
        self._load_state()
        self.ex = ccxt.bybit({
            "apiKey": api_key, "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        if testnet:
            self.ex.set_sandbox_mode(True)
        self.ex.load_markets()
        # Best-effort one-way position mode — the breakeven-stop update posts
        # positionIdx 0, which requires one-way (non-hedged) mode. Non-fatal:
        # the account may already be one-way or the call may be unsupported.
        try:
            self.ex.set_position_mode(False)
        except Exception as e:  # pragma: no cover - exchange/account dependent
            log.debug("set_position_mode(one-way): %s", e)
        log.warning("LIVE executor initialized on %s",
                    "TESTNET" if testnet else "*** MAINNET ***")

    # ------------------------------------------------------------------ #
    # persisted breaker state — the guard that matters most is only useful if a
    # process bounce cannot wipe it
    # ------------------------------------------------------------------ #
    def _load_state(self) -> None:
        s = read_json(self.state_path)
        if not s:
            return
        # Trust the account-SPECIFIC risk state (day-loss anchor + streak) unless
        # a DIFFERENT non-empty stamp proves the file belongs to another
        # account/env. A present-but-UNSTAMPED file is our own legacy state (the
        # prior version wrote no stamp): trust it and re-stamp on the next save —
        # discarding it would DISARM both live breakers on the first upgrade load
        # (a persisted losing streak / daily-loss anchor would silently reset,
        # fail-DANGER). Only a genuine foreign stamp (e.g. a different account's
        # poisoned equity=1000 anchor) is dropped so the anchor re-seeds from
        # live equity. Tracked positions + cooldowns are restored either way —
        # they self-correct against the live book (own = tracked ∩ live).
        # (Tests can no longer poison the real file: they use an isolated root.)
        foreign = bool(s.get("account")) and s.get("account") != self._account_fp
        try:
            if not foreign:
                self.consecutive_losses = int(s.get("consecutive_losses", 0))
                self.day_anchor = dict(s.get("day_anchor", self.day_anchor))
            else:
                log.warning("live state at %s carries a DIFFERENT account stamp — "
                            "dropping its daily-loss anchor + streak, keeping "
                            "tracked positions for cap accounting", self.state_path)
            # Restore the positions the bot itself opened, so the portfolio caps
            # still count the bot's OWN pre-restart book (own-book accounting
            # must survive a process bounce, or a restart silently resets
            # concurrency/risk to zero and lets it over-concentrate). The meta
            # (tf/time_stop/opened_at/entry_id) keeps the time-stop, tf-budget
            # and journaling working across restarts; the full plan (TP ladder /
            # breakeven state) is not restored — those positions stay protected
            # by their exchange-side SL/TP.
            tracked = s.get("tracked")
            if isinstance(tracked, dict):
                for sym, meta in tracked.items():
                    self._tracked.setdefault(str(sym), dict(meta or {}))
            else:  # legacy list format
                for sym in (s.get("tracked_symbols") or []):
                    self._tracked.setdefault(str(sym), {})
            now = datetime.now(timezone.utc).timestamp()
            for sym, until in dict(s.get("cooldown") or {}).items():
                if float(until) > now:
                    self._cooldown[str(sym)] = float(until)
            log.info("restored live breaker state: streak=%d, day_anchor=%s, "
                     "tracked=%d symbols, cooldowns=%d", self.consecutive_losses,
                     self.day_anchor, len(self._tracked), len(self._cooldown))
        except (TypeError, ValueError) as e:
            log.warning("live breaker state unreadable (%s) — starting fresh", e)

    @staticmethod
    def _tracked_meta(state: dict) -> dict:
        """Persistable meta of one tracked entry (the plan object itself is not
        serialized; its scheduling facts are)."""
        plan = state.get("plan")
        return {
            "tf": state.get("tf") or (getattr(plan, "timeframe", "") if plan else ""),
            "time_stop": int(state.get("time_stop")
                             or (getattr(plan, "time_stop_candles", 0) if plan else 0) or 0),
            "opened_at": state.get("opened_at") or "",
            "entry_id": state.get("entry_id") or "",
        }

    def _save_state(self) -> None:
        write_json(self.state_path, {
            "account": self._account_fp,
            "consecutive_losses": self.consecutive_losses,
            "day_anchor": self.day_anchor,
            "tracked": {sym: self._tracked_meta(st)
                        for sym, st in self._tracked.items()},
            "tracked_symbols": list(self._tracked.keys()),  # back-compat mirror
            "cooldown": dict(self._cooldown),
            "updated_at": utcnow_iso(),
        })

    def _roll_day_anchor(self, equity: float) -> None:
        """Reset the daily-loss reference at the UTC day boundary."""
        today = datetime.now(timezone.utc).date().isoformat()
        if self.day_anchor.get("date") != today:
            self.day_anchor = {"date": today, "equity": equity}
            self._save_state()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, app_cfg, interactive: bool = True) -> "BybitExecutor":
        """Walk the safety stack; raise LiveTradingRefused on the first gate
        that fails, with an actionable message."""
        if not app_cfg.allow_live:
            raise LiveTradingRefused(
                "ALLOW_LIVE_TRADING is not 'true' in .env — live trading is off.")
        if app_cfg.execution_mode != "live":
            raise LiveTradingRefused(
                "execution.mode is not 'live' in config.yaml.")
        if not app_cfg.api_key or not app_cfg.api_secret:
            raise LiveTradingRefused(
                "API_KEY / API_SECRET missing in .env (use testnet.bybit.com keys first).")
        testnet = True
        if not app_cfg.execution_testnet:
            if not app_cfg.allow_mainnet:
                raise LiveTradingRefused(
                    "execution.testnet=false requires execution.allow_mainnet=true "
                    "(double opt-in). Forcing testnet is the default for a reason.")
            testnet = False
        if os.getenv("LIVE_TRADING_CONFIRM", "") != CONFIRM_PHRASE:
            if interactive:
                print(f"\n*** {'TESTNET' if testnet else 'MAINNET — REAL MONEY'} "
                      f"LIVE TRADING ***\nType {CONFIRM_PHRASE} to continue: ", end="")
                if input().strip() != CONFIRM_PHRASE:
                    raise LiveTradingRefused("confirmation phrase not entered.")
            else:
                raise LiveTradingRefused(
                    f"set LIVE_TRADING_CONFIRM={CONFIRM_PHRASE} in the environment "
                    f"for non-interactive live sessions.")
        return cls(app_cfg.risk, app_cfg.api_key, app_cfg.api_secret, testnet)

    # ------------------------------------------------------------------ #
    def equity(self) -> float:
        try:
            bal = self.ex.fetch_balance()
            return float(bal.get("USDT", {}).get("total") or 0.0)
        except Exception as e:
            log.error("fetch_balance failed: %s", e)
            return 0.0

    def free_usdt(self) -> float:
        """Spendable (FREE) USDT margin — gates whether a new position can be
        funded. Distinct from equity() (total wallet value): a new trade must be
        affordable out of free margin, not just backed by total equity.

        A genuine free==0.0 (margin fully deployed) must be treated as 0.0, NOT
        backfilled from total — `free or total` would report total when free is a
        legitimate zero (0.0 is falsy) and silently defeat the preflight at the
        exact boundary it exists to guard. Only fall back to total if 'free' is
        absent (None)."""
        try:
            bal = self.ex.fetch_balance()
            # Unified Trading Account (Bybit's default): margin is POOLED across
            # the account, so the amount available to open a new position is the
            # account-level `totalAvailableBalance`, NOT the per-coin USDT 'free'
            # (which reads ~0 once other positions consume margin). Prefer it.
            for acct in (((bal.get("info") or {}).get("result") or {}).get("list") or []):
                if acct.get("accountType") == "UNIFIED":
                    tab = acct.get("totalAvailableBalance")
                    if tab not in (None, ""):
                        return float(tab)
            usdt = bal.get("USDT", {})
            free = usdt.get("free")
            return float(free if free is not None else (usdt.get("total") or 0.0))
        except Exception as e:
            log.error("fetch_balance (free) failed: %s", e)
            return 0.0

    def _fetch_positions_raw(self) -> list[PositionState]:
        """Exchange positions as-is — NO time-stop side effects. close_position
        must use this (not open_positions) or the two mutually recurse and can
        spam duplicate close orders."""
        out: list[PositionState] = []
        try:
            for p in self.ex.fetch_positions(params={"category": "linear"}):
                contracts = float(p.get("contracts") or 0)
                if contracts == 0:
                    continue
                side = Side.LONG if p.get("side") == "long" else Side.SHORT
                # timeframe comes from OUR tracked meta (the exchange has no
                # concept of it); manual positions correctly stay "".
                tf = str((self._tracked.get(p["symbol"]) or {}).get("tf") or "")
                entry = float(p.get("entryPrice") or 0)
                stop = float(p.get("stopLossPrice") or 0)
                # Snapshot the open risk-in-quote (qty × stop distance) so
                # portfolio_allows' max_portfolio_risk_pct cap actually counts
                # this position — PositionState.plan otherwise defaults to {} and
                # every live position would contribute 0 open risk.
                risk = contracts * abs(entry - stop) if (entry > 0 and stop > 0) else 0.0
                out.append(PositionState(
                    id=str(p.get("id") or p["symbol"]), symbol=p["symbol"],
                    timeframe=tf, side=side,
                    entry=entry,
                    qty_initial=contracts, qty_open=contracts,
                    stop=stop,
                    leverage=float(p.get("leverage") or 1),
                    margin=float(p.get("initialMargin") or 0),
                    unrealized_pnl=float(p.get("unrealizedPnl") or 0),
                    opened_at=str(p.get("datetime") or ""),
                    plan={"risk_amount": risk},
                ))
        except Exception as e:
            log.error("fetch_positions failed: %s", e)
        return out

    def open_positions(self) -> list[PositionState]:
        """Open positions, enforcing the time-stop on the bot's own trades.
        Expired positions are closed AFTER the scan (never mid-iteration) and a
        close-pending window stops duplicate close orders while the exchange
        still reports the position during settlement."""
        now = datetime.now(timezone.utc).timestamp()
        out: list[PositionState] = []
        expired: list[PositionState] = []
        for pos in self._fetch_positions_raw():
            recently_closed = now - self._close_pending.get(pos.symbol, 0.0) < 180
            if not recently_closed and self._duration_exceeded(pos):
                expired.append(pos)
                continue
            if not recently_closed:
                self._close_pending.pop(pos.symbol, None)
            out.append(pos)
        for pos in expired:
            self._close_pending[pos.symbol] = now
            if not self.close_position(pos.id, 0.0, "time_stop"):
                # Close order failed (transient exchange error): the position is
                # STILL OPEN. Keep it in the book so reconcile_closed does not
                # phantom-close it, clear the pending window so the next loop
                # retries the time-stop. (close_position kept it tracked.)
                out.append(pos)
                self._close_pending.pop(pos.symbol, None)
        return out

    def _duration_exceeded(self, pos: PositionState) -> bool:
        """Has this position exceeded its max trade duration (wall-clock bars)?

        The timeframe/time-stop come from the bot's tracked meta (persisted
        across restarts), falling back to the in-memory plan. Positions the bot
        does not track — the owner's MANUAL trades — have no timeframe and are
        never time-stopped: the bot must not close trades it did not open.
        `opened_at_candle` is a backtest concept and is deliberately ignored.
        """
        st = self._tracked.get(pos.symbol) or {}
        plan = st.get("plan")
        tf = (getattr(plan, "timeframe", "") if plan else "") or st.get("tf") or pos.timeframe
        time_stop = int((getattr(plan, "time_stop_candles", 0) if plan else 0)
                        or st.get("time_stop") or 0)
        if time_stop <= 0:
            time_stop = int(self.cfg.max_trade_duration_candles or 0)
        if time_stop <= 0 or not tf or pos.symbol not in self._tracked:
            return False
        opened = st.get("opened_at") or pos.opened_at
        return self._wall_clock_candles_elapsed(opened, tf) >= time_stop

    def _wall_clock_candles_elapsed(self, opened_at: str, timeframe: str) -> int:
        """Approximate candles elapsed from opened_at ISO timestamp to now."""
        if not opened_at:
            return 0
        try:
            opened = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
            elapsed_sec = (datetime.now(timezone.utc) - opened).total_seconds()
            return int(elapsed_sec / timeframe_to_seconds(timeframe)) if timeframe else 0
        except Exception:
            return 0

    # ------------------------------------------------------------------ #
    # order-size preflight: the exchange only accepts step-grid quantities and
    # has per-symbol minimums; sending raw plan quantities either fails (BTC at
    # 45 USDT notional) or silently trades a different size than the risk math
    # assumed (ccxt TRUNCATEs). Everything downstream must use REAL numbers.
    # ------------------------------------------------------------------ #
    def _preflight_qty(self, symbol: str, plan: TradePlan) -> Optional[float]:
        """Quantize the entry qty to the exchange grid; None = untradeable size."""
        try:
            m = self.ex.market(symbol)
            qty = float(self.ex.amount_to_precision(symbol, plan.qty))
        except Exception as e:  # InvalidOrder: below minimum amount precision
            log.info("LIVE entry skipped: %s qty %.8g below exchange precision "
                     "(%s)", symbol, plan.qty, str(e)[:120])
            audit("live_entry_skipped", {"symbol": symbol,
                                         "reason": "qty below exchange minimum",
                                         "qty": plan.qty})
            return None
        min_qty = float((m.get("limits", {}).get("amount", {}) or {}).get("min") or 0)
        min_notional = float(((m.get("info") or {}).get("lotSizeFilter") or {})
                             .get("minNotionalValue") or 0)
        if qty <= 0 or qty < min_qty or qty * plan.entry < min_notional:
            log.info("LIVE entry skipped: %s qty %.8g below min qty %.8g / "
                     "min notional %.4g", symbol, qty, min_qty, min_notional)
            audit("live_entry_skipped", {"symbol": symbol,
                                         "reason": "below exchange minimums",
                                         "qty": qty, "min_qty": min_qty,
                                         "min_notional": min_notional})
            return None
        tol = float(getattr(self.cfg, "max_qty_truncation_pct", 0.10) or 0.10)
        if plan.qty > 0 and (plan.qty - qty) / plan.qty > tol:
            log.info("LIVE entry skipped: %s step-truncation %.8g -> %.8g exceeds "
                     "%.0f%% tolerance", symbol, plan.qty, qty, tol * 100)
            audit("live_entry_skipped", {"symbol": symbol,
                                         "reason": "qty truncation beyond tolerance",
                                         "plan_qty": plan.qty, "stepped_qty": qty})
            return None
        return qty

    def _build_tp_legs(self, symbol: str, filled_qty: float,
                       take_profits) -> list[tuple[float, float, float]]:
        """(price, r_multiple, qty) TP legs from the ACTUAL filled qty, on the
        exchange step grid, every leg >= min qty, summing exactly to the fill.

        Integer-step arithmetic avoids float dust. Legs too small to exist are
        merged BACKWARD into the previous rung (earlier profit-taking — the
        conservative direction); if nothing survives, one 100% leg at the first
        TP price. Never returns an unplaceable leg.
        """
        tps = list(take_profits or [])
        if filled_qty <= 0 or not tps:
            return []
        m = self.ex.market(symbol)
        step = float(m.get("precision", {}).get("amount") or 0) or 0.0
        min_qty = float((m.get("limits", {}).get("amount", {}) or {}).get("min") or 0)
        if step <= 0:
            return [(tps[0].price, tps[0].r_multiple, filled_qty)]
        total_steps = int(round(filled_qty / step))
        min_steps = max(1, int(math.ceil((min_qty or step) / step)))
        if total_steps < min_steps:
            return []  # cannot even place one reduce-only leg
        # allocate integer steps per rung, remainder to the last rung
        steps = [int(total_steps * tp.allocation) for tp in tps[:-1]]
        steps.append(total_steps - sum(steps))
        # merge sub-minimum rungs backward into the previous rung
        for i in range(len(steps) - 1, 0, -1):
            if 0 < steps[i] < min_steps:
                steps[i - 1] += steps[i]
                steps[i] = 0
        # first rung too small -> push it forward into the next surviving rung
        if 0 < steps[0] < min_steps:
            nxt = next((j for j in range(1, len(steps)) if steps[j] > 0), None)
            if nxt is None:
                return [(tps[0].price, tps[0].r_multiple, total_steps * step)]
            steps[nxt] += steps[0]
            steps[0] = 0
        legs = [(tps[i].price, tps[i].r_multiple, steps[i] * step)
                for i in range(len(steps)) if steps[i] >= min_steps]
        if not legs:
            legs = [(tps[0].price, tps[0].r_multiple, total_steps * step)]
        if len(legs) < len(tps):
            audit("live_tp_ladder_degraded", {
                "symbol": symbol, "planned_rungs": len(tps),
                "placed_rungs": len(legs), "filled_qty": filled_qty})
        return legs

    def _position_qty(self, symbol: str) -> float:
        """Actual open contracts for a symbol (fill-confirmation fallback)."""
        for p in self._fetch_positions_raw():
            if p.symbol == symbol:
                return float(p.qty_open)
        return 0.0

    # ------------------------------------------------------------------ #
    def open_position(self, plan: TradePlan) -> Optional[str]:
        symbol = plan.symbol
        equity = self.equity()
        self._roll_day_anchor(equity)
        ok, why = self.breakers.allow_entry(equity, self.day_anchor["equity"],
                                            self.consecutive_losses)
        if not ok:
            log.warning("LIVE entry blocked: %s", why)
            audit("live_entry_blocked", {"symbol": symbol, "reason": why})
            return None

        # ---- edge safety catch: never deploy real capital against an
        # unvalidated / negative out-of-sample edge (default ON, live-only).
        if getattr(self.cfg, "require_validated_edge", True):
            ok_edge, edge_why = validated_edge(self.breakers.root)
            if not ok_edge:
                log.warning("LIVE entry blocked by edge safety catch: %s. Set "
                            "risk.require_validated_edge: false in config.yaml to "
                            "override deliberately.", edge_why)
                audit("live_entry_blocked", {"symbol": symbol,
                                             "reason": f"edge catch: {edge_why}"})
                return None

        # ---- live trade-selection filters (measured bleed-reducers). Paper and
        # the backtest still take these segments so calibration keeps learning.
        if plan.regime and plan.regime in tuple(getattr(self.cfg, "live_blocked_regimes", ()) or ()):
            log.info("LIVE entry skipped: %s regime '%s' is live-blocked",
                     symbol, plan.regime)
            audit("live_entry_skipped", {"symbol": symbol,
                                         "reason": f"blocked regime {plan.regime}"})
            return None
        kind = getattr(plan, "kind", "") or ""
        if kind and kind in tuple(getattr(self.cfg, "live_blocked_setups", ()) or ()):
            log.info("LIVE entry skipped: %s setup kind '%s' is live-blocked",
                     symbol, kind)
            audit("live_entry_skipped", {"symbol": symbol,
                                         "reason": f"blocked setup {kind}"})
            return None
        # Per-timeframe slot budget (e.g. 15m capped at ~20% of concurrency).
        budget = (getattr(self.cfg, "live_tf_budget", {}) or {}).get(plan.timeframe)
        if budget is not None:
            max_slots = max(1, int(self.cfg.max_concurrent_positions * float(budget)))
            in_tf = sum(1 for st in self._tracked.values()
                        if (st.get("tf") or getattr(st.get("plan"), "timeframe", None))
                        == plan.timeframe)
            if in_tf >= max_slots:
                log.info("LIVE entry skipped: %s tf budget reached (%d/%d %s slots)",
                         symbol, in_tf, max_slots, plan.timeframe)
                audit("live_entry_skipped", {"symbol": symbol,
                                             "reason": f"tf budget {plan.timeframe}",
                                             "slots": max_slots})
                return None
        # Cooldown after a losing close: no immediate re-entry into the same chop.
        now_ts = datetime.now(timezone.utc).timestamp()
        until = self._cooldown.get(symbol, 0.0)
        if until > now_ts:
            log.info("LIVE entry skipped: %s in post-loss cooldown for %.0f more min",
                     symbol, (until - now_ts) / 60)
            return None
        self._cooldown.pop(symbol, None)

        # ---- order-size preflight: quantize to the exchange grid and rewrite
        # the plan's sizing numbers so every later gate uses REAL quantities.
        qty = self._preflight_qty(symbol, plan)
        if qty is None:
            return None
        plan.qty = qty
        plan.notional = qty * plan.entry
        plan.margin = plan.notional / max(plan.leverage, 1.0)
        plan.risk_amount = qty * abs(plan.entry - plan.stop_loss)

        # Positions on the exchange may include MANUAL trades on this account.
        # Never stack onto a symbol that already holds ANY position (manual or
        # bot) — that would double exposure or fight a manual trade.
        all_positions = self.open_positions()
        if any(p.symbol == symbol for p in all_positions):
            log.info("LIVE entry skipped: already holding a %s position", symbol)
            return None
        # Apply the portfolio caps (concurrency / margin% / risk%) to the bot's
        # OWN book only, so foreign manual positions do not consume the bot's
        # allocation. The free-margin preflight below is the account-level guard
        # that keeps new exposure within the actually-available balance.
        own = [p for p in all_positions if p.symbol in self._tracked]
        ok, why = portfolio_allows(self.cfg, own, plan, equity)
        if not ok:
            log.info("LIVE entry skipped: %s", why)
            return None

        # ---- free-margin preflight: only open if enough FREE USDT is available
        # to fund this position's margin ("check for N USDT availability before
        # opening"). Uses free balance, not total equity.
        required_margin = float(plan.margin)
        free = self.free_usdt()
        if free < required_margin:
            log.info("LIVE entry skipped: free balance %.2f USDT < required margin "
                     "%.2f USDT for %s", free, required_margin, symbol)
            audit("live_entry_skipped", {"symbol": symbol,
                                         "reason": "insufficient free margin",
                                         "free": round(free, 2),
                                         "required": round(required_margin, 2)})
            return None

        # ---- stale-candle fail-safe (must have a recently-closed bar)
        try:
            candles = self.ex.fetch_ohlcv(symbol, plan.timeframe, limit=3)
            if candles and len(candles) >= 2:
                last_close_ms = candles[-2][0]
                last_candle = datetime.fromtimestamp(last_close_ms / 1000.0,
                                                     tz=timezone.utc)
                if CircuitBreakers.is_stale(last_candle,
                                            timeframe_to_seconds(plan.timeframe)):
                    log.warning("LIVE entry on %s rejected: candle data is stale",
                                symbol)
                    audit("live_entry_blocked", {"symbol": symbol,
                                                 "reason": "stale candle data"})
                    return None
        except Exception as e:
            log.warning("LIVE entry on %s: could not verify candle freshness (%s) - "
                        "rejecting to fail safe", symbol, e)
            audit("live_entry_blocked", {"symbol": symbol,
                                         "reason": f"candle freshness check failed: {e}"})
            return None

        # ---- microstructure preflight: one ticker feeds the spread guard, the
        # funding tilt, and the IOC limit price.
        buy = plan.side == Side.LONG
        try:
            ticker = self.ex.fetch_ticker(symbol)
            bid = float(ticker.get("bid") or 0)
            ask = float(ticker.get("ask") or 0)
            if bid <= 0 or ask <= 0:
                raise ValueError("no bid/ask")
        except Exception as e:
            log.warning("LIVE entry on %s: no usable ticker (%s) - rejecting to "
                        "fail safe", symbol, e)
            audit("live_entry_blocked", {"symbol": symbol,
                                         "reason": f"ticker unavailable: {e}"})
            return None
        spread_bps = (ask - bid) / ((ask + bid) / 2.0) * 1e4
        max_spread = float(getattr(self.cfg, "max_spread_bps", 25.0) or 0)
        if max_spread > 0 and spread_bps > max_spread:
            log.info("LIVE entry skipped: %s spread %.1fbps > %.1fbps cap",
                     symbol, spread_bps, max_spread)
            audit("live_entry_skipped", {"symbol": symbol, "reason": "wide spread",
                                         "spread_bps": round(spread_bps, 1)})
            return None
        # Funding tilt: do not open a position that immediately PAYS extreme
        # funding (longs pay positive rates; shorts pay negative ones).
        max_funding = float(getattr(self.cfg, "max_entry_funding_rate", 0.0) or 0)
        try:
            frate = float((ticker.get("info") or {}).get("fundingRate") or 0.0)
        except (TypeError, ValueError):
            frate = 0.0
        pays = (buy and frate > 0) or ((not buy) and frate < 0)
        if max_funding > 0 and pays and abs(frate) > max_funding:
            log.info("LIVE entry skipped: %s would pay funding %.4f%% > %.4f%% cap",
                     symbol, abs(frate) * 100, max_funding * 100)
            audit("live_entry_skipped", {"symbol": symbol,
                                         "reason": "pays extreme funding",
                                         "funding_rate": frate})
            return None

        side = "buy" if buy else "sell"
        try:
            # ---- leverage: round UP to integer, then re-check safety ceiling
            lev = int(math.ceil(plan.leverage))
            rm = RiskManager(self.cfg)
            safe_lev = rm._max_safe_leverage(plan.side, plan.entry, plan.stop_loss)
            if lev > safe_lev:
                log.warning("LIVE entry on %s rejected: ceiling leverage %dx exceeds "
                            "safe stop-before-liquidation ceiling %.2fx",
                            symbol, lev, safe_lev)
                audit("live_entry_rejected", {"symbol": symbol,
                                              "reason": "leverage exceeds safe ceiling",
                                              "leverage": lev, "safe_leverage": safe_lev})
                return None
            liq = rm.liquidation_price(plan.side, plan.entry, lev)
            if rm._liq_before_stop(plan.side, liq, plan.stop_loss):
                log.warning("LIVE entry on %s rejected: liquidation %.6g is closer "
                            "than stop %.6g", symbol, liq, plan.stop_loss)
                return None
            try:
                self.ex.set_leverage(lev, symbol, params={"category": "linear"})
            except Exception as e:  # already set / not supported for symbol
                log.debug("set_leverage: %s", e)

            # ---- entry: IOC limit at ref*(1±cap) — a market order with bounded
            # slippage. An unfilled IOC simply expires; we do not chase.
            cap = float(getattr(self.cfg, "entry_slippage_cap_pct", 0.0015) or 0.0015)
            limit_price = ask * (1 + cap) if buy else bid * (1 - cap)
            entry = self.ex.create_order(
                symbol, "limit", side, qty,
                float(self.ex.price_to_precision(symbol, limit_price)),
                params={"category": "linear", "timeInForce": "IOC",
                        "stopLoss": self.ex.price_to_precision(symbol, plan.stop_loss),
                        "slTriggerBy": "LastPrice"})
        except Exception as e:
            log.error("LIVE entry failed for %s: %s", symbol, e)
            audit("live_entry_failed", {"symbol": symbol, "error": str(e)})
            return None

        # ---- fill confirmation: everything downstream (TP ladder, tracking,
        # risk accounting) must be sized from what ACTUALLY filled.
        entry_id = str(entry.get("id") or f"{symbol}@{utcnow_iso()}")
        filled = float(entry.get("filled") or 0.0)
        if filled <= 0:
            try:
                o = self.ex.fetch_order(entry_id, symbol, params={"category": "linear"})
                filled = float(o.get("filled") or 0.0)
            except Exception as e:
                log.warning("could not confirm fill for %s (%s) — checking the "
                            "position book", symbol, e)
                filled = self._position_qty(symbol)
        if filled <= 0:
            log.info("LIVE entry on %s: IOC expired unfilled — skipping", symbol)
            audit("live_entry_unfilled", {"symbol": symbol, "order": entry_id})
            return None
        # Rewrite the plan to the ACTUAL fill so the caller journals the real
        # position size (an IOC can fill partially), not the requested size.
        plan.qty = filled
        plan.notional = filled * plan.entry
        plan.margin = plan.notional / max(plan.leverage, 1.0)
        plan.risk_amount = filled * abs(plan.entry - plan.stop_loss)
        audit("live_open", {"symbol": symbol, "side": side, "qty": filled,
                            "order": entry_id, "testnet": self.testnet})
        log.warning("LIVE OPEN %s %s qty=%.6g (order %s)",
                    side.upper(), symbol, filled, entry_id)

        # Reduce-only TP ladder from the REAL fill; entry is protected by the
        # attached stop either way.
        tp_side = "sell" if buy else "buy"
        legs = self._build_tp_legs(symbol, filled, plan.take_profits)
        placed: list[list] = []
        for price, r_mult, leg_qty in legs:
            try:
                self.ex.create_order(
                    symbol, "limit", tp_side, leg_qty,
                    float(self.ex.price_to_precision(symbol, price)),
                    params={"category": "linear", "reduceOnly": True})
                placed.append([price, r_mult, leg_qty / filled, False])
            except Exception as e:
                log.error("TP order failed (%s @ %s): %s — position remains "
                          "protected by its stop", symbol, price, e)
                audit("live_tp_failed", {"symbol": symbol, "price": price,
                                         "error": str(e)})

        # ---- start partial-TP tracking for breakeven-stop management. Meta
        # (tf/time_stop/opened_at) persists so the time-stop, tf-budget, and
        # close reconciliation survive restarts.
        self._tracked[symbol] = {
            "entry_id": entry_id,
            "plan": plan,
            "tf": plan.timeframe,
            "time_stop": int(plan.time_stop_candles or 0),
            "opened_at": utcnow_iso(),
            "tps": placed,
            "breakeven_moved": False,
        }
        self._save_state()   # persist so a restart still counts this position
        self.sync_take_profits(symbol)
        return entry_id

    def close_position(self, pos_id: str, price: float, reason: str) -> bool:
        """Market reduce-only close of the whole position for `pos_id`'s symbol.
        Returns True only when the close order was accepted (position closing).

        Untracks the symbol ONLY on a confirmed close: if the reduce-only order
        raises (transient exchange error), the position is STILL OPEN, so we keep
        it tracked so the caller can retry the time-stop and reconcile_closed does
        not phantom-close a live position. Uses the RAW position fetch —
        open_positions() enforces the time-stop by calling back here, so calling
        it from here would recurse.
        """
        ok = False
        for p in self._fetch_positions_raw():
            if p.id != pos_id and p.symbol != pos_id:
                continue
            side = "sell" if p.side == Side.LONG else "buy"
            try:
                self.ex.create_order(p.symbol, "market", side, p.qty_open, None,
                                     params={"category": "linear",
                                             "reduceOnly": True})
                audit("live_close", {"symbol": p.symbol, "reason": reason})
                log.warning("LIVE CLOSE %s (%s)", p.symbol, reason)
                self._tracked.pop(p.symbol, None)
                self._save_state()
                ok = True
            except Exception as e:
                # Keep it tracked — the position is still open; retry next loop.
                log.error("LIVE close failed for %s: %s — will retry", p.symbol, e)
                audit("live_close_failed", {"symbol": p.symbol, "error": str(e)})
        return ok

    def sync_take_profits(self, symbol: str) -> None:
        """Poll closed orders, mark filled TP rungs, and move stop to breakeven
        once the first TP rung fills. Safe to call repeatedly.
        """
        state = self._tracked.get(symbol)
        if not state or "plan" not in state:
            # Unknown or restored-after-restart marker (no plan) — nothing to
            # manage here; the position stays protected by its exchange SL/TP.
            return
        plan: TradePlan = state["plan"]
        if not plan or not state.get("tps"):
            # No TP legs were actually placed (ladder fully degraded/failed) —
            # nothing to poll and no breakeven trigger exists.
            return

        try:
            since = int((datetime.now(timezone.utc).timestamp() - 86400) * 1000)
            closed = self.ex.fetch_closed_orders(symbol, since=since,
                                                 params={"category": "linear"})
        except Exception as e:
            log.debug("fetch_closed_orders failed for %s: %s", symbol, e)
            return

        tp_side = "sell" if plan.side == Side.LONG else "buy"
        price_tol = plan.entry * 0.001  # 0.1% tolerance for price matching
        for order in closed:
            if order.get("side") != tp_side:
                continue
            if not order.get("reduceOnly"):
                continue
            avg_price = float(order.get("average") or order.get("price") or 0.0)
            if avg_price <= 0:
                continue
            for tp in state["tps"]:
                if tp[3]:
                    continue
                if abs(avg_price - tp[0]) <= price_tol:
                    tp[3] = True

        # First TP filled -> move stop to breakeven (entry) once.
        if state["tps"][0][3] and not state["breakeven_moved"]:
            self._move_stop_to_breakeven(symbol, plan.entry)
            state["breakeven_moved"] = True

    def _move_stop_to_breakeven(self, symbol: str, price: float) -> None:
        """Update the position's attached stop-loss to the breakeven entry price."""
        try:
            price_str = self.ex.price_to_precision(symbol, price)
            self.ex.private_post_v5_position_trading_stop({
                "category": "linear",
                "symbol": symbol,
                "stopLoss": price_str,
                "positionIdx": 0,
            })
            audit("live_breakeven_stop", {"symbol": symbol, "stop": price})
            log.warning("LIVE breakeven stop set on %s @ %s", symbol, price_str)
        except Exception as e:
            log.error("Failed to move stop to breakeven for %s: %s — "
                      "MANUAL INTERVENTION MAY BE REQUIRED", symbol, e)
            audit("live_breakeven_stop_failed", {"symbol": symbol,
                                                 "stop": price, "error": str(e)})

    def record_trade_result(self, win: bool) -> None:
        """Feed the losing-streak breaker (called by the runner on fills)."""
        self.consecutive_losses = 0 if win else self.consecutive_losses + 1
        self._save_state()

    def _closed_pnl_win(self, symbol: str) -> Optional[bool]:
        """Win/loss of a just-closed position from Bybit's closed-PnL ledger.

        The ledger has one row per closing ORDER, not per position lifecycle —
        the bot's own designed exit (TP1 profit + breakeven remainder ≈ −fees)
        would read as a loss from the LAST row alone. So SUM every closing row
        since the position opened (opened_at is tracked/persisted). Without a
        known open time, fall back to the single latest row. Returns None when
        undeterminable (the streak breaker is left unchanged, not guessed)."""
        try:
            raw = symbol.replace("/", "").replace(":USDT", "")
            params: dict = {"category": "linear", "symbol": raw, "limit": 50}
            opened = (self._tracked.get(symbol) or {}).get("opened_at") or ""
            if opened:
                try:
                    ts = datetime.fromisoformat(str(opened).replace("Z", "+00:00"))
                    params["startTime"] = int(ts.timestamp() * 1000) - 60_000
                except ValueError:
                    params["limit"] = 1
            else:
                params["limit"] = 1
            resp = self.ex.private_get_v5_position_closed_pnl(params)
            rows = ((resp or {}).get("result") or {}).get("list") or []
            if not rows:
                return None
            return sum(float(r.get("closedPnl") or 0.0) for r in rows) > 0.0
        except Exception as e:
            log.warning("could not determine closed PnL for %s (%s) — losing-streak "
                        "breaker not advanced", symbol, e)
            return None

    def reconcile_closed(self, prev_symbols, open_symbols) -> list[dict]:
        """Detect positions that closed since the last loop (exchange SL/TP, the
        time-stop, or a manual close), feed the losing-streak breaker with each
        one's realized win/loss, and start the post-loss re-entry cooldown.
        Returns [{symbol, win, entry_id}] so the caller can journal the closes —
        the exchange closes positions without a local callback, so this is the
        only place the live bot learns a trade ended."""
        results: list[dict] = []
        closed = set(prev_symbols) - set(open_symbols)
        for symbol in closed:
            st = self._tracked.get(symbol) or {}
            win = self._closed_pnl_win(symbol)
            if win is not None:
                self.record_trade_result(win)   # persists state
            if win is False and int(getattr(self.cfg, "cooldown_bars_after_loss", 0) or 0) > 0:
                tf = st.get("tf") or getattr(st.get("plan"), "timeframe", "") or ""
                try:
                    secs = (self.cfg.cooldown_bars_after_loss
                            * timeframe_to_seconds(tf)) if tf else \
                        int(getattr(self.cfg, "cooldown_fallback_minutes", 120)) * 60
                except ValueError:
                    secs = int(getattr(self.cfg, "cooldown_fallback_minutes", 120)) * 60
                self._cooldown[symbol] = (datetime.now(timezone.utc).timestamp()
                                          + max(60, secs))
                log.info("post-loss cooldown set on %s for %.0f min",
                         symbol, max(60, secs) / 60)
            results.append({"symbol": symbol, "win": win,
                            "entry_id": st.get("entry_id") or ""})
            self._tracked.pop(symbol, None)
        if closed:
            self._save_state()   # keep the persisted tracked/cooldown state current
        return results
