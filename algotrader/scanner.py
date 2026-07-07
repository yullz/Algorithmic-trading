"""Portfolio-aware universe scanner.

Fetches the whole universe concurrently (AsyncDataFeed + incremental cache),
generates signals per symbol × timeframe, enriches surviving candidates with
derivatives (funding/OI) and relative-strength evidence, sizes them, then
ranks and prunes at the PORTFOLIO level:

  * rank = EV(R) × confidence × market_bias_factor(side, BTC regime)
  * one signal per symbol (best timeframe wins)
  * correlation guard: a candidate whose 30d hourly returns correlate above
    cfg.risk.correlation_cap with an already-accepted pick is suppressed —
    ten long altcoin positions are one levered BTC bet wearing ten hats.

The scanner only *proposes*; executors decide what actually gets opened.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pandas as pd

from . import regime as regime_mod
from .config import AppConfig
from .models import Side, TradePlan
from .reporting import plan_to_dict
from .risk.manager import RiskManager
from .signals.engine import SignalEngine
from .utils.logging import get_logger, utcnow_iso

log = get_logger("scanner")

_BTC = "BTC/USDT:USDT"


class Scanner:
    def __init__(self, cfg: AppConfig, engine: SignalEngine, risk: RiskManager,
                 feed):
        """`feed` is an AsyncDataFeed (algotrader.data.feed)."""
        self.cfg = cfg
        self.engine = engine
        self.risk = risk
        self.feed = feed
        self._pool = ThreadPoolExecutor(max_workers=4)
        scan_raw = (cfg.raw or {}).get("scanner", {})
        # Only this many top-ranked candidates get funding/OI enrichment...
        self.enrich_top = int(scan_raw.get("enrich_top", 25))
        # ...and only this many survive into the published picks.
        self.top_n = int(scan_raw.get("top_n", 15))

    # ------------------------------------------------------------------ #
    async def universe(self) -> list[str]:
        if self.cfg.universe_mode != "top_volume":
            return list(self.cfg.symbols)
        from .data.universe import get_universe
        return await asyncio.to_thread(
            get_universe, self.cfg.exchange_id, self.cfg.universe_size,
            self.cfg.cache_dir, 3600, list(self.cfg.symbols))

    # ------------------------------------------------------------------ #
    async def scan(self) -> dict:
        t0 = time.monotonic()
        symbols = await self.universe()
        tfs = list(self.cfg.timeframes)
        ctx_tf = self.cfg.context_timeframe
        limit = self.cfg.lookback_candles

        pairs = [(s, tf, limit) for s in symbols for tf in tfs]
        pairs += [(s, ctx_tf, limit) for s in symbols if ctx_tf not in tfs]
        data = await self.feed.fetch_many(pairs)

        fetch_errors = sum(1 for v in data.values() if v is None)
        btc_ctx = data.get((_BTC, ctx_tf))
        if btc_ctx is None:
            btc_ctx = data.get((_BTC, tfs[-1]))
        btc_regime = regime_mod.classify(btc_ctx) if btc_ctx is not None else "range"

        # ---- pass 1: rule engine over every symbol × timeframe (thread pool)
        loop = asyncio.get_running_loop()
        jobs = []
        for s in symbols:
            for tf in tfs:
                df = data.get((s, tf))
                if df is None or len(df) < 60:
                    continue
                htf = data.get((s, ctx_tf))
                btc_df = data.get((_BTC, tf))
                jobs.append(loop.run_in_executor(
                    self._pool, self._generate_one, s, tf, df, htf, btc_df))
        candidates = [c for c in await asyncio.gather(*jobs) if c is not None]

        # ---- preliminary ranking; only the head of the list is worth the
        # extra derivatives API calls (2 per candidate, rate-limited).
        for c in candidates:
            c["rank"] = self._rank_of(c, btc_regime)
        candidates.sort(key=lambda c: -c["rank"])
        n_all = len(candidates)
        candidates = candidates[:self.enrich_top]
        if n_all > len(candidates):
            log.info("enriching top %d of %d candidates (rest dropped by rank)",
                     len(candidates), n_all)

        # ---- pass 2: enrich candidates with funding/OI evidence + re-sizing
        candidates = await self._enrich_with_derivatives(candidates, data)

        # ---- portfolio-level ranking and pruning
        picks, suppressed, accepted_objs = self._rank_and_prune(
            candidates, data, tfs, btc_regime)

        # ---- per-symbol market summary for the dashboard heatmap
        cand_syms = {c["symbol"] for c in candidates}
        pick_syms = {p["symbol"] for p in picks}
        market = []
        bars_24h = {"15m": 96, "1h": 24, "4h": 6, "1d": 1}.get(tfs[0], 24)
        for s in symbols:
            df = data.get((s, tfs[0]))
            if df is None or len(df) < bars_24h + 1:
                continue
            last = float(df["close"].iloc[-1])
            chg = last / float(df["close"].iloc[-(bars_24h + 1)]) - 1.0
            market.append({"symbol": s, "last": last,
                           "chg24h_pct": round(chg * 100, 2),
                           "candidate": s in cand_syms, "picked": s in pick_syms})

        out = {
            "plans": picks,
            # live objects for executors — strip before JSON serialization
            "plan_objects": [c["plan"] for c in accepted_objs],
            "signal_objects": [c["signal"] for c in accepted_objs],
            "market": market,
            "suppressed_by_correlation": suppressed,
            "universe_size": len(symbols),
            "pairs_scanned": len(pairs),
            "fetch_errors": fetch_errors,
            "candidates": len(candidates),
            "btc_regime": btc_regime,
            "scanned_at": utcnow_iso(),
            "duration_sec": round(time.monotonic() - t0, 2),
        }
        log.info("scan: %d symbols, %d candidates -> %d picks (%.1fs, btc=%s)",
                 len(symbols), len(candidates), len(picks), out["duration_sec"],
                 btc_regime)
        return out

    # ------------------------------------------------------------------ #
    def _generate_one(self, symbol: str, tf: str, df: pd.DataFrame,
                      htf: Optional[pd.DataFrame],
                      btc_df: Optional[pd.DataFrame]) -> Optional[dict]:
        try:
            extra = []
            if btc_df is not None and symbol != _BTC:
                try:
                    from .indicators.indicators import relative_strength_evidence
                    extra = relative_strength_evidence(df, btc_df)
                except ImportError:
                    pass
            sig = self.engine.generate(df, symbol, tf, htf=htf,
                                       extra_evidence=extra or None)
            if sig is None:
                return None
            plan = self.risk.build_plan(sig)
            if plan is None or not plan.is_actionable(self.cfg.risk):
                return None
            return {"signal": sig, "plan": plan, "symbol": symbol, "tf": tf}
        except Exception as e:
            log.error("generate failed for %s %s: %s", symbol, tf, e)
            return None

    async def _enrich_with_derivatives(self, candidates: list[dict],
                                       data: dict) -> list[dict]:
        """Funding/OI evidence only for symbols that already have a candidate
        (a couple of extra API calls each — cheap at candidate scale). If the
        enriched signal no longer passes the gates, the candidate dies: crowded
        funding against the trade is real information."""
        try:
            from .data.derivatives import fetch_derivatives_evidence
        except ImportError:
            return candidates
        ex = getattr(self.feed, "exchange", None)
        if ex is None:
            return candidates

        sem = asyncio.Semaphore(4)

        async def deriv_for(c: dict) -> list:
            df = data.get((c["symbol"], c["tf"]))
            try:
                chg = float(df["close"].iloc[-1] / df["close"].iloc[-13] - 1.0) \
                    if df is not None and len(df) > 13 else 0.0
                async with sem:
                    return await fetch_derivatives_evidence(ex, c["symbol"], chg)
            except Exception:
                return []

        all_deriv = await asyncio.gather(*(deriv_for(c) for c in candidates))

        out: list[dict] = []
        for c, deriv in zip(candidates, all_deriv):
            df = data.get((c["symbol"], c["tf"]))
            if not deriv:
                out.append(c)
                continue
            htf = data.get((c["symbol"], self.cfg.context_timeframe))
            btc_df = data.get((_BTC, c["tf"]))
            extra = list(deriv)
            if btc_df is not None and c["symbol"] != _BTC:
                try:
                    from .indicators.indicators import relative_strength_evidence
                    extra += relative_strength_evidence(df, btc_df)
                except ImportError:
                    pass
            sig = self.engine.generate(df, c["symbol"], c["tf"], htf=htf,
                                       extra_evidence=extra)
            if sig is None:
                log.info("%s %s dropped after derivatives evidence", c["symbol"], c["tf"])
                continue
            plan = self.risk.build_plan(sig)
            if plan is None or not plan.is_actionable(self.cfg.risk):
                continue
            out.append({"signal": sig, "plan": plan,
                        "symbol": c["symbol"], "tf": c["tf"]})
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _rank_of(c: dict, btc_regime: str) -> float:
        plan: TradePlan = c["plan"]
        side: Side = plan.side
        return float(plan.expected_value_r * plan.confidence *
                     regime_mod.market_bias_factor(side, btc_regime))

    def _rank_and_prune(self, candidates: list[dict], data: dict,
                        tfs: list[str], btc_regime: str,
                        ) -> tuple[list[dict], list[dict], list[dict]]:
        # one candidate per symbol: best rank across timeframes
        best: dict[str, dict] = {}
        for c in candidates:
            r = self._rank_of(c, btc_regime)   # re-rank: enrichment moved EV
            c["rank"] = r
            if c["symbol"] not in best or r > best[c["symbol"]]["rank"]:
                best[c["symbol"]] = c
        ordered = sorted(best.values(), key=lambda c: -c["rank"])

        ordered = ordered[:self.top_n]

        # correlation guard (30d of the lowest timeframe's closes)
        corr_tf = tfs[0]
        rets: dict[str, pd.Series] = {}
        for c in ordered:
            df = data.get((c["symbol"], corr_tf))
            if df is not None and len(df) > 100:
                rets[c["symbol"]] = df["close"].pct_change().iloc[-500:]

        accepted: list[dict] = []
        suppressed: list[dict] = []
        cap = self.cfg.risk.correlation_cap
        for c in ordered:
            too_similar = None
            for a in accepted:
                if a["plan"].side != c["plan"].side:
                    continue
                ra, rc = rets.get(a["symbol"]), rets.get(c["symbol"])
                if ra is None or rc is None:
                    continue
                joined = pd.concat([ra, rc], axis=1, join="inner").dropna()
                if len(joined) > 50 and abs(joined.corr().iloc[0, 1]) > cap:
                    too_similar = a["symbol"]
                    break
            entry = dict(plan_to_dict(c["plan"]), rank=round(c["rank"], 4))
            if too_similar:
                entry["suppressed_by"] = too_similar
                suppressed.append(entry)
            else:
                accepted.append(c)

        picks = [dict(plan_to_dict(c["plan"]), rank=round(c["rank"], 4))
                 for c in accepted]
        return picks, suppressed, accepted
