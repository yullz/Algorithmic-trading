"""Event-driven backtester.

Walks history bar-by-bar, asks the SignalEngine for a signal using only data
available up to that bar (no lookahead), sizes it with the RiskManager, then
simulates the trade forward to a win (reached +1R first) or loss (hit stop
first). If both the stop and target fall inside the same candle, it assumes the
stop hit first — the conservative, capital-preserving assumption.

Outputs aggregate performance AND per-factor empirical win rates, which are
written to calibration.json so the live engine reports measured, not assumed,
hit rates.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..models import RiskConfig, Side
from ..risk.manager import RiskManager
from ..signals.engine import SignalEngine


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR (price-only)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


@dataclass
class BacktestResult:
    trades: list[dict] = field(default_factory=list)
    factor_win_rate: dict[str, float] = field(default_factory=dict)
    kind_win_rate: dict[str, float] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    equity_curve: list[float] = field(default_factory=list)  # cumulative R

    def to_report(self) -> dict:
        return {
            "summary": self.summary,
            "factor_win_rate": {k: round(v, 4) for k, v in self.factor_win_rate.items()},
            "kind_win_rate": {k: round(v, 4) for k, v in self.kind_win_rate.items()},
            "equity_curve": [round(x, 4) for x in self.equity_curve],
            "n_trades": len(self.trades),
        }

    def calibration_dict(self, min_samples: int = 25,
                         half_life_candles: int = 252) -> dict:
        """Per-factor empirical win rates (only factors with enough samples) plus
        ladder-aware realized payoffs, so the risk engine's EV stays consistent
        with the measured win rate.

        Besides the global per-factor rate, regime- and timeframe-conditioned
        keys are emitted ("factor|regime" and "factor|regime|tf") whenever they
        have enough samples of their own — a factor that only works in trends
        must not be credited in chop. winrate.calibrated_rate() walks the
        fallback chain at read time.

        Each calibrated factor is exported as a dict containing the live
        (Bayesian-shrunk) rate, the raw empirical rate, the recency-weighted
        rate, sample counts, and a Wilson-score 95% confidence interval.
        Aggregate payoffs (_overall, _avg_win_r, _avg_loss_r) are also
        recency-weighted.
        """
        if not self.trades:
            return {
                "_overall": self.summary.get("win_rate", 0.5),
                "_avg_win_r": self.summary.get("avg_win_r", 1.0),
                "_avg_loss_r": abs(self.summary.get("avg_loss_r", 1.0)),
            }

        max_index = max(t["entry_idx"] for t in self.trades)
        decay = math.log(2) / half_life_candles

        def _weight(t: dict) -> float:
            return math.exp(-decay * (max_index - t["entry_idx"]))

        stats: dict[str, dict] = defaultdict(
            lambda: {"raw_wins": 0, "raw_n": 0,
                     "weighted_wins": 0.0, "weighted_n": 0.0}
        )
        for t in self.trades:
            regime, tf = t.get("regime", ""), t.get("tf", "")
            is_win = int(t["win"])
            w = _weight(t)
            for f in t["factors"]:
                keys = [f]
                if regime:
                    keys.append(f"{f}|{regime}")
                    if tf:
                        keys.append(f"{f}|{regime}|{tf}")
                for k in keys:
                    s = stats[k]
                    s["raw_n"] += 1
                    s["raw_wins"] += is_win
                    s["weighted_n"] += w
                    s["weighted_wins"] += w * is_win

        prior = 0.5
        prior_strength = 10.0
        z = 1.96

        calib: dict = {}
        for k, s in stats.items():
            raw_n = s["raw_n"]
            if raw_n < min_samples:
                continue
            raw_rate = s["raw_wins"] / raw_n
            weighted_rate = (
                s["weighted_wins"] / s["weighted_n"]
                if s["weighted_n"] > 0 else raw_rate
            )
            eff_n = s["weighted_n"]
            shrunk_rate = (
                eff_n * weighted_rate + prior_strength * prior
            ) / (eff_n + prior_strength)
            wilson_lower, wilson_upper = _wilson_interval(raw_rate, raw_n, z)
            calib[k] = {
                "rate": round(shrunk_rate, 6),
                "raw": round(raw_rate, 6),
                "weighted": round(weighted_rate, 6),
                "n": raw_n,
                "eff_n": round(eff_n, 2),
                "wilson_lower": round(wilson_lower, 6),
                "wilson_upper": round(wilson_upper, 6),
            }

        # Recency-weighted aggregate payoffs.
        total_weight = 0.0
        weighted_win_sum = 0.0
        win_weight_sum = 0.0
        loss_weight_sum = 0.0
        weighted_win_r = 0.0
        weighted_loss_r = 0.0
        for t in self.trades:
            w = _weight(t)
            total_weight += w
            weighted_win_sum += w * int(t["win"])
            r = t["r"]
            if t["win"]:
                weighted_win_r += w * r
                win_weight_sum += w
            else:
                weighted_loss_r += w * abs(r)
                loss_weight_sum += w

        calib["_overall"] = round(
            weighted_win_sum / total_weight, 4
        ) if total_weight else self.summary.get("win_rate", 0.5)
        calib["_avg_win_r"] = round(
            weighted_win_r / win_weight_sum, 3
        ) if win_weight_sum else self.summary.get("avg_win_r", 1.0)
        calib["_avg_loss_r"] = round(
            weighted_loss_r / loss_weight_sum, 3
        ) if loss_weight_sum else abs(self.summary.get("avg_loss_r", 1.0))
        return calib

    def to_dataset(self) -> "pd.DataFrame":
        """One row per simulated trade for the ML meta-model: sparse factor
        strengths (factor__<name> columns), signal context, and the outcome.
        Rows are time-ordered per symbol; the trainer must split by time."""
        rows = []
        for t in self.trades:
            row = {
                "symbol": t.get("symbol", ""), "tf": t.get("tf", ""),
                "entry_time": t.get("entry_time", ""),
                "side": 1 if t["side"] == "LONG" else -1,
                "kind": t.get("kind", ""), "regime": t.get("regime", ""),
                "confidence": t.get("confidence", 0.0),
                "score": t.get("score", 0.0),
                "n_families": t.get("n_families", 0),
                "n_factors": len(t.get("factors", [])),
                "rule_win_rate": t.get("rule_win_rate", 0.5),
                "stop_pct": t.get("stop_pct", 0.0),
                "r": t["r"], "win": int(t["win"]),
            }
            for name, s in (t.get("factor_strengths") or {}).items():
                row[f"factor__{name}"] = s
            rows.append(row)
        return pd.DataFrame(rows).fillna(0.0)

    def write_calibration(self, path: str, min_samples: int = 25,
                          half_life_candles: int = 252) -> dict:
        calib = self.calibration_dict(min_samples, half_life_candles)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2)
        return calib


def _wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson-score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(
        p * (1 - p) / n + z * z / (4 * n * n)
    ) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


class Backtester:
    def __init__(self, engine: SignalEngine, risk: RiskManager,
                 horizon: int = 48, warmup: int = 210):
        self.engine = engine
        self.risk = risk
        self.horizon = horizon
        self.warmup = warmup

    def run(self, df: pd.DataFrame, symbol: str, timeframe: str,
            htf_full: pd.DataFrame | None = None,
            start: int | None = None, end: int | None = None) -> BacktestResult:
        """Walk history with no lookahead. If htf_full (a higher-timeframe frame)
        is provided, the same HTF context used live is sliced to each bar so HTF
        factors get calibrated identically to production. `start`/`end` restrict
        the bar range in which NEW signals are generated (used by walk-forward)."""
        n = len(df)
        trades: list[dict] = []
        # HTF bar duration, used to exclude the still-forming higher-timeframe bar
        # (a resampled bar aggregates its whole window, incl. future sub-bars, so
        # using the in-progress one would leak lookahead).
        htf_period = None
        if htf_full is not None and len(htf_full) > 1:
            htf_period = htf_full.index.to_series().diff().median()
        # ATR for adaptive volatility-target sizing
        atr = _atr_series(df, period=14)
        median_atr = atr.median()
        i = max(self.warmup, start if start is not None else self.warmup)
        stop_at = min(n - 1, end if end is not None else n - 1)
        while i < stop_at:
            sl = df.iloc[: i + 1]
            htf = None
            if htf_full is not None and htf_period is not None:
                now = sl.index[-1]
                htf = htf_full[htf_full.index + htf_period <= now]  # closed bars only
            sig = self.engine.generate(sl, symbol, timeframe, htf=htf)
            if sig is None:
                i += 1
                continue
            current_atr = float(atr.iloc[i]) if i < len(atr) and pd.notna(atr.iloc[i]) else None
            plan = self.risk.build_plan(
                sig,
                current_atr=current_atr,
                median_atr_lookback=float(median_atr) if pd.notna(median_atr) else None,
                opened_at_candle=i,
            )
            if plan is None or not plan.is_actionable(self.risk.cfg):
                i += 1
                continue
            outcome, exit_i, r = self._simulate(df, i, plan)
            trades.append({
                "entry_idx": i, "exit_idx": exit_i, "side": plan.side.value,
                "entry": plan.entry, "stop": plan.stop_loss, "r": r,
                "win": outcome == "win", "factors": [e.name for e in sig.evidence],
                "kind": sig.kind.value,
                # context for regime-conditioned calibration + the ML dataset
                "symbol": symbol, "tf": timeframe,
                "regime": getattr(sig, "regime", "") or "",
                "confidence": sig.confidence, "score": sig.score,
                "n_families": len(getattr(sig, "families", []) or []),
                "families": list(getattr(sig, "families", []) or []),
                "factor_strengths": {e.name: e.strength for e in sig.evidence},
                "rule_win_rate": sig.base_win_rate,
                "stop_pct": abs(plan.entry - plan.stop_loss) / plan.entry,
                "entry_time": str(sl.index[-1]),
            })
            i = max(exit_i, i + 1)  # no overlapping trades
        return self._aggregate(trades)

    def _simulate(self, df: pd.DataFrame, entry_i: int, plan) -> tuple[str, int, float]:
        """Simulate the FULL take-profit ladder with partial exits and a
        move-to-breakeven-after-TP1 stop, returning the net realized R after
        round-trip taker fees and slippage. A win is realized_R > 0 — so the
        win rate and the realized payoff come from the same model and cannot
        be inconsistent."""
        n = len(df)
        sgn = plan.side.sign
        entry, stop = plan.entry, plan.stop_loss
        stop_dist = abs(entry - stop)
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        end = min(entry_i + 1 + self.horizon, n)

        # Round-trip execution costs in R units so simulation is net-of-fee.
        cfg = self.risk.cfg
        fee_rate = cfg.taker_fee * 2 + cfg.slippage_pct * 2
        fees_r = fee_rate * entry / stop_dist if stop_dist > 0 else 0.0

        remaining = 1.0
        realized_r = 0.0
        cur_stop = stop
        hit = [False] * len(plan.take_profits)

        time_stop = plan.time_stop_candles
        for j in range(entry_i + 1, end):
            # Time stop: close at market if the trade exceeded its max duration.
            if time_stop > 0 and (j - entry_i) >= time_stop:
                realized_r += remaining * (sgn * (close[j] - entry) / stop_dist - fees_r)
                return ("win" if realized_r > 0 else "loss"), j, float(realized_r)
            # Stop is checked first on ambiguous bars (conservative).
            stopped = (low[j] <= cur_stop) if plan.side == Side.LONG else (high[j] >= cur_stop)
            if stopped:
                r_at_stop = sgn * (cur_stop - entry) / stop_dist  # ~0 if breakeven
                realized_r += remaining * (r_at_stop - fees_r)
                return ("win" if realized_r > 0 else "loss"), j, float(realized_r)
            for k, tp in enumerate(plan.take_profits):
                reached = (high[j] >= tp.price) if plan.side == Side.LONG else (low[j] <= tp.price)
                if not hit[k] and reached:
                    hit[k] = True
                    realized_r += tp.allocation * (tp.r_multiple - fees_r)
                    remaining -= tp.allocation
                    if k == 0:
                        cur_stop = entry  # lock in: move stop to breakeven
            if remaining <= 1e-9:
                return ("win" if realized_r > 0 else "loss"), j, float(realized_r)

        # timeout: close the remainder at the horizon close
        j = end - 1
        realized_r += remaining * (sgn * (close[j] - entry) / stop_dist - fees_r)
        return ("win" if realized_r > 0 else "loss"), j, float(realized_r)

    def _aggregate(self, trades: list[dict]) -> BacktestResult:
        res = BacktestResult(trades=trades)
        if not trades:
            res.summary = {"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0}
            return res

        rs = np.array([t["r"] for t in trades])
        wins = rs[rs > 0]
        losses = rs[rs <= 0]
        win_rate = float((rs > 0).mean())
        expectancy = float(rs.mean())
        if losses.sum() != 0:
            profit_factor = float(wins.sum() / abs(losses.sum()))
        else:
            # No losing trades in the sample: profit factor is mathematically
            # infinite. Cap to a large finite sentinel so reports stay valid JSON
            # (json.dump emits `Infinity`, which browsers' JSON.parse rejects)
            # and downstream numeric comparisons keep working.
            profit_factor = 999.99 if wins.sum() > 0 else 0.0

        # equity curve in R for drawdown
        eq = np.cumsum(rs)
        peak = np.maximum.accumulate(eq)
        max_dd = float((peak - eq).max()) if len(eq) else 0.0
        res.equity_curve = [0.0] + [float(x) for x in eq]

        # per-factor
        f_wins, f_tot = defaultdict(int), defaultdict(int)
        k_wins, k_tot = defaultdict(int), defaultdict(int)
        for t in trades:
            for f in t["factors"]:
                f_tot[f] += 1
                f_wins[f] += int(t["win"])
            k_tot[t["kind"]] += 1
            k_wins[t["kind"]] += int(t["win"])
        res.factor_win_rate = {f: f_wins[f] / f_tot[f] for f in f_tot}
        res.kind_win_rate = {k: k_wins[k] / k_tot[k] for k in k_tot}
        res.summary = {
            "trades": len(trades),
            "win_rate": round(win_rate, 4),
            "expectancy_r": round(expectancy, 4),
            "profit_factor": round(profit_factor, 3),
            "max_drawdown_r": round(max_dd, 3),
            "avg_win_r": round(float(wins.mean()) if len(wins) else 0.0, 3),
            "avg_loss_r": round(float(losses.mean()) if len(losses) else 0.0, 3),
        }
        return res


def walk_forward(df: pd.DataFrame, symbol: str, timeframe: str, cfg,
                 htf_full: pd.DataFrame | None = None, folds: int = 4,
                 train_frac: float = 0.4, warmup: int = 210, horizon: int = 48,
                 min_samples: int = 6) -> dict | None:
    """Anchored (expanding-window) walk-forward. The first `train_frac` of usable
    history is training-only; the remainder is split into `folds` out-of-sample
    segments. Each segment is traded with calibration learned ONLY from the bars
    before it — the honest test of whether the edge generalizes. Returns the
    aggregated OOS result plus per-fold stats.
    """
    n = len(df)
    usable_end = n - 1
    if usable_end - warmup < max(folds * 30, 120):
        return None  # not enough data to split meaningfully
    train_end0 = warmup + int(train_frac * (usable_end - warmup))
    seg = max(1, (usable_end - train_end0) // folds)
    oos_trades: list[dict] = []
    fold_rows: list[dict] = []
    for k in range(folds):
        a = train_end0 + k * seg
        b = usable_end if k == folds - 1 else train_end0 + (k + 1) * seg
        if a >= b:
            continue
        # Calibrate on everything BEFORE the test window (expanding, uncalibrated engine).
        def _engine(calibration: dict) -> SignalEngine:
            return SignalEngine(cfg.min_confidence, cfg.min_confluence, calibration,
                                min_families=getattr(cfg, "min_families", 2),
                                htf_veto=getattr(cfg, "htf_veto", False),
                                regime_gating=getattr(cfg, "regime_gating", True),
                                max_stop_atr_mult=cfg.risk.max_stop_atr_mult)

        train = Backtester(_engine({}), RiskManager(cfg.risk, {}), horizon, warmup).run(
            df, symbol, timeframe, htf_full, start=warmup, end=a)
        calib = train.calibration_dict(min_samples)
        # Trade the OOS window with that calibration only.
        test = Backtester(_engine(calib), RiskManager(cfg.risk, calib), horizon, warmup).run(
            df, symbol, timeframe, htf_full, start=a, end=b)
        oos_trades.extend(test.trades)
        fold_rows.append({
            "fold": k + 1, "train_bars": a - warmup, "test_bars": b - a,
            "trades": test.summary.get("trades", 0),
            "win_rate": test.summary.get("win_rate", 0.0),
            "expectancy_r": test.summary.get("expectancy_r", 0.0),
        })
    agg = Backtester(SignalEngine(), RiskManager(cfg.risk))._aggregate(oos_trades)
    return {"oos": agg, "folds": fold_rows}
