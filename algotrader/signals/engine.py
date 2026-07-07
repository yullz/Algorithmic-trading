"""Signal engine: indicators + candlesticks + chart patterns + liquidity +
divergences + HTF context -> a single directional Signal with structure-aware
stop/target references, regime gating, and an optional ML meta-model blend.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..indicators import indicators as ind
from ..models import Bias, Evidence, PatternMatch, Side, Signal, SetupKind
from ..patterns import candlestick, chart_patterns
from ..winrate import blend_win_rate, estimate_win_rate
from .. import regime as regime_mod
from . import confluence

# Optional evidence modules (added incrementally; the engine must keep working
# while they don't exist yet, and pick them up automatically once they do).
try:
    from ..patterns import liquidity as _liquidity
except ImportError:      # pragma: no cover
    _liquidity = None
try:
    from ..patterns import harmonic as _harmonic
except ImportError:      # pragma: no cover
    _harmonic = None
try:
    from ..patterns import wyckoff as _wyckoff
except ImportError:      # pragma: no cover
    _wyckoff = None
try:
    from ..patterns import continuation as _continuation
except ImportError:      # pragma: no cover
    _continuation = None
try:
    from ..indicators import divergence as _divergence
except ImportError:      # pragma: no cover
    _divergence = None
try:
    from ..indicators import volume_profile as _volume_profile
except ImportError:      # pragma: no cover
    _volume_profile = None
try:
    from ..indicators import market_structure as _market_structure
except ImportError:      # pragma: no cover
    _market_structure = None
try:
    from ..indicators import volume_breakout as _volume_breakout
except ImportError:      # pragma: no cover
    _volume_breakout = None
try:
    from ..indicators import cross_asset as _cross_asset
except ImportError:      # pragma: no cover
    _cross_asset = None


class SignalEngine:
    def __init__(self, min_confidence: float = 0.55, min_confluence: int = 3,
                 calibration: Optional[dict] = None, min_families: int = 2,
                 htf_veto: bool = False, regime_gating: bool = True,
                 max_stop_atr_mult: float = 3.0, meta_model=None,
                 calibration_path: Optional[str] = None,
                 staleness_days: int = 30):
        self.min_confidence = min_confidence
        self.min_confluence = min_confluence          # raw agreeing pieces
        self.min_families = max(2, int(min_families))  # independent families
        self.htf_veto = htf_veto
        self.regime_gating = regime_gating
        self.max_stop_atr_mult = max_stop_atr_mult
        self.calibration = calibration or {}
        self.calibration_path = calibration_path
        self.staleness_days = staleness_days
        # Object with predict_for_signal(df, indf, evidence, agg, regime,
        # timeframe) -> (prob, weight, contribs) | None. See algotrader/ml/.
        self.meta_model = meta_model

    # ------------------------------------------------------------------ #
    def gather_evidence(self, df: pd.DataFrame, htf: Optional[pd.DataFrame] = None,
                        indf: Optional[pd.DataFrame] = None,
                        extra_evidence: Optional[list[Evidence]] = None,
                        btc_df: Optional[pd.DataFrame] = None,
                        symbol: Optional[str] = None,
                        ) -> tuple[list[Evidence], list[PatternMatch]]:
        """All evidence + the raw PatternMatches (kept for their geometric
        anchors, which to_evidence() cannot carry)."""
        if indf is None:
            indf = ind.compute_all(df)
        evidence: list[Evidence] = list(ind.read_evidence(indf))

        patterns: list[PatternMatch] = []
        patterns.extend(candlestick.detect(df))
        patterns.extend(chart_patterns.detect(df))
        if _liquidity is not None:
            patterns.extend(_liquidity.detect(df))
        if _harmonic is not None:
            patterns.extend(_harmonic.detect(df))
        if _continuation is not None:
            patterns.extend(_continuation.detect(df))
        if _wyckoff is not None:
            for item in _wyckoff.detect(df):
                if isinstance(item, Evidence):
                    evidence.append(item)
                else:
                    patterns.append(item)
        for pm in patterns:
            evidence.append(pm.to_evidence())

        if _liquidity is not None:
            evidence.extend(_liquidity.detect_sr_retest(df))

        if _divergence is not None:
            evidence.extend(_divergence.read_evidence(indf))
        if _volume_profile is not None:
            evidence.extend(_volume_profile.read_evidence(df))
        if _market_structure is not None:
            evidence.extend(_market_structure.read_evidence(df))
        if _volume_breakout is not None:
            evidence.extend(_volume_breakout.read_evidence(df))
        if _cross_asset is not None and btc_df is not None:
            evidence.append(_cross_asset.btc_regime_evidence(btc_df))
            evidence.extend(_cross_asset.sector_strength_evidence(df, btc_df))
            if symbol and "ETH" in symbol.upper():
                evidence.extend(_cross_asset.eth_btc_spread_evidence(df, btc_df))

        if htf is not None and len(htf) > 200:
            evidence.append(self._htf_context(htf))
        if extra_evidence:
            evidence.extend(extra_evidence)

        # Stamp any evidence that arrived without a candle index with the latest
        # bar so the confluence scorer can apply time-decay weighting.
        latest_idx = len(df) - 1
        for e in evidence:
            if e is not None and e.candle_index is None:
                e.candle_index = latest_idx
        return [e for e in evidence if e is not None], patterns

    def _htf_context(self, htf: pd.DataFrame) -> Optional[Evidence]:
        """Higher-timeframe trend acts as a directional filter / confluence vote."""
        e20 = ind.ema(htf["close"], 20).iloc[-1]
        e50 = ind.ema(htf["close"], 50).iloc[-1]
        price = htf["close"].iloc[-1]
        idx = len(htf) - 1
        if price > e20 > e50:
            return Evidence("htf_uptrend", "structure", Bias.BULLISH, 0.6, 0.57,
                            "higher timeframe uptrend", family="trend",
                            candle_index=idx).clamp()
        if price < e20 < e50:
            return Evidence("htf_downtrend", "structure", Bias.BEARISH, 0.6, 0.57,
                            "higher timeframe downtrend", family="trend",
                            candle_index=idx).clamp()
        return Evidence("htf_range", "structure", Bias.NEUTRAL, 0.2, 0.5,
                        "higher timeframe range", family="trend",
                        candle_index=idx).clamp()

    # ------------------------------------------------------------------ #
    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str,
                 htf: Optional[pd.DataFrame] = None,
                 extra_evidence: Optional[list[Evidence]] = None,
                 btc_df: Optional[pd.DataFrame] = None,
                 ) -> Optional[Signal]:
        if len(df) < 60:
            return None
        indf = ind.compute_all(df)
        evidence, patterns = self.gather_evidence(df, htf, indf=indf,
                                                  extra_evidence=extra_evidence,
                                                  btc_df=btc_df, symbol=symbol)
        calibration_error = self._calibration_error_proxy()
        agg = confluence.score(evidence, current_index=len(df) - 1,
                               calibration_error=calibration_error)
        side: Side = agg["side"]
        if side == Side.FLAT:
            return None
        if len(agg["agreeing"]) < self.min_confluence:
            return None
        # Require independence: uncorrelated families must agree, so a single
        # trend firing several correlated indicators cannot pass alone.
        if agg["n_families"] < self.min_families:
            return None
        if agg["confidence"] < self.min_confidence:
            return None

        # Hard higher-timeframe veto (optional; default is the soft vote that
        # already flowed through confluence scoring).
        if self.htf_veto and self._htf_opposes(evidence, side):
            return None

        # Regime gate: some setup kinds are not tradeable in some regimes.
        regime_label = regime_mod.classify(df, indf)
        kind: SetupKind = agg["kind"]
        if self.regime_gating and not regime_mod.allows_setup(kind, side, regime_label):
            return None

        last = indf.iloc[-1]
        entry_ref = float(last["close"])
        atr_val = float(last["atr"]) if not np.isnan(last["atr"]) else entry_ref * 0.01

        structure_stop, structure_target = self._structure_anchors(
            patterns, agg["agreeing"], side, entry_ref)
        stop_ref = self._resolve_stop(df, side, entry_ref, atr_val, structure_stop)

        base_wr = estimate_win_rate(
            agg["agreeing"], self.calibration,
            regime=regime_label, timeframe=timeframe,
            calibration_path=self.calibration_path,
            staleness_days=self.staleness_days,
        )

        ml_prob, ml_weight, ml_contribs = None, 0.0, []
        if self.meta_model is not None:
            vol_pct = float(last.get("volatility_percentile", 0.0)
                            if not pd.isna(last.get("volatility_percentile"))
                            else 0.0)
            atr_pct = float(last.get("atr_percentile", 0.0)
                            if not pd.isna(last.get("atr_percentile"))
                            else 0.0)
            out = self.meta_model.predict_for_signal(
                evidence=agg["agreeing"], agg=agg, regime=regime_label,
                timeframe=timeframe, rule_win_rate=base_wr,
                stop_pct=abs(entry_ref - stop_ref) / max(entry_ref, 1e-9),
                side_sign=side.sign,
                volatility_percentile=vol_pct,
                atr_percentile=atr_pct,
                entry_time=str(indf.index[-1]))
            if out is not None:
                ml_prob, ml_weight, ml_contribs = out
                base_wr = blend_win_rate(base_wr, ml_prob, ml_weight)

        return Signal(
            symbol=symbol, timeframe=timeframe, side=side, kind=kind,
            entry_ref=entry_ref, stop_ref=stop_ref,
            confidence=agg["confidence"], score=agg["score"],
            evidence=agg["agreeing"], base_win_rate=base_wr,
            note=f"{len(agg['agreeing'])} agreeing factors "
                 f"({agg['n_families']} families), regime={regime_label}",
            structure_stop=structure_stop, structure_target=structure_target,
            regime=regime_label, families=agg["families"],
            ml_prob=ml_prob, ml_weight=ml_weight, ml_contribs=ml_contribs,
        )

    # ------------------------------------------------------------------ #
    def _calibration_error_proxy(self) -> float:
        """Simple proxy for how far the recent rule-based win rate is from
        a neutral 0.5 baseline. Returned in [0, 1]; larger values nudge the
        confluence confidence back toward the base rate."""
        overall = self.calibration.get("_overall")
        if overall is None:
            return 0.0
        return min(abs(float(overall) - 0.5) * 2.0, 1.0)

    @staticmethod
    def _htf_opposes(evidence: list[Evidence], side: Side) -> bool:
        for e in evidence:
            if e.name == "htf_downtrend" and side == Side.LONG:
                return True
            if e.name == "htf_uptrend" and side == Side.SHORT:
                return True
        return False

    @staticmethod
    def _structure_anchors(patterns: list[PatternMatch], agreeing: list[Evidence],
                           side: Side, entry: float,
                           ) -> tuple[Optional[float], Optional[float]]:
        """Stop/target anchors from the strongest agreeing chart pattern.

        A pattern's invalidation level is only usable as a stop if it sits on
        the loss side of entry; its measured-move target only if it sits on the
        profit side. Anything else is geometry from a stale/incompatible match.
        """
        agreeing_names = {e.name for e in agreeing}
        want = Bias.BULLISH if side == Side.LONG else Bias.BEARISH
        best: Optional[PatternMatch] = None
        for pm in patterns:
            if pm.bias != want or pm.name not in agreeing_names:
                continue
            if pm.invalidation_level is None and pm.target_level is None:
                continue
            if best is None or pm.confidence > best.confidence:
                best = pm
        if best is None:
            return None, None

        stop = best.invalidation_level
        if stop is not None:
            if not ((side == Side.LONG and stop < entry) or
                    (side == Side.SHORT and stop > entry)):
                stop = None
        target = best.target_level
        if target is not None:
            if not ((side == Side.LONG and target > entry) or
                    (side == Side.SHORT and target < entry)):
                target = None
        return stop, target

    def _resolve_stop(self, df: pd.DataFrame, side: Side, entry: float,
                      atr_val: float, structure_stop: Optional[float]) -> float:
        """Prefer the pattern's invalidation level, then swing structure, then
        ATR distance — but never let the stop drift beyond max_stop_atr_mult
        ATRs from entry (a deep 20-bar swing must not balloon risk distance)."""
        stop = structure_stop if structure_stop is not None else \
            self._structural_stop(df, side, entry, atr_val)
        max_dist = self.max_stop_atr_mult * atr_val
        if abs(entry - stop) > max_dist:
            stop = entry - max_dist if side == Side.LONG else entry + max_dist
        return stop

    @staticmethod
    def _structural_stop(df: pd.DataFrame, side: Side, entry: float, atr_val: float) -> float:
        """Prefer recent swing structure; fall back to ATR distance."""
        window = 20
        recent = df.iloc[-window:]
        if side == Side.LONG:
            swing = float(recent["low"].min())
            atr_stop = entry - 1.5 * atr_val
            return min(swing, atr_stop) if swing < entry else atr_stop
        else:
            swing = float(recent["high"].max())
            atr_stop = entry + 1.5 * atr_val
            return max(swing, atr_stop) if swing > entry else atr_stop
