"""Confluence scoring: fold many Evidence readings into one directional view.

Crucially, evidence is grouped into *independent families* (trend, momentum,
mean-reversion, volume, volatility, candlestick-reversal, chart-structure,
divergence, derivatives, relative-strength...). Six correlated trend indicators
firing together are ONE piece of information, not six — so the confluence count
and confidence are driven by how many independent families agree, not by the
raw number of Evidence objects. This prevents a single trend from masquerading
as high-confluence confirmation.

NEUTRAL evidence (doji, htf_range, squeeze) is not ignored: it dilutes
directionality, so a conflicted tape lowers confidence instead of being
invisible.

Emitters MUST set Evidence.family explicitly for any new evidence name, using
one of CANONICAL_FAMILIES. The setup-kind registry below covers known legacy
and incoming pattern names; anything unknown lands in the single shared "other"
bucket (which can never inflate the independent-family count by more than one).
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

from ..models import Bias, Evidence, Side, SetupKind

# The canonical correlation families. Keep this list short and meaningful:
# a family should represent one *source of information*, not one indicator.
CANONICAL_FAMILIES = (
    "trend",             # EMA stacks, ADX direction, Supertrend, Ichimoku, PSAR, HTF trend
    "momentum",          # MACD, ROC, momentum candles, flags/pennant thrust
    "mean_reversion",    # RSI/Stoch/StochRSI/CCI/W%R extremes, band breaks
    "volume",            # volume spikes, OBV/CMF/MFI flow
    "volatility",        # BB/Keltner squeeze, expansion, ATR percentile
    "candle_reversal",   # single/multi-bar reversal candles
    "chart",             # classical chart patterns (H&S, triangles, wedges...)
    "structure",         # S/R levels, trendline breaks, volume-profile levels
    "liquidity",         # SFP/sweeps, order blocks, FVG
    "divergence",        # RSI/MACD/OBV divergences
    "derivatives",       # funding rate, open interest
    "relative_strength", # performance vs BTC / market
)

# Legacy names -> family (new evidence must set Evidence.family instead).
_FAMILY = {
    # trend
    "ema_stack_bull": "trend", "ema_stack_bear": "trend",
    "adx_trend_bull": "trend", "adx_trend_bear": "trend",
    "htf_uptrend": "trend", "htf_downtrend": "trend", "htf_range": "trend",
    # momentum
    "macd_bull_cross": "momentum", "macd_bear_cross": "momentum",
    "marubozu": "momentum",
    "three_white_soldiers": "momentum", "three_black_crows": "momentum",
    # mean reversion / oscillators
    "rsi_oversold": "mean_reversion", "rsi_overbought": "mean_reversion",
    "bb_lower_break": "mean_reversion", "bb_upper_break": "mean_reversion",
    "stoch_bull": "mean_reversion", "stoch_bear": "mean_reversion",
    # volume
    "volume_spike": "volume",
    # candlestick reversals
    "hammer": "candle_reversal", "shooting_star": "candle_reversal",
    "doji": "candle_reversal",
    "bullish_engulfing": "candle_reversal", "bearish_engulfing": "candle_reversal",
    "piercing_line": "candle_reversal", "dark_cloud_cover": "candle_reversal",
    "tweezer_top": "candle_reversal", "tweezer_bottom": "candle_reversal",
    "morning_star": "candle_reversal", "evening_star": "candle_reversal",
    # chart structure
    "double_top": "chart", "double_bottom": "chart",
    "head_and_shoulders": "chart", "inverse_head_and_shoulders": "chart",
    "ascending_triangle": "chart", "descending_triangle": "chart",
    "symmetrical_triangle_break_up": "chart", "symmetrical_triangle_break_down": "chart",
    "range_breakout_up": "chart", "range_breakout_down": "chart",
    # harmonic
    "bullish_gartley": "chart", "bearish_gartley": "chart",
    "bullish_bat": "chart", "bearish_bat": "chart",
    "bullish_butterfly": "chart", "bearish_butterfly": "chart",
    # continuation
    "bull_pennant": "chart", "bear_pennant": "chart",
    "rising_channel_break_up": "chart", "rising_channel_bounce": "chart",
    "falling_channel_break_down": "chart", "falling_channel_bounce": "chart",
    # wyckoff
    "wyckoff_spring": "structure", "wyckoff_upthrust": "structure",
    "sign_of_strength": "structure", "sign_of_weakness": "structure",
    # liquidity retest
    "sr_break_retest_bull": "liquidity", "sr_break_retest_bear": "liquidity",
}

# --------------------------------------------------------------------------- #
# Setup-kind registry
# --------------------------------------------------------------------------- #
# Each entry is ((source_matcher, name_matcher), kind, name_is_substring).
# * source_matcher: source must start with this ("" matches any source).
# * name_matcher: exact name when name_is_substring=False, substring otherwise.
# Exact entries are checked before substring entries; heuristic substring
# matching is only used as a fallback for unregistered names.
SETUP_KIND_REGISTRY: list[tuple[tuple[str, str], SetupKind, bool]] = [
    # ---------------- indicator / exact ----------------
    (("indicator", "ema_stack_bull"), SetupKind.MOMENTUM, False),
    (("indicator", "ema_stack_bear"), SetupKind.MOMENTUM, False),
    (("indicator", "adx_trend_bull"), SetupKind.MOMENTUM, False),
    (("indicator", "adx_trend_bear"), SetupKind.MOMENTUM, False),
    (("indicator", "macd_bull_cross"), SetupKind.MOMENTUM, False),
    (("indicator", "macd_bear_cross"), SetupKind.MOMENTUM, False),
    (("indicator", "rsi_oversold"), SetupKind.MEAN_REVERSION, False),
    (("indicator", "rsi_overbought"), SetupKind.MEAN_REVERSION, False),
    (("indicator", "bb_lower_break"), SetupKind.MEAN_REVERSION, False),
    (("indicator", "bb_upper_break"), SetupKind.MEAN_REVERSION, False),
    (("indicator", "stoch_bull"), SetupKind.MEAN_REVERSION, False),
    (("indicator", "stoch_bear"), SetupKind.MEAN_REVERSION, False),
    (("indicator", "volume_spike"), SetupKind.BREAKOUT, False),
    # Phase-2b indicator additions
    (("indicator", "ichimoku_bull"), SetupKind.MOMENTUM, False),
    (("indicator", "ichimoku_bear"), SetupKind.MOMENTUM, False),
    (("indicator", "ichimoku_tk_cross_bull"), SetupKind.MOMENTUM, False),
    (("indicator", "ichimoku_tk_cross_bear"), SetupKind.MOMENTUM, False),
    (("indicator", "supertrend_bull"), SetupKind.MOMENTUM, False),
    (("indicator", "supertrend_bear"), SetupKind.MOMENTUM, False),
    (("indicator", "squeeze_breakout_up"), SetupKind.BREAKOUT, False),
    (("indicator", "squeeze_breakout_down"), SetupKind.BREAKOUT, False),
    (("indicator", "squeeze_on"), SetupKind.MEAN_REVERSION, False),
    (("indicator", "stochrsi_oversold_turn"), SetupKind.MEAN_REVERSION, False),
    (("indicator", "stochrsi_overbought_turn"), SetupKind.MEAN_REVERSION, False),
    (("indicator", "mfi_oversold"), SetupKind.MEAN_REVERSION, False),
    (("indicator", "mfi_overbought"), SetupKind.MEAN_REVERSION, False),
    (("indicator", "cci_extreme_low"), SetupKind.MEAN_REVERSION, False),
    (("indicator", "cci_extreme_high"), SetupKind.MEAN_REVERSION, False),
    (("indicator", "cmf_bull"), SetupKind.MOMENTUM, False),
    (("indicator", "cmf_bear"), SetupKind.MOMENTUM, False),
    (("indicator", "psar_flip_bull"), SetupKind.MOMENTUM, False),
    (("indicator", "psar_flip_bear"), SetupKind.MOMENTUM, False),
    (("indicator", "ribbon_aligned_bull"), SetupKind.MOMENTUM, False),
    (("indicator", "ribbon_aligned_bear"), SetupKind.MOMENTUM, False),
    (("indicator", "obv_trend_bull"), SetupKind.MOMENTUM, False),
    (("indicator", "obv_trend_bear"), SetupKind.MOMENTUM, False),
    (("indicator", "vwap_reclaim_bull"), SetupKind.MEAN_REVERSION, False),
    (("indicator", "vwap_reject_bear"), SetupKind.MEAN_REVERSION, False),
    (("indicator", "rs_outperform_btc"), SetupKind.MOMENTUM, False),
    (("indicator", "rs_underperform_btc"), SetupKind.MOMENTUM, False),

    # ---------------- structure / HTF ----------------
    (("structure", "htf_uptrend"), SetupKind.MOMENTUM, False),
    (("structure", "htf_downtrend"), SetupKind.MOMENTUM, False),
    (("structure", "htf_range"), SetupKind.MEAN_REVERSION, False),
    (("structure", "price_at_poc_support"), SetupKind.REVERSAL, False),
    (("structure", "price_at_poc_resist"), SetupKind.REVERSAL, False),
    (("structure", "price_above_value"), SetupKind.MEAN_REVERSION, False),
    (("structure", "price_below_value"), SetupKind.MEAN_REVERSION, False),

    # ---------------- derivatives ----------------
    (("derivatives", "funding_extreme_long"), SetupKind.REVERSAL, False),
    (("derivatives", "funding_extreme_short"), SetupKind.REVERSAL, False),
    (("derivatives", "oi_rising_with_price"), SetupKind.MOMENTUM, False),
    (("derivatives", "oi_rising_against_price"), SetupKind.MOMENTUM, False),

    # ---------------- candlestick / exact ----------------
    (("candlestick", "doji"), SetupKind.REVERSAL, False),
    (("candlestick", "hammer"), SetupKind.REVERSAL, False),
    (("candlestick", "shooting_star"), SetupKind.REVERSAL, False),
    (("candlestick", "bullish_engulfing"), SetupKind.REVERSAL, False),
    (("candlestick", "bearish_engulfing"), SetupKind.REVERSAL, False),
    (("candlestick", "bullish_harami"), SetupKind.REVERSAL, False),
    (("candlestick", "bearish_harami"), SetupKind.REVERSAL, False),
    (("candlestick", "bullish_harami_cross"), SetupKind.REVERSAL, False),
    (("candlestick", "bearish_harami_cross"), SetupKind.REVERSAL, False),
    (("candlestick", "outside_bar_bull"), SetupKind.REVERSAL, False),
    (("candlestick", "outside_bar_bear"), SetupKind.REVERSAL, False),
    (("candlestick", "piercing_line"), SetupKind.REVERSAL, False),
    (("candlestick", "dark_cloud_cover"), SetupKind.REVERSAL, False),
    (("candlestick", "tweezer_top"), SetupKind.REVERSAL, False),
    (("candlestick", "tweezer_bottom"), SetupKind.REVERSAL, False),
    (("candlestick", "morning_star"), SetupKind.REVERSAL, False),
    (("candlestick", "evening_star"), SetupKind.REVERSAL, False),

    # ---------------- chart / exact ----------------
    (("chart", "double_top"), SetupKind.REVERSAL, False),
    (("chart", "double_bottom"), SetupKind.REVERSAL, False),
    (("chart", "triple_top"), SetupKind.REVERSAL, False),
    (("chart", "triple_bottom"), SetupKind.REVERSAL, False),
    (("chart", "head_and_shoulders"), SetupKind.REVERSAL, False),
    (("chart", "inverse_head_and_shoulders"), SetupKind.REVERSAL, False),
    (("chart", "ascending_triangle"), SetupKind.BREAKOUT, False),
    (("chart", "descending_triangle"), SetupKind.BREAKOUT, False),
    (("chart", "symmetrical_triangle_break_up"), SetupKind.BREAKOUT, False),
    (("chart", "symmetrical_triangle_break_down"), SetupKind.BREAKOUT, False),
    (("chart", "rising_wedge"), SetupKind.REVERSAL, False),
    (("chart", "falling_wedge"), SetupKind.REVERSAL, False),
    (("chart", "bull_flag"), SetupKind.CONTINUATION, False),
    (("chart", "bear_flag"), SetupKind.CONTINUATION, False),
    (("chart", "cup_and_handle"), SetupKind.CONTINUATION, False),
    (("chart", "rounding_bottom"), SetupKind.REVERSAL, False),
    (("chart", "sr_bounce_long"), SetupKind.REVERSAL, False),
    (("chart", "sr_bounce_short"), SetupKind.REVERSAL, False),
    (("chart", "sr_break_up"), SetupKind.BREAKOUT, False),
    (("chart", "sr_break_down"), SetupKind.BREAKOUT, False),
    (("chart", "trendline_break_up"), SetupKind.BREAKOUT, False),
    (("chart", "trendline_break_down"), SetupKind.BREAKOUT, False),
    (("chart", "range_breakout_up"), SetupKind.BREAKOUT, False),
    (("chart", "range_breakout_down"), SetupKind.BREAKOUT, False),
    (("chart", "marubozu"), SetupKind.MOMENTUM, False),
    (("chart", "inside_bar"), SetupKind.CONTINUATION, False),
    (("chart", "three_white_soldiers"), SetupKind.CONTINUATION, False),
    (("chart", "three_black_crows"), SetupKind.CONTINUATION, False),
    (("chart", "rising_three_methods"), SetupKind.CONTINUATION, False),
    (("chart", "falling_three_methods"), SetupKind.CONTINUATION, False),

    # ---------------- liquidity ----------------
    (("chart", "sfp_long"), SetupKind.REVERSAL, False),
    (("chart", "sfp_short"), SetupKind.REVERSAL, False),
    (("chart", "bullish_ob_retest"), SetupKind.CONTINUATION, False),
    (("chart", "bearish_ob_retest"), SetupKind.CONTINUATION, False),
    (("chart", "fvg_fill_bull"), SetupKind.CONTINUATION, False),
    (("chart", "fvg_fill_bear"), SetupKind.CONTINUATION, False),
    (("liquidity", "sr_break_retest_bull"), SetupKind.CONTINUATION, False),
    (("liquidity", "sr_break_retest_bear"), SetupKind.CONTINUATION, False),

    # ---------------- harmonic patterns ----------------
    (("chart", "bullish_gartley"), SetupKind.REVERSAL, False),
    (("chart", "bearish_gartley"), SetupKind.REVERSAL, False),
    (("chart", "bullish_bat"), SetupKind.REVERSAL, False),
    (("chart", "bearish_bat"), SetupKind.REVERSAL, False),
    (("chart", "bullish_butterfly"), SetupKind.REVERSAL, False),
    (("chart", "bearish_butterfly"), SetupKind.REVERSAL, False),

    # ---------------- continuation patterns ----------------
    (("chart", "bull_pennant"), SetupKind.CONTINUATION, False),
    (("chart", "bear_pennant"), SetupKind.CONTINUATION, False),
    (("chart", "rising_channel_break_up"), SetupKind.BREAKOUT, False),
    (("chart", "rising_channel_bounce"), SetupKind.CONTINUATION, False),
    (("chart", "falling_channel_break_down"), SetupKind.BREAKOUT, False),
    (("chart", "falling_channel_bounce"), SetupKind.CONTINUATION, False),

    # ---------------- wyckoff ----------------
    (("chart", "wyckoff_spring"), SetupKind.REVERSAL, False),
    (("chart", "wyckoff_upthrust"), SetupKind.REVERSAL, False),
    (("structure", "sign_of_strength"), SetupKind.MOMENTUM, False),
    (("structure", "sign_of_weakness"), SetupKind.MOMENTUM, False),

    # ---------------- incoming subagent modules (substring rules) ----------------
    # market structure
    (("", "market_structure"), SetupKind.BREAKOUT, True),
    (("", "swing_"), SetupKind.REVERSAL, True),
    (("", "higher_"), SetupKind.CONTINUATION, True),
    (("", "lower_"), SetupKind.CONTINUATION, True),
    (("", "msb_"), SetupKind.BREAKOUT, True),
    (("", "bos_"), SetupKind.BREAKOUT, True),
    (("", "choch_"), SetupKind.BREAKOUT, True),
    # volume breakout
    (("", "volume_breakout"), SetupKind.BREAKOUT, True),
    (("", "volume_confirmed_breakout"), SetupKind.BREAKOUT, True),
    (("", "vpin_breakout"), SetupKind.BREAKOUT, True),
    (("", "obv_breakout"), SetupKind.BREAKOUT, True),
    # cross-asset / relative strength
    (("", "btc_regime_"), SetupKind.MOMENTUM, True),
    (("", "eth_btc_"), SetupKind.MOMENTUM, True),
    (("", "sector_"), SetupKind.MOMENTUM, True),
    # harmonic
    (("", "harmonic_"), SetupKind.REVERSAL, True),
    (("", "gartley_"), SetupKind.REVERSAL, True),
    (("", "bat_"), SetupKind.REVERSAL, True),
    (("", "butterfly_"), SetupKind.REVERSAL, True),
    (("", "crab_"), SetupKind.REVERSAL, True),
    (("", "shark_"), SetupKind.REVERSAL, True),
    # wyckoff
    (("", "wyckoff_"), SetupKind.REVERSAL, True),
    (("", "wyckoff_spring"), SetupKind.REVERSAL, True),
    (("", "wyckoff_upthrust"), SetupKind.REVERSAL, True),
    # continuation
    (("", "continuation_"), SetupKind.CONTINUATION, True),
    (("", "flag_continuation"), SetupKind.CONTINUATION, True),
    (("", "pennant_continuation"), SetupKind.CONTINUATION, True),

    # ---------------- source-agnostic substring fallbacks ----------------
    # Indicators / overlays
    (("", "ema_stack_"), SetupKind.MOMENTUM, True),
    (("", "adx_trend_"), SetupKind.MOMENTUM, True),
    (("", "supertrend_"), SetupKind.MOMENTUM, True),
    (("", "macd_"), SetupKind.MOMENTUM, True),
    (("", "volume_spike"), SetupKind.BREAKOUT, True),
    (("", "ichimoku_"), SetupKind.MOMENTUM, True),
    (("", "psar_"), SetupKind.MOMENTUM, True),
    (("", "ribbon_"), SetupKind.MOMENTUM, True),
    (("", "obv_"), SetupKind.MOMENTUM, True),
    (("", "cmf_"), SetupKind.MOMENTUM, True),
    (("", "cci_"), SetupKind.MEAN_REVERSION, True),
    (("", "stochrsi_"), SetupKind.MEAN_REVERSION, True),
    (("", "mfi_"), SetupKind.MEAN_REVERSION, True),
    (("", "vwap_"), SetupKind.MEAN_REVERSION, True),
    (("", "squeeze_"), SetupKind.BREAKOUT, True),
    (("", "rs_"), SetupKind.MOMENTUM, True),
    (("", "htf_"), SetupKind.MOMENTUM, True),
    # Structure / volume profile
    (("", "price_at_poc_"), SetupKind.REVERSAL, True),
    (("", "price_above_value"), SetupKind.MEAN_REVERSION, True),
    (("", "price_below_value"), SetupKind.MEAN_REVERSION, True),
    # Derivatives
    (("", "funding_"), SetupKind.REVERSAL, True),
    (("", "oi_"), SetupKind.MOMENTUM, True),
    # Chart patterns
    (("", "double_"), SetupKind.REVERSAL, True),
    (("", "triple_"), SetupKind.REVERSAL, True),
    (("", "head_and_shoulders"), SetupKind.REVERSAL, True),
    (("", "wedge"), SetupKind.REVERSAL, True),
    (("", "flag"), SetupKind.CONTINUATION, True),
    (("", "cup_and_handle"), SetupKind.CONTINUATION, True),
    (("", "rounding_bottom"), SetupKind.REVERSAL, True),
    (("", "inside_bar"), SetupKind.CONTINUATION, True),
    (("", "three_white_soldiers"), SetupKind.CONTINUATION, True),
    (("", "three_black_crows"), SetupKind.CONTINUATION, True),
    (("", "rising_three_methods"), SetupKind.CONTINUATION, True),
    (("", "falling_three_methods"), SetupKind.CONTINUATION, True),
    (("", "sr_bounce_"), SetupKind.REVERSAL, True),
    (("", "sr_break_"), SetupKind.BREAKOUT, True),
    (("", "trendline_break_"), SetupKind.BREAKOUT, True),
    (("", "range_breakout_"), SetupKind.BREAKOUT, True),
    # Liquidity
    (("", "sfp_"), SetupKind.REVERSAL, True),
    (("", "ob_retest"), SetupKind.CONTINUATION, True),
    (("", "fvg_fill_"), SetupKind.CONTINUATION, True),
    # Candlesticks
    (("", "hammer"), SetupKind.REVERSAL, True),
    (("", "shooting_star"), SetupKind.REVERSAL, True),
    (("", "doji"), SetupKind.REVERSAL, True),
    (("", "engulfing"), SetupKind.REVERSAL, True),
    (("", "harami"), SetupKind.REVERSAL, True),
    (("", "piercing_line"), SetupKind.REVERSAL, True),
    (("", "dark_cloud_cover"), SetupKind.REVERSAL, True),
    (("", "tweezer_"), SetupKind.REVERSAL, True),
    (("", "morning_star"), SetupKind.REVERSAL, True),
    (("", "evening_star"), SetupKind.REVERSAL, True),
    (("", "marubozu"), SetupKind.MOMENTUM, True),
    (("", "outside_bar_"), SetupKind.REVERSAL, True),
    # Legacy / generic
    (("", "breakout"), SetupKind.BREAKOUT, True),
    (("", "range"), SetupKind.BREAKOUT, True),
    (("", "triangle"), SetupKind.BREAKOUT, True),
    (("", "oversold"), SetupKind.MEAN_REVERSION, True),
    (("", "overbought"), SetupKind.MEAN_REVERSION, True),
    (("", "bb_"), SetupKind.MEAN_REVERSION, True),
    (("", "stoch_"), SetupKind.MEAN_REVERSION, True),
    (("", "rsi_"), SetupKind.MEAN_REVERSION, True),
    (("", "engulf"), SetupKind.REVERSAL, True),
    (("", "star"), SetupKind.REVERSAL, True),
    (("", "hidden_bull_divergence"), SetupKind.CONTINUATION, True),
    (("", "hidden_bear_divergence"), SetupKind.CONTINUATION, True),
    (("", "_divergence"), SetupKind.REVERSAL, True),
]

# Time-decay parameters.
DECAY_PER_CANDLE = 0.2
MIN_DECAY_WEIGHT = 0.5


def family_of(e: "Evidence | str") -> str:
    """Family of an Evidence (preferred) or a bare name (legacy callers)."""
    if isinstance(e, Evidence):
        if e.family:
            return e.family
        return _FAMILY.get(e.name, "other")
    return _FAMILY.get(e, "other")


def families(evidence: list[Evidence]) -> set[str]:
    return {family_of(e) for e in evidence}


def _kind_of(e: Evidence) -> SetupKind:
    """Classify evidence into a SetupKind using the registry, with heuristic
    substring fallback for unregistered legacy names."""
    # Exact matches first.
    for (src, name), kind, is_sub in SETUP_KIND_REGISTRY:
        if is_sub:
            continue
        if e.source == src and e.name == name:
            return kind

    # Substring / prefix matches.
    for (src, name), kind, is_sub in SETUP_KIND_REGISTRY:
        if not is_sub:
            continue
        if src and not e.source.startswith(src):
            continue
        if name in e.name:
            return kind

    # Heuristic fallback (the old brittle logic, kept for backwards safety).
    if "breakout" in e.name or "range" in e.name or "triangle" in e.name:
        return SetupKind.BREAKOUT
    if "oversold" in e.name or "overbought" in e.name or "bb_" in e.name:
        return SetupKind.MEAN_REVERSION
    if e.source == "chart" or "engulf" in e.name or "star" in e.name:
        return SetupKind.REVERSAL

    warnings.warn(
        f"Cannot classify evidence name '{e.name}' (source='{e.source}') into a SetupKind; "
        f"add it to SETUP_KIND_REGISTRY in confluence.py.",
        RuntimeWarning,
        stacklevel=3,
    )
    return SetupKind.MOMENTUM


def _validate_families(evidence: list[Evidence]) -> None:
    """Warn once per unknown/missing family so emitters notice quickly."""
    seen: set[str] = set()
    for e in evidence:
        fam = family_of(e)
        if fam == "other":
            key = f"missing:{e.name}"
            if key not in seen:
                seen.add(key)
                warnings.warn(
                    f"Evidence '{e.name}' has no explicit family and is not in the "
                    f"legacy family map; it will be bucketed as 'other'. "
                    f"Set Evidence.family to one of {CANONICAL_FAMILIES}.",
                    RuntimeWarning,
                    stacklevel=4,
                )
        elif fam not in CANONICAL_FAMILIES:
            key = f"unknown:{fam}:{e.name}"
            if key not in seen:
                seen.add(key)
                warnings.warn(
                    f"Evidence '{e.name}' uses unknown family '{fam}'. "
                    f"Canonical families are: {CANONICAL_FAMILIES}.",
                    RuntimeWarning,
                    stacklevel=4,
                )


def _time_decay_weight(e: Evidence, current_index: Optional[int]) -> float:
    """Weight older evidence down within the validity window.

    Uses exp(-decay * gap) with a floor so stale-but-valid evidence never
    disappears entirely.
    """
    if current_index is None or e.candle_index is None:
        return 1.0
    gap = current_index - e.candle_index
    if gap <= 0:
        return 1.0
    return max(math.exp(-DECAY_PER_CANDLE * gap), MIN_DECAY_WEIGHT)


def recalibrate_confidence(
    raw_confidence: float,
    evidence_list: list[Evidence],
    calibration_error: Optional[float],
) -> float:
    """Conservatively nudge confidence toward the base rate (0.5) when the
    rule-based calibration looks unreliable.

    `calibration_error` is expected in [0, 1]; larger values mean the recent
    rule-based win rate is further from a neutral 0.5 and the raw confidence
    should be dampened. The nudge is capped so raw_confidence remains dominant.
    """
    if calibration_error is None or calibration_error <= 0 or not evidence_list:
        return raw_confidence
    calibration_error = min(max(float(calibration_error), 0.0), 1.0)
    # Nudge toward 0.5; at most ~25% of the distance, scaled by error.
    nudge = 0.25 * calibration_error * (0.5 - raw_confidence)
    return min(max(raw_confidence + nudge, 0.0), 1.0)


def score(
    evidence: list[Evidence],
    current_index: Optional[int] = None,
    calibration_error: Optional[float] = None,
) -> dict:
    """Combine evidence into a signed score and a chosen side.

    Returns: side, score (signed), confidence (0..1), agreeing (evidence
    supporting the chosen side), n_families (independent families agreeing),
    families (their names), kind (dominant SetupKind).
    """
    if not evidence:
        return {"side": Side.FLAT, "score": 0.0, "confidence": 0.0,
                "agreeing": [], "n_families": 0, "families": [],
                "kind": SetupKind.MOMENTUM}

    _validate_families(evidence)

    def w(e: Evidence) -> float:
        return _time_decay_weight(e, current_index)

    bull = sum(e.strength * w(e) for e in evidence if e.bias == Bias.BULLISH)
    bear = sum(e.strength * w(e) for e in evidence if e.bias == Bias.BEARISH)
    # Neutral evidence signals a conflicted/undecided tape. It dilutes
    # directionality at half weight (it is information, but weaker than an
    # outright opposing vote).
    neutral = sum(e.strength * w(e) for e in evidence if e.bias == Bias.NEUTRAL)
    net = bull - bear
    total = bull + bear + 0.5 * neutral + 1e-9

    if net > 0:
        side = Side.LONG
        agreeing = [e for e in evidence if e.bias == Bias.BULLISH]
    elif net < 0:
        side = Side.SHORT
        agreeing = [e for e in evidence if e.bias == Bias.BEARISH]
    else:
        side = Side.FLAT
        agreeing = []

    fams = families(agreeing)
    n_families = len(fams)
    directionality = abs(net) / total                 # 0..1, how one-sided
    family_factor = min(n_families / 3.0, 1.0)         # saturates at 3 families
    raw_confidence = 0.5 * directionality + 0.5 * (directionality * family_factor)
    raw_confidence = min(max(raw_confidence, 0.0), 1.0)
    confidence = recalibrate_confidence(raw_confidence, evidence, calibration_error)

    return {
        "side": side,
        "score": net,
        "confidence": confidence,
        "agreeing": agreeing,
        "n_families": n_families,
        "families": sorted(fams),
        "kind": _dominant_kind(agreeing),
    }


def _dominant_kind(agreeing: list[Evidence]) -> SetupKind:
    from collections import Counter
    if not agreeing:
        return SetupKind.MOMENTUM
    kinds = [_kind_of(e) for e in agreeing]
    return Counter(kinds).most_common(1)[0][0]
