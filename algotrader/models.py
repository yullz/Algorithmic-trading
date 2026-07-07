"""
Core data contracts shared across the whole system.

Every module (indicators, patterns, signals, risk, backtest) speaks in terms
of the types defined here. Keep this file dependency-light so it can be
imported from anywhere without circular imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --------------------------------------------------------------------------- #
# Direction / side
# --------------------------------------------------------------------------- #
class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

    @property
    def sign(self) -> int:
        return {Side.LONG: 1, Side.SHORT: -1, Side.FLAT: 0}[self]


class Bias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SetupKind(str, Enum):
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"


# --------------------------------------------------------------------------- #
# Pattern / indicator evidence
# --------------------------------------------------------------------------- #
@dataclass
class Evidence:
    """A single piece of evidence contributing to a signal.

    `strength` is a subjective 0..1 weight of how strong this individual
    reading is *right now*. `base_win_rate` is the historical/empirical hit
    rate of this factor when it fires (populated by the backtest calibrator;
    a literature prior is used until calibrated).
    """
    name: str                       # e.g. "rsi_oversold", "bullish_engulfing"
    source: str                     # "indicator" | "candlestick" | "chart" | "structure"
    bias: Bias
    strength: float                 # 0..1 confidence of this reading now
    base_win_rate: float = 0.5      # 0..1 empirical hit rate (calibrated)
    note: str = ""
    # Correlation family (see signals/confluence.CANONICAL_FAMILIES). Explicit
    # beats name-guessing: emitters MUST set this for any new evidence name.
    family: str = ""
    # Optional index of the candle this evidence was derived from. Used for
    # time-decay weighting inside the confluence scorer.
    candle_index: Optional[int] = None

    def clamp(self) -> "Evidence":
        self.strength = max(0.0, min(1.0, self.strength))
        self.base_win_rate = max(0.0, min(1.0, self.base_win_rate))
        return self


@dataclass
class PatternMatch:
    """A detected chart/candlestick pattern."""
    name: str
    kind: SetupKind
    bias: Bias
    confidence: float               # 0..1
    start_idx: int
    end_idx: int
    # Optional geometric anchors (price levels) used for targets/stops.
    breakout_level: Optional[float] = None
    target_level: Optional[float] = None
    invalidation_level: Optional[float] = None
    base_win_rate: float = 0.5
    note: str = ""
    family: str = ""                # correlation family for the derived Evidence

    def to_evidence(self) -> Evidence:
        return Evidence(
            name=self.name,
            source="chart" if self.kind in (SetupKind.BREAKOUT, SetupKind.CONTINUATION,
                                            SetupKind.REVERSAL) else "candlestick",
            bias=self.bias,
            strength=self.confidence,
            base_win_rate=self.base_win_rate,
            note=self.note,
            family=self.family,
            candle_index=self.end_idx,
        ).clamp()


# --------------------------------------------------------------------------- #
# Risk / trade plan
# --------------------------------------------------------------------------- #
@dataclass
class RiskConfig:
    account_equity: float = 1000.0        # quote currency (e.g. USDT)
    risk_per_trade_pct: float = 0.01      # fraction of equity risked to stop (1%)
    max_leverage: float = 5.0             # user hard cap (3-5x requested)
    default_leverage: float = 3.0
    maker_fee: float = 0.0002             # 0.02%
    taker_fee: float = 0.0005             # 0.05%
    slippage_pct: float = 0.0005          # assumed slippage per side
    maintenance_margin_rate: float = 0.005  # 0.5% (conservative tier-1 approx)
    max_margin_alloc_pct: float = 0.20    # never commit >20% equity as margin/trade
    min_reward_risk: float = 1.0          # reject setups whose expected win < expected loss
    # Distance of stop from entry as ATR multiple (structure stop overrides this)
    atr_stop_mult: float = 1.5
    # Take-profit R multiples and the fraction of position closed at each
    tp_r_multiples: tuple = (1.0, 2.0, 3.0)
    tp_allocations: tuple = (0.4, 0.35, 0.25)
    # Hard cap on stop distance as an ATR multiple (a deep 20-bar swing low must
    # not silently create a huge risk distance).
    max_stop_atr_mult: float = 3.0
    # ---- portfolio-level constraints (scanner + executors) ----
    max_concurrent_positions: int = 6
    max_total_margin_pct: float = 0.60    # sum of margins across open positions
    correlation_cap: float = 0.8          # >this return-correlation = same bet
    # ---- circuit breakers (live/paper executors) ----
    max_daily_loss_pct: float = 0.03      # halt new entries after -3% day
    max_consecutive_losses: int = 5       # halt after N straight losers
    # ---- adaptive sizing ----
    adaptive_sizing_mode: str = "none"              # none | volatility_target | kelly
    volatility_target_atr_percentile: float = 50.0  # median ATR used as neutral
    kelly_fraction: float = 0.25                    # quarter-Kelly cap
    # ---- time stop ----
    max_trade_duration_candles: int = 0             # 0 = disabled
    # ---- regime-dependent sizing ----
    volatile_regime_size_factor: float = 0.7


@dataclass
class TakeProfit:
    price: float
    r_multiple: float
    allocation: float               # fraction of position closed here (sums to 1)


@dataclass
class TradePlan:
    """The complete, execution-ready futures trade specification."""
    symbol: str
    timeframe: str
    side: Side
    entry: float
    stop_loss: float
    take_profits: list[TakeProfit]
    leverage: float
    # sizing
    qty: float                      # position size in base units (contracts)
    notional: float                 # qty * entry (quote)
    margin: float                   # notional / leverage
    risk_amount: float              # quote at risk if stopped (before fees)
    liquidation_price: float
    # quality
    reward_risk: float
    expected_win_rate: float        # calibrated estimate, 0..1
    expected_value_r: float         # EV in R units (accounts for fees)
    confidence: float               # 0..1 overall confidence
    fees_estimate: float            # round-trip fees + slippage (quote)
    rationale: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)  # plain-English why profit/loss
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""            # ISO timestamp
    valid_until: str = ""           # signals decay; re-evaluate after this
    # signal context surfaced to the dashboard/API
    regime: str = ""
    families: list[str] = field(default_factory=list)
    ml_prob: Optional[float] = None
    ml_weight: float = 0.0
    ml_contribs: list[str] = field(default_factory=list)
    # time-stop bookkeeping
    time_stop_candles: int = 0
    opened_at_candle: int = -1

    def is_actionable(self, cfg: RiskConfig) -> bool:
        return (
            self.side != Side.FLAT
            and self.reward_risk >= cfg.min_reward_risk
            and self.expected_value_r > 0
            and self.qty > 0
        )


@dataclass
class Signal:
    """Aggregated engine output before it is turned into a sized TradePlan."""
    symbol: str
    timeframe: str
    side: Side
    kind: SetupKind
    entry_ref: float                # reference/last price used
    stop_ref: float                 # structural or ATR-based stop reference
    confidence: float               # 0..1
    score: float                    # raw confluence score (signed)
    evidence: list[Evidence] = field(default_factory=list)
    base_win_rate: float = 0.5      # blended empirical base rate
    note: str = ""
    # Structure-derived anchors from the strongest agreeing chart pattern.
    # The risk manager prefers these over generic swing/ATR levels when set.
    structure_stop: Optional[float] = None
    structure_target: Optional[float] = None
    regime: str = ""                # market regime label at signal time
    families: list[str] = field(default_factory=list)  # agreeing families
    # ML meta-model output (None when model absent/insufficiently trained)
    ml_prob: Optional[float] = None
    ml_weight: float = 0.0
    ml_contribs: list[str] = field(default_factory=list)
    # Continuous, stationary indicator values at signal time (ind_* keys) fed to
    # the ML meta-model and recorded in the backtest dataset. See
    # indicators.numeric_context.
    numeric_context: dict = field(default_factory=dict)
