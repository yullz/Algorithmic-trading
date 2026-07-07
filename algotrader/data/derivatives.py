"""Derivatives-market evidence: funding rate and open interest.

This is the only evidence in the system that comes from outside the candle
series — it reads crowd *positioning* rather than price. Extreme funding
means one side of the perp market is crowded and paying to stay in; rising
open interest tells you whether a move is being driven by new positions
(conviction) or by closing ones (exhaustion).

Honesty and design notes:
  * Many symbols/exchanges do not support one or both endpoints. Each fetch
    is wrapped independently and a failure yields a *partial* (possibly
    empty) evidence list at debug log level — 150-symbol scans must not spam
    warnings or die because a long-tail perp lacks OI history.
  * Price context is passed in as `last_price_change` (fractional change over
    the caller's chosen window) rather than fetched here: the caller controls
    the lookback, and this module cannot introduce lookahead or a redundant
    price request.
  * The classification rules are pure functions of already-fetched numbers,
    shared verbatim between the async and sync entry points, so scanner and
    single-symbol tooling can never drift apart.
  * base_win_rate values are conservative priors (0.53-0.54) awaiting
    backtest calibration, like every other evidence family.

Mainnet public data only; no testnet wiring in the data layer.
"""
from __future__ import annotations

from typing import Optional, Sequence

from ..models import Bias, Evidence
from ..utils.logging import get_logger

log = get_logger(__name__)

# |8h funding| at or beyond this is "extreme": the crowded side pays real
# money to hold (5 bp per interval ~ 5.5%/yr against the position).
FUNDING_EXTREME = 0.0005
# Full evidence strength at 20 bp per interval — genuinely painful funding.
FUNDING_FULL_SCALE = 0.002
# OI change is measured over the last ~12 observations (hourly history).
OI_WINDOW = 12
OI_RISE_PCT = 0.05          # >5% OI growth = meaningful new positioning
OI_STRENGTH = 0.4
BASE_WR_FUNDING = 0.54
BASE_WR_OI = 0.53

_FAMILY = "derivatives"


# --------------------------------------------------------------------------- #
# Pure classification rules (shared by the async and sync fetchers)
# --------------------------------------------------------------------------- #
def funding_evidence(rate: Optional[float]) -> Optional[Evidence]:
    """Contrarian read of extreme funding.

    Deeply negative funding = shorts are crowded and paying longs -> BULLISH
    squeeze fuel. Mirror for positive. Strength scales linearly with |rate|
    up to FUNDING_FULL_SCALE.
    """
    if rate is None:
        return None
    strength = min(abs(rate) / FUNDING_FULL_SCALE, 1.0)
    if rate <= -FUNDING_EXTREME:
        return Evidence(
            "funding_extreme_long", "derivatives", Bias.BULLISH, strength,
            BASE_WR_FUNDING,
            f"funding {rate:+.4%}: shorts crowded, paying longs to hold",
            family=_FAMILY,
        ).clamp()
    if rate >= FUNDING_EXTREME:
        return Evidence(
            "funding_extreme_short", "derivatives", Bias.BEARISH, strength,
            BASE_WR_FUNDING,
            f"funding {rate:+.4%}: longs crowded, paying shorts to hold",
            family=_FAMILY,
        ).clamp()
    return None


def open_interest_evidence(oi_values: Sequence[float],
                           last_price_change: float) -> Optional[Evidence]:
    """OI growth read against the caller-supplied price direction.

    OI up >5% while price rises = new longs funding the move (continuation
    long). OI up while price falls = new shorts pressing (bearish). Flat
    price context yields nothing — OI alone has no direction.
    """
    vals = [float(v) for v in oi_values if v is not None and float(v) > 0]
    vals = vals[-OI_WINDOW:]
    if len(vals) < 3:
        return None
    change = (vals[-1] - vals[0]) / vals[0]
    if change <= OI_RISE_PCT:
        return None
    if last_price_change > 0:
        return Evidence(
            "oi_rising_with_price", "derivatives", Bias.BULLISH, OI_STRENGTH,
            BASE_WR_OI,
            f"OI +{change:.1%} with rising price: new longs driving the move",
            family=_FAMILY,
        ).clamp()
    if last_price_change < 0:
        return Evidence(
            "oi_rising_against_price", "derivatives", Bias.BEARISH, OI_STRENGTH,
            BASE_WR_OI,
            f"OI +{change:.1%} while price falls: new shorts pressing down",
            family=_FAMILY,
        ).clamp()
    return None


# --------------------------------------------------------------------------- #
# Payload extraction (tolerant of unified/raw field differences)
# --------------------------------------------------------------------------- #
def _extract_funding_rate(payload) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    rate = payload.get("fundingRate")
    if rate is None:
        rate = (payload.get("info") or {}).get("fundingRate")
    try:
        return float(rate) if rate is not None else None
    except (TypeError, ValueError):
        return None


def _extract_oi_values(history) -> list[float]:
    """Open-interest series from fetch_open_interest_history output. Prefers
    quote-denominated value, falls back to contract amount, then raw info."""
    vals: list[float] = []
    for row in history or []:
        if not isinstance(row, dict):
            continue
        v = row.get("openInterestValue")
        if v is None:
            v = row.get("openInterestAmount")
        if v is None:
            v = (row.get("info") or {}).get("openInterest")
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return vals


# --------------------------------------------------------------------------- #
# Fetchers
# --------------------------------------------------------------------------- #
async def fetch_derivatives_evidence(async_exchange, symbol: str,
                                     last_price_change: float = 0.0,
                                     ) -> list[Evidence]:
    """Funding + OI evidence for `symbol` via a ccxt.async_support exchange.

    Each endpoint fails independently and silently-ish (debug log): the
    result is whatever evidence could actually be gathered, possibly [].
    """
    out: list[Evidence] = []
    try:
        fr = await async_exchange.fetch_funding_rate(symbol)
        ev = funding_evidence(_extract_funding_rate(fr))
        if ev is not None:
            out.append(ev)
    except Exception as e:
        log.debug("funding rate unavailable for %s: %s", symbol, e)
    try:
        hist = await async_exchange.fetch_open_interest_history(
            symbol, timeframe="1h", limit=OI_WINDOW + 2)
        ev = open_interest_evidence(_extract_oi_values(hist), last_price_change)
        if ev is not None:
            out.append(ev)
    except Exception as e:
        log.debug("open interest history unavailable for %s: %s", symbol, e)
    return out


def fetch_derivatives_evidence_sync(exchange, symbol: str,
                                    last_price_change: float = 0.0,
                                    ) -> list[Evidence]:
    """Sync twin of fetch_derivatives_evidence (same rules, sync ccxt
    exchange) for run_scan-style single-threaded usage."""
    out: list[Evidence] = []
    try:
        fr = exchange.fetch_funding_rate(symbol)
        ev = funding_evidence(_extract_funding_rate(fr))
        if ev is not None:
            out.append(ev)
    except Exception as e:
        log.debug("funding rate unavailable for %s: %s", symbol, e)
    try:
        hist = exchange.fetch_open_interest_history(
            symbol, timeframe="1h", limit=OI_WINDOW + 2)
        ev = open_interest_evidence(_extract_oi_values(hist), last_price_change)
        if ev is not None:
            out.append(ev)
    except Exception as e:
        log.debug("open interest history unavailable for %s: %s", symbol, e)
    return out
