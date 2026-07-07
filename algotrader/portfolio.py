"""Portfolio exposure analysis.

Aggregates open positions by sector, BTC-correlation bucket and side so the
dashboard can surface concentration risks at a glance.
"""
from __future__ import annotations

from typing import Optional

from .execution.base import PositionState
from .models import Side


# Hardcoded sector map (base token -> sector).  Everything else falls to Other.
_SECTOR_MAP = {
    "BTC": "Layer1",
    "ETH": "Layer1",
    "SOL": "Alt-L1",
    "AVAX": "Alt-L1",
    "BNB": "Alt-L1",
    "ARB": "L2",
    "OP": "L2",
    "MATIC": "L2",
    "LINK": "Infra",
    "DOGE": "Meme",
    "SHIB": "Meme",
    "PEPE": "Meme",
    "WIF": "Meme",
}


def _base_token(symbol: str) -> str:
    """Extract base token from CCXT perpetual symbol formats.

    Examples:
        BTC/USDT:USDT -> BTC
        BTC/USDT      -> BTC
        SOL-PERP      -> SOL
    """
    raw = symbol.split(":")[0].split("/")[0].split("-")[0]
    return raw.upper()


def _notional(pos: PositionState) -> float:
    """Mark-to-market notional for an open position.

    Uses last known price when available, otherwise entry price.
    """
    price = pos.last_price if pos.last_price else pos.entry
    return price * pos.qty_open


class ExposureAnalyzer:
    """Break down open positions by sector, correlation bucket and side."""

    @classmethod
    def by_sector(cls, positions: list[PositionState],
                  symbol_metadata: Optional[dict] = None) -> list[dict]:
        """Group positions by sector and sum long/short notional + margin.

        `symbol_metadata` is accepted for API symmetry but the sector is
        currently resolved from a hardcoded base-token map; unknown tokens are
        reported as ``Other``.
        """
        metadata = symbol_metadata or {}
        groups: dict[str, dict] = {}
        for pos in positions:
            sector = metadata.get(pos.symbol, {}).get("sector")
            if not sector:
                sector = _SECTOR_MAP.get(_base_token(pos.symbol), "Other")
            g = groups.setdefault(sector, {
                "name": sector,
                "long_notional": 0.0,
                "short_notional": 0.0,
                "net_notional": 0.0,
                "margin": 0.0,
                "count": 0,
            })
            notional = _notional(pos)
            margin = pos.margin
            g["count"] += 1
            g["margin"] += margin
            if pos.side == Side.LONG:
                g["long_notional"] += notional
                g["net_notional"] += notional
            elif pos.side == Side.SHORT:
                g["short_notional"] += notional
                g["net_notional"] -= notional
        return [dict(v) for v in groups.values()]

    @classmethod
    def by_correlation_bucket(cls, positions: list[PositionState],
                              correlation_matrix: Optional[dict] = None) -> list[dict]:
        """Group positions by absolute correlation with BTC.

        Buckets:
            low    : abs(correlation) < 0.4
            medium : 0.4 <= abs(correlation) <= 0.7
            high   : abs(correlation) > 0.7

        Positions without a correlation entry are treated as ``low``.
        """
        corr = correlation_matrix or {}
        btc = "BTC/USDT:USDT"
        buckets = {
            "low": cls._bucket("low"),
            "medium": cls._bucket("medium"),
            "high": cls._bucket("high"),
        }
        for pos in positions:
            symbol_corr = corr.get(pos.symbol, {}).get(btc)
            if symbol_corr is None:
                symbol_corr = corr.get(btc, {}).get(pos.symbol)
            if symbol_corr is None:
                bucket_key = "low"
            else:
                abs_corr = abs(float(symbol_corr))
                if abs_corr > 0.7:
                    bucket_key = "high"
                elif abs_corr >= 0.4:
                    bucket_key = "medium"
                else:
                    bucket_key = "low"
            b = buckets[bucket_key]
            notional = _notional(pos)
            margin = pos.margin
            b["count"] += 1
            b["margin"] += margin
            if pos.side == Side.LONG:
                b["long_notional"] += notional
                b["net_notional"] += notional
            elif pos.side == Side.SHORT:
                b["short_notional"] += notional
                b["net_notional"] -= notional
        return [dict(v) for v in buckets.values()]

    @staticmethod
    def _bucket(name: str) -> dict:
        return {
            "name": name,
            "long_notional": 0.0,
            "short_notional": 0.0,
            "net_notional": 0.0,
            "margin": 0.0,
            "count": 0,
        }

    @classmethod
    def by_side(cls, positions: list[PositionState]) -> dict:
        """Gross long/short notional, margin and position counts."""
        long_notional = 0.0
        short_notional = 0.0
        long_margin = 0.0
        short_margin = 0.0
        long_count = 0
        short_count = 0
        for pos in positions:
            notional = _notional(pos)
            if pos.side == Side.LONG:
                long_notional += notional
                long_margin += pos.margin
                long_count += 1
            elif pos.side == Side.SHORT:
                short_notional += notional
                short_margin += pos.margin
                short_count += 1
        net = long_notional - short_notional
        gross = long_notional + short_notional
        return {
            "long": {
                "notional": long_notional,
                "margin": long_margin,
                "count": long_count,
            },
            "short": {
                "notional": short_notional,
                "margin": short_margin,
                "count": short_count,
            },
            "net": net,
            "gross": gross,
        }

    @classmethod
    def analyze(cls, positions: list[PositionState],
                symbol_metadata: Optional[dict] = None,
                correlation_matrix: Optional[dict] = None) -> dict:
        """Full exposure snapshot used by ``GET /api/exposure``."""
        return {
            "sectors": cls.by_sector(positions, symbol_metadata),
            "correlation_buckets": cls.by_correlation_bucket(
                positions, correlation_matrix),
            "sides": cls.by_side(positions),
        }
