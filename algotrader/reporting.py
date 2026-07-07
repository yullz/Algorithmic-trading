"""Serialization helpers shared by the paper trader and the dashboard."""
from __future__ import annotations

import json
import os

from .models import TradePlan


def plan_to_dict(p: TradePlan) -> dict:
    return {
        "symbol": p.symbol,
        "timeframe": p.timeframe,
        "side": p.side.value,
        "entry": p.entry,
        "stop_loss": p.stop_loss,
        "take_profits": [{"price": t.price, "r": t.r_multiple, "alloc": t.allocation}
                         for t in p.take_profits],
        "leverage": p.leverage,
        "qty": p.qty,
        "notional": p.notional,
        "margin": p.margin,
        "risk_amount": p.risk_amount,
        "liquidation_price": p.liquidation_price,
        "reward_risk": p.reward_risk,
        "expected_win_rate": p.expected_win_rate,
        "expected_value_r": p.expected_value_r,
        "confidence": p.confidence,
        "fees_estimate": p.fees_estimate,
        "warnings": p.warnings,
        "rationale": p.rationale,
        "explanation": p.explanation,
        "created_at": p.created_at,
        "regime": p.regime,
        "families": p.families,
        "ml_prob": p.ml_prob,
        "ml_weight": p.ml_weight,
        "ml_contribs": p.ml_contribs,
        "ml_ev_r": p.ml_ev_r,
        "time_stop_candles": p.time_stop_candles,
        "opened_at_candle": p.opened_at_candle,
    }


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    try:
        os.replace(tmp, path)  # atomic; dashboard never reads a half-written file
    except PermissionError:
        # On Windows os.replace can fail if the target is currently open.
        # Try remove-then-rename; if still blocked, fall back to direct write.
        try:
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp, path)
        except PermissionError:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, default=str)
            try:
                os.remove(tmp)
            except OSError:
                pass


def read_json(path: str, default: dict | None = None) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}
