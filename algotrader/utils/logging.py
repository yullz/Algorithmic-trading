"""Structured logging + a JSONL trade audit log."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

_LOG_DIR = os.path.join(os.getcwd(), "logs")


def get_logger(name: str = "algotrader") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logger.addHandler(h)
    return logger


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit(event: str, payload: dict[str, Any], path: str = "logs/audit.jsonl") -> None:
    """Append-only audit trail for every signal/trade. Never raises."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        record = {"ts": utcnow_iso(), "event": event, **payload}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:  # audit must never break trading logic
        get_logger().warning("audit write failed", exc_info=True)
