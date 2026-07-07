"""SQLite-backed historical store for scans, signals, trades and positions.

The store is designed to be safe to call from the hot scan loop: every write
is committed immediately so a crash never loses more than the in-flight
operation.  Queries are simple string-based filters suitable for the dashboard
history view.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

from .models import SetupKind, Side
from .utils.logging import audit, get_logger, utcnow_iso

log = get_logger("history")

DEFAULT_DB_PATH = "reports/history.db"


@dataclass
class SignalHistory:
    """Flat representation of a signal ready for the history table."""
    symbol: str
    timeframe: str
    side: Side
    kind: SetupKind
    entry: float
    stop: float
    confidence: float
    score: float
    win_rate: float
    expected_value_r: float
    rationale: list[str]
    timestamp: Optional[str] = None


@dataclass
class TradeHistory:
    """Flat representation of an opened trade ready for the history table."""
    symbol: str
    side: Side
    entry: float
    stop: float
    qty: float
    leverage: float
    margin: float
    timestamp: Optional[str] = None


class HistoryStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH, root: str = "."):
        self.db_path = os.path.join(root, db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ #
    # connection helpers
    # ------------------------------------------------------------------ #
    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    btc_regime TEXT,
                    n_symbols INTEGER,
                    top_n INTEGER,
                    error_msg TEXT
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id INTEGER NOT NULL,
                    timestamp TEXT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT,
                    side TEXT,
                    kind TEXT,
                    entry REAL,
                    stop REAL,
                    confidence REAL,
                    score REAL,
                    win_rate REAL,
                    expected_value_r REAL,
                    rationale_json TEXT,
                    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_signals_scan ON signals(scan_id);
                CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
                CREATE INDEX IF NOT EXISTS idx_signals_timeframe ON signals(timeframe);
                CREATE INDEX IF NOT EXISTS idx_signals_side ON signals(side);
                CREATE INDEX IF NOT EXISTS idx_signals_kind ON signals(kind);
                CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id INTEGER,
                    signal_id INTEGER,
                    timestamp TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT,
                    entry REAL,
                    stop REAL,
                    qty REAL,
                    leverage REAL,
                    margin REAL,
                    outcome TEXT,
                    realized_r REAL,
                    fees_r REAL,
                    closed_at TEXT,
                    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE,
                    FOREIGN KEY (signal_id) REFERENCES signals(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_trades_scan ON trades(scan_id);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome);
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
                CREATE INDEX IF NOT EXISTS idx_trades_closed ON trades(closed_at);

                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT,
                    opened_at TEXT,
                    closed_at TEXT,
                    status TEXT,
                    mtm_pnl REAL,
                    FOREIGN KEY (trade_id) REFERENCES trades(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
                CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
                """
            )
            conn.commit()

    # ------------------------------------------------------------------ #
    # writes
    # ------------------------------------------------------------------ #
    def record_scan(self, timestamp: str, btc_regime: str, n_symbols: int,
                    top_n: int, error_msg: str = "") -> int:
        """Persist a scan summary and return the scan id."""
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO scans (timestamp, btc_regime, n_symbols, top_n, error_msg)
                   VALUES (?, ?, ?, ?, ?)""",
                (timestamp, btc_regime or "", n_symbols, top_n, error_msg or ""),
            )
            conn.commit()
            scan_id = cur.lastrowid
        audit("history_scan", {"scan_id": scan_id, "btc_regime": btc_regime,
                               "n_symbols": n_symbols, "top_n": top_n})
        return scan_id

    def record_signals(self, scan_id: int, signals: list[SignalHistory]) -> dict[str, int]:
        """Persist signals for a scan.  Returns a map symbol -> signal_id."""
        ids: dict[str, int] = {}
        if not signals:
            return ids
        ts = utcnow_iso()
        with self._connect() as conn:
            for sig in signals:
                rationale = sig.rationale if isinstance(sig.rationale, list) else []
                cur = conn.execute(
                    """INSERT INTO signals
                       (scan_id, timestamp, symbol, timeframe, side, kind, entry, stop,
                        confidence, score, win_rate, expected_value_r, rationale_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (scan_id,
                     sig.timestamp or ts,
                     sig.symbol,
                     sig.timeframe,
                     sig.side.value if isinstance(sig.side, Side) else str(sig.side),
                     sig.kind.value if isinstance(sig.kind, SetupKind) else str(sig.kind),
                     float(sig.entry),
                     float(sig.stop),
                     float(sig.confidence),
                     float(sig.score),
                     float(sig.win_rate),
                     float(sig.expected_value_r),
                     json.dumps(rationale, default=str)),
                )
                ids[sig.symbol] = cur.lastrowid
            conn.commit()
        audit("history_signals", {"scan_id": scan_id, "count": len(ids)})
        return ids

    def record_trade(self, scan_id: int, signal_id: Optional[int],
                     trade: TradeHistory) -> int:
        """Persist a newly opened trade and its associated open position."""
        ts = trade.timestamp or utcnow_iso()
        side = trade.side.value if isinstance(trade.side, Side) else str(trade.side)
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO trades
                   (scan_id, signal_id, timestamp, symbol, side, entry, stop, qty,
                    leverage, margin, outcome)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (scan_id, signal_id, ts, trade.symbol, side,
                 float(trade.entry), float(trade.stop), float(trade.qty),
                 float(trade.leverage), float(trade.margin), "open"),
            )
            trade_id = cur.lastrowid
            conn.execute(
                """INSERT INTO positions
                   (trade_id, symbol, side, opened_at, status, mtm_pnl)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (trade_id, trade.symbol, side, ts, "open", 0.0),
            )
            conn.commit()
        audit("history_trade_open", {"trade_id": trade_id, "scan_id": scan_id,
                                     "signal_id": signal_id, "symbol": trade.symbol})
        return trade_id

    def update_position(self, trade_id: int, symbol: str, side: str,
                        opened_at: str, closed_at: Optional[str],
                        status: str, mtm_pnl: float,
                        outcome: Optional[str] = None,
                        realized_r: Optional[float] = None,
                        fees_r: Optional[float] = None) -> None:
        """Update the position lifecycle row; if the position is closing,
        also update the parent trade with outcome and realized R metrics.
        """
        side_val = side.value if isinstance(side, Side) else str(side)
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO positions (trade_id, symbol, side, opened_at,
                                            closed_at, status, mtm_pnl)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(trade_id) DO UPDATE SET
                       closed_at=excluded.closed_at,
                       status=excluded.status,
                       mtm_pnl=excluded.mtm_pnl""",
                (trade_id, symbol, side_val, opened_at, closed_at, status,
                 float(mtm_pnl)),
            )
            if status == "closed" and closed_at is not None:
                conn.execute(
                    """UPDATE trades
                       SET outcome = COALESCE(?, outcome),
                           realized_r = COALESCE(?, realized_r),
                           fees_r = COALESCE(?, fees_r),
                           closed_at = ?
                       WHERE id = ?""",
                    (outcome, realized_r, fees_r, closed_at, trade_id),
                )
            conn.commit()
        audit("history_position_update",
              {"trade_id": trade_id, "symbol": symbol, "status": status,
               "mtm_pnl": mtm_pnl, "outcome": outcome})

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_where(filters: dict[str, Any], aliases: dict[str, str]) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        for key, value in filters.items():
            if value is None or value == "":
                continue
            col = aliases.get(key)
            if col is None:
                continue
            if key in ("from", "from_"):
                clauses.append(f"{col} >= ?")
                values.append(value)
            elif key == "to":
                clauses.append(f"{col} <= ?")
                values.append(value)
            elif key == "limit":
                continue
            else:
                clauses.append(f"{col} = ?")
                values.append(value)
        return " AND ".join(clauses), values

    def get_signals(self, filters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        filters = filters or {}
        aliases = {
            "symbol": "s.symbol",
            "timeframe": "s.timeframe",
            "side": "s.side",
            "kind": "s.kind",
            "regime": "sc.btc_regime",
            "from": "s.timestamp",
            "from_": "s.timestamp",
            "to": "s.timestamp",
        }
        where, values = self._build_where(filters, aliases)
        sql = """SELECT s.*, sc.btc_regime
                 FROM signals s
                 JOIN scans sc ON sc.id = s.scan_id"""
        if where:
            sql += " WHERE " + where
        sql += " ORDER BY s.timestamp DESC"
        limit = filters.get("limit")
        if limit:
            sql += " LIMIT ?"
            values.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, values).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_trades(self, filters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        filters = filters or {}
        aliases = {
            "symbol": "symbol",
            "outcome": "outcome",
            "from": "timestamp",
            "from_": "timestamp",
            "to": "timestamp",
        }
        where, values = self._build_where(filters, aliases)
        sql = "SELECT * FROM trades"
        if where:
            sql += " WHERE " + where
        sql += " ORDER BY timestamp DESC"
        limit = filters.get("limit")
        if limit:
            sql += " LIMIT ?"
            values.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, values).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_scan_summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            total_scans = conn.execute(
                "SELECT COUNT(*) FROM scans").fetchone()[0]
            total_signals = conn.execute(
                "SELECT COUNT(*) FROM signals").fetchone()[0]
            total_trades = conn.execute(
                "SELECT COUNT(*) FROM trades").fetchone()[0]
            closed = conn.execute(
                """SELECT COUNT(*), AVG(realized_r), SUM(CASE WHEN realized_r > 0 THEN 1 ELSE 0 END)
                   FROM trades WHERE outcome IS NOT NULL AND outcome != 'open'""").fetchone()
            closed_count, avg_r, wins = closed
        summary = {
            "total_scans": total_scans,
            "total_signals": total_signals,
            "total_trades": total_trades,
            "closed_trades": closed_count or 0,
            "win_rate": (wins / closed_count) if closed_count else None,
            "avg_realized_r": avg_r,
        }
        summary["by_month"] = self._by_month()
        summary["by_regime"] = self._by_regime()
        summary["by_setup_kind"] = self._by_setup_kind()
        return summary

    def _by_month(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT strftime('%Y-%m', closed_at) AS month,
                          COUNT(*) AS trades,
                          AVG(realized_r) AS avg_r,
                          SUM(CASE WHEN realized_r > 0 THEN 1 ELSE 0 END) AS wins
                   FROM trades
                   WHERE closed_at IS NOT NULL AND outcome != 'open'
                   GROUP BY month
                   ORDER BY month DESC""").fetchall()
        return [_row_to_dict(r) for r in rows]

    def _by_regime(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT sc.btc_regime,
                          COUNT(*) AS trades,
                          AVG(t.realized_r) AS avg_r,
                          SUM(CASE WHEN t.realized_r > 0 THEN 1 ELSE 0 END) AS wins
                   FROM trades t
                   JOIN scans sc ON sc.id = t.scan_id
                   WHERE t.closed_at IS NOT NULL AND t.outcome != 'open'
                   GROUP BY sc.btc_regime
                   ORDER BY trades DESC""").fetchall()
        return [_row_to_dict(r) for r in rows]

    def _by_setup_kind(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT s.kind,
                          COUNT(*) AS trades,
                          AVG(t.realized_r) AS avg_r,
                          SUM(CASE WHEN t.realized_r > 0 THEN 1 ELSE 0 END) AS wins
                   FROM trades t
                   JOIN signals s ON s.id = t.signal_id
                   WHERE t.closed_at IS NOT NULL AND t.outcome != 'open'
                   GROUP BY s.kind
                   ORDER BY trades DESC""").fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_win_rate_by(self, field: str) -> list[dict[str, Any]]:
        """Return win-rate breakdown by one of: symbol, side, kind, regime."""
        field = field.lower()
        allowed = {"symbol", "side", "kind", "regime"}
        if field not in allowed:
            raise ValueError(f"field must be one of {allowed}")
        if field == "regime":
            group_col = "sc.btc_regime"
            join = "JOIN scans sc ON sc.id = t.scan_id"
        elif field == "kind":
            group_col = "s.kind"
            join = "JOIN signals s ON s.id = t.signal_id"
        else:
            group_col = f"t.{field}"
            join = ""
        sql = f"""SELECT {group_col} AS {field},
                          COUNT(*) AS trades,
                          AVG(t.realized_r) AS avg_r,
                          SUM(CASE WHEN t.realized_r > 0 THEN 1 ELSE 0 END) AS wins
                   FROM trades t
                   {join}
                   WHERE t.closed_at IS NOT NULL AND t.outcome != 'open'
                   GROUP BY {field}
                   ORDER BY trades DESC"""
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}
