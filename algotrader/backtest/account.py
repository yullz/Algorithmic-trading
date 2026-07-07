"""Account-level backtest: turn per-trade R multiples into a real P&L path.

The R-multiple engine measures per-trade edge but never exercises compounding,
concurrent positions competing for capital, or ruin. This layers a shared-equity
account on top: trades open at their entry time, hold for their measured
duration, are capped at the concurrent-position limit, sized as a fraction of
CURRENT equity (so wins and losses compound), and the realized path yields
currency CAGR, max drawdown, and time-under-water. A Monte-Carlo reshuffle of
the trade sequence estimates ruin probability and the drawdown you should brace
for — the numbers that decide whether an R-edge is actually survivable.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
import pandas as pd

_TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60,
           "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440}


def simulate_account(trades: list[dict], cfg, mc_runs: int = 500,
                     ruin_frac: float = 0.5, seed: int = 7) -> dict:
    """Simulate a shared-equity account from per-trade R outcomes.

    `trades` are the backtester's trade dicts (entry_time, tf, entry_idx,
    exit_idx, r). `cfg` supplies account_equity, risk_per_trade_pct, and
    max_concurrent_positions. Returns currency metrics + a Monte-Carlo ruin
    estimate.
    """
    if not trades:
        return {"present": False}
    start = float(cfg.account_equity)
    risk_pct = float(cfg.risk_per_trade_pct)
    max_conc = max(1, int(cfg.max_concurrent_positions))

    # Build (entry_time, exit_time, r) events; exit_time = entry + measured hold.
    evs = []
    for t in trades:
        et = pd.to_datetime(t.get("entry_time"), errors="coerce", utc=True)
        if pd.isna(et):
            continue
        dur = max(1, int(t.get("exit_idx", 0)) - int(t.get("entry_idx", 0)))
        xt = et + pd.Timedelta(minutes=_TF_MIN.get(t.get("tf", ""), 60) * dur)
        evs.append((et, xt, float(t["r"])))
    if len(evs) < 5:
        return {"present": False}
    evs.sort(key=lambda e: e[0])

    equity = start
    open_heap: list = []       # (exit_time, risk_amount, r)
    curve: list = []           # (timestamp, equity) at each close
    skipped = 0

    def _close_due(now) -> None:
        nonlocal equity
        while open_heap and open_heap[0][0] <= now:
            _xt, ra, r = heapq.heappop(open_heap)
            equity += ra * r
            curve.append((_xt, equity))

    for (et, xt, r) in evs:
        _close_due(et)
        if len(open_heap) >= max_conc:
            skipped += 1
            continue
        heapq.heappush(open_heap, (xt, equity * risk_pct, r))
    while open_heap:                       # close whatever remains
        _xt, ra, r = heapq.heappop(open_heap)
        equity += ra * r
        curve.append((_xt, equity))

    curve.sort(key=lambda c: c[0])
    eq = np.array([e for _, e in curve], dtype=float)
    eq = np.maximum(eq, 1e-9)              # guard against wipeout going negative
    times = [t for t, _ in curve]

    # ---- realized-path metrics ----
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = float(dd.max()) if len(dd) else 0.0
    tuw = float((eq < peak).mean()) if len(eq) else 0.0
    span_years = max((times[-1] - evs[0][0]).total_seconds() / (365.25 * 86400), 1e-6)
    total_return = equity / start - 1.0
    cagr = (equity / start) ** (1.0 / span_years) - 1.0 if equity > 0 else -1.0
    pct = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])
    sharpe = float(pct.mean() / pct.std(ddof=1)) if len(pct) > 1 and pct.std(ddof=1) > 0 else 0.0

    # ---- Monte-Carlo ruin / drawdown from reshuffled trade sequences ----
    r_series = np.array([e[2] for e in evs], dtype=float)
    rng = np.random.default_rng(seed)
    ruins, mc_dd = 0, np.empty(mc_runs)
    for i in range(mc_runs):
        order = rng.permutation(len(r_series))
        e = start
        pk = start
        worst = 0.0
        ruined = False
        for idx in order:
            e *= (1.0 + r_series[idx] * risk_pct)
            pk = max(pk, e)
            worst = max(worst, (pk - e) / pk)
            if e <= start * ruin_frac:
                ruined = True
        ruins += int(ruined)
        mc_dd[i] = worst
    ruin_prob = ruins / mc_runs
    dd_p95 = float(np.quantile(mc_dd, 0.95))

    return {
        "present": True,
        "start_equity": round(start, 2),
        "final_equity": round(float(equity), 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "time_under_water_pct": round(tuw * 100, 1),
        "sharpe": round(sharpe, 3),
        "span_years": round(span_years, 2),
        "n_trades": len(evs),
        "skipped_full_book": skipped,
        "ruin_prob": round(ruin_prob, 4),
        "ruin_frac": ruin_frac,
        "mc_drawdown_p95_pct": round(dd_p95 * 100, 1),
        "equity_curve": [round(float(x), 2) for x in eq[::max(1, len(eq) // 300)]],
    }
