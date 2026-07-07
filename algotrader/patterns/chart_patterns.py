"""Multi-bar chart pattern detection based on swing-pivot structure.

Detects: double/triple tops & bottoms, head & shoulders (+ inverse),
ascending / descending / symmetrical triangles, rising / falling wedges,
bull / bear flags, cup & handle, rounding bottom, horizontal S/R levels
(touch-counted), trendline breaks, and range breakouts. Each match carries
geometric anchors (breakout / target / invalidation) so the risk engine can
build structural stops and measured-move targets.

These are heuristic detectors — they approximate what a chartist sees, not a
formal proof. Confidence reflects how cleanly the geometry fits. Breakouts
require a CONFIRMED close through the level (no touch-fires), and the generic
Donchian range breakout is dropped whenever a more specific pattern explains
the same move (no double-counting in confluence).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Bias, PatternMatch, SetupKind

_FLAT_THR = 0.0008     # 0.08%/bar normalized slope = "flat"


def _pivots(series: np.ndarray, order: int = 3):
    """Return (high_idx, low_idx) local extrema with `order` bars on each side."""
    highs, lows = [], []
    n = len(series)
    for i in range(order, n - order):
        window = series[i - order:i + order + 1]
        if series[i] == window.max() and (window.argmax() == order):
            highs.append(i)
        if series[i] == window.min() and (window.argmin() == order):
            lows.append(i)
    return highs, lows


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> float:
    if len(close) < 2:
        return float(high[-1] - low[-1]) or 1e-9
    prev = close[:-1]
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - prev), np.abs(low[1:] - prev)))
    tail = tr[-n:] if len(tr) >= n else tr
    return float(max(tail.mean(), 1e-9))


def _r_squared(x: np.ndarray, y: np.ndarray, coef) -> float:
    fit = np.polyval(coef, x)
    ss_res = float(((y - fit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def _fresh_break(prev_close: float, last_close: float, level: float,
                 bullish: bool) -> bool:
    """True only on the bar the close FIRST crosses `level` in the given
    direction — previous bar on the wrong side, current bar through it.

    Without this gate a breakout/reversal pattern re-emits on EVERY bar while
    price merely remains beyond the level, which inflates the confluence score
    and pollutes each factor's win-rate calibration with stale repeats of the
    same move.
    """
    if bullish:
        return prev_close <= level < last_close
    return prev_close >= level > last_close


def _cluster_levels(prices: list[float], band: float) -> list[tuple[float, int]]:
    """Group pivot prices within `band` of the running cluster mean ->
    [(level, touches)] sorted by touches desc."""
    out: list[tuple[float, int]] = []
    for p in sorted(prices):
        if out and abs(p - out[-1][0]) <= band:
            lvl, n = out[-1]
            out[-1] = ((lvl * n + p) / (n + 1), n + 1)
        else:
            out.append((p, 1))
    return sorted(out, key=lambda t: -t[1])


def get_sr_levels(df: pd.DataFrame, lookback: int = 120,
                  max_levels: int = 12) -> list[tuple[float, int]]:
    """Horizontal support/resistance levels with touch counts (for the
    dashboard and for the S/R evidence below)."""
    if len(df) < 40:
        return []
    d = df.iloc[-lookback:] if len(df) > lookback else df
    high, low, close = (d["high"].to_numpy(), d["low"].to_numpy(),
                        d["close"].to_numpy())
    atr = _atr(high, low, close)
    h3, l3 = _pivots(close, 3)
    h5, l5 = _pivots(close, 5)
    prices = [float(close[j]) for j in set(h3 + l3 + h5 + l5)]
    return _cluster_levels(prices, 0.5 * atr)[:max_levels]


def detect(df: pd.DataFrame, lookback: int = 120, order: int = 3) -> list[PatternMatch]:
    if len(df) < 40:
        return []
    d = df.iloc[-lookback:] if len(df) > lookback else df
    high = d["high"].to_numpy()
    low = d["low"].to_numpy()
    close = d["close"].to_numpy()
    base = len(df) - len(d)  # offset to map back to full-frame indices
    out: list[PatternMatch] = []

    hi_idx, lo_idx = _pivots(close, order)
    hi5, lo5 = _pivots(close, 5)
    last = len(d) - 1
    last_close = close[-1]
    prev_close = close[-2]
    atr = _atr(high, low, close)
    mean_price = max(float(close.mean()), 1e-9)

    def tol(a, b, pct=0.02):
        return abs(a - b) <= pct * max(abs(a), abs(b), 1e-9)

    # ---------------- Double top / bottom ----------------
    if len(hi_idx) >= 2:
        a, b = hi_idx[-2], hi_idx[-1]
        if tol(close[a], close[b]) and (b - a) >= order:
            trough = close[a:b].min()
            neckline = trough
            if _fresh_break(prev_close, last_close, neckline, bullish=False):
                height = max(close[a], close[b]) - trough
                out.append(PatternMatch(
                    "double_top", SetupKind.REVERSAL, Bias.BEARISH,
                    0.62, base + a, base + b,
                    breakout_level=neckline, target_level=neckline - height,
                    invalidation_level=max(close[a], close[b]),
                    note="two equal highs, neckline broken", family="chart"))
    if len(lo_idx) >= 2:
        a, b = lo_idx[-2], lo_idx[-1]
        if tol(close[a], close[b]) and (b - a) >= order:
            peak = close[a:b].max()
            neckline = peak
            if _fresh_break(prev_close, last_close, neckline, bullish=True):
                height = peak - min(close[a], close[b])
                out.append(PatternMatch(
                    "double_bottom", SetupKind.REVERSAL, Bias.BULLISH,
                    0.62, base + a, base + b,
                    breakout_level=neckline, target_level=neckline + height,
                    invalidation_level=min(close[a], close[b]),
                    note="two equal lows, neckline broken", family="chart"))

    # ---------------- Triple top / bottom ----------------
    if len(hi_idx) >= 3:
        a, b, c3 = hi_idx[-3], hi_idx[-2], hi_idx[-1]
        if (tol(close[a], close[b], 0.025) and tol(close[b], close[c3], 0.025)
                and tol(close[a], close[c3], 0.025) and (c3 - a) >= 2 * order):
            neck = float(close[a:c3].min())
            if _fresh_break(prev_close, last_close, neck, bullish=False):
                top = float(max(close[a], close[b], close[c3]))
                out.append(PatternMatch(
                    "triple_top", SetupKind.REVERSAL, Bias.BEARISH,
                    0.64, base + a, base + c3,
                    breakout_level=neck, target_level=neck - (top - neck),
                    invalidation_level=top,
                    note="three equal highs, neckline broken", family="chart"))
    if len(lo_idx) >= 3:
        a, b, c3 = lo_idx[-3], lo_idx[-2], lo_idx[-1]
        if (tol(close[a], close[b], 0.025) and tol(close[b], close[c3], 0.025)
                and tol(close[a], close[c3], 0.025) and (c3 - a) >= 2 * order):
            neck = float(close[a:c3].max())
            if _fresh_break(prev_close, last_close, neck, bullish=True):
                bot = float(min(close[a], close[b], close[c3]))
                out.append(PatternMatch(
                    "triple_bottom", SetupKind.REVERSAL, Bias.BULLISH,
                    0.64, base + a, base + c3,
                    breakout_level=neck, target_level=neck + (neck - bot),
                    invalidation_level=bot,
                    note="three equal lows, neckline broken", family="chart"))

    # ---------------- Head & shoulders (+ inverse) ----------------
    if len(hi_idx) >= 3:
        l, h, r = hi_idx[-3], hi_idx[-2], hi_idx[-1]
        # head must clear both shoulders by >=1%, shoulders within 2% of each other
        if close[h] > max(close[l], close[r]) * 1.01 and tol(close[l], close[r], 0.02):
            neck = min(close[l:h].min(), close[h:r].min()) if r > h else close[l:r].min()
            if _fresh_break(prev_close, last_close, neck, bullish=False):
                height = close[h] - neck
                out.append(PatternMatch(
                    "head_and_shoulders", SetupKind.REVERSAL, Bias.BEARISH,
                    0.66, base + l, base + r,
                    breakout_level=neck, target_level=neck - height,
                    invalidation_level=close[h], note="H&S neckline broken",
                    family="chart"))
    if len(lo_idx) >= 3:
        l, h, r = lo_idx[-3], lo_idx[-2], lo_idx[-1]
        if close[h] < min(close[l], close[r]) * 0.99 and tol(close[l], close[r], 0.02):
            neck = max(close[l:h].max(), close[h:r].max()) if r > h else close[l:r].max()
            if _fresh_break(prev_close, last_close, neck, bullish=True):
                height = neck - close[h]
                out.append(PatternMatch(
                    "inverse_head_and_shoulders", SetupKind.REVERSAL, Bias.BULLISH,
                    0.66, base + l, base + r,
                    breakout_level=neck, target_level=neck + height,
                    invalidation_level=close[h], note="inverse H&S neckline broken",
                    family="chart"))

    # ---------------- Triangles (converging trendlines) ----------------
    win = 20
    if len(hi_idx) >= 2 and len(lo_idx) >= 2 and len(d) > win + 1:
        hs = hi_idx[-3:] if len(hi_idx) >= 3 else hi_idx[-2:]
        ls = lo_idx[-3:] if len(lo_idx) >= 3 else lo_idx[-2:]
        hslope = np.polyfit(hs, close[hs], 1)[0] / mean_price   # fraction/bar
        lslope = np.polyfit(ls, close[ls], 1)[0] / mean_price
        is_flat = lambda s: abs(s) <= _FLAT_THR
        rising = lambda s: s > _FLAT_THR
        falling = lambda s: s < -_FLAT_THR
        prior_high = float(high[-(win + 1):-1].max())            # excludes current bar
        prior_low = float(low[-(win + 1):-1].min())
        rng = prior_high - prior_low
        if is_flat(hslope) and rising(lslope):        # ascending -> bullish
            if _fresh_break(prev_close, last_close, prior_high, bullish=True):
                out.append(PatternMatch(
                    "ascending_triangle", SetupKind.BREAKOUT, Bias.BULLISH, 0.6,
                    base + ls[0], last + base, breakout_level=prior_high,
                    target_level=prior_high + rng, invalidation_level=prior_low,
                    note="flat top, rising lows, broke prior high", family="chart"))
        elif is_flat(lslope) and falling(hslope):     # descending -> bearish
            if _fresh_break(prev_close, last_close, prior_low, bullish=False):
                out.append(PatternMatch(
                    "descending_triangle", SetupKind.BREAKOUT, Bias.BEARISH, 0.6,
                    base + hs[0], last + base, breakout_level=prior_low,
                    target_level=prior_low - rng, invalidation_level=prior_high,
                    note="flat bottom, falling highs, broke prior low", family="chart"))
        elif falling(hslope) and rising(lslope):      # symmetrical -> either way
            if _fresh_break(prev_close, last_close, prior_high, bullish=True):
                out.append(PatternMatch(
                    "symmetrical_triangle_break_up", SetupKind.BREAKOUT, Bias.BULLISH,
                    0.55, base + ls[0], last + base, breakout_level=prior_high,
                    target_level=prior_high + rng, invalidation_level=prior_low,
                    family="chart"))
            elif _fresh_break(prev_close, last_close, prior_low, bullish=False):
                out.append(PatternMatch(
                    "symmetrical_triangle_break_down", SetupKind.BREAKOUT, Bias.BEARISH,
                    0.55, base + hs[0], last + base, breakout_level=prior_low,
                    target_level=prior_low - rng, invalidation_level=prior_high,
                    family="chart"))

        # ---------------- Wedges (both lines slope the SAME way, converging) --
        hfit = np.polyfit(hs, close[hs], 1)
        lfit = np.polyfit(ls, close[ls], 1)
        up_line = float(np.polyval(hfit, last))
        lo_line = float(np.polyval(lfit, last))
        start = min(hs[0], ls[0])
        gap_start = float(np.polyval(hfit, start) - np.polyval(lfit, start))
        gap_end = up_line - lo_line
        converging = gap_start > 0 and gap_end < 0.7 * gap_start
        hs_n, ls_n = hfit[0] / mean_price, lfit[0] / mean_price
        if (converging and rising(hs_n) and rising(ls_n)
                and _fresh_break(prev_close, last_close, lo_line, bullish=False)):
            out.append(PatternMatch(
                "rising_wedge", SetupKind.REVERSAL, Bias.BEARISH, 0.58,
                base + start, last + base, breakout_level=lo_line,
                target_level=lo_line - gap_start,
                invalidation_level=float(close[hs].max()),
                note="rising wedge support broken", family="chart"))
        if (converging and falling(hs_n) and falling(ls_n)
                and _fresh_break(prev_close, last_close, up_line, bullish=True)):
            out.append(PatternMatch(
                "falling_wedge", SetupKind.REVERSAL, Bias.BULLISH, 0.58,
                base + start, last + base, breakout_level=up_line,
                target_level=up_line + gap_start,
                invalidation_level=float(close[ls].min()),
                note="falling wedge resistance broken", family="chart"))

    # ---------------- Flags (impulse pole + tight drift + fresh break) -------
    for cons_n in (5, 8, 12):
        if len(d) < cons_n + 14:
            continue
        cons_h = float(high[-(cons_n + 1):-1].max())
        cons_l = float(low[-(cons_n + 1):-1].min())
        cons_rng = cons_h - cons_l
        pole_a, pole_b = -(cons_n + 13), -(cons_n + 1)
        pole_move = float(close[pole_b] - close[pole_a])
        drift = float(close[-2] - close[pole_b])
        if (pole_move >= 3 * atr and cons_rng <= 0.4 * abs(pole_move)
                and drift <= 0 and prev_close <= cons_h and last_close > cons_h):
            out.append(PatternMatch(
                "bull_flag", SetupKind.CONTINUATION, Bias.BULLISH, 0.6,
                base + len(d) + pole_a, last + base, breakout_level=cons_h,
                target_level=cons_h + abs(pole_move), invalidation_level=cons_l,
                note=f"pole {pole_move/atr:.1f} ATR, {cons_n}-bar flag broken up",
                family="chart"))
            break
        if (pole_move <= -3 * atr and cons_rng <= 0.4 * abs(pole_move)
                and drift >= 0 and prev_close >= cons_l and last_close < cons_l):
            out.append(PatternMatch(
                "bear_flag", SetupKind.CONTINUATION, Bias.BEARISH, 0.6,
                base + len(d) + pole_a, last + base, breakout_level=cons_l,
                target_level=cons_l - abs(pole_move), invalidation_level=cons_h,
                note=f"pole {pole_move/atr:.1f} ATR, {cons_n}-bar flag broken down",
                family="chart"))
            break

    # ---------------- Cup & handle (bullish only) -----------------------------
    handle_n = 10
    for W in (40, 60, 80):
        if len(d) < W + 2:
            continue
        seg = close[-(W + 1):-1]
        cup, handle = seg[:-handle_n], seg[-handle_n:]
        if len(cup) < 20:
            continue
        m = int(cup.argmin())
        if not (len(cup) / 3 <= m <= 2 * len(cup) / 3):
            continue
        q = max(len(cup) // 4, 3)
        rim_l, rim_r = float(cup[:q].max()), float(cup[-q:].max())
        if abs(rim_l - rim_r) > 0.05 * max(rim_l, rim_r):
            continue
        rim = max(rim_l, rim_r)
        depth = rim - float(cup.min())
        if depth < 2 * atr or float(handle.min()) < rim - 0.38 * depth:
            continue
        if _fresh_break(prev_close, last_close, rim, bullish=True):
            out.append(PatternMatch(
                "cup_and_handle", SetupKind.CONTINUATION, Bias.BULLISH, 0.6,
                base + len(d) - 1 - W, last + base, breakout_level=rim,
                target_level=rim + depth, invalidation_level=float(handle.min()),
                note=f"{W}-bar cup, shallow handle, rim cleared", family="chart"))
            break

    # ---------------- Rounding bottom ----------------------------------------
    for W in (40, 60, 80):
        if len(d) < W + 2:
            continue
        seg = close[-(W + 1):-1]
        x = np.arange(len(seg), dtype=float)
        coef = np.polyfit(x, seg, 2)
        if coef[0] <= 0:
            continue
        vertex = -coef[1] / (2 * coef[0])
        if not (0.25 * len(seg) <= vertex <= 0.75 * len(seg)):
            continue
        if _r_squared(x, seg, coef) < 0.5:
            continue
        rim = float(seg[:max(len(seg) // 5, 5)].max())
        if _fresh_break(prev_close, last_close, rim, bullish=True):
            out.append(PatternMatch(
                "rounding_bottom", SetupKind.REVERSAL, Bias.BULLISH, 0.55,
                base + len(d) - 1 - W, last + base, breakout_level=rim,
                target_level=rim + (rim - float(seg.min())),
                invalidation_level=float(seg.min()),
                note=f"{W}-bar saucer, rim cleared", family="chart"))
            break

    # ---------------- Horizontal S/R (touch-counted) --------------------------
    sr_prices = [float(close[j]) for j in set(hi_idx + lo_idx + hi5 + lo5)]
    levels = [(p, t) for p, t in _cluster_levels(sr_prices, 0.5 * atr) if t >= 3]
    sr_best: PatternMatch | None = None
    for p, t in levels:
        conf = 0.5 + 0.05 * min(t, 5)
        above = sorted(lp for lp, _ in levels if lp > p)
        below = sorted((lp for lp, _ in levels if lp < p), reverse=True)
        cand: PatternMatch | None = None
        if low[-1] <= p <= last_close and prev_close > p:
            cand = PatternMatch(
                "sr_bounce_long", SetupKind.REVERSAL, Bias.BULLISH, conf,
                last + base, last + base, breakout_level=p,
                target_level=above[0] if above else None,
                invalidation_level=p - 0.6 * atr,
                note=f"bounced off {t}-touch support", family="structure")
        elif high[-1] >= p >= last_close and prev_close < p:
            cand = PatternMatch(
                "sr_bounce_short", SetupKind.REVERSAL, Bias.BEARISH, conf,
                last + base, last + base, breakout_level=p,
                target_level=below[0] if below else None,
                invalidation_level=p + 0.6 * atr,
                note=f"rejected at {t}-touch resistance", family="structure")
        elif last_close > p and prev_close <= p:
            cand = PatternMatch(
                "sr_break_up", SetupKind.BREAKOUT, Bias.BULLISH, min(conf + 0.05, 1),
                last + base, last + base, breakout_level=p,
                target_level=above[0] if above else p + 2 * atr,
                invalidation_level=p - 0.6 * atr,
                note=f"closed above {t}-touch resistance", family="structure")
        elif last_close < p and prev_close >= p:
            cand = PatternMatch(
                "sr_break_down", SetupKind.BREAKOUT, Bias.BEARISH, min(conf + 0.05, 1),
                last + base, last + base, breakout_level=p,
                target_level=below[0] if below else p - 2 * atr,
                invalidation_level=p + 0.6 * atr,
                note=f"closed below {t}-touch support", family="structure")
        if cand is not None and (sr_best is None or cand.confidence > sr_best.confidence):
            sr_best = cand
    if sr_best is not None:
        out.append(sr_best)

    # ---------------- Trendline breaks ----------------------------------------
    if len(lo_idx) >= 3:
        xs = np.array(lo_idx[-4:] if len(lo_idx) >= 4 else lo_idx[-3:])
        ys = close[xs]
        coef = np.polyfit(xs, ys, 1)
        if coef[0] / mean_price > _FLAT_THR and _r_squared(xs.astype(float), ys, coef) >= 0.8:
            line_last = float(np.polyval(coef, last))
            line_prev = float(np.polyval(coef, last - 1))
            if last_close < line_last and prev_close >= line_prev:
                out.append(PatternMatch(
                    "trendline_break_down", SetupKind.BREAKOUT, Bias.BEARISH, 0.55,
                    base + int(xs[0]), last + base, breakout_level=line_last,
                    invalidation_level=line_last + atr,
                    note=f"uptrend line ({len(xs)} touches) broken", family="structure"))
    if len(hi_idx) >= 3:
        xs = np.array(hi_idx[-4:] if len(hi_idx) >= 4 else hi_idx[-3:])
        ys = close[xs]
        coef = np.polyfit(xs, ys, 1)
        if coef[0] / mean_price < -_FLAT_THR and _r_squared(xs.astype(float), ys, coef) >= 0.8:
            line_last = float(np.polyval(coef, last))
            line_prev = float(np.polyval(coef, last - 1))
            if last_close > line_last and prev_close <= line_prev:
                out.append(PatternMatch(
                    "trendline_break_up", SetupKind.BREAKOUT, Bias.BULLISH, 0.55,
                    base + int(xs[0]), last + base, breakout_level=line_last,
                    invalidation_level=line_last - atr,
                    note=f"downtrend line ({len(xs)} touches) broken", family="structure"))

    # ---------------- Range breakout (Donchian) ----------------
    window = 20
    if len(d) > window + 1:
        prior_high = high[-(window + 1):-1].max()
        prior_low = low[-(window + 1):-1].min()
        rng_h = prior_high - prior_low
        if _fresh_break(prev_close, last_close, prior_high, bullish=True):
            out.append(PatternMatch(
                "range_breakout_up", SetupKind.BREAKOUT, Bias.BULLISH, 0.55,
                last + base - window, last + base, breakout_level=prior_high,
                target_level=prior_high + rng_h, invalidation_level=prior_low,
                note=f"broke {window}-bar high", family="chart"))
        elif _fresh_break(prev_close, last_close, prior_low, bullish=False):
            out.append(PatternMatch(
                "range_breakout_down", SetupKind.BREAKOUT, Bias.BEARISH, 0.55,
                last + base - window, last + base, breakout_level=prior_low,
                target_level=prior_low - rng_h, invalidation_level=prior_high,
                note=f"broke {window}-bar low", family="chart"))

    return _dedupe(out)


def _dedupe(matches: list[PatternMatch]) -> list[PatternMatch]:
    """Drop generic matches that a more specific pattern already explains:
    the Donchian range breakout duplicates any specific same-direction
    breakout, and a triple top/bottom subsumes the double variant."""
    names = {m.name for m in matches}
    drop: set[str] = set()
    specific_bull = names & {"ascending_triangle", "symmetrical_triangle_break_up",
                             "falling_wedge", "bull_flag", "cup_and_handle",
                             "sr_break_up", "trendline_break_up"}
    specific_bear = names & {"descending_triangle", "symmetrical_triangle_break_down",
                             "rising_wedge", "bear_flag", "sr_break_down",
                             "trendline_break_down"}
    if specific_bull:
        drop.add("range_breakout_up")
    if specific_bear:
        drop.add("range_breakout_down")
    if "triple_top" in names:
        drop.add("double_top")
    if "triple_bottom" in names:
        drop.add("double_bottom")
    return [m for m in matches if m.name not in drop]
