"""Technical indicators, implemented in pure pandas/numpy (no TA-Lib needed).

Each function takes a price DataFrame (columns: open, high, low, close, volume)
and returns a Series/DataFrame aligned to the input index. `compute_all`
attaches everything as columns; `read_evidence` turns the latest row into a
list of `Evidence` for the confluence engine.

Honesty/design notes:
  * NO lookahead anywhere: every value at row i is computed from rows <= i.
    The Ichimoku spans are shifted FORWARD (+26), so the cloud value stored at
    the current row was computed from data 26 bars ago — reading the current
    row's `senkou_a`/`senkou_b` is exactly "price vs the cloud drawn today"
    with zero future information.
  * Trend indicators (Ichimoku, SuperTrend, PSAR, EMA ribbon) mostly emit
    evidence only on FLIPS / fresh alignments: their steady state is already
    captured by the EMA stack, and re-stating it every bar would just stuff
    the `trend` correlation family without adding information.
  * Williams %R and Aroon are computed as columns for other modules / the ML
    feature set but deliberately emit NO evidence (they duplicate Stoch and
    ADX respectively — double-counting is how confluence engines lie).
  * base_win_rate values are conservative priors; the backtest calibrator
    overwrites them with measured, regime-conditioned rates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..models import Bias, Evidence


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # A window with zero average loss is a clean rally -> RSI is 100 (max
    # overbought), NOT neutral. Without this the fillna(50) below would suppress
    # rsi_overbought evidence exactly when price is most extended (common on low
    # timeframes). Warmup rows (avg_gain NaN) still fall through to 50.
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - sig
    return line, sig, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid + k * sd, mid, mid - k * sd


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    low_k = df["low"].rolling(k, min_periods=k).min()
    high_k = df["high"].rolling(k, min_periods=k).max()
    pct_k = 100 * (df["close"] - low_k) / (high_k - low_k).replace(0, np.nan)
    pct_d = pct_k.rolling(d, min_periods=d).mean()
    return pct_k.fillna(50.0), pct_d.fillna(50.0)


def adx(df: pd.DataFrame, n: int = 14):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df)
    atr_n = tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / n, min_periods=n, adjust=False).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / n, min_periods=n, adjust=False).mean() / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_n = dx.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    return adx_n.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"]).cumsum()


def vwap(df: pd.DataFrame, window: int = 96) -> pd.Series:
    """ROLLING anchored VWAP over the trailing `window` bars.

    The previous implementation accumulated from the start of whatever frame
    was fetched, so the value depended on the arbitrary fetch length — useless
    as a level. A rolling anchor (default 96 bars = 1 day of 15m / 4 days of
    1h) gives a stable, fetch-length-independent reference. min_periods=1
    keeps the early bars defined (they degrade gracefully to the cumulative
    VWAP of the available history).
    """
    w = max(1, min(len(df), window))
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).rolling(w, min_periods=1).sum()
    vv = df["volume"].rolling(w, min_periods=1).sum().replace(0, np.nan)
    return pv / vv


def roc(close: pd.Series, n: int = 10) -> pd.Series:
    return 100 * (close / close.shift(n) - 1)


# --------------------------------------------------------------------------- #
# New primitives (Phase 2b expansion)
# --------------------------------------------------------------------------- #
def ichimoku(df: pd.DataFrame, tenkan_n: int = 9, kijun_n: int = 26,
             senkou_n: int = 52):
    """Ichimoku Kinko Hyo (9/26/52), WITHOUT the chikou span (pure lookback).

    `senkou_a`/`senkou_b` are shifted forward by `kijun_n` bars, so the value
    at row i was computed from rows <= i - 26. Reading the current row gives
    the cloud as plotted under today's candle — no future data involved.
    """
    h, l = df["high"], df["low"]
    tenkan = (h.rolling(tenkan_n, min_periods=tenkan_n).max()
              + l.rolling(tenkan_n, min_periods=tenkan_n).min()) / 2
    kijun = (h.rolling(kijun_n, min_periods=kijun_n).max()
             + l.rolling(kijun_n, min_periods=kijun_n).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_n)
    senkou_b = ((h.rolling(senkou_n, min_periods=senkou_n).max()
                 + l.rolling(senkou_n, min_periods=senkou_n).min()) / 2).shift(kijun_n)
    return tenkan, kijun, senkou_a, senkou_b


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0):
    """SuperTrend (ATR trailing band). Returns (line, direction) where
    direction is +1 (price above line / uptrend) or -1, 0 during warmup.

    Inherently recursive (the final bands ratchet against their previous
    values), so this is the one indicator that needs a per-bar loop — over a
    500-bar frame that is negligible.
    """
    a = atr(df, n).to_numpy(dtype=float)
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    ub, lb = hl2 + mult * a, hl2 - mult * a
    m = len(df)
    line = np.full(m, np.nan)
    dir_ = np.zeros(m)
    valid = np.flatnonzero(~np.isnan(a))
    if len(valid) == 0:
        return (pd.Series(line, index=df.index),
                pd.Series(dir_, index=df.index))
    start = int(valid[0])
    fu, fl, d = ub[start], lb[start], 1
    for i in range(start, m):
        if i > start:
            fu = ub[i] if (ub[i] < fu or c[i - 1] > fu) else fu
            fl = lb[i] if (lb[i] > fl or c[i - 1] < fl) else fl
        if d == 1 and c[i] < fl:
            d = -1
        elif d == -1 and c[i] > fu:
            d = 1
        line[i] = fl if d == 1 else fu
        dir_[i] = d
    return pd.Series(line, index=df.index), pd.Series(dir_, index=df.index)


def keltner(df: pd.DataFrame, n: int = 20, mult: float = 2.0):
    """Keltner Channel: EMA(close, n) +/- mult * ATR(n)."""
    mid = ema(df["close"], n)
    a = atr(df, n)
    return mid + mult * a, mid - mult * a


def donchian(df: pd.DataFrame, n: int = 20):
    """Donchian channel (includes the current bar — pure lookback)."""
    up = df["high"].rolling(n, min_periods=n).max()
    lo = df["low"].rolling(n, min_periods=n).min()
    return up, lo, (up + lo) / 2


def stochrsi(close: pd.Series, n: int = 14, stoch_n: int = 14,
             k: int = 3, d: int = 3):
    """Stochastic RSI (14,14,3,3), 0..100 scaled."""
    r = rsi(close, n)
    lo = r.rolling(stoch_n, min_periods=stoch_n).min()
    hi = r.rolling(stoch_n, min_periods=stoch_n).max()
    raw = 100 * (r - lo) / (hi - lo).replace(0, np.nan)
    k_line = raw.rolling(k, min_periods=k).mean()
    d_line = k_line.rolling(d, min_periods=d).mean()
    return k_line, d_line


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Money Flow Index — volume-weighted RSI analogue."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    raw = tp * df["volume"]
    delta = tp.diff()
    pos = raw.where(delta > 0, 0.0).rolling(n, min_periods=n).sum()
    neg = raw.where(delta < 0, 0.0).rolling(n, min_periods=n).sum()
    ratio = pos / neg.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50.0)


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = sma(tp, n)
    mean_dev = (tp - ma).abs().rolling(n, min_periods=n).mean()
    return (tp - ma) / (0.015 * mean_dev.replace(0, np.nan))


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Williams %R (-100..0). Column only — duplicates Stochastic."""
    hh = df["high"].rolling(n, min_periods=n).max()
    ll = df["low"].rolling(n, min_periods=n).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Chaikin Money Flow: signed accumulation/distribution over n bars."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    mult = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    mfv = (mult * df["volume"]).fillna(0.0)
    return (mfv.rolling(n, min_periods=n).sum()
            / df["volume"].rolling(n, min_periods=n).sum().replace(0, np.nan))


def psar(df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02,
         af_max: float = 0.2):
    """Parabolic SAR. Returns (sar, direction) with direction +1/-1 (0 warmup).

    Recursive by construction (AF ratchets, SAR clamps to prior bars), so a
    per-bar loop is unavoidable — and cheap at scan sizes.
    """
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    m = len(df)
    sar = np.full(m, np.nan)
    dir_ = np.zeros(m)
    if m < 3:
        return (pd.Series(sar, index=df.index),
                pd.Series(dir_, index=df.index))
    up = h[1] + l[1] >= h[0] + l[0]      # initial trend guess from bar 0->1
    ep = h[1] if up else l[1]            # extreme point
    s = l[0] if up else h[0]
    af = af_start
    sar[1] = s
    dir_[1] = 1 if up else -1
    for i in range(2, m):
        s = s + af * (ep - s)
        if up:
            s = min(s, l[i - 1], l[i - 2])   # SAR may not enter recent lows
            if l[i] < s:                     # reversal
                up, s, ep, af = False, ep, l[i], af_start
            elif h[i] > ep:
                ep, af = h[i], min(af + af_step, af_max)
        else:
            s = max(s, h[i - 1], h[i - 2])
            if h[i] > s:
                up, s, ep, af = True, ep, h[i], af_start
            elif l[i] < ep:
                ep, af = l[i], min(af + af_step, af_max)
        sar[i] = s
        dir_[i] = 1 if up else -1
    return pd.Series(sar, index=df.index), pd.Series(dir_, index=df.index)


def aroon(df: pd.DataFrame, n: int = 25):
    """Aroon up/down (0..100). Columns only — ADX already covers trend
    strength as evidence. Vectorized via sliding windows (no .apply loop)."""
    win = n + 1
    m = len(df)
    if m < win:
        nanser = pd.Series(np.full(m, np.nan), index=df.index)
        return nanser, nanser.copy()
    hw = sliding_window_view(df["high"].to_numpy(dtype=float), win)
    lw = sliding_window_view(df["low"].to_numpy(dtype=float), win)
    # bars since the MOST RECENT extreme (ties -> latest occurrence)
    since_hi = np.argmax(hw[:, ::-1], axis=1)
    since_lo = np.argmin(lw[:, ::-1], axis=1)
    pad = np.full(win - 1, np.nan)
    up = np.concatenate([pad, 100.0 * (n - since_hi) / n])
    dn = np.concatenate([pad, 100.0 * (n - since_lo) / n])
    return pd.Series(up, index=df.index), pd.Series(dn, index=df.index)


def atr_percentile(df: pd.DataFrame, atr_series: pd.Series,
                   window: int = 100) -> pd.Series:
    """Percentile (0..1) of ATR-relative volatility vs the trailing window.
    Consumed by regime detection and the ML feature set; no evidence."""
    ratio = atr_series / df["close"].replace(0, np.nan)
    return ratio.rolling(window, min_periods=20).rank(pct=True)


def volatility_percentile(atr_series: pd.Series,
                          lookback: int = 100) -> pd.Series:
    """Percentile (0..1) of raw ATR vs its trailing `lookback` window.

    Used by `read_evidence` to regime-shift overbought/oversold thresholds.
    """
    return atr_series.rolling(lookback, min_periods=20).rank(pct=True)


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ema(df["close"], 20)
    out["ema50"] = ema(df["close"], 50)
    out["ema200"] = ema(df["close"], 200)
    out["rsi"] = rsi(df["close"], 14)
    line, sig, hist = macd(df["close"])
    out["macd"], out["macd_signal"], out["macd_hist"] = line, sig, hist
    out["atr"] = atr(df, 14)
    up, mid, low = bollinger(df["close"], 20, 2.0)
    out["bb_up"], out["bb_mid"], out["bb_low"] = up, mid, low
    out["stoch_k"], out["stoch_d"] = stochastic(df)
    out["adx"], out["plus_di"], out["minus_di"] = adx(df)
    out["obv"] = obv(df)
    out["vwap"] = vwap(df)
    out["roc"] = roc(df["close"], 10)
    out["vol_sma20"] = sma(df["volume"], 20)

    # ---- Phase 2b additions ----
    tenkan, kijun, senkou_a, senkou_b = ichimoku(df)
    out["tenkan"], out["kijun"] = tenkan, kijun
    out["senkou_a"], out["senkou_b"] = senkou_a, senkou_b
    st_line, st_dir = supertrend(df, 10, 3.0)
    out["supertrend"], out["supertrend_dir"] = st_line, st_dir
    out["kc_up"], out["kc_low"] = keltner(df, 20, 2.0)
    # BB fully inside Keltner = volatility compression (TTM-style squeeze). The
    # canonical TTM squeeze uses a TIGHTER Keltner (1.5x) so BB-inside-KC is a
    # meaningful compression; at 2.0 the channel is so wide the squeeze fires
    # constantly and dilutes the signal. Keep kc_up/kc_low at 2.0 for any generic
    # Keltner consumers and use a dedicated 1.5x band for the squeeze test.
    # NaN comparisons are False, so the warmup region is simply "no squeeze".
    kc_sq_up, kc_sq_low = keltner(df, 20, 1.5)
    out["squeeze_on"] = (out["bb_up"] < kc_sq_up) & (out["bb_low"] > kc_sq_low)
    out["dc_up"], out["dc_low"], out["dc_mid"] = donchian(df, 20)
    out["stochrsi_k"], out["stochrsi_d"] = stochrsi(df["close"])
    out["mfi"] = mfi(df, 14)
    out["cci"] = cci(df, 20)
    out["willr"] = williams_r(df, 14)
    out["cmf"] = cmf(df, 20)
    ps, ps_dir = psar(df)
    out["psar"], out["psar_dir"] = ps, ps_dir
    out["aroon_up"], out["aroon_down"] = aroon(df, 25)
    for span in (8, 13, 21, 34, 55):
        out[f"ema{span}"] = ema(df["close"], span)
    out["atr_percentile"] = atr_percentile(df, out["atr"], 100)
    out["volatility_percentile"] = volatility_percentile(out["atr"], 100)
    return out


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
def _ok(*vals) -> bool:
    """All values present and not NaN."""
    return all(v is not None and not pd.isna(v) for v in vals)


def read_evidence(ind: pd.DataFrame) -> list[Evidence]:
    """Interpret the most recent indicator row into weighted Evidence.

    Priors on base_win_rate are conservative literature/rule-of-thumb values;
    the backtester overwrites them with measured rates via the calibration file.
    Every check reads only the last row (comparing against previous rows where
    a cross/flip must be detected) — no lookahead.
    """
    if len(ind) < 2:
        return []
    r = ind.iloc[-1]
    prev = ind.iloc[-2]
    ev: list[Evidence] = []

    def add(name, bias, strength, base=0.52, note="", family=""):
        ev.append(Evidence(name, "indicator", bias, strength, base, note,
                           family=family).clamp())

    # Volatility regime: adapt oscillator overbought/oversold thresholds.
    # Use the ATR/price percentile (stationary), NOT the raw-ATR percentile:
    # raw ATR scales with price level, so for a symbol whose price drifted up
    # over the window the raw percentile is biased high and wrongly widens the
    # overbought/oversold bands. atr_percentile normalizes by close first.
    vol_pct = float(r.get("atr_percentile", np.nan))
    if pd.isna(vol_pct):
        vol_pct = float(r.get("volatility_percentile", np.nan))
    if pd.isna(vol_pct):
        vol_pct = 0.5
    if vol_pct > 0.8:
        rsi_ob, rsi_os = 80.0, 20.0
        stoch_ob, stoch_os = 90.0, 10.0
        macd_hist_z = 2.0
        add("volatility_regime_high", Bias.NEUTRAL, 0.4, 0.5,
            f"vol percentile {vol_pct:.0%}", family="volatility")
    elif vol_pct < 0.2:
        rsi_ob, rsi_os = 65.0, 35.0
        stoch_ob, stoch_os = 75.0, 25.0
        macd_hist_z = 1.0
        add("volatility_regime_low", Bias.NEUTRAL, 0.4, 0.5,
            f"vol percentile {vol_pct:.0%}", family="volatility")
    else:
        rsi_ob, rsi_os = 70.0, 30.0
        stoch_ob, stoch_os = 80.0, 20.0
        macd_hist_z = 1.5

    # Trend regime via EMA stack
    if not np.isnan(r.get("ema50", np.nan)) and not np.isnan(r.get("ema200", np.nan)):
        if r["ema20"] > r["ema50"] > r["ema200"]:
            add("ema_stack_bull", Bias.BULLISH, 0.7, 0.56, "20>50>200 uptrend",
                family="trend")
        elif r["ema20"] < r["ema50"] < r["ema200"]:
            add("ema_stack_bear", Bias.BEARISH, 0.7, 0.56, "20<50<200 downtrend",
                family="trend")

    # RSI with volatility-adjusted thresholds
    if r["rsi"] < rsi_os:
        add("rsi_oversold", Bias.BULLISH, (rsi_os - r["rsi"]) / rsi_os + 0.3, 0.55,
            f"RSI {r['rsi']:.0f}", family="mean_reversion")
    elif r["rsi"] > rsi_ob:
        add("rsi_overbought", Bias.BEARISH, (r["rsi"] - rsi_ob) / (100 - rsi_ob) + 0.3, 0.55,
            f"RSI {r['rsi']:.0f}", family="mean_reversion")

    # MACD cross (unchanged) + volatility-scaled histogram extreme
    if prev["macd"] <= prev["macd_signal"] and r["macd"] > r["macd_signal"]:
        add("macd_bull_cross", Bias.BULLISH, 0.6, 0.53, family="momentum")
    elif prev["macd"] >= prev["macd_signal"] and r["macd"] < r["macd_signal"]:
        add("macd_bear_cross", Bias.BEARISH, 0.6, 0.53, family="momentum")

    if _ok(r.get("macd_hist")):
        hist_window = ind["macd_hist"].iloc[-20:].dropna()
        if len(hist_window) >= 5:
            hist_std = float(hist_window.std(ddof=0))
            if hist_std > 0:
                if r["macd_hist"] > macd_hist_z * hist_std:
                    add("macd_overbought", Bias.BEARISH, 0.5, 0.52,
                        f"MACD hist {r['macd_hist']:.4f} > {macd_hist_z}σ",
                        family="momentum")
                elif r["macd_hist"] < -macd_hist_z * hist_std:
                    add("macd_oversold", Bias.BULLISH, 0.5, 0.52,
                        f"MACD hist {r['macd_hist']:.4f} < -{macd_hist_z}σ",
                        family="momentum")

    # Bollinger reversion
    if r["close"] < r["bb_low"]:
        add("bb_lower_break", Bias.BULLISH, 0.5, 0.52, "below lower band",
            family="mean_reversion")
    elif r["close"] > r["bb_up"]:
        add("bb_upper_break", Bias.BEARISH, 0.5, 0.52, "above upper band",
            family="mean_reversion")

    # Stochastic with volatility-adjusted thresholds
    if r["stoch_k"] < stoch_os and r["stoch_k"] > r["stoch_d"]:
        add("stoch_bull", Bias.BULLISH, 0.5, 0.52, family="mean_reversion")
    elif r["stoch_k"] > stoch_ob and r["stoch_k"] < r["stoch_d"]:
        add("stoch_bear", Bias.BEARISH, 0.5, 0.52, family="mean_reversion")

    # ADX trend strength gate (directional, only counts if trend is real)
    if r["adx"] > 25:
        if r["plus_di"] > r["minus_di"]:
            add("adx_trend_bull", Bias.BULLISH, min(r["adx"] / 50, 1.0), 0.55,
                f"ADX {r['adx']:.0f}", family="trend")
        else:
            add("adx_trend_bear", Bias.BEARISH, min(r["adx"] / 50, 1.0), 0.55,
                f"ADX {r['adx']:.0f}", family="trend")

    # Volume confirmation
    if not np.isnan(r.get("vol_sma20", np.nan)) and r["volume"] > 1.5 * r["vol_sma20"]:
        bias = Bias.BULLISH if r["close"] > prev["close"] else Bias.BEARISH
        add("volume_spike", bias, 0.4, 0.51, "vol > 1.5x avg", family="volume")

    # ------------------------------------------------------------------ #
    # Phase 2b evidence
    # ------------------------------------------------------------------ #
    # Ichimoku: cloud position + Tenkan/Kijun state, TK cross on latest bar
    if _ok(r.get("tenkan"), r.get("kijun"), r.get("senkou_a"), r.get("senkou_b")):
        cloud_top = max(r["senkou_a"], r["senkou_b"])
        cloud_bot = min(r["senkou_a"], r["senkou_b"])
        above, below = r["close"] > cloud_top, r["close"] < cloud_bot
        if above and r["tenkan"] > r["kijun"]:
            add("ichimoku_bull", Bias.BULLISH, 0.65, 0.56,
                "price above cloud, tenkan>kijun", family="trend")
        elif below and r["tenkan"] < r["kijun"]:
            add("ichimoku_bear", Bias.BEARISH, 0.65, 0.56,
                "price below cloud, tenkan<kijun", family="trend")
        if _ok(prev.get("tenkan"), prev.get("kijun")):
            if prev["tenkan"] <= prev["kijun"] and r["tenkan"] > r["kijun"] and above:
                add("ichimoku_tk_cross_bull", Bias.BULLISH, 0.55, 0.54,
                    "TK cross up above cloud", family="trend")
            elif prev["tenkan"] >= prev["kijun"] and r["tenkan"] < r["kijun"] and below:
                add("ichimoku_tk_cross_bear", Bias.BEARISH, 0.55, 0.54,
                    "TK cross down below cloud", family="trend")

    # SuperTrend: only FLIPS carry new information (steady state = EMA stack)
    if _ok(r.get("supertrend_dir"), prev.get("supertrend_dir")):
        if prev["supertrend_dir"] == -1 and r["supertrend_dir"] == 1:
            add("supertrend_bull", Bias.BULLISH, 0.6, 0.55,
                "supertrend flipped up", family="trend")
        elif prev["supertrend_dir"] == 1 and r["supertrend_dir"] == -1:
            add("supertrend_bear", Bias.BEARISH, 0.6, 0.55,
                "supertrend flipped down", family="trend")

    # Squeeze: BB inside Keltner = compression; a recent release that breaks
    # 20-bar structure is a directional expansion signal.
    if bool(r.get("squeeze_on", False)):
        add("squeeze_on", Bias.NEUTRAL, 0.4, 0.5,
            "compression — expansion likely, direction unknown",
            family="volatility")
    elif len(ind) >= 22 and bool(ind["squeeze_on"].iloc[-6:-1].any()):
        prior_high = float(ind["high"].iloc[-21:-1].max())   # excludes current bar
        prior_low = float(ind["low"].iloc[-21:-1].min())
        if r["close"] > prior_high:
            add("squeeze_breakout_up", Bias.BULLISH, 0.65, 0.56,
                "squeeze released, broke 20-bar high", family="volatility")
        elif r["close"] < prior_low:
            add("squeeze_breakout_down", Bias.BEARISH, 0.65, 0.56,
                "squeeze released, broke 20-bar low", family="volatility")

    # StochRSI turn from an extreme
    if _ok(r.get("stochrsi_k"), r.get("stochrsi_d")):
        if r["stochrsi_k"] < 20 and r["stochrsi_k"] > r["stochrsi_d"]:
            add("stochrsi_oversold_turn", Bias.BULLISH, 0.5, 0.53,
                f"StochRSI {r['stochrsi_k']:.0f} turning up", family="mean_reversion")
        elif r["stochrsi_k"] > 80 and r["stochrsi_k"] < r["stochrsi_d"]:
            add("stochrsi_overbought_turn", Bias.BEARISH, 0.5, 0.53,
                f"StochRSI {r['stochrsi_k']:.0f} turning down", family="mean_reversion")

    # MFI extremes (volume-weighted flow -> volume family)
    if _ok(r.get("mfi")):
        if r["mfi"] < 20:
            add("mfi_oversold", Bias.BULLISH, 0.5, 0.53, f"MFI {r['mfi']:.0f}",
                family="volume")
        elif r["mfi"] > 80:
            add("mfi_overbought", Bias.BEARISH, 0.5, 0.53, f"MFI {r['mfi']:.0f}",
                family="volume")

    # CCI extremes, strength scaled by how stretched the reading is
    if _ok(r.get("cci")):
        if r["cci"] < -150:
            add("cci_extreme_low", Bias.BULLISH, min(abs(r["cci"]) / 300, 1.0), 0.52,
                f"CCI {r['cci']:.0f}", family="mean_reversion")
        elif r["cci"] > 150:
            add("cci_extreme_high", Bias.BEARISH, min(r["cci"] / 300, 1.0), 0.52,
                f"CCI {r['cci']:.0f}", family="mean_reversion")

    # Chaikin Money Flow: sustained signed flow
    if _ok(r.get("cmf")):
        if r["cmf"] > 0.15:
            add("cmf_bull", Bias.BULLISH, 0.45, 0.53, f"CMF {r['cmf']:.2f}",
                family="volume")
        elif r["cmf"] < -0.15:
            add("cmf_bear", Bias.BEARISH, 0.45, 0.53, f"CMF {r['cmf']:.2f}",
                family="volume")

    # Parabolic SAR flip on the latest bar
    if _ok(r.get("psar_dir"), prev.get("psar_dir")):
        if prev["psar_dir"] == -1 and r["psar_dir"] == 1:
            add("psar_flip_bull", Bias.BULLISH, 0.5, 0.53, "PSAR flipped below price",
                family="trend")
        elif prev["psar_dir"] == 1 and r["psar_dir"] == -1:
            add("psar_flip_bear", Bias.BEARISH, 0.5, 0.53, "PSAR flipped above price",
                family="trend")

    # EMA ribbon: only FRESH strict alignment (was not aligned 3 bars ago)
    ribbon = ("ema8", "ema13", "ema21", "ema34", "ema55")
    if len(ind) >= 4 and all(_ok(r.get(c)) for c in ribbon):
        r3 = ind.iloc[-4]

        def _bull(row):
            return (_ok(*(row.get(c) for c in ribbon))
                    and row["ema8"] > row["ema13"] > row["ema21"]
                    > row["ema34"] > row["ema55"])

        def _bear(row):
            return (_ok(*(row.get(c) for c in ribbon))
                    and row["ema8"] < row["ema13"] < row["ema21"]
                    < row["ema34"] < row["ema55"])

        if _bull(r) and not _bull(r3):
            add("ribbon_aligned_bull", Bias.BULLISH, 0.6, 0.55,
                "EMA ribbon freshly aligned up", family="trend")
        elif _bear(r) and not _bear(r3):
            add("ribbon_aligned_bear", Bias.BEARISH, 0.6, 0.55,
                "EMA ribbon freshly aligned down", family="trend")

    # OBV trend agreement: 20-bar regression slopes of OBV and price agree,
    # and both moves are meaningful (total window move >= 1.5 sigma of the
    # window — a pure linear ramp scores ~3.5, pure noise ~0).
    if len(ind) >= 20 and "obv" in ind.columns:
        x = np.arange(20, dtype=float)
        o = ind["obv"].iloc[-20:].to_numpy(dtype=float)
        c = ind["close"].iloc[-20:].to_numpy(dtype=float)
        if not (np.isnan(o).any() or np.isnan(c).any()):
            o_norm = np.polyfit(x, o, 1)[0] * 19 / (np.std(o) + 1e-9)
            c_norm = np.polyfit(x, c, 1)[0] * 19 / (np.std(c) + 1e-9)
            if o_norm > 1.5 and c_norm > 1.5:
                add("obv_trend_bull", Bias.BULLISH, 0.45, 0.53,
                    "OBV confirms price uptrend", family="volume")
            elif o_norm < -1.5 and c_norm < -1.5:
                add("obv_trend_bear", Bias.BEARISH, 0.45, 0.53,
                    "OBV confirms price downtrend", family="volume")

    # VWAP reclaim/reject: cross on the latest bar after >=3 bars on the
    # other side (prev bar counts as one of the three).
    if len(ind) >= 5 and _ok(r.get("vwap"), prev.get("vwap")):
        closes3 = ind["close"].iloc[-4:-1]
        vwaps3 = ind["vwap"].iloc[-4:-1]
        if not vwaps3.isna().any():
            if r["close"] > r["vwap"] and (closes3 < vwaps3).all():
                add("vwap_reclaim_bull", Bias.BULLISH, 0.5, 0.53,
                    "reclaimed VWAP after >=3 bars below", family="mean_reversion")
            elif r["close"] < r["vwap"] and (closes3 > vwaps3).all():
                add("vwap_reject_bear", Bias.BEARISH, 0.5, 0.53,
                    "rejected at VWAP after >=3 bars above", family="mean_reversion")

    return ev


# --------------------------------------------------------------------------- #
# Cross-symbol helper (called by the scanner, NOT by read_evidence)
# --------------------------------------------------------------------------- #
def relative_strength_evidence(df: pd.DataFrame, btc_df: pd.DataFrame,
                               lookback: int = 20,
                               threshold: float = 0.05) -> list[Evidence]:
    """Relative strength vs BTC: 20-bar return spread on the shared index.

    The scanner passes the symbol frame and the BTC frame of the SAME
    timeframe; alignment is on the index intersection so partially-listed or
    gappy symbols cannot fabricate a spread. Returns [] when the overlap is
    too short (<30 bars) to be meaningful.
    """
    if df is None or btc_df is None or "close" not in df or "close" not in btc_df:
        return []
    idx = df.index.intersection(btc_df.index)
    if len(idx) < max(30, lookback + 1):
        return []
    s = df.loc[idx, "close"]
    b = btc_df.loc[idx, "close"]
    s0, b0 = float(s.iloc[-1 - lookback]), float(b.iloc[-1 - lookback])
    if s0 <= 0 or b0 <= 0:
        return []
    excess = (float(s.iloc[-1]) / s0 - 1) - (float(b.iloc[-1]) / b0 - 1)
    if excess > threshold:
        return [Evidence("rs_outperform_btc", "indicator", Bias.BULLISH,
                         min(excess / 0.15, 1.0), 0.54,
                         f"+{excess:.1%} vs BTC over {lookback} bars",
                         family="relative_strength").clamp()]
    if excess < -threshold:
        return [Evidence("rs_underperform_btc", "indicator", Bias.BEARISH,
                         min(-excess / 0.15, 1.0), 0.54,
                         f"{excess:.1%} vs BTC over {lookback} bars",
                         family="relative_strength").clamp()]
    return []
