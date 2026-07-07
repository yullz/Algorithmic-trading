"""Unit tests for algotrader/indicators and the new Phase-2 modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrader.indicators import indicators as ind
from algotrader.indicators import cross_asset, derivatives, market_structure, volume_breakout
from algotrader.models import Bias


EXPECTED_COLUMNS = {
    "open", "high", "low", "close", "volume",
    "ema20", "ema50", "ema200",
    "rsi", "macd", "macd_signal", "macd_hist",
    "atr", "bb_up", "bb_mid", "bb_low",
    "stoch_k", "stoch_d", "adx", "plus_di", "minus_di",
    "obv", "vwap", "roc", "vol_sma20",
    "tenkan", "kijun", "senkou_a", "senkou_b",
    "supertrend", "supertrend_dir",
    "kc_up", "kc_low", "squeeze_on",
    "dc_up", "dc_low", "dc_mid",
    "stochrsi_k", "stochrsi_d",
    "mfi", "cci", "willr", "cmf",
    "psar", "psar_dir",
    "aroon_up", "aroon_down",
    "ema8", "ema13", "ema21", "ema34", "ema55",
    "atr_percentile", "volatility_percentile",
}


def test_compute_all_output_columns(ohlcv_frame):
    out = ind.compute_all(ohlcv_frame)
    missing = EXPECTED_COLUMNS - set(out.columns)
    assert not missing, f"Missing indicator columns: {missing}"
    assert len(out) == len(ohlcv_frame)
    assert out.index.equals(ohlcv_frame.index)


def test_read_evidence_clamps_strength(ohlcv_frame):
    """Evidence strengths and base rates produced by read_evidence stay in [0, 1]."""
    indf = ind.compute_all(ohlcv_frame)

    # Extreme overbought RSI -> raw strength formula would be > 1.
    indf.loc[indf.index[-1], "rsi"] = 100.0
    ev = ind.read_evidence(indf)
    assert ev
    for e in ev:
        assert 0.0 <= e.strength <= 1.0, f"{e.name} strength {e.strength} out of bounds"
        assert 0.0 <= e.base_win_rate <= 1.0

    # Negative RSI -> raw strength formula would be > 1; must clamp to 1.0.
    indf2 = ind.compute_all(ohlcv_frame)
    indf2.loc[indf2.index[-1], "rsi"] = -10.0
    ev2 = ind.read_evidence(indf2)
    rsi_ev = [e for e in ev2 if "rsi" in e.name]
    assert rsi_ev
    assert rsi_ev[0].strength == pytest.approx(1.0, abs=1e-9)


def test_rolling_indicators_no_lookahead(ohlcv_frame):
    """For rolling/windowed indicators, value at row i must equal the value
    computed from only rows 0..i (inclusive) once the longest window is warm.
    """
    full = ind.compute_all(ohlcv_frame)
    # Start after the longest warm-up (200-bar EMA / 100-bar percentile).
    start = 250
    cols = ["ema20", "rsi", "bb_up", "stoch_k", "atr", "adx",
            "aroon_up", "willr", "vwap", "atr_percentile", "volatility_percentile"]
    for i in range(start, len(ohlcv_frame)):
        incremental = ind.compute_all(ohlcv_frame.iloc[: i + 1])
        for col in cols:
            full_val = full[col].iloc[i]
            inc_val = incremental[col].iloc[-1]
            if pd.isna(full_val):
                assert pd.isna(inc_val), f"{col}@{i}: NaN mismatch"
            else:
                assert full_val == pytest.approx(inc_val, rel=1e-9, abs=1e-12), \
                    f"{col}@{i}: lookahead detected"


# --------------------------------------------------------------------------- #
# Volatility regime-aware thresholds
# --------------------------------------------------------------------------- #
def test_volatility_percentile_present_and_bounded(ohlcv_frame):
    out = ind.compute_all(ohlcv_frame)
    assert "volatility_percentile" in out.columns
    vp = out["volatility_percentile"].dropna()
    assert ((vp >= 0.0) & (vp <= 1.0)).all()


def test_read_evidence_volatility_regime_thresholds(ohlcv_frame):
    """High vol widens RSI/Stoch thresholds; low vol tightens them. The regime
    is now driven by atr_percentile (ATR/price, stationary), not raw-ATR rank."""
    indf = ind.compute_all(ohlcv_frame)

    # High-vol regime: RSI must exceed 80 to be overbought.
    indf.loc[indf.index[-1], "atr_percentile"] = 0.95
    indf.loc[indf.index[-1], "rsi"] = 75.0
    ev = ind.read_evidence(indf)
    assert not any(e.name == "rsi_overbought" for e in ev)
    assert any(e.name == "volatility_regime_high" for e in ev)

    # Low-vol regime: RSI above 65 is overbought.
    indf2 = ind.compute_all(ohlcv_frame)
    indf2.loc[indf2.index[-1], "atr_percentile"] = 0.05
    indf2.loc[indf2.index[-1], "rsi"] = 68.0
    ev2 = ind.read_evidence(indf2)
    assert any(e.name == "rsi_overbought" for e in ev2)
    assert any(e.name == "volatility_regime_low" for e in ev2)


def test_rsi_clean_rally_reads_overbought_not_neutral():
    """A monotonic rally has zero average loss -> RSI must be ~100 (overbought),
    not the misleading neutral 50 the old .fillna(50) produced."""
    close = pd.Series(np.linspace(100.0, 200.0, 60))  # strictly increasing
    r = ind.rsi(close, 14)
    assert r.iloc[-1] > 95.0
    valid = r.dropna()
    assert ((valid >= 0.0) & (valid <= 100.0)).all()


def test_squeeze_uses_tighter_keltner():
    """The TTM squeeze band is 1.5x Keltner, so it fires strictly less often than
    a 2.0x band would (fewer, more meaningful compressions)."""
    rng = np.random.default_rng(3)
    n = 200
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    df = pd.DataFrame({
        "open": close, "high": close + 0.4, "low": close - 0.4,
        "close": close, "volume": rng.uniform(100, 200, n),
    })
    out = ind.compute_all(df)
    wide = (out["bb_up"] < out["kc_up"]) & (out["bb_low"] > out["kc_low"])  # 2.0x
    tight = out["squeeze_on"]                                               # 1.5x
    assert int(tight.sum()) <= int(wide.sum())


def test_numeric_context_is_stationary_across_price_levels():
    """The continuous ML context normalizes price-scaled features (distance-to-MA
    in ATRs), so the same shape at price 100 vs 10000 yields ~equal values."""
    rng = np.random.default_rng(9)
    close = 100 + np.cumsum(rng.normal(0, 0.4, 250))
    vol = rng.uniform(100, 200, len(close))

    def frame(scale):
        c = close * scale
        return pd.DataFrame({"open": c, "high": c + 0.4 * scale,
                             "low": c - 0.4 * scale, "close": c, "volume": vol})

    lo = ind.numeric_context(ind.compute_all(frame(1.0)))
    hi = ind.numeric_context(ind.compute_all(frame(100.0)))

    assert "ind_rsi" in lo and "ind_dist_ema50_atr" in lo
    assert "atr_percentile" in lo and "ind_macd_hist_atr" in lo
    # Bounded oscillator identical; ATR-normalized distance scale-invariant.
    assert abs(lo["ind_rsi"] - hi["ind_rsi"]) < 1e-6
    assert abs(lo["ind_dist_ema50_atr"] - hi["ind_dist_ema50_atr"]) < 1e-3


# --------------------------------------------------------------------------- #
# Market structure
# --------------------------------------------------------------------------- #
def test_market_structure_detects_swing_pivots():
    """A clear series of higher highs/higher lows produces structure evidence."""
    n = 60
    base = np.linspace(100, 130, n)
    # Inject order-3 fractal swing lows/highs; keep the LAST pivot within the
    # freshness window (last 10 bars) so read_evidence emits it.
    # Swing low around index 12.
    base[9:16] = [104, 102, 98, 96, 99, 103, 108]
    # Swing high around index 22.
    base[19:26] = [108, 112, 114, 116, 113, 110, 108]
    # Final higher-low near the end (index 54) so the comparison is fresh.
    base[51:58] = [122, 119, 117, 116, 118, 121, 124]
    df = pd.DataFrame({
        "open": base,
        "high": base + 0.5,
        "low": base - 0.5,
        "close": base,
        "volume": np.ones(n) * 1000.0,
    })
    ev = market_structure.read_evidence(df)
    names = {e.name for e in ev}
    assert "swing_low" in names or "swing_high" in names
    assert any(n in names for n in ("higher_high", "higher_low", "lower_high", "lower_low"))
    for e in ev:
        assert e.family == "structure"
        assert 0.0 <= e.strength <= 1.0


def test_market_structure_empty_for_short_frame():
    df = pd.DataFrame({
        "open": [1.0], "high": [1.1], "low": [0.9],
        "close": [1.0], "volume": [100.0],
    })
    assert market_structure.read_evidence(df) == []


# --------------------------------------------------------------------------- #
# Volume-confirmed breakout
# --------------------------------------------------------------------------- #
def test_volume_breakout_up():
    n = 40
    close = np.linspace(100.0, 119.0, n)
    high = close + 0.5
    low = close - 0.5
    volume = np.full(n, 1000.0)
    # Force a 20-bar Donchian break on the last bar with heavy volume.
    close[-1] = 130.0
    high[-1] = 130.5
    volume[-1] = 5000.0
    df = pd.DataFrame({
        "open": close - 0.1, "high": high, "low": low,
        "close": close, "volume": volume,
    })
    ev = volume_breakout.read_evidence(df)
    up = [e for e in ev if e.name == "volume_confirmed_breakout_up"]
    assert up
    assert up[0].bias == Bias.BULLISH
    assert up[0].family == "volume"


def test_volume_breakout_requires_volume():
    n = 40
    close = np.linspace(100.0, 119.0, n)
    close[-1] = 130.0
    df = pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 1000.0),
    })
    assert not volume_breakout.read_evidence(df)


# --------------------------------------------------------------------------- #
# Derivatives rank
# --------------------------------------------------------------------------- #
def test_funding_oi_rank_emits_extremes():
    rng = np.random.default_rng(123)
    base = pd.Series(rng.normal(0.0, 1.0, 60))
    funding = base.copy()
    funding.iloc[-1] = 5.0   # extreme high -> extreme_long (bearish)
    oi = base.copy()
    oi.iloc[-1] = -5.0       # extreme low -> extreme_short (bearish)
    ev = derivatives.funding_oi_rank_evidence(funding, oi)
    names = {e.name for e in ev}
    assert "funding_extreme_long" in names
    assert "oi_extreme_short" in names
    for e in ev:
        assert e.family == "derivatives"


def test_derivatives_synthetic_from_ohlcv(ohlcv_frame):
    funding, oi = derivatives.synthetic_from_ohlcv(ohlcv_frame)
    assert len(funding) == len(ohlcv_frame)
    assert len(oi) == len(ohlcv_frame)
    ev = derivatives.funding_oi_rank_evidence(funding, oi)
    # Synthetic proxies are well-formed enough to produce evidence or safely [] if
    # no extremes. Either way no exception is raised.
    assert isinstance(ev, list)


# --------------------------------------------------------------------------- #
# Cross-asset context
# --------------------------------------------------------------------------- #
def test_btc_regime_evidence(ohlcv_frame):
    ev = cross_asset.btc_regime_evidence(ohlcv_frame)
    assert ev.family == "trend"
    assert ev.bias in (Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL)


def test_eth_btc_spread_evidence():
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    btc = 40000.0 + np.cumsum(np.random.default_rng(7).normal(0, 50, n))
    eth = 0.055 * btc + np.cumsum(np.random.default_rng(8).normal(0, 2, n))
    btc_df = pd.DataFrame({
        "open": btc, "high": btc + 10, "low": btc - 10,
        "close": btc, "volume": np.ones(n) * 1000.0,
    }, index=idx)
    eth_df = pd.DataFrame({
        "open": eth, "high": eth + 1, "low": eth - 1,
        "close": eth, "volume": np.ones(n) * 1000.0,
    }, index=idx)
    ev = cross_asset.eth_btc_spread_evidence(eth_df, btc_df)
    for e in ev:
        assert e.family == "relative_strength"
        assert e.name in ("eth_btc_bullish", "eth_btc_bearish")


def test_sector_strength_evidence():
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    btc = np.linspace(100.0, 110.0, n)
    alt = np.linspace(100.0, 106.0, n)
    # Make the alt coin materially out-perform BTC in the last 21 bars.
    alt[-21:] = np.linspace(alt[-21], 130.0, 21)
    btc_df = pd.DataFrame({
        "open": btc, "high": btc + 0.5, "low": btc - 0.5,
        "close": btc, "volume": np.ones(n),
    }, index=idx)
    alt_df = pd.DataFrame({
        "open": alt, "high": alt + 0.5, "low": alt - 0.5,
        "close": alt, "volume": np.ones(n),
    }, index=idx)
    ev = cross_asset.sector_strength_evidence(alt_df, btc_df)
    assert any(e.name == "sector_outperform" for e in ev)
    for e in ev:
        assert e.family == "relative_strength"
