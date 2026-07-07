"""Feature construction shared by training and inference.

The training rows come from BacktestResult.to_dataset() (one row per simulated
trade). At inference the SAME feature vector is rebuilt from the live signal
context, aligned to the exact column list frozen at training time — any factor
the model never saw is silently dropped, any missing one is zero.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Optional

import pandas as pd

from ..models import Evidence
from ..signals.confluence import family_of

# Numeric context columns present in the dataset (see backtest.engine.to_dataset).
# volatility_percentile and atr_percentile are optional symbol-context numerics.
CONTEXT_COLS = ("confidence", "score", "n_families", "n_factors",
                "rule_win_rate", "stop_pct", "side",
                "volatility_percentile", "atr_percentile")
# Categorical columns one-hot encoded as {col}__{value}.
CATEGORICAL = ("kind", "regime", "tf")
LABEL_COL = "win"

# Bump this whenever build_matrix's feature-engineering LOGIC changes in a way
# that makes a previously-trained model's inputs incompatible. The trained model
# stores this string as its schema_version and MetaModel.load() refuses to load a
# model built under a different version. Crucially, the exact column SET is
# allowed to differ between runs — a real 150-symbol backtest legitimately
# produces dozens of factor__ columns that a small probe never would — because
# signal_row() aligns every inference row to the model's *persisted*
# feature_columns and zero-fills anything unseen. So only a logic change is
# fatal, not a larger/smaller universe. (The old guard hashed the column set from
# a 2-factor synthetic probe, which could essentially never match a real model,
# silently disabling the meta-model forever.)
FEATURE_LOGIC_VERSION = "2"


def _trend_family_strength(ds: pd.DataFrame) -> pd.Series:
    """Aggregate trend-family factor strength (max of trend evidence)."""
    trend_cols = [
        c for c in ds.columns
        if c.startswith("factor__") and family_of(c.removeprefix("factor__")) == "trend"
    ]
    if trend_cols:
        return ds[trend_cols].max(axis=1).fillna(0.0)
    return pd.Series(0.0, index=ds.index)


def _parse_entry_time(ds: pd.DataFrame) -> pd.Series:
    """Best-effort parse of signal timestamp for cyclical features."""
    if "entry_time" not in ds.columns:
        return pd.Series(pd.NaT, index=ds.index)
    return pd.to_datetime(ds["entry_time"], errors="coerce")


def _add_interactions(X: pd.DataFrame, ds: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and engineered features after base columns are built."""
    # Trend-family strength × regime dummy
    trend_family = _trend_family_strength(ds)
    regime_cols = [c for c in X.columns if c.startswith("regime__")]
    for c in regime_cols:
        regime = c.removeprefix("regime__")
        X[f"trend_family_x_regime__{regime}"] = trend_family * X[c]

    # Volatility percentile × setup-kind dummy
    if "volatility_percentile" in ds.columns:
        vol_pct = ds["volatility_percentile"].astype(float).fillna(0.0)
    else:
        vol_pct = pd.Series(0.0, index=ds.index)
    kind_cols = [c for c in X.columns if c.startswith("kind__")]
    for c in kind_cols:
        kind = c.removeprefix("kind__")
        X[f"vol_pct_x_kind__{kind}"] = vol_pct * X[c]

    # n_families × confidence
    if "n_families" in X.columns and "confidence" in X.columns:
        X["n_families_x_confidence"] = X["n_families"] * X["confidence"]

    # Hour-of-day and day-of-week of signal timestamp
    ts = _parse_entry_time(ds)
    if ts.notna().any():
        X["signal_hour"] = ts.dt.hour.astype(float).fillna(0.0)
        X["signal_dow"] = ts.dt.dayofweek.astype(float).fillna(0.0)

    return X


def build_matrix(ds: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Dataset -> (X, y) with stable, sorted feature columns."""
    y = ds[LABEL_COL].astype(int)
    X = ds[[c for c in CONTEXT_COLS if c in ds.columns]].astype(float).copy()
    for col in CATEGORICAL:
        if col in ds.columns:
            dummies = pd.get_dummies(ds[col].astype(str), prefix=col,
                                     prefix_sep="__", dtype=float)
            X = pd.concat([X, dummies], axis=1)
    factor_cols = sorted(c for c in ds.columns if c.startswith("factor__"))
    if factor_cols:
        X = pd.concat([X, ds[factor_cols].astype(float)], axis=1)
    # Continuous, normalized indicator values (ind_rsi, ind_dist_ema50_atr, ...).
    ind_cols = sorted(c for c in ds.columns if c.startswith("ind_"))
    if ind_cols:
        X = pd.concat([X, ds[ind_cols].astype(float)], axis=1)
    X = _add_interactions(X, ds)
    return X[sorted(X.columns)], y


def compute_schema_hash(columns: list[str],
                        hyperparams: Optional[dict] = None) -> str:
    """Deterministic hash of the feature columns + model hyperparameters."""
    payload = {
        "columns": sorted(columns),
        "hyperparams": {k: v for k, v in sorted((hyperparams or {}).items())
                        if v is not None},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()




def signal_row(feature_columns: list[str], *, evidence: list[Evidence],
               confidence: float, score: float, n_families: int,
               rule_win_rate: float, stop_pct: float, side_sign: int,
               kind: str, regime: str, timeframe: str,
               volatility_percentile: float = 0.0,
               atr_percentile: float = 0.0,
               numeric_context: Optional[dict] = None,
               entry_time: Optional[str | datetime] = None) -> pd.DataFrame:
    """One inference row aligned to the training columns (missing -> 0)."""
    row: dict[str, float] = {c: 0.0 for c in feature_columns}

    def put(col: str, val: float) -> None:
        if col in row:
            row[col] = float(val)

    put("confidence", confidence)
    put("score", score)
    put("n_families", n_families)
    put("n_factors", len(evidence))
    put("rule_win_rate", rule_win_rate)
    put("stop_pct", stop_pct)
    put("side", side_sign)
    put("volatility_percentile", volatility_percentile)
    put("atr_percentile", atr_percentile)
    # Continuous indicator values (ind_* and the two percentiles). Any column the
    # model never saw is ignored; any the model expects but is absent stays 0.
    for _k, _v in (numeric_context or {}).items():
        put(_k, _v)
    put(f"kind__{kind}", 1.0)
    put(f"regime__{regime}", 1.0)
    put(f"tf__{timeframe}", 1.0)
    for e in evidence:
        put(f"factor__{e.name}", e.strength)

    # Engineered interaction features (must match build_matrix).
    trend_family = max(
        (e.strength for e in evidence if family_of(e) == "trend"),
        default=0.0,
    )
    put(f"trend_family_x_regime__{regime}",
        trend_family * row.get(f"regime__{regime}", 0.0))
    put(f"vol_pct_x_kind__{kind}",
        volatility_percentile * row.get(f"kind__{kind}", 0.0))
    put("n_families_x_confidence", n_families * confidence)

    if entry_time is not None:
        try:
            ts = pd.to_datetime(entry_time)
            put("signal_hour", float(ts.hour))
            put("signal_dow", float(ts.dayofweek))
        except Exception:
            pass

    return pd.DataFrame([row], columns=feature_columns)
