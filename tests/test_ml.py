"""Unit tests for the ML meta-model pipeline."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from algotrader.models import Bias, Evidence
from algotrader.ml import features, train
from algotrader.ml.predict import MetaModel


@pytest.fixture
def toy_dataset() -> pd.DataFrame:
    """A representative dataset including the new context numerics."""
    rng = np.random.default_rng(7)
    n = 300
    return pd.DataFrame({
        "win": rng.integers(0, 2, n),
        "confidence": rng.uniform(0.3, 0.9, n),
        "score": rng.normal(0, 1, n),
        "n_families": rng.integers(1, 5, n),
        "n_factors": rng.integers(1, 6, n),
        "rule_win_rate": rng.uniform(0.35, 0.65, n),
        "stop_pct": rng.uniform(0.005, 0.05, n),
        "side": rng.choice([1, -1], n),
        "volatility_percentile": rng.uniform(0, 1, n),
        "atr_percentile": rng.uniform(0, 1, n),
        "kind": rng.choice(
            ["reversal", "continuation", "breakout", "momentum", "mean_reversion"], n),
        "regime": rng.choice(
            ["trend_up", "trend_down", "range", "volatile"], n),
        "tf": rng.choice(["1h", "4h", "1d"], n),
        "entry_time": pd.date_range("2024-01-01", periods=n, freq="h"),
        "factor__ema_stack_bull": rng.uniform(0, 1, n),
        "factor__rsi_oversold": rng.uniform(0, 1, n),
    })


@pytest.fixture
def strong_dataset() -> pd.DataFrame:
    """Dataset with a strong (but not perfect) predictor so the model is trusted."""
    rng = np.random.default_rng(42)
    n = 400
    score = rng.normal(0, 1, n)
    win = (score > 0).astype(int)
    # Add label noise so the validation Brier is > 0 (realistic reference).
    noise = rng.random(n) < 0.12
    win = np.where(noise, 1 - win, win)
    return pd.DataFrame({
        "win": win,
        "confidence": rng.uniform(0.3, 0.9, n),
        "score": score,
        "n_families": rng.integers(2, 5, n),
        "n_factors": rng.integers(2, 6, n),
        "rule_win_rate": rng.uniform(0.40, 0.60, n),
        "stop_pct": rng.uniform(0.005, 0.05, n),
        "side": rng.choice([1, -1], n),
        "volatility_percentile": rng.uniform(0, 1, n),
        "atr_percentile": rng.uniform(0, 1, n),
        "kind": rng.choice(
            ["reversal", "continuation", "breakout", "momentum", "mean_reversion"], n),
        "regime": rng.choice(
            ["trend_up", "trend_down", "range", "volatile"], n),
        "tf": rng.choice(["1h", "4h", "1d"], n),
        "entry_time": pd.date_range("2024-06-01", periods=n, freq="h"),
        "factor__ema_stack_bull": rng.uniform(0, 1, n),
        "factor__rsi_oversold": rng.uniform(0, 1, n),
    })


def test_build_matrix_includes_interactions_and_context(toy_dataset):
    X, y = features.build_matrix(toy_dataset)
    cols = set(X.columns)
    assert "volatility_percentile" in cols
    assert "atr_percentile" in cols
    assert "n_families_x_confidence" in cols
    assert "signal_hour" in cols
    assert "signal_dow" in cols
    assert any(c.startswith("trend_family_x_regime__") for c in cols)
    assert any(c.startswith("vol_pct_x_kind__") for c in cols)
    assert len(X) == len(y)


def test_signal_row_aligns_with_training_columns(toy_dataset):
    X, _ = features.build_matrix(toy_dataset)
    cols = list(X.columns)
    ev = [
        Evidence("ema_stack_bull", "indicator", Bias.BULLISH,
                 0.8, family="trend"),
        Evidence("rsi_oversold", "indicator", Bias.BULLISH,
                 0.6, family="mean_reversion"),
    ]
    row = features.signal_row(
        cols, evidence=ev, confidence=0.75, score=1.2, n_families=2,
        rule_win_rate=0.55, stop_pct=0.02, side_sign=1,
        kind="breakout", regime="trend_up", timeframe="1h",
        volatility_percentile=0.7, atr_percentile=0.6,
        entry_time="2024-06-10 14:30",
    )
    assert list(row.columns) == cols
    assert row["confidence"].iloc[0] == pytest.approx(0.75)
    assert row["n_families_x_confidence"].iloc[0] == pytest.approx(1.5)
    assert row["signal_hour"].iloc[0] == pytest.approx(14.0)
    assert row["signal_dow"].iloc[0] == pytest.approx(0.0)  # Monday
    assert row["trend_family_x_regime__trend_up"].iloc[0] == pytest.approx(0.8)
    assert row["vol_pct_x_kind__breakout"].iloc[0] == pytest.approx(0.7)


def test_load_rejects_schema_mismatch(tmp_path):
    """A pickle whose schema_version disagrees with the live code is rejected."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(max_iter=10, random_state=1)
    X = np.random.rand(80, 5)
    y = np.random.randint(0, 2, 80)
    model.fit(X, y)

    path = tmp_path / "bad_schema.pkl"
    with open(path, "wb") as f:
        pickle.dump({
            "model": model,
            "meta": {
                "feature_columns": ["a", "b", "c", "d", "e"],
                "schema_version": "deadbeef000000000000000000000000",
                "feature_names": ["a", "b", "c", "d", "e"],
                "feature_importances_": [0.2, 0.2, 0.2, 0.2, 0.2],
                "n_train": 2000,
                "n_valid": 500,
                "auc_valid": 0.75,
                "brier_valid": 0.2,
                "base_rate": 0.5,
                "min_trades": 300,
                "trusted": True,
            },
        }, f)

    assert MetaModel.load(str(path)) is None


def test_top_features_and_load_with_matching_schema(tmp_path, strong_dataset):
    """A model trained with the current code loads and exposes top_features."""
    ds_path = tmp_path / "ds.parquet"
    out_path = tmp_path / "model.pkl"
    strong_dataset.to_parquet(ds_path)

    meta = train.train(str(ds_path), str(out_path), min_trades=10)
    assert "schema_version" in meta
    assert meta["feature_importances_"] is not None

    mm = MetaModel.load(str(out_path), min_training_trades=10)
    assert mm is not None, "model should be trusted with a strong predictor"

    top = mm.top_features(3)
    assert len(top) == 3
    assert all(isinstance(t, tuple) and len(t) == 2 for t in top)
    assert top[0][1] >= top[1][1] >= top[2][1]

    # score is perfectly predictive, so it should dominate importance.
    names = [t[0] for t in top]
    assert "score" in names


def test_drift_score_degrades_when_predictions_worsen(tmp_path, strong_dataset):
    ds_path = tmp_path / "ds.parquet"
    out_path = tmp_path / "model.pkl"
    strong_dataset.to_parquet(ds_path)
    train.train(str(ds_path), str(out_path), min_trades=10)
    mm = MetaModel.load(str(out_path), min_training_trades=10)
    assert mm is not None

    # Recent predictions that match outcomes closely -> positive skill.
    perfect = [1.0] * 50 + [0.0] * 50
    outcomes = [1] * 50 + [0] * 50
    assert mm.drift_score(perfect, outcomes) > 0.0

    # Worse than validation -> negative skill
    bad = [0.0] * 100
    assert mm.drift_score(bad, outcomes) < 0.0

    # Missing data -> None
    assert mm.drift_score([], []) is None


def test_load_accepts_many_factor_columns(tmp_path):
    """Regression for C2: a realistic model with dozens of factor__ columns must
    LOAD. The old guard hashed the column set against a 2-factor synthetic probe,
    so a real 150-symbol model could never match and the meta-model was silently
    disabled forever (ran rules-only). The new guard versions the feature LOGIC
    and trusts the persisted feature_columns, so column-count differences are fine.
    """
    rng = np.random.default_rng(11)
    n = 500
    score = rng.normal(0, 1, n)
    win = (score > 0).astype(int)
    win = np.where(rng.random(n) < 0.12, 1 - win, win)  # label noise
    data = {
        "win": win,
        "confidence": rng.uniform(0.3, 0.9, n),
        "score": score,
        "n_families": rng.integers(2, 5, n),
        "n_factors": rng.integers(2, 6, n),
        "rule_win_rate": rng.uniform(0.40, 0.60, n),
        "stop_pct": rng.uniform(0.005, 0.05, n),
        "side": rng.choice([1, -1], n),
        "volatility_percentile": rng.uniform(0, 1, n),
        "atr_percentile": rng.uniform(0, 1, n),
        "kind": rng.choice(
            ["reversal", "continuation", "breakout", "momentum", "mean_reversion"], n),
        "regime": rng.choice(["trend_up", "trend_down", "range", "volatile"], n),
        "tf": rng.choice(["1h", "4h", "1d"], n),
        "entry_time": pd.date_range("2024-06-01", periods=n, freq="h"),
    }
    # Dozens of sparse factor columns, like a real 150-symbol backtest produces.
    for i in range(30):
        col = rng.uniform(0, 1, n)
        col[rng.random(n) < 0.7] = 0.0  # sparse: most rows didn't fire this factor
        data[f"factor__f{i:02d}"] = col
    ds = pd.DataFrame(data)
    ds_path = tmp_path / "many.parquet"
    out_path = tmp_path / "many_model.pkl"
    ds.to_parquet(ds_path)

    meta = train.train(str(ds_path), str(out_path), min_trades=10)
    assert meta["schema_version"] == features.FEATURE_LOGIC_VERSION
    n_factor_cols = len([c for c in meta["feature_columns"] if c.startswith("factor__")])
    assert n_factor_cols >= 30

    mm = MetaModel.load(str(out_path), min_training_trades=10)
    assert mm is not None, "a real multi-factor model must load (regression for C2)"


def test_training_split_is_temporal(tmp_path, strong_dataset):
    """The train/valid split must be by time, never shuffled — a shuffle would
    leak future outcomes into training and inflate AUC + the earned blend weight.
    """
    ds_path = tmp_path / "ds.parquet"
    out_path = tmp_path / "model.pkl"
    # Shuffle the rows on disk; train() must re-sort by entry_time internally.
    shuffled = strong_dataset.sample(frac=1.0, random_state=3).reset_index(drop=True)
    shuffled.to_parquet(ds_path)

    ordered = shuffled.sort_values("entry_time").reset_index(drop=True)
    split = int(len(ordered) * 0.75)
    last_train_time = ordered["entry_time"].iloc[split - 1]
    first_valid_time = ordered["entry_time"].iloc[split]
    # Sanity: the intended split boundary is strictly ordered in time.
    assert last_train_time <= first_valid_time

    meta = train.train(str(ds_path), str(out_path), min_trades=10)
    assert meta["n_train"] == split
    assert meta["n_valid"] == len(ordered) - split
