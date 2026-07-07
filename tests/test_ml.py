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


def test_build_matrix_and_signal_row_include_numeric_indicators():
    """Continuous indicator values (ind_* + percentiles) flow into the feature
    matrix at train time and are filled from numeric_context at inference."""
    ds = pd.DataFrame({
        "win": [0, 1, 0, 1, 0, 1],
        "confidence": [0.5, 0.6, 0.7, 0.8, 0.55, 0.62],
        "score": [1.0, -1.0, 2.0, -2.0, 0.5, 1.0],
        "n_families": [1, 2, 3, 4, 2, 3], "n_factors": [1, 2, 3, 4, 3, 2],
        "rule_win_rate": [0.5] * 6, "stop_pct": [0.02] * 6,
        "side": [1, -1, 1, -1, 1, -1],
        "kind": ["breakout"] * 6, "regime": ["trend_up"] * 6, "tf": ["1h"] * 6,
        "entry_time": pd.date_range("2024-01-01", periods=6, freq="h"),
        "atr_percentile": [0.3] * 6, "volatility_percentile": [0.4] * 6,
        "ind_rsi": [55, 60, 65, 40, 50, 58],
        "ind_dist_ema50_atr": [0.5, -0.2, 1.0, -1.5, 0.1, 0.3],
        "factor__ema_stack_bull": [0.8, 0, 0.9, 0, 0.7, 0.6],
    })
    X, y = features.build_matrix(ds)
    cols = set(X.columns)
    assert {"ind_rsi", "ind_dist_ema50_atr", "atr_percentile",
            "volatility_percentile"} <= cols

    row = features.signal_row(
        list(X.columns), evidence=[], confidence=0.7, score=1.0, n_families=2,
        rule_win_rate=0.55, stop_pct=0.02, side_sign=1, kind="breakout",
        regime="trend_up", timeframe="1h",
        numeric_context={"ind_rsi": 62.0, "ind_dist_ema50_atr": 0.9,
                         "atr_percentile": 0.7, "volatility_percentile": 0.5})
    assert list(row.columns) == list(X.columns)
    assert row["ind_rsi"].iloc[0] == pytest.approx(62.0)
    assert row["ind_dist_ema50_atr"].iloc[0] == pytest.approx(0.9)
    assert row["atr_percentile"].iloc[0] == pytest.approx(0.7)


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


def test_training_is_time_ordered_regardless_of_input_order(tmp_path, strong_dataset):
    """Training must be by TIME, never row order — a shuffle would leak future
    outcomes into training. train() re-sorts by entry_time internally, so a
    shuffled-on-disk dataset must yield IDENTICAL metrics to a pre-sorted one.
    """
    sorted_ds = strong_dataset.sort_values("entry_time").reset_index(drop=True)
    shuffled_ds = strong_dataset.sample(frac=1.0, random_state=3).reset_index(drop=True)
    p1, p2 = tmp_path / "sorted.parquet", tmp_path / "shuffled.parquet"
    o1, o2 = tmp_path / "a.pkl", tmp_path / "b.pkl"
    sorted_ds.to_parquet(p1)
    shuffled_ds.to_parquet(p2)

    m1 = train.train(str(p1), str(o1), min_trades=10)
    m2 = train.train(str(p2), str(o2), min_trades=10)

    # Order-invariance == no dependence on shuffle == no leakage from row order.
    assert m1["auc_valid"] == m2["auc_valid"]
    assert m1["brier_valid"] == m2["brier_valid"]
    assert m1["n_train"] == m2["n_train"]
    assert m1["n_valid"] == m2["n_valid"]


def test_isotonic_calibration_and_brier_gate(tmp_path, strong_dataset):
    """A skillful model is calibrated, beats the base-rate Brier, and is trusted."""
    ds_path = tmp_path / "ds.parquet"
    out_path = tmp_path / "model.pkl"
    strong_dataset.to_parquet(ds_path)

    meta = train.train(str(ds_path), str(out_path), min_trades=10)
    assert meta["calibrated"] is True
    assert meta["brier_valid"] < meta["brier_baseline"], "should have calibration skill"
    assert meta["n_folds"] >= 1

    mm = MetaModel.load(str(out_path), min_training_trades=10)
    assert mm is not None and mm.weight > 0, "skillful, calibrated model must be trusted"


def test_reward_head_trains_persists_and_loads(tmp_path):
    """The E[R] reward head is trained on a dataset with an 'r' column, evaluated
    OOS by Spearman rank-skill, persisted, and exposed via MetaModel."""
    import pickle
    rng = np.random.default_rng(21)
    n = 400
    score = rng.normal(0, 1, n)
    r = score * 1.5 + rng.normal(0, 0.5, n)   # realized R strongly tied to score
    win = (r > 0).astype(int)
    ds = pd.DataFrame({
        "win": win, "r": r,
        "confidence": rng.uniform(0.3, 0.9, n), "score": score,
        "n_families": rng.integers(2, 5, n), "n_factors": rng.integers(2, 6, n),
        "rule_win_rate": rng.uniform(0.4, 0.6, n), "stop_pct": rng.uniform(0.005, 0.05, n),
        "side": rng.choice([1, -1], n),
        "kind": rng.choice(
            ["reversal", "continuation", "breakout", "momentum", "mean_reversion"], n),
        "regime": rng.choice(["trend_up", "trend_down", "range", "volatile"], n),
        "tf": rng.choice(["1h", "4h", "1d"], n),
        "entry_time": pd.date_range("2024-01-01", periods=n, freq="h"),
        "ind_rsi": rng.uniform(20, 80, n),
        "factor__ema_stack_bull": rng.uniform(0, 1, n),
    })
    ds_path = tmp_path / "rew.parquet"
    out = tmp_path / "m.pkl"
    ds.to_parquet(ds_path)

    meta = train.train(str(ds_path), str(out), min_trades=10)
    assert meta["has_reward_head"] is True
    assert meta["reward_spearman"] > 0.1
    assert meta["reward_trusted"] is True

    blob = pickle.load(open(out, "rb"))
    assert blob["reward_model"] is not None

    mm = MetaModel.load(str(out), min_training_trades=10)
    assert mm is not None and mm.reward_available is True
