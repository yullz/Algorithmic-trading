"""Train the meta-model on backtester trade outcomes.

    python backtest.py --export-dataset          # writes reports/dataset.parquet
    python -m algotrader.ml.train                # trains models/meta_model.pkl

Honesty rules baked in:
  * Evaluation is a PURGED, anchored walk-forward over time — never a shuffled
    split. Training rows whose label-outcome window overlaps a validation fold
    are purged (the label is realized over `horizon_bars` of the row's own
    timeframe), so the reported OOS AUC/Brier do not leak the future.
  * The deployed model is ISOTONICALLY CALIBRATED (CalibratedClassifierCV,
    cv='prefit') on a held-out time tail, so predict_proba is a real probability
    — it is consumed literally in log-odds blending and position sizing.
  * Trust is earned from OOS AUC *and* OOS Brier skill (must beat the base-rate
    Brier); an unskilled or miscalibrated model contributes 0 to the blend.
  * class_weight is NOT used: reweighting distorts the probability scale, and the
    probability is consumed literally downstream.
  * The model can only re-rank rule-approved setups — the blended win rate stays
    inside the global 30–78% cap.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from ..utils.logging import get_logger, utcnow_iso
from .features import FEATURE_LOGIC_VERSION, LABEL_COL, build_matrix

log = get_logger("ml.train")

# Minutes per timeframe, used to translate the bar-based label horizon into a
# wall-clock window for purging leakage across fold boundaries.
_TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
}

# Default hyperparameters. early_stopping is OFF: HGB's internal early-stopping
# split is drawn at RANDOM, which would leak future rows into the stop criterion
# on time-ordered market data.
DEFAULT_HYPERPARAMS = {
    "max_depth": 4,
    "learning_rate": 0.08,
    "max_iter": 250,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 7,
}


def _label_end_times(ds: pd.DataFrame, entry_times: pd.Series,
                     horizon_bars: int) -> pd.Series:
    """When each trade's outcome becomes known: entry_time + horizon_bars of its
    own timeframe. Used to purge training rows that leak across a fold boundary."""
    tf_col = ds.get("tf")
    if tf_col is None:
        minutes = pd.Series(60.0, index=ds.index)
    else:
        minutes = tf_col.map(_TF_MINUTES).fillna(60.0).astype(float)
    return entry_times + pd.to_timedelta(minutes * horizon_bars, unit="m")


def _purged_walk_forward(X: pd.DataFrame, y: pd.Series, entry_times: pd.Series,
                         label_end: pd.Series, n_folds: int,
                         hyperparams: dict) -> tuple[list[float], list[float]]:
    """Anchored, purged walk-forward. Returns (aucs, briers) per fold.

    Fold k trades the k-th time-slice out-of-sample; the training set is every
    row BEFORE the fold whose label window closes before the fold begins — so a
    trade whose outcome is only known during the validation window is purged.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import brier_score_loss, roc_auc_score

    n = len(X)
    start = int(n * 0.4)                      # first 40% is a training-only anchor
    seg = max(1, (n - start) // max(1, n_folds))
    positions = np.arange(n)
    aucs: list[float] = []
    briers: list[float] = []
    for k in range(n_folds):
        a = start + k * seg
        b = n if k == n_folds - 1 else start + (k + 1) * seg
        if a >= b or a >= n:
            continue
        valid_start_time = entry_times.iloc[a]
        # NaT label_end (unparseable time) -> comparison False -> row excluded.
        purged = (label_end < valid_start_time).to_numpy()
        train_mask = purged & (positions < a)
        yva = y.iloc[a:b]
        if train_mask.sum() < 30 or yva.nunique() < 2:
            continue
        m = HistGradientBoostingClassifier(**hyperparams)
        m.fit(X.iloc[train_mask], y.iloc[train_mask])
        p = m.predict_proba(X.iloc[a:b])[:, 1]
        aucs.append(float(roc_auc_score(yva, p)))
        briers.append(float(brier_score_loss(yva, p)))
    return aucs, briers


def train(dataset_path: str = "reports/dataset.parquet",
          out_path: str = "models/meta_model.pkl",
          min_trades: int = 300, horizon_bars: int = 48,
          n_folds: int = 4) -> dict:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance

    ds = pd.read_parquet(dataset_path)
    if LABEL_COL not in ds.columns or len(ds) < 50:
        raise SystemExit(f"dataset too small ({len(ds)} rows) — run a longer/deeper "
                         f"backtest first (backtest.py --deep --export-dataset)")

    # Time order across the whole pool (ISO strings sort correctly).
    ds = ds.sort_values("entry_time").reset_index(drop=True)
    X, y = build_matrix(ds)
    entry_times = pd.to_datetime(ds["entry_time"], errors="coerce", utc=True)
    label_end = _label_end_times(ds, entry_times, horizon_bars)

    base_rate = float(y.mean())
    # Brier of a constant base-rate predictor — the "no-skill" reference.
    brier_baseline = base_rate * (1.0 - base_rate)

    # --- honest OOS metrics via purged anchored walk-forward ---
    aucs, briers = _purged_walk_forward(
        X, y, entry_times, label_end, n_folds, DEFAULT_HYPERPARAMS)
    auc = float(np.mean(aucs)) if aucs else 0.5
    auc_std = float(np.std(aucs)) if aucs else 0.0
    brier = float(np.mean(briers)) if briers else float(brier_baseline)
    brier_std = float(np.std(briers)) if briers else 0.0

    # --- deployed model: fit on an early time block, isotonic-calibrate on the
    #     held-out time tail (cv='prefit' keeps calibration strictly out-of-fit) ---
    n = len(ds)
    cal_split = int(n * 0.80)
    positions = np.arange(n)
    valid_start_time = entry_times.iloc[cal_split]
    fit_mask = (label_end < valid_start_time).to_numpy() & (positions < cal_split)
    can_calibrate = fit_mask.sum() >= 20 and y.iloc[cal_split:].nunique() >= 2
    if not can_calibrate:
        fit_mask = positions < cal_split  # fall back to a plain time split

    base = HistGradientBoostingClassifier(**DEFAULT_HYPERPARAMS)
    base.fit(X.iloc[fit_mask], y.iloc[fit_mask])

    model = base
    if can_calibrate:
        try:
            try:
                # sklearn >= 1.6: wrap the fitted model so calibration does not refit it.
                from sklearn.frozen import FrozenEstimator
                calibrated = CalibratedClassifierCV(
                    FrozenEstimator(base), method="isotonic")
            except ImportError:  # sklearn < 1.6
                calibrated = CalibratedClassifierCV(
                    base, method="isotonic", cv="prefit")
            calibrated.fit(X.iloc[cal_split:], y.iloc[cal_split:])
            model = calibrated
        except Exception as e:  # pragma: no cover - defensive
            log.warning("isotonic calibration failed (%s); deploying uncalibrated base", e)

    # Feature importances from the base estimator (permutation on the held-out tail).
    feature_names = list(X.columns)
    try:
        pi = permutation_importance(
            base, X.iloc[cal_split:], y.iloc[cal_split:],
            n_repeats=5, random_state=7, scoring="neg_brier_score")
        feature_importances = [float(v) for v in pi.importances_mean]
    except Exception:  # pragma: no cover - defensive
        feature_importances = [0.0] * len(feature_names)

    n_train = int(fit_mask.sum())
    n_valid = int(n - cal_split)
    trusted = bool(n_train >= min_trades and auc > 0.53 and brier < brier_baseline)

    meta = {
        "feature_columns": list(X.columns),
        "schema_version": FEATURE_LOGIC_VERSION,
        "feature_names": feature_names,
        "feature_importances_": feature_importances,
        "n_train": n_train, "n_valid": n_valid, "n_folds": len(aucs),
        "auc_valid": round(auc, 4), "auc_std": round(auc_std, 4),
        "brier_valid": round(brier, 4), "brier_std": round(brier_std, 4),
        "brier_baseline": round(float(brier_baseline), 4),
        "base_rate": round(base_rate, 4),
        "calibrated": bool(model is not base),
        "trained_at": utcnow_iso(),
        "min_trades": min_trades,
        "trusted": trusted,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "meta": meta}, f)

    log.info("purged WF: AUC=%.3f±%.3f Brier=%.3f±%.3f (baseline %.3f) over %d folds; "
             "fit n=%d, calib n=%d, calibrated=%s -> %s",
             auc, auc_std, brier, brier_std, brier_baseline, len(aucs),
             n_train, n_valid, meta["calibrated"], out_path)
    if not trusted:
        log.warning("model NOT trusted (needs n_train>=%d, OOS AUC>0.53, OOS Brier<%.3f; "
                    "got n=%d, AUC=%.3f, Brier=%.3f) — the live blend will ignore it "
                    "until a bigger/better backtest retrains it.",
                    min_trades, brier_baseline, n_train, auc, brier)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="reports/dataset.parquet")
    ap.add_argument("--out", default="models/meta_model.pkl")
    ap.add_argument("--min-trades", type=int, default=300)
    ap.add_argument("--horizon-bars", type=int, default=48,
                    help="label horizon in bars (must match backtest label_horizon)")
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()
    meta = train(args.dataset, args.out, args.min_trades,
                 horizon_bars=args.horizon_bars, n_folds=args.folds)
    print("\n===== META-MODEL =====")
    for k, v in meta.items():
        if k not in ("feature_columns", "feature_names", "feature_importances_"):
            print(f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
