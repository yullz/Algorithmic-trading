"""Train the meta-model on backtester trade outcomes.

    python backtest.py --export-dataset          # writes reports/dataset.parquet
    python -m algotrader.ml.train                # trains models/meta_model.pkl

Honesty rules baked in:
  * TIME-ORDERED split only (last 25% of trades are the validation set).
    Shuffled CV on overlapping market data leaks the future into training.
  * The saved metadata carries n_train and validation AUC; MetaModel derives
    its blend weight from those and refuses to participate when the model has
    not demonstrated out-of-sample skill (AUC <= 0.53) or has too few trades.
  * The model can only re-rank rule-approved setups — it cannot invent trades,
    and the blended win rate stays inside the global 30–78% cap.
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

# Current default hyperparameters for the meta-model.
DEFAULT_HYPERPARAMS = {
    "max_depth": 4,
    "learning_rate": 0.08,
    "max_iter": 250,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "random_state": 7,
}


def train(dataset_path: str = "reports/dataset.parquet",
          out_path: str = "models/meta_model.pkl",
          min_trades: int = 300) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import brier_score_loss, roc_auc_score

    ds = pd.read_parquet(dataset_path)
    if LABEL_COL not in ds.columns or len(ds) < 50:
        raise SystemExit(f"dataset too small ({len(ds)} rows) — run a longer/deeper "
                         f"backtest first (backtest.py --deep --export-dataset)")

    # Time order across the whole pool (ISO strings sort correctly).
    ds = ds.sort_values("entry_time").reset_index(drop=True)
    X, y = build_matrix(ds)
    split = int(len(ds) * 0.75)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_va, y_va = X.iloc[split:], y.iloc[split:]

    base_rate = float(y.mean())
    class_weight = "balanced" if base_rate < 0.4 or base_rate > 0.6 else None

    model = HistGradientBoostingClassifier(
        **DEFAULT_HYPERPARAMS, class_weight=class_weight)
    model.fit(X_tr, y_tr)

    p_va = model.predict_proba(X_va)[:, 1]
    auc = float(roc_auc_score(y_va, p_va)) if y_va.nunique() > 1 else 0.5
    brier = float(brier_score_loss(y_va, p_va))

    feature_names = list(X.columns)
    if hasattr(model, "feature_importances_"):
        feature_importances = [float(v) for v in model.feature_importances_]
    else:
        # HistGradientBoostingClassifier does not expose native feature
        # importances in this scikit-learn build; fall back to permutation
        # importance on the validation set (honest, albeit slower).
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(
            model, X_va, y_va, n_repeats=5, random_state=7,
            scoring="neg_brier_score")
        feature_importances = [float(v) for v in pi.importances_mean]

    schema_version = FEATURE_LOGIC_VERSION

    meta = {
        "feature_columns": list(X.columns),
        "schema_version": schema_version,
        "feature_names": feature_names,
        "feature_importances_": feature_importances,
        "n_train": int(split), "n_valid": int(len(ds) - split),
        "auc_valid": round(auc, 4), "brier_valid": round(brier, 4),
        "base_rate": round(base_rate, 4),
        "trained_at": utcnow_iso(),
        "min_trades": min_trades,
        "trusted": bool(split >= min_trades and auc > 0.53),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "meta": meta}, f)

    log.info("trained on %d, validated on %d: AUC=%.3f Brier=%.3f base=%.0f%% -> %s",
             meta["n_train"], meta["n_valid"], auc, brier, base_rate * 100, out_path)
    if not meta["trusted"]:
        log.warning("model NOT trusted (needs >=%d training trades and OOS AUC>0.53; "
                    "got n=%d, AUC=%.3f) — the live blend will ignore it until a "
                    "bigger/better backtest retrains it", min_trades, split, auc)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="reports/dataset.parquet")
    ap.add_argument("--out", default="models/meta_model.pkl")
    ap.add_argument("--min-trades", type=int, default=300)
    args = ap.parse_args()
    meta = train(args.dataset, args.out, args.min_trades)
    print("\n===== META-MODEL =====")
    for k, v in meta.items():
        if k not in ("feature_columns", "feature_names", "feature_importances_"):
            print(f"  {k:14s} {v}")


if __name__ == "__main__":
    main()
