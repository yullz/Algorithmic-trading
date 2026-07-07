"""Inference wrapper the SignalEngine plugs into (engine.meta_model).

MetaModel.predict_for_signal returns (probability, blend_weight, contribs) or
None when the model should stay out of the decision. The blend weight is
earned, not assumed:

    weight = min(0.5, (auc_valid - 0.53) * 5) * min(1, n_train / 2000)

so an untrained or unskilled model contributes nothing, and even a strong one
never outweighs the calibrated rules (cap 0.5). Contributions are computed by
zeroing each active factor and measuring the probability delta — crude but
honest, and it names the factors the model actually leaned on.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np

from ..models import Evidence
from ..utils.logging import get_logger
from .features import FEATURE_LOGIC_VERSION, signal_row

log = get_logger("ml.predict")


class MetaModel:
    def __init__(self, model, meta: dict, reward_model=None):
        self.model = model
        self.meta = meta
        self.reward_model = reward_model
        # The reward (E[R]) head is used for ranking only when it earned OOS skill.
        self.reward_available = bool(reward_model is not None
                                     and meta.get("reward_trusted"))
        auc = float(meta.get("auc_valid", 0.5))
        brier = float(meta.get("brier_valid", 1.0))
        brier_baseline = float(meta.get("brier_baseline", 1.0))
        n_train = int(meta.get("n_train", 0))
        min_trades = int(meta.get("min_trades", 300))
        # Trust is earned from OOS ranking skill (AUC) AND OOS calibration skill
        # (Brier must beat a constant base-rate predictor). A high-AUC but
        # miscalibrated model would otherwise size positions off distorted
        # probabilities, since predict_proba is consumed literally in the blend.
        if n_train < min_trades or auc <= 0.53 or brier >= brier_baseline:
            self.weight = 0.0
        else:
            self.weight = min(0.5, (auc - 0.53) * 5.0) * min(1.0, n_train / 2000.0)

    @classmethod
    def load(cls, path: str = "models/meta_model.pkl",
             min_training_trades: int | None = None) -> Optional["MetaModel"]:
        """None when missing/unreadable or schema-drifted — engine runs rules-only."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                blob = pickle.load(f)
            model = blob["model"]
            meta = blob["meta"]
            reward_model = blob.get("reward_model")

            # Feature-logic guard. A real 150-symbol model legitimately has a
            # different (larger) column set than any small probe, so we do NOT
            # compare column hashes — signal_row() aligns inference rows to the
            # persisted feature_columns and zero-fills unseen columns. We only
            # reject a model built under an incompatible feature-engineering
            # LOGIC version, or one missing its persisted feature_columns.
            model_version = meta.get("schema_version")
            if model_version != FEATURE_LOGIC_VERSION:
                log.warning("meta-model feature-logic mismatch (expected %s, got %s) — "
                            "running rules-only", FEATURE_LOGIC_VERSION, model_version)
                return None
            if not meta.get("feature_columns"):
                log.warning("meta-model missing feature_columns — running rules-only")
                return None

            if min_training_trades is not None:
                meta["min_trades"] = min_training_trades
            mm = cls(model, meta, reward_model=reward_model)
            if mm.weight <= 0:
                log.warning("meta-model loaded but not trusted (n=%s, AUC=%s) — "
                            "running rules-only", meta.get("n_train"),
                            meta.get("auc_valid"))
                return None
            log.info("meta-model active: AUC=%.3f n=%d blend weight=%.0f%%",
                     meta.get("auc_valid", 0.5), meta.get("n_train", 0),
                     mm.weight * 100)
            return mm
        except Exception as e:  # corrupt pickle, sklearn version drift, ...
            log.warning("failed to load meta-model (%s) — running rules-only", e)
            return None

    # ------------------------------------------------------------------ #
    def predict_for_signal(self, *, evidence: list[Evidence], agg: dict,
                           regime: str, timeframe: str, rule_win_rate: float,
                           stop_pct: float, side_sign: int,
                           volatility_percentile: float = 0.0,
                           atr_percentile: float = 0.0,
                           numeric_context: Optional[dict] = None,
                           entry_time: Optional[str] = None,
                           ) -> Optional[tuple[float, float, list[str], Optional[float]]]:
        if self.weight <= 0:
            return None
        cols = self.meta["feature_columns"]
        row = signal_row(
            cols, evidence=evidence, confidence=agg["confidence"],
            score=agg["score"], n_families=agg["n_families"],
            rule_win_rate=rule_win_rate, stop_pct=stop_pct,
            side_sign=side_sign, kind=agg["kind"].value, regime=regime,
            timeframe=timeframe,
            volatility_percentile=volatility_percentile,
            atr_percentile=atr_percentile,
            numeric_context=numeric_context,
            entry_time=entry_time)
        try:
            prob = float(self.model.predict_proba(row)[0, 1])
        except Exception as e:
            log.warning("meta-model predict failed (%s) — skipping", e)
            return None
        ev_r = None
        if self.reward_available and self.reward_model is not None:
            try:
                ev_r = float(self.reward_model.predict(row)[0])
            except Exception:  # pragma: no cover - defensive
                ev_r = None
        contribs = self._contributions(row, prob)
        return prob, self.weight, contribs, ev_r

    def drift_score(self, recent_predictions: list[float],
                    recent_outcomes: list[int]) -> Optional[float]:
        """Brier-skill-score degradation vs the validation Brier score.

        Returns 1 - Brier_recent / Brier_valid.  Positive means the model is
        still beating its validation score; negative means degradation.
        """
        from sklearn.metrics import brier_score_loss
        if (not recent_predictions or not recent_outcomes
                or len(recent_predictions) != len(recent_outcomes)):
            return None
        y_true = np.asarray(recent_outcomes)
        y_prob = np.asarray(recent_predictions)
        if len(np.unique(y_true)) < 2:
            return None
        recent_brier = float(brier_score_loss(y_true, y_prob))
        ref_brier = float(self.meta.get("brier_valid", 0.25))
        if ref_brier <= 0:
            return None
        return 1.0 - recent_brier / ref_brier

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Most important features by stored feature_importances_."""
        importances = self.meta.get("feature_importances_")
        names = self.meta.get("feature_names")
        if not importances or not names or len(importances) != len(names):
            return []
        ranked = sorted(zip(names, importances), key=lambda x: -x[1])
        return ranked[:n]

    def _contributions(self, row, prob: float, top: int = 3) -> list[str]:
        """Zero-out deltas on the active factor features (max 15 checks)."""
        active = [c for c in row.columns
                  if c.startswith("factor__") and row.iloc[0][c] != 0.0][:15]
        deltas: list[tuple[str, float]] = []
        for c in active:
            probe = row.copy()
            probe.iloc[0, probe.columns.get_loc(c)] = 0.0
            try:
                p2 = float(self.model.predict_proba(probe)[0, 1])
            except Exception:
                continue
            deltas.append((c.removeprefix("factor__"), prob - p2))
        deltas.sort(key=lambda x: -abs(x[1]))
        return [f"{name} ({delta:+.02f})" for name, delta in deltas[:top]]
