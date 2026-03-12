"""
Meta-label model (hedge fund technique).
Secondary model that predicts whether the primary ensemble signal should be trusted.
Only outputs predictions when meta-confidence is above threshold.
"""
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Tuple, Dict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from config import get_config

logger = logging.getLogger(__name__)
cfg = get_config()


class MetaLabelModel:
    """
    Two-layer prediction architecture.
    Layer 1: ensemble prediction
    Layer 2: meta-model decides if we should trust the signal
    """

    def __init__(self):
        meta_cfg = cfg.get("meta_model", {})
        self.confidence_threshold = meta_cfg.get("confidence_threshold", 0.60)
        self.min_meta_confidence = meta_cfg.get("min_meta_confidence", 0.55)
        self.model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self._fitted = False
        self._prediction_history: list = []
        self._accuracy_window = 20

    def _build_meta_features(
        self,
        ensemble: "EnsemblePredictor",
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Build meta-features for the secondary model.
        These capture signal quality, market context, and recent model performance.
        """
        proba = ensemble.predict_proba(X)
        prob_up = proba[:, 1]
        confidence = np.abs(prob_up - 0.5) * 2  # 0 = random, 1 = certain

        meta_feats = {}
        meta_feats["model_confidence"] = confidence
        meta_feats["prob_up"] = prob_up
        meta_feats["prob_extreme"] = ((prob_up > 0.7) | (prob_up < 0.3)).astype(float)

        # Volatility regime from features
        for col_key in ["5m_vol_regime", "5m_adx", "5m_bb_width"]:
            if col_key in X.columns:
                meta_feats[col_key] = X[col_key].values
            else:
                meta_feats[col_key] = np.zeros(len(X))

        # Distance from EMA
        for col_key in ["5m_dist_ema_20", "5m_dist_vwap"]:
            if col_key in X.columns:
                meta_feats[col_key] = X[col_key].abs().values
            else:
                meta_feats[col_key] = np.zeros(len(X))

        # Rolling accuracy from prediction history (lagged)
        recent_accuracy = np.full(len(X), 0.5)
        meta_feats["recent_accuracy"] = recent_accuracy

        return np.column_stack(list(meta_feats.values()))

    def fit(
        self,
        ensemble,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "MetaLabelModel":
        """
        Fit meta-model on validation set.
        Target: 1 if primary model was correct, 0 if wrong.
        """
        logger.info("Fitting meta-label model...")
        proba = ensemble.predict_proba(X_val)
        primary_pred = (proba[:, 1] >= 0.5).astype(int)
        y_arr = y_val.values if hasattr(y_val, "values") else np.array(y_val)

        min_len = min(len(primary_pred), len(y_arr))
        meta_target = (primary_pred[-min_len:] == y_arr[-min_len:]).astype(int)

        meta_features = self._build_meta_features(ensemble, X_val.iloc[-min_len:])
        meta_scaled = self.scaler.fit_transform(meta_features)

        self.model.fit(meta_scaled, meta_target)
        self._fitted = True

        train_acc = (self.model.predict(meta_scaled) == meta_target).mean()
        logger.info(f"Meta-model fitted. Train accuracy (is_correct): {train_acc:.4f}")
        return self

    def validate(
        self,
        ensemble,
        X: pd.DataFrame,
        primary_proba: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (should_trade, meta_confidence) arrays.
        should_trade = 1 if meta-model thinks signal is reliable.
        """
        if not self._fitted:
            n = len(X)
            return np.ones(n, dtype=int), np.ones(n) * 0.5

        meta_features = self._build_meta_features(ensemble, X)
        meta_scaled = self.scaler.transform(meta_features)
        meta_proba = self.model.predict_proba(meta_scaled)[:, 1]  # P(correct)

        primary_confidence = np.abs(primary_proba[:, 1] - 0.5) * 2
        should_trade = (
            (meta_proba >= self.min_meta_confidence) &
            (primary_confidence >= (self.confidence_threshold - 0.5) * 2)
        ).astype(int)

        return should_trade, meta_proba

    def validate_single(
        self,
        ensemble,
        X_single: pd.DataFrame,
        primary_proba: float,
    ) -> Dict:
        """Validate a single prediction point."""
        X = X_single if isinstance(X_single, pd.DataFrame) else pd.DataFrame([X_single])
        proba_2d = np.array([[1 - primary_proba, primary_proba]])
        should_trade, meta_conf = self.validate(ensemble, X, proba_2d)
        return {
            "should_trade": bool(should_trade[0]),
            "meta_confidence": float(meta_conf[0]),
        }

    def update_history(self, predicted: int, actual: int, confidence: float):
        """Update prediction history for rolling accuracy."""
        self._prediction_history.append({
            "predicted": predicted,
            "actual": actual,
            "correct": int(predicted == actual),
            "confidence": confidence,
        })
        if len(self._prediction_history) > 200:
            self._prediction_history.pop(0)

    def get_recent_accuracy(self, n: int = None) -> float:
        n = n or self._accuracy_window
        if len(self._prediction_history) < 5:
            return 0.5
        recent = self._prediction_history[-n:]
        return np.mean([r["correct"] for r in recent])

    def save(self, path: str = "models/meta_model.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = "models/meta_model.joblib") -> "MetaLabelModel":
        return joblib.load(path)
