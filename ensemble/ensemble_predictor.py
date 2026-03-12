"""
Ensemble prediction system using weighted averaging and stacking.
Combines XGBoost, LightGBM, RF, LR, LSTM, and Transformer predictions.
"""
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from config import get_config

logger = logging.getLogger(__name__)
cfg = get_config()


class EnsemblePredictor:
    """
    Ensemble model combining multiple base models via:
    1. Weighted average (default)
    2. Stacking with meta-learner
    """

    def __init__(self, models: Dict = None):
        ensemble_cfg = cfg.get("ensemble", {})
        self.weights = ensemble_cfg.get("weights", {})
        self.method = ensemble_cfg.get("method", "weighted_average")
        self.models = models or {}
        self.stacking_model = None
        self.stacking_scaler = None
        self._fitted_stacking = False

    def _get_model_proba(self, name: str, model, X) -> np.ndarray:
        """Get probability predictions from a model, handling LR scaler wrapper."""
        try:
            if isinstance(model, dict) and "model" in model:
                # LR with scaler
                X_scaled = model["scaler"].transform(
                    X.values if hasattr(X, "values") else X
                )
                return model["model"].predict_proba(X_scaled)
            else:
                return model.predict_proba(X)
        except Exception as e:
            logger.warning(f"Model {name} prediction failed: {e}")
            return np.full((len(X), 2), 0.5)

    def predict_proba(self, X) -> np.ndarray:
        """Ensemble prediction."""
        if not self.models:
            return np.full((len(X), 2), 0.5)

        if self._fitted_stacking and self.stacking_model is not None:
            return self._predict_stacking(X)
        else:
            return self._predict_weighted(X)

    def _predict_weighted(self, X) -> np.ndarray:
        """Weighted average of model probabilities."""
        weighted_proba = np.zeros((len(X), 2))
        total_weight = 0.0

        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0 / len(self.models))
            proba = self._get_model_proba(name, model, X)
            # Align length (deep models pad)
            if len(proba) != len(X):
                proba = np.full((len(X), 2), 0.5)
            weighted_proba += weight * proba
            total_weight += weight

        return weighted_proba / max(total_weight, 1e-8)

    def _predict_stacking(self, X) -> np.ndarray:
        """Stacking meta-learner prediction."""
        meta_features = self._build_meta_features(X)
        meta_scaled = self.stacking_scaler.transform(meta_features)
        return self.stacking_model.predict_proba(meta_scaled)

    def _build_meta_features(self, X) -> np.ndarray:
        """Build meta-feature matrix from base model predictions."""
        base_probas = []
        for name, model in self.models.items():
            proba = self._get_model_proba(name, model, X)
            if len(proba) != len(X):
                proba = np.full((len(X), 2), 0.5)
            base_probas.append(proba[:, 1])
        return np.column_stack(base_probas)

    def fit_stacking(self, X_val, y_val) -> "EnsemblePredictor":
        """Fit stacking meta-learner on validation set."""
        logger.info("Fitting stacking ensemble meta-learner...")
        meta_features = self._build_meta_features(X_val)

        self.stacking_scaler = StandardScaler()
        meta_scaled = self.stacking_scaler.fit_transform(meta_features)

        self.stacking_model = LogisticRegression(C=1.0, max_iter=1000)
        y_arr = y_val.values if hasattr(y_val, "values") else np.array(y_val)

        # Align lengths (deep models may have padding)
        min_len = min(len(meta_scaled), len(y_arr))
        self.stacking_model.fit(meta_scaled[-min_len:], y_arr[-min_len:])
        self._fitted_stacking = True
        logger.info("Stacking ensemble fitted.")
        return self

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_individual_probas(self, X) -> Dict[str, np.ndarray]:
        """Return individual model probas for diagnostics."""
        result = {}
        for name, model in self.models.items():
            proba = self._get_model_proba(name, model, X)
            result[name] = proba[:, 1] if len(proba) == len(X) else np.full(len(X), 0.5)
        return result

    def save(self, path: str = "models/ensemble_model.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Ensemble saved to {path}")

    @staticmethod
    def load(path: str = "models/ensemble_model.joblib") -> "EnsemblePredictor":
        return joblib.load(path)
