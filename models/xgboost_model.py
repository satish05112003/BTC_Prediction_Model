"""
XGBoost model wrapper.
"""
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from config import get_config

logger = logging.getLogger(__name__)


class XGBoostModel:
    def __init__(self, params: dict = None):
        cfg = get_config()
        default_params = cfg.get("models", {}).get("xgboost", {})
        self.params = params or default_params
        self.model = None
        self.feature_names = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "XGBoostModel":
        import xgboost as xgb

        self.feature_names = (
            list(X_train.columns) if hasattr(X_train, "columns") else None
        )

        # Separate early stopping from model params
        early_stopping = self.params.get("early_stopping_rounds", 50)
        params = {k: v for k, v in self.params.items()
                  if k not in ("early_stopping_rounds",)}

        # Add early stopping directly to classifier
        if X_val is not None and y_val is not None:
            params["early_stopping_rounds"] = early_stopping
            eval_set = [(X_val, y_val)]
        else:
            eval_set = None

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        logger.info("XGBoost trained successfully.")
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        if self.model and self.feature_names:
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
        return pd.Series()

    def save(self, path: str = "models/xgboost_model.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"XGBoost model saved to {path}")

    @staticmethod
    def load(path: str = "models/xgboost_model.joblib") -> "XGBoostModel":
        return joblib.load(path)