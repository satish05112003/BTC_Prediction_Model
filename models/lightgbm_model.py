"""
LightGBM model wrapper.
"""
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from config import get_config

logger = logging.getLogger(__name__)


class LightGBMModel:
    def __init__(self, params: dict = None):
        cfg = get_config()
        default_params = cfg.get("models", {}).get("lightgbm", {})
        self.params = params or default_params
        self.model = None
        self.feature_names = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "LightGBMModel":
        import lightgbm as lgb

        self.feature_names = (
            list(X_train.columns) if hasattr(X_train, "columns") else None
        )

        params = {k: v for k, v in self.params.items()
                  if k not in ("early_stopping_rounds", "n_estimators")}
        params["n_estimators"] = self.params.get("n_estimators", 500)
        params["verbose"] = -1

        callbacks = [lgb.log_evaluation(period=-1)]
        if X_val is not None:
            early_stop = self.params.get("early_stopping_rounds", 50)
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stop, verbose=False))

        self.model = lgb.LGBMClassifier(**params)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        logger.info("LightGBM trained.")
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

    def save(self, path: str = "models/lightgbm_model.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = "models/lightgbm_model.joblib") -> "LightGBMModel":
        return joblib.load(path)
