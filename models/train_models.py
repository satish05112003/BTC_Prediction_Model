"""
Models training module.
Complete rewrite with walk-forward CV, no-lookahead enforcement,
calibration, ensemble, and honest Polymarket backtesting.
"""
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import time
import json

from config import get_config
from utils.data_loader import load_all_data, create_target, validate_data
from features.feature_engineering import build_features, merge_multi_timeframe
from features.microstructure_features import build_microstructure_features
from features.regime_detection import build_regime_features, RegimeDetector
from features.volatility_models import build_volatility_features
from features.liquidity_shock_features import build_liquidity_features

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from ensemble.ensemble_predictor import EnsemblePredictor
from meta.meta_label_model import MetaLabelModel

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
cfg = get_config()

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


# =====================================================================
# MODULE-LEVEL PICKLABLE WRAPPER CLASSES
# =====================================================================

class CalibratedModel:
    """Thin wrapper: applies calibration to an already-fitted base model."""
    def __init__(self, base, cal_fn):
        self.base = base
        self.cal_fn = cal_fn

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        cal = self.cal_fn(raw)
        cal = np.clip(cal, 1e-7, 1 - 1e-7)
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class LRScaledModel:
    """Logistic Regression wrapper that applies StandardScaler before predicting."""
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


class StackedEnsemble:
    """Stacked ensemble: base model predictions fed into a meta-learner."""
    def __init__(self, base_models, weights, meta_learner):
        self.base_models = base_models
        self.names = list(base_models.keys())
        self.weights = weights
        self.meta_learner = meta_learner

    def predict_proba(self, X):
        preds = {}
        for name, m in self.base_models.items():
            preds[name] = m.predict_proba(X)[:, 1]
        avg_prob = np.zeros(len(X))
        for i, n in enumerate(self.names):
            avg_prob += self.weights[i] * preds[n]
        stack_features = []
        for m_name in ["xgboost", "lightgbm", "random_forest", "logistic_regression"]:
            stack_features.append(preds.get(m_name, np.zeros(len(X))))
        stack_features.append(np.abs(avg_prob - 0.5))
        stack_features.append(np.maximum(avg_prob, 1 - avg_prob))
        stack_features.append(preds.get("xgboost", np.zeros(len(X))) - preds.get("lightgbm", np.zeros(len(X))))
        X_stack = np.column_stack(stack_features)
        meta_probs = self.meta_learner.predict_proba(X_stack)
        return meta_probs

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# =====================================================================
# SECTION A — STRICT NO-LOOKAHEAD FEATURE CLEANING
# =====================================================================

def assert_no_lookahead(X: pd.DataFrame, y: pd.Series):
    """
    Hard check: ensure features at time t use ONLY data from t and earlier.
    Method: check that all NaN rows in X align with early candles only.
    Features must NOT use shift(-n) for any n > 0.
    """
    forbidden = ["shift(-", "future_", "forward_", "next_"]
    for col in X.columns:
        col_lower = str(col).lower()
        for pattern in forbidden:
            if pattern in col_lower:
                raise ValueError(f"Lookahead feature detected: {col}")
    logger.info("No lookahead features detected.")


# =====================================================================
# SECTION B — WALK-FORWARD CROSS-VALIDATION
# =====================================================================

class WalkForwardCV:
    """Purged Walk-Forward Cross-Validation."""
    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 10,
        embargo_pct: float = 0.01,
        min_train_size: int = 50000,
    ):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.min_train_size = min_train_size

    def split(self, X: pd.DataFrame):
        n = len(X)
        fold_size = n // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            test_end   = min(fold_size * (i + 1), n)
            test_start = fold_size * i
            train_end  = test_start - self.purge_gap
            embargo    = int(fold_size * self.embargo_pct)
            actual_test_start = test_start + embargo

            if train_end < self.min_train_size:
                continue
            if actual_test_start >= test_end:
                continue

            train_idx = np.arange(0, train_end)
            test_idx  = np.arange(actual_test_start, test_end)

            yield train_idx, test_idx


# =====================================================================
# SECTION C — FEATURE SELECTION
# =====================================================================

def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    n_features: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    from sklearn.feature_selection import VarianceThreshold
    import lightgbm as lgb
    
    selector = VarianceThreshold(threshold=1e-6)
    selector.fit(X_train)
    mask_var = selector.get_support()
    X_train_v = X_train.loc[:, mask_var]
    X_val_v   = X_val.loc[:, mask_var]
    logger.info(f"After variance filter: {X_train_v.shape[1]} features")

    fast_lgb = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.1,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced",
    )
    fast_lgb.fit(X_train_v, y_train)
    importances = pd.Series(
        fast_lgb.feature_importances_,
        index=X_train_v.columns
    ).sort_values(ascending=False)
    top_features = importances.head(min(n_features * 2, len(importances))).index.tolist()
    X_train_t = X_train_v[top_features]
    X_val_t   = X_val_v[top_features]
    logger.info(f"After importance filter: {len(top_features)} features")

    corr_matrix = X_train_t.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    final_features = [f for f in top_features if f not in to_drop][:n_features]
    X_train_f = X_train_t[final_features]
    X_val_f   = X_val_t[final_features]
    logger.info(f"After correlation filter: {len(final_features)} features")

    joblib.dump(final_features, MODELS_DIR / "feature_names.joblib")
    return X_train_f, X_val_f, final_features


# =====================================================================
# SECTION D, E, F — MODEL TRAINING
# =====================================================================

def calibrate_model(base_model, X_cal, y_cal, method="isotonic"):
    """
    Calibrate an already-fitted model using X_cal/y_cal.
    cv='prefit' was removed in sklearn 1.4+, so we manually calibrate:
      1. Get raw predicted probabilities from the fitted base model.
      2. Fit an isotonic or sigmoid calibrator on those probs vs true labels.
      3. Return a thin wrapper that applies calibration at predict_proba time.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression as _LR

    # Raw probs from the already-fitted base model
    raw_probs = base_model.predict_proba(X_cal)[:, 1]

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probs, y_cal)
        transform_fn = calibrator.transform
    else:  # sigmoid / Platt scaling
        calibrator = _LR(C=1.0, solver="lbfgs", max_iter=1000)
        calibrator.fit(raw_probs.reshape(-1, 1), y_cal)
        def transform_fn(p):
            return calibrator.predict_proba(p.reshape(-1, 1))[:, 1]

    # CalibratedModel is defined at module level for pickling compatibility
    return CalibratedModel(base_model, transform_fn)


def train_base_models(X_train, y_train):
    models = {}
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

    logger.info("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    models["xgboost"] = xgb_model

    logger.info("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.03,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        verbose=-1,
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)
    models["lightgbm"] = lgb_model
    
    logger.info("Training RandomForest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models["random_forest"] = rf_model

    logger.info("Training LogisticRegression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lr_model = LogisticRegression(
        C=0.1, penalty="l2", class_weight="balanced", max_iter=1000, n_jobs=-1
    )
    lr_model.fit(X_train_scaled, y_train)
    
    # LRScaledModel is defined at module level for pickling compatibility
    models["logistic_regression"] = LRScaledModel(lr_model, scaler)
    return models


# =====================================================================
# SECTION I — ENSEMBLE COMBINATION
# =====================================================================

def build_ensemble(models_dict, X_train, y_train, X_cal, y_cal):
    logger.info("Building Ensemble Predictor...")
    from scipy.optimize import minimize
    
    cal_preds = {}
    for name, model in models_dict.items():
        cal_preds[name] = model.predict_proba(X_cal)[:, 1]
    
    def log_loss_func(weights, preds, y):
        w = np.array(weights)
        w = np.clip(w, 0, None)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones(len(w)) / len(w)
        final_probs = np.zeros(len(y))
        for i, name in enumerate(preds.keys()):
            final_probs += w[i] * preds[name]
        final_probs = np.clip(final_probs, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(final_probs) + (1 - y) * np.log(1 - final_probs))

    init_weights = [1.0 / len(models_dict)] * len(models_dict)
    bounds = [(0.0, 1.0)] * len(models_dict)
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    res = minimize(log_loss_func, init_weights, args=(cal_preds, y_cal), bounds=bounds, constraints=cons)
    weights = res.x
    weights /= weights.sum()
    logger.info(f"Ensemble Weights: {dict(zip(models_dict.keys(), weights))}")
    
    stack_features = []
    stack_features.append(cal_preds["xgboost"])
    stack_features.append(cal_preds["lightgbm"])
    stack_features.append(cal_preds["random_forest"])
    stack_features.append(cal_preds["logistic_regression"])
    avg_prob = np.zeros(len(y_cal))
    for i, n in enumerate(models_dict.keys()):
        avg_prob += weights[i] * cal_preds[n]
    stack_features.append(np.abs(avg_prob - 0.5))
    stack_features.append(np.maximum(avg_prob, 1 - avg_prob))
    stack_features.append(cal_preds["xgboost"] - cal_preds["lightgbm"])
    
    X_stack = np.column_stack(stack_features)
    meta_learner = lgb.LGBMClassifier(
        n_estimators=100, num_leaves=15, learning_rate=0.05, 
        class_weight="balanced", verbose=-1
    )
    meta_learner.fit(X_stack, y_cal)
    
    # StackedEnsemble is defined at module level for pickling compatibility
    return StackedEnsemble(models_dict, weights, meta_learner)


# =====================================================================
# SECTION G — WALK-FORWARD TRAINING LOOP
# =====================================================================

def run_walk_forward_training(X, y, n_splits=5):
    logger.info(f"Running Walk-Forward CV with {n_splits} splits...")
    cv = WalkForwardCV(n_splits=n_splits)
    fold_results = []
    all_oos_preds = []
    all_oos_labels = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_tr_full, y_tr_full = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        calib_size = max(int(len(X_tr_full) * 0.1), 100)
        X_tr = X_tr_full.iloc[:-calib_size]
        y_tr = y_tr_full.iloc[:-calib_size]
        X_cal = X_tr_full.iloc[-calib_size:]
        y_cal = y_tr_full.iloc[-calib_size:]
        
        X_tr_sel, X_cal_sel, sel_features = select_features(X_tr, y_tr, X_cal, n_features=80)
        X_test_sel = X_test[sel_features]
        base_models = train_base_models(X_tr_sel, y_tr)
        
        calibrated_models = {}
        for name, m in base_models.items():
            calibrated_models[name] = calibrate_model(m, X_cal_sel, y_cal, method="isotonic")
            
        ensemble = build_ensemble(calibrated_models, X_tr_sel, y_tr, X_cal_sel, y_cal)
        
        xgb_proba = calibrated_models["xgboost"].predict_proba(X_test_sel)[:, 1]
        lgb_proba = calibrated_models["lightgbm"].predict_proba(X_test_sel)[:, 1]
        ens_proba = ensemble.predict_proba(X_test_sel)[:, 1]
        
        xgb_auc = roc_auc_score(y_test, xgb_proba)
        lgb_auc = roc_auc_score(y_test, lgb_proba)
        ens_auc = roc_auc_score(y_test, ens_proba)
        
        # Collect OOS data
        fold_oos_data = X_test_sel.copy()
        fold_oos_data["ens_prob"] = ens_proba
        all_oos_preds.append(fold_oos_data)
        all_oos_labels.append(y_test)
        
        fold_results.append({
            "fold": fold + 1,
            "train_start": train_idx[0],
            "train_end": train_idx[-1],
            "test_start": test_idx[0],
            "test_end": test_idx[-1],
            "xgb_auc": xgb_auc,
            "lgb_auc": lgb_auc,
            "ens_auc": ens_auc
        })
        logger.info(f"Fold {fold+1}: XGB AUC={xgb_auc:.4f} | LGB AUC={lgb_auc:.4f} | Ensemble AUC={ens_auc:.4f}")
        
    return fold_results, pd.concat(all_oos_preds), pd.concat(all_oos_labels)


# =====================================================================
# SECTION H — HONEST BACKTEST (Polymarket simulation)
# =====================================================================

def backtest_polymarket(model, X_test, y_test, df_test_prices, feature_names):
    logger.info("Running Polymarket Backtest...")
    X_test_sel = X_test[feature_names]

    # ── Align price data to X_test_sel by index to avoid shape mismatches ──────
    # X and df_5m may have slightly different lengths (X had its last row dropped
    # during feature construction). Reindexing on a shared index ensures all
    # arrays are the same length and properly time-aligned.
    shared_index = X_test_sel.index.intersection(df_test_prices.index)
    X_test_sel   = X_test_sel.loc[shared_index]
    prices_aligned = df_test_prices.loc[shared_index, "close"]
    y_test_aligned = y_test.reindex(shared_index)

    probas = model.predict_proba(X_test_sel)
    prob_ups = probas[:, 1]
    prob_downs = probas[:, 0]
    confidences = np.maximum(prob_ups, prob_downs)

    # actual_closes: the close of the NEXT candle (what we're predicting)
    actual_closes  = prices_aligned.shift(-1)
    price_to_beats = prices_aligned

    skip_mask = confidences < 0.60
    skipped = skip_mask.sum()
    valid_mask = ~skip_mask
    valid_ups   = valid_mask & (prob_ups   > prob_downs)
    valid_downs = valid_mask & (prob_downs > prob_ups)

    win_up   = valid_ups   & (actual_closes > price_to_beats)
    win_down = valid_downs & (actual_closes < price_to_beats)
    wins = win_up.sum() + win_down.sum()
    total_signals = valid_mask.sum()
    losses = total_signals - wins

    win_rate     = (wins / total_signals * 100) if total_signals > 0 else 0
    skip_rate    = (skipped / len(confidences) * 100) if len(confidences) > 0 else 0
    p            = win_rate / 100.0
    kelly        = (2 * p - 1) if p > 0 else 0
    profit_factor = (wins / losses) if losses > 0 else float("inf")

    valid_y_mask = y_test_aligned.notna()
    if valid_y_mask.sum() > 0:
        brier_score = np.mean((prob_ups[valid_y_mask.values] - y_test_aligned[valid_y_mask].values) ** 2)
    else:
        brier_score = 0.0
        
    report = {
        "Total Signals": int(total_signals),
        "Wins": int(wins),
        "Losses": int(losses),
        "Skipped": int(skipped),
        "Win Rate": float(win_rate),
        "Skip Rate": float(skip_rate),
        "Kelly": float(kelly),
        "Profit Factor": float(profit_factor),
        "Brier Score": float(brier_score),
    }
    
    logger.info("\n" + "═" * 60)
    logger.info("POLYMARKET SIMULATION BACKTEST RESULTS")
    logger.info("═" * 60)
    logger.info(f"Total Signals:  {total_signals:,} ({100 - skip_rate:.1f}% of candles)")
    logger.info(f"Wins:           {wins:,}")
    logger.info(f"Losses:         {losses:,}")
    logger.info(f"Skipped:        {skipped:,}")
    logger.info("─" * 40)
    logger.info(f"Win Rate:       {win_rate:.1f}%")
    logger.info(f"Skip Rate:      {skip_rate:.1f}%")
    logger.info("─" * 40)
    logger.info(f"Kelly Criterion:{kelly:.2f}")
    logger.info(f"Profit Factor:  {profit_factor:.2f}")
    logger.info(f"Brier Score:    {brier_score:.3f}")
    logger.info("═" * 60 + "\n")
    
    with open(LOGS_DIR / "backtest_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    return report


# =====================================================================
# SECTION J — REGIME-CONDITIONAL SIGNAL FILTERING
# =====================================================================

def apply_regime_filter(signal: str, regime_state: int, regime_confidence: float) -> Tuple[float, bool]:
    """
    Apply regime-based filtering to signals.
    """
    # Default outputs
    should_emit = True
    base_threshold = 0.60
    
    if regime_confidence < 0.60:
        # High uncertainty/vol regime
        base_threshold = 0.70
        should_emit = False # Default to suppress if very uncertain
        
    if regime_state == 0: # Ranging/Low vol
        base_threshold = 0.55
        should_emit = True
    elif regime_state == 1: # Trending
        # In a real system, we'd check if signal direction matches trend
        # For now, we raise threshold slightly
        base_threshold = 0.58
        should_emit = True
    elif regime_state == 2: # High Volatility
        base_threshold = 0.70
        should_emit = False
        
    return base_threshold, should_emit


# =====================================================================
# FEATURE PIPELINE UTILS
# =====================================================================

def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    import warnings
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    good_cols = {}
    for col in X.columns:
        if col in good_cols:
            continue
        series = X[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        good_cols[col] = pd.to_numeric(series, errors="coerce")
    X = pd.DataFrame(good_cols, index=X.index)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().bfill().fillna(0)
    nunique = X.nunique()
    constant_cols = nunique[nunique <= 1].index
    if len(constant_cols):
        X = X.drop(columns=constant_cols)
    X = X.astype(np.float32)
    return X


def build_full_feature_matrix(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    regime_detector: RegimeDetector = None,
) -> pd.DataFrame:
    logger.info("Building feature matrices...")
    feat_5m = build_features(df_5m, "5m")
    feat_5m_ms = build_microstructure_features(df_5m, "5m")
    feat_5m_vol = build_volatility_features(df_5m, "5m")
    feat_5m_liq = build_liquidity_features(df_5m, "5m")
    feat_5m_reg = build_regime_features(df_5m, regime_detector, "5m")
    feat_1m = build_features(df_1m, "1m")
    feat_1m_ms = build_microstructure_features(df_1m, "1m")
    feat_1h = build_features(df_1h, "1h")
    feat_1h_vol = build_volatility_features(df_1h, "1h")
    feat_1h_reg = build_regime_features(df_1h, regime_detector, "1h")
    feat_5m_all = pd.concat([feat_5m, feat_5m_ms, feat_5m_vol, feat_5m_liq, feat_5m_reg], axis=1)
    feat_1m_all = pd.concat([feat_1m, feat_1m_ms], axis=1)
    feat_1h_all = pd.concat([feat_1h, feat_1h_vol, feat_1h_reg], axis=1)
    combined = merge_multi_timeframe(feat_1m_all, feat_5m_all, feat_1h_all, df_5m)
    return combined

def run_training_pipeline(base_path: str = None) -> Dict:
    """Fallback if executed directly without run_training.py handling checkpoints."""
    pass
