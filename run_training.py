"""
Safe training runner with checkpoints.
Saves progress after each step so crashes don't lose everything.
"""
import os
import sys
import joblib
import logging
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

from utils.logger import setup_logging
from config import load_config, get_config

load_config()
cfg = get_config()
setup_logging("INFO", "logs/system.log")
logger = logging.getLogger(__name__)

Path("models").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

def checkpoint_exists(name):
    return Path(f"models/checkpoint_{name}.joblib").exists()

def save_checkpoint(name, data):
    joblib.dump(data, f"models/checkpoint_{name}.joblib")
    logger.info(f"Checkpoint saved: {name}")

def load_checkpoint(name):
    return joblib.load(f"models/checkpoint_{name}.joblib")

# Import all new pipeline functions
from utils.data_loader import load_all_data, validate_data, create_target
from features.regime_detection import RegimeDetector
import lightgbm as lgb
from models.train_models import (
    build_full_feature_matrix, 
    clean_features, 
    assert_no_lookahead,
    run_walk_forward_training,
    select_features,
    train_base_models,
    calibrate_model,
    build_ensemble,
    backtest_polymarket
)

def run_pipeline():
    # ── STEP 1: Load Data & Validate ──────────────────────────────────────────────
    if checkpoint_exists("data"):
        logger.info("STEP 1: Loading verified data from checkpoint...")
        df_1m, df_5m, df_1h = load_checkpoint("data")
    else:
        logger.info("STEP 1: Loading and validating data...")
        df_1m, df_5m, df_1h = load_all_data()
        df_1m = validate_data(df_1m, "1m")
        df_5m = validate_data(df_5m, "5m")
        df_1h = validate_data(df_1h, "1h")
        save_checkpoint("data", (df_1m, df_5m, df_1h))

    # ── STEP 2: Regime Detector ───────────────────────────────────────────────────
    if checkpoint_exists("regime"):
        logger.info("STEP 2: Loading regime detector from checkpoint...")
        regime_detector = load_checkpoint("regime")
    else:
        logger.info("STEP 2: Fitting regime detector on training portion...")
        train_size = int(len(df_5m) * 0.9)
        regime_detector = RegimeDetector(n_states=3)
        regime_detector.fit(df_5m.iloc[:train_size])
        regime_detector.save("models/regime_detector.joblib")
        save_checkpoint("regime", regime_detector)

    # ── STEP 3: Build Features ────────────────────────────────────────────────────
    if checkpoint_exists("features"):
        logger.info("STEP 3: Loading features from checkpoint...")
        feature_matrix = load_checkpoint("features")
    else:
        logger.info("STEP 3: Building features...")
        feature_matrix = build_full_feature_matrix(df_1m, df_5m, df_1h, regime_detector)
        save_checkpoint("features", feature_matrix)

    # ── STEP 4: Create Target & Clean Features ────────────────────────────────────
    if checkpoint_exists("dataset"):
        logger.info("STEP 4: Loading dataset from checkpoint...")
        X, target = load_checkpoint("dataset")
    else:
        logger.info("STEP 4: Creating target and cleaning features...")
        target = create_target(df_5m, forward_periods=1, threshold_pct=0.03)
        
        aligned = feature_matrix.loc[feature_matrix.index.isin(df_5m.index)]
        target_aligned = target.reindex(aligned.index)
        
        # Last index has no future close, drop it for both but keep NaN targets in X
        aligned = aligned.iloc[:-1]
        target_aligned = target_aligned.iloc[:-1]
        
        X = clean_features(aligned)
        
        logger.info("Asserting no lookahead features...")
        assert_no_lookahead(X, target_aligned)
        
        save_checkpoint("dataset", (X, target_aligned))

    # Define validation and train masks
    valid_mask = target.notna()
    X_clean = X[valid_mask]
    y_clean = target[valid_mask]

    # ── STEP 5: Walk Forward CV ───────────────────────────────────────────────────
    if checkpoint_exists("wfcv"):
        logger.info("STEP 5: Loading WFCV results from checkpoint...")
        fold_results, X_oos, y_oos = load_checkpoint("wfcv")
    else:
        logger.info("STEP 5: Running Walk Forward CV (no lookahead)...")
        fold_results, X_oos, y_oos = run_walk_forward_training(X_clean, y_clean, n_splits=5)
        save_checkpoint("wfcv", (fold_results, X_oos, y_oos))

    # Log WFCV Table
    mean_auc = np.mean([r["ens_auc"] for r in fold_results])
    std_auc = np.std([r["ens_auc"] for r in fold_results])
    logger.info("┌─────────┬───────────┬──────────┬──────────┬──────────┬──────────┐")
    logger.info("│ Fold    │ Train     │ Test     │ XGB AUC  │ LGB AUC  │ Ens AUC  │")
    logger.info("├─────────┼───────────┼──────────┼──────────┼──────────┼──────────┤")
    for r in fold_results:
        logger.info(f"│ Fold {r['fold']:<2} │ {r['train_start']:<5}-{r['train_end']:<5} │ {r['test_start']:<4}-{r['test_end']:<4} │ "
                    f"{r['xgb_auc']:<8.4f} │ {r['lgb_auc']:<8.4f} │ {r['ens_auc']:<8.4f} │")
    logger.info("├─────────┼───────────┼──────────┼──────────┼──────────┼──────────┤")
    logger.info(f"│ MEAN    │           │          │ {np.mean([r['xgb_auc'] for r in fold_results]):<8.4f} │ {np.mean([r['lgb_auc'] for r in fold_results]):<8.4f} │ {mean_auc:<8.4f} │")
    logger.info(f"│ STD     │           │          │ {np.std([r['xgb_auc'] for r in fold_results]):<8.4f} │ {np.std([r['lgb_auc'] for r in fold_results]):<8.4f} │ {std_auc:<8.4f} │")
    logger.info("└─────────┴───────────┴──────────┴──────────┴──────────┴──────────┘")

    if mean_auc < 0.53:
        logger.warning("HIGH VARIANCE / LOW AUC — likely still overfitting or no real edge.")

    # ── STEP 8: Final Model Training & HPO Placeholder ────────────────────────────
    if checkpoint_exists("final"):
        logger.info("STEP 8: Loading final trained models...")
        final_data = load_checkpoint("final")
        calibrated_models = final_data["models"]
        final_features = final_data["features"]
    else:
        logger.info("STEP 8: Final model training on 90% of data...")
        X_train_full = X_clean.iloc[:int(len(X_clean)*0.9)]
        y_train_full = y_clean.iloc[:int(len(y_clean)*0.9)]
        
        calib_size = max(int(len(X_train_full) * 0.1), 100)
        X_tr = X_train_full.iloc[:-calib_size]
        y_tr = y_train_full.iloc[:-calib_size]
        X_cal = X_train_full.iloc[-calib_size:]
        y_cal = y_train_full.iloc[-calib_size:]
        
        X_tr_sel, X_cal_sel, final_features = select_features(X_tr, y_tr, X_cal, n_features=80)
        
        base_models = train_base_models(X_tr_sel, y_tr)
        
        calibrated_models = {}
        for name, m in base_models.items():
            calibrated_models[name] = calibrate_model(m, X_cal_sel, y_cal, method="isotonic")
            joblib.dump(calibrated_models[name], f"models/{name}_model.joblib")
            
        save_checkpoint("final", {"models": calibrated_models, "features": final_features})
        
    # ── STEP 13: Ensemble & Meta Model ────────────────────────────────────────────
    # Load checkpoint only if it is in the expected dict format {ensemble, meta}.
    # A stale checkpoint (raw EnsemblePredictor) is deleted so we rebuild below.
    _ens_loaded = False
    if checkpoint_exists("ensemble"):
        ens_data = load_checkpoint("ensemble")
        if isinstance(ens_data, dict) and "ensemble" in ens_data and "meta" in ens_data:
            logger.info("STEP 13: Loading ensemble from checkpoint...")
            ensemble = ens_data["ensemble"]
            meta_model = ens_data["meta"]
            _ens_loaded = True
        else:
            logger.warning("STEP 13: Stale/incompatible ensemble checkpoint detected. Deleting and rebuilding...")
            Path("models/checkpoint_ensemble.joblib").unlink()

    if not _ens_loaded:
        logger.info("STEP 13: Building calibrated ensemble and meta-learner...")
        X_train_full = X_clean.iloc[:int(len(X_clean)*0.9)]
        y_train_full = y_clean.iloc[:int(len(y_clean)*0.9)]
        calib_size = max(int(len(X_train_full) * 0.1), 100)
        X_tr = X_train_full.iloc[:-calib_size]
        y_tr = y_train_full.iloc[:-calib_size]
        X_cal = X_train_full.iloc[-calib_size:]
        y_cal = y_train_full.iloc[-calib_size:]

        X_tr_sel = X_tr[final_features]
        X_cal_sel = X_cal[final_features]

        ensemble = build_ensemble(calibrated_models, X_tr_sel, y_tr, X_cal_sel, y_cal)
        joblib.dump(ensemble, "models/ensemble_model.joblib")

        # Meta model training on Out-Of-Sample data from Walk-Forward CV
        logger.info("Training meta-label model on OOS data...")
        oos_correct = ((X_oos["ens_prob"] >= 0.5).astype(int) == y_oos).astype(int)

        meta_model = lgb.LGBMClassifier(n_estimators=100, class_weight="balanced", verbose=-1)
        meta_model.fit(X_oos, oos_correct.values)
        joblib.dump(meta_model, "models/meta_model.joblib")

        save_checkpoint("ensemble", {"ensemble": ensemble, "meta": meta_model})

    # ── STEP 16: Polymarket Backtest ──────────────────────────────────────────────
    logger.info("STEP 16: Honest Simulation...")
    X_test_full = X.iloc[int(len(X)*0.9):]     # Includes ambiguous NaNs for realism
    y_test_full = target.iloc[int(len(target)*0.9):] 
    df_test_prices = df_5m.iloc[int(len(df_5m)*0.9):]
    
    report = backtest_polymarket(ensemble, X_test_full, y_test_full, df_test_prices, final_features)
    
    if report["Win Rate"] < 50.0:
        logger.warning(f"Win rate is ONLY {report['Win Rate']:.1f}% < 50%. Saving anyway, but model may lose money.")
    else:
        logger.info("Models saved and training successful.")

if __name__ == "__main__":
    run_pipeline()