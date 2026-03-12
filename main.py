"""
BTC Prediction Model - Main Entry Point
========================================
Production-grade BTC 5-minute direction prediction system.

Usage:
    python main.py              # Run full system (train + live)
    python main.py --train      # Train models only
    python main.py --live       # Live prediction only (requires trained models)
    python main.py --backtest   # Run backtesting evaluation
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# ── Bootstrap path ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.logger import setup_logging
from config import load_config, get_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="BTC Direction Prediction System for Polymarket"
    )
    parser.add_argument("--train", action="store_true", help="Train models only")
    parser.add_argument("--live", action="store_true", help="Run live prediction only")
    parser.add_argument("--backtest", action="store_true", help="Run backtest evaluation")
    parser.add_argument("--config", type=str, default=None, help="Custom config path")
    parser.add_argument("--data-path", type=str, default=None, help="Override data path")
    return parser.parse_args()


def run_training(data_path: str = None):
    """Execute full training pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting Model Training Pipeline")
    logger.info("=" * 60)

    from models.train_models import run_training_pipeline
    results = run_training_pipeline(base_path=data_path)

    logger.info("\n✅ Training Complete!")
    logger.info(f"Trained {len(results['models'])} base models + ensemble + meta-model")
    logger.info("Models saved to: models/")
    return results


async def run_live_system():
    """Start the full live prediction system."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting Live Prediction System")
    logger.info("=" * 60)

    from live.price_stream import PriceStreamManager
    from live.prediction_engine import PredictionEngine

    # Initialize stream manager
    stream_manager = PriceStreamManager()

    # Initialize prediction engine
    engine = PredictionEngine(stream_manager)
    engine.load_models()

    if not engine._loaded:
        logger.error("Failed to load models. Run training first: python main.py --train")
        return

    # Initialize Telegram bot (new BTCPredictionBot)
    from tg_bot.bot import BTCPredictionBot
    bot = BTCPredictionBot(
        stream_manager=stream_manager,
        prediction_engine=engine
    )

    # Register bot as primary signal callback — it stores the signal and
    # auto-sends the formatted message to Telegram
    engine.add_signal_callback(bot.on_new_signal)

    # Also echo signals to console for local debugging
    async def console_log(signal):
        print("\n" + "=" * 50)
        print(signal.format_signal())
        print("=" * 50 + "\n")

    engine.add_signal_callback(console_log)

    # Start bot — disclaimer message is sent automatically inside start()
    await bot.start()

    logger.info("All systems initialized. Starting streams...")

    # Run all tasks concurrently
    tasks = [
        asyncio.create_task(stream_manager.start(), name="streams"),
        asyncio.create_task(engine.run(), name="predictions"),
    ]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
    finally:
        for task in tasks:
            task.cancel()
        stream_manager.stop()
        await bot.stop()
        logger.info("System shutdown complete.")


def run_backtest():
    """Run backtesting on historical data."""
    logger = logging.getLogger(__name__)
    logger.info("Running backtest evaluation...")

    try:
        import joblib
        import pandas as pd
        import numpy as np
        from sklearn.metrics import classification_report, confusion_matrix

        from utils.data_loader import load_all_data, create_target
        from models.train_models import (
            build_full_feature_matrix,
            clean_features
        )

        # Load models
        ensemble = joblib.load("models/ensemble_model.joblib")
        regime_detector = joblib.load("models/regime_detector.joblib")
        feature_names = joblib.load("models/feature_names.joblib")

        # Load data
        df_1m, df_5m, df_1h = load_all_data()

        # Build features
        feature_matrix = build_full_feature_matrix(df_1m, df_5m, df_1h, regime_detector)
        # Prepare dataset: Create target and align indices
        logger.info("Aligning features and target...")
        y = create_target(df_5m)
        
        # Ensure indices match
        shared_idx = feature_matrix.index.intersection(y.index)
        X = feature_matrix.loc[shared_idx]
        y = y.loc[shared_idx].dropna()
        X = X.loc[y.index]
        
        X = clean_features(X)

        # Use last 20% as test set
        n = len(X)
        test_start = int(n * 0.8)
        X_test = X.iloc[test_start:]
        y_test = y.iloc[test_start:]

        # Align feature names
        aligned_cols = []
        for col in feature_names:
            if col in X_test.columns:
                aligned_cols.append(col)
            else:
                X_test[col] = 0.0
                aligned_cols.append(col)
        X_test = X_test[aligned_cols]

        # Evaluate
        logger.info(f"Evaluating ensemble on {len(X_test)} samples...")
        proba = ensemble.predict_proba(X_test)
        
        # Handle proba shape if it's already just probabilities for class 1 or a full matrix
        if proba.ndim == 2:
            prob_up = proba[:, 1]
        else:
            prob_up = proba
            
        preds = (prob_up >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, prob_up)

        logger.info("\n" + classification_report(y_test, preds, target_names=["DOWN", "UP"]))
        logger.info("\nConfusion Matrix:")
        logger.info(str(confusion_matrix(y_test, preds)))
        logger.info(f"\nFinal Test Accuracy: {acc*100:.2f}%")
        logger.info(f"ROC AUC: {auc:.4f}")

    except FileNotFoundError:
        logger.error("Models not found. Run training first: python main.py --train")
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)


def main():
    args = parse_args()

    # Load configuration
    if args.config:
        load_config(args.config)
    else:
        load_config()

    cfg = get_config()
    setup_logging(
        log_level=cfg.get("system", {}).get("log_level", "INFO"),
        log_file=cfg.get("logging", {}).get("system_log", "logs/system.log")
    )

    logger = logging.getLogger(__name__)
    logger.info("BTC Prediction Model v1.0.0 starting...")

    if args.train:
        run_training(data_path=args.data_path)

    elif args.live:
        asyncio.run(run_live_system())

    elif args.backtest:
        run_backtest()

    else:
        # Default: train then go live
        logger.info("Running full pipeline: train → live")
        models_exist = (
            Path("models/ensemble_model.joblib").exists() and
            Path("models/meta_model.joblib").exists()
        )

        if not models_exist:
            logger.info("No trained models found. Starting training...")
            run_training(data_path=args.data_path)
        else:
            logger.info("Trained models found. Skipping training.")

        logger.info("Starting live prediction system...")
        asyncio.run(run_live_system())


if __name__ == "__main__":
    main()
