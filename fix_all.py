"""
Run this once to fix all pandas compatibility issues automatically.
"""
import os
import re

fixes = {
    "features/volatility_models.py": [
        ('garch_vol.fillna(method="ffill")', "garch_vol.ffill()"),
        ('garch_vol.fillna(method="bfill")', "garch_vol.bfill()"),
    ],
    "features/feature_engineering.py": [
        ('.fillna(method="ffill")', ".ffill()"),
        ('.fillna(method="bfill")', ".bfill()"),
    ],
    "features/microstructure_features.py": [
        ('.fillna(method="ffill")', ".ffill()"),
        ('.fillna(method="bfill")', ".bfill()"),
    ],
    "features/regime_detection.py": [
        ('.fillna(method="ffill")', ".ffill()"),
        ('.fillna(method="bfill")', ".bfill()"),
    ],
    "features/liquidity_shock_features.py": [
        ('.fillna(method="ffill")', ".ffill()"),
        ('.fillna(method="bfill")', ".bfill()"),
    ],
    "models/train_models.py": [
        ('.fillna(method="ffill")', ".ffill()"),
        ('.fillna(method="bfill")', ".bfill()"),
    ],
    "utils/data_loader.py": [
        ('infer_datetime_format=True', ""),
        ('.fillna(method="ffill")', ".ffill()"),
        ('.fillna(method="bfill")', ".bfill()"),
    ],
    "ensemble/ensemble_predictor.py": [
        ('.fillna(method="ffill")', ".ffill()"),
        ('.fillna(method="bfill")', ".bfill()"),
    ],
    "meta/meta_label_model.py": [
        ('.fillna(method="ffill")', ".ffill()"),
        ('.fillna(method="bfill")', ".bfill()"),
    ],
    "live/prediction_engine.py": [
        ('.fillna(method="ffill")', ".ffill()"),
        ('.fillna(method="bfill")', ".bfill()"),
    ],
    "logs/prediction_logger.py": [
        ('.fillna(method="ffill")', ".ffill()"),
        ('.fillna(method="bfill")', ".bfill()"),
    ],
}

print("Fixing all pandas compatibility issues...")
for filepath, replacements in fixes.items():
    if not os.path.exists(filepath):
        print(f"  SKIP (not found): {filepath}")
        continue
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    original = content
    for old, new in replacements:
        content = content.replace(old, new)
    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  FIXED: {filepath}")
    else:
        print(f"  OK (no changes): {filepath}")

print("\nAll fixes applied!")
print("Now run: python main.py --train")