"""
Liquidity shock and order flow spike detection features.
Identifies sudden abnormal events that often precede large moves.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_liquidity_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Build liquidity shock detection features."""
    feat = pd.DataFrame(index=df.index)
    p = f"{prefix}_" if prefix else ""

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── Volume shock ratio ────────────────────────────────────────────────────
    vol_ma_20 = volume.rolling(20, min_periods=10).mean()
    vol_ma_50 = volume.rolling(50, min_periods=25).mean()
    vol_std_20 = volume.rolling(20, min_periods=10).std()

    feat[f"{p}vol_shock_ratio_20"] = volume / vol_ma_20.replace(0, np.nan)
    feat[f"{p}vol_shock_ratio_50"] = volume / vol_ma_50.replace(0, np.nan)
    feat[f"{p}vol_shock_zscore"] = (volume - vol_ma_20) / vol_std_20.replace(0, np.nan)

    # Binary shock signals
    feat[f"{p}vol_shock_2x"] = (feat[f"{p}vol_shock_ratio_20"] > 2.0).astype(int)
    feat[f"{p}vol_shock_3x"] = (feat[f"{p}vol_shock_ratio_20"] > 3.0).astype(int)

    # ── Abnormal volatility spike ─────────────────────────────────────────────
    candle_range = (high - low) / close
    range_ma = candle_range.rolling(20, min_periods=10).mean()
    range_std = candle_range.rolling(20, min_periods=10).std()

    feat[f"{p}range_zscore"] = (candle_range - range_ma) / range_std.replace(0, np.nan)
    feat[f"{p}volatility_spike"] = (feat[f"{p}range_zscore"] > 2.0).astype(int)

    # ── Price displacement from VWAP ──────────────────────────────────────────
    typical = (high + low + close) / 3
    vwap = (typical * volume).rolling(20, min_periods=10).sum() / volume.rolling(20, min_periods=10).sum().replace(0, np.nan)
    feat[f"{p}vwap_displacement"] = (close - vwap) / vwap.replace(0, np.nan)
    feat[f"{p}vwap_displacement_abs"] = feat[f"{p}vwap_displacement"].abs()
    feat[f"{p}vwap_extreme"] = (feat[f"{p}vwap_displacement_abs"] > 0.005).astype(int)

    # ── Trade burst frequency ─────────────────────────────────────────────────
    # Proxy: count of bars in last 10 with vol > 1.5x average
    is_burst = (volume > vol_ma_20 * 1.5).astype(int)
    feat[f"{p}burst_count_10"] = is_burst.rolling(10, min_periods=5).sum()
    feat[f"{p}burst_count_20"] = is_burst.rolling(20, min_periods=10).sum()

    # ── Combined liquidity shock score ────────────────────────────────────────
    shock_score = (
        0.3 * feat[f"{p}vol_shock_zscore"].clip(-5, 5) / 5 +
        0.3 * feat[f"{p}range_zscore"].clip(-5, 5) / 5 +
        0.2 * feat[f"{p}vwap_displacement_abs"].clip(0, 0.02) / 0.02 +
        0.2 * feat[f"{p}burst_count_10"] / 10
    )
    feat[f"{p}liquidity_shock_score"] = shock_score.clip(0, 1)

    # ── Large candle detection ────────────────────────────────────────────────
    ret = (close - df["open"]) / df["open"]
    ret_abs = ret.abs()
    ret_ma = ret_abs.rolling(50, min_periods=25).mean()
    ret_std = ret_abs.rolling(50, min_periods=25).std()

    feat[f"{p}candle_zscore"] = (ret_abs - ret_ma) / ret_std.replace(0, np.nan)
    feat[f"{p}big_candle"] = (feat[f"{p}candle_zscore"] > 2.0).astype(int)

    # ── Momentum exhaustion ────────────────────────────────────────────────────
    # Strong move with shrinking volume → possible reversal
    momentum_3 = close.diff(3)
    vol_trend = volume.rolling(3, min_periods=1).mean() / volume.rolling(10, min_periods=5).mean()
    feat[f"{p}momentum_exhaustion"] = (
        (momentum_3.abs() > momentum_3.abs().rolling(20, min_periods=10).quantile(0.75)) &
        (vol_trend < 0.8)
    ).astype(int)

    # ── Gap detection ─────────────────────────────────────────────────────────
    gap = (df["open"] - close.shift(1)) / close.shift(1)
    feat[f"{p}gap"] = gap
    feat[f"{p}gap_up"] = (gap > 0.001).astype(int)
    feat[f"{p}gap_down"] = (gap < -0.001).astype(int)

    return feat
