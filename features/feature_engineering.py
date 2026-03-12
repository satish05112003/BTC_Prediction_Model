"""
Feature engineering pipeline.
Generates 100+ technical, microstructure, and statistical features.
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional

from utils import indicators as ind
from config import get_config

logger = logging.getLogger(__name__)
cfg = get_config()


def no_lookahead(func):
    """Wraps feature function to verify output has no lookahead."""
    def wrapper(df, *args, **kwargs):
        result = func(df, *args, **kwargs)
        # Feature at time t should use only data up to t
        # Verified by checking NaN patterns align with beginning, not end
        has_nans = result.isna().sum().sum() > 0 if isinstance(result, pd.DataFrame) else result.isna().sum() > 0
        if has_nans:
            last_valid = result.last_valid_index()
            last_data  = df.index[-1]
            if last_valid is not None and last_valid < last_data:
                raise ValueError(f"Feature {func.__name__} may have lookahead")
        return result
    return wrapper


@no_lookahead
def build_features(df: pd.DataFrame, timeframe_label: str = "") -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    prefix = f"{timeframe_label}_" if timeframe_label else ""

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    open_ = df["open"]

    feat_cfg = cfg.get("features", {})

    # EMAs and SMAs
    for p in feat_cfg.get("ema_periods", [5, 9, 21, 50, 100, 200]):
        feat[f"{prefix}ema_{p}"] = ind.ema(close, p)
        feat[f"{prefix}dist_ema_{p}"] = (close - feat[f"{prefix}ema_{p}"]) / feat[f"{prefix}ema_{p}"]

    for p in feat_cfg.get("sma_periods", [10, 20, 50, 100, 200]):
        feat[f"{prefix}sma_{p}"] = ind.sma(close, p)
        feat[f"{prefix}dist_sma_{p}"] = (close - feat[f"{prefix}sma_{p}"]) / feat[f"{prefix}sma_{p}"]

    # EMA crossovers
    feat[f"{prefix}ema_cross_9_21"] = feat[f"{prefix}ema_9"] - feat[f"{prefix}ema_21"]
    feat[f"{prefix}ema_cross_21_50"] = feat[f"{prefix}ema_21"] - feat[f"{prefix}ema_50"]
    feat[f"{prefix}ema_cross_50_200"] = feat[f"{prefix}ema_50"] - feat[f"{prefix}ema_200"]

    # MACD
    macd_fast = feat_cfg.get("macd_fast", 12)
    macd_slow = feat_cfg.get("macd_slow", 26)
    macd_sig = feat_cfg.get("macd_signal", 9)
    macd_line, signal_line, histogram = ind.macd(close, macd_fast, macd_slow, macd_sig)
    feat[f"{prefix}macd"] = macd_line
    feat[f"{prefix}macd_signal"] = signal_line
    feat[f"{prefix}macd_hist"] = histogram
    feat[f"{prefix}macd_hist_delta"] = histogram.diff()

    # Parabolic SAR
    feat[f"{prefix}psar"] = ind.parabolic_sar(high, low)
    feat[f"{prefix}psar_dist"] = (close - feat[f"{prefix}psar"]) / close

    # ADX
    adx_period = feat_cfg.get("adx_period", 14)
    adx_val, di_plus, di_minus = ind.adx(high, low, close, adx_period)
    feat[f"{prefix}adx"] = adx_val
    feat[f"{prefix}di_plus"] = di_plus
    feat[f"{prefix}di_minus"] = di_minus
    feat[f"{prefix}di_diff"] = di_plus - di_minus

    # RSI
    rsi_period = feat_cfg.get("rsi_period", 14)
    feat[f"{prefix}rsi"] = ind.rsi(close, rsi_period)
    feat[f"{prefix}rsi_delta"] = feat[f"{prefix}rsi"].diff()
    feat[f"{prefix}rsi_overbought"] = (feat[f"{prefix}rsi"] > 70).astype(int)
    feat[f"{prefix}rsi_oversold"] = (feat[f"{prefix}rsi"] < 30).astype(int)

    stoch_k, stoch_d = ind.stochastic_rsi(close, feat_cfg.get("stoch_rsi_period", 14))
    feat[f"{prefix}stoch_k"] = stoch_k
    feat[f"{prefix}stoch_d"] = stoch_d
    feat[f"{prefix}stoch_diff"] = stoch_k - stoch_d

    for roc_p in [5, 10, 20]:
        feat[f"{prefix}roc_{roc_p}"] = ind.roc(close, roc_p)

    feat[f"{prefix}momentum_5"] = ind.momentum(close, 5)
    feat[f"{prefix}momentum_10"] = ind.momentum(close, 10)
    feat[f"{prefix}williams_r"] = ind.williams_r(high, low, close, 14)

    # Volatility
    atr_period = feat_cfg.get("atr_period", 14)
    atr_val = ind.atr(high, low, close, atr_period)
    feat[f"{prefix}atr"] = atr_val
    feat[f"{prefix}atr_pct"] = atr_val / close

    bb_period = feat_cfg.get("bb_period", 20)
    bb_std = feat_cfg.get("bb_std", 2.0)
    bb_upper, bb_mid, bb_lower, bb_width, bb_pct = ind.bollinger_bands(close, bb_period, bb_std)
    feat[f"{prefix}bb_upper"] = bb_upper
    feat[f"{prefix}bb_lower"] = bb_lower
    feat[f"{prefix}bb_width"] = bb_width
    feat[f"{prefix}bb_pct"] = bb_pct

    kc_upper, kc_mid, kc_lower = ind.keltner_channels(high, low, close)
    feat[f"{prefix}kc_upper"] = kc_upper
    feat[f"{prefix}kc_lower"] = kc_lower
    feat[f"{prefix}kc_width"] = (kc_upper - kc_lower) / kc_mid

    for hv_p in [5, 10, 20]:
        feat[f"{prefix}hist_vol_{hv_p}"] = ind.historical_volatility(close, hv_p)

    # Volume
    feat[f"{prefix}vwap"] = ind.vwap(high, low, close, volume)
    feat[f"{prefix}dist_vwap"] = (close - feat[f"{prefix}vwap"]) / feat[f"{prefix}vwap"]
    feat[f"{prefix}obv"] = ind.obv(close, volume)
    feat[f"{prefix}obv_ema"] = ind.ema(feat[f"{prefix}obv"], 10)
    feat[f"{prefix}obv_diff"] = feat[f"{prefix}obv"] - feat[f"{prefix}obv_ema"]
    feat[f"{prefix}vol_sma_10"] = ind.volume_sma(volume, 10)
    feat[f"{prefix}vol_sma_20"] = ind.volume_sma(volume, 20)
    feat[f"{prefix}vol_spike"] = ind.volume_spike_ratio(volume, 20)
    feat[f"{prefix}vol_delta"] = volume.diff()

    # Candle structure
    candle_range = high - low
    body = (close - open_).abs()
    upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low

    feat[f"{prefix}candle_range"] = candle_range
    feat[f"{prefix}body_size"] = body
    feat[f"{prefix}body_ratio"] = body / candle_range.replace(0, np.nan)
    feat[f"{prefix}upper_wick"] = upper_wick
    feat[f"{prefix}lower_wick"] = lower_wick
    feat[f"{prefix}wick_ratio"] = (upper_wick + lower_wick) / candle_range.replace(0, np.nan)
    feat[f"{prefix}bullish_candle"] = (close > open_).astype(int)
    feat[f"{prefix}candle_return"] = (close - open_) / open_

    # Rolling returns
    log_ret = np.log(close / close.shift(1))
    for w in [3, 5, 10, 20]:
        feat[f"{prefix}roll_ret_{w}"] = log_ret.rolling(w, min_periods=w//2).sum()
        feat[f"{prefix}roll_ret_std_{w}"] = log_ret.rolling(w, min_periods=w//2).std()

    # Time features
    feat[f"{prefix}hour"] = df.index.hour
    feat[f"{prefix}minute"] = df.index.minute
    feat[f"{prefix}day_of_week"] = df.index.dayofweek
    feat[f"{prefix}hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    feat[f"{prefix}hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

    # Support/Resistance
    feat[f"{prefix}high_20"] = high.rolling(20, min_periods=10).max()
    feat[f"{prefix}low_20"] = low.rolling(20, min_periods=10).min()
    feat[f"{prefix}dist_high_20"] = (close - feat[f"{prefix}high_20"]) / close
    feat[f"{prefix}dist_low_20"] = (close - feat[f"{prefix}low_20"]) / close

    return feat


def merge_multi_timeframe(
    feat_1m: pd.DataFrame,
    feat_5m: pd.DataFrame,
    feat_1h: pd.DataFrame,
    base_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forward-fill higher timeframe features onto 5m index.
    """
    result = feat_5m.copy()

    # Add 1m features resampled to 5m
    feat_1m_5m = feat_1m.resample("5min").last()
    for col in feat_1m.columns:
        try:
            series = feat_1m_5m[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            reindexed = series.reindex(result.index, method="ffill")
            if isinstance(reindexed, pd.Series):
                result[col] = reindexed
        except Exception:
            pass

    # Add 1h features forward-filled
    for col in feat_1h.columns:
        try:
            series = feat_1h[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            reindexed = series.reindex(result.index, method="ffill")
            if isinstance(reindexed, pd.Series):
                result[col] = reindexed
        except Exception:
            pass

    return result