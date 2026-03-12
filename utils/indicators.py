"""
Core technical indicator implementations using pandas/numpy.
Vectorized for performance on large datasets.
"""
import numpy as np
import pandas as pd
from typing import Optional


# ─── Trend ────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period//2).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def parabolic_sar(high: pd.Series, low: pd.Series,
                  af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
    n = len(high)
    sar = np.zeros(n)
    bull = True
    af = af_start
    ep = low.iloc[0]
    sar[0] = high.iloc[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if bull:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low.iloc[i - 1], low.iloc[max(i - 2, 0)])
            if low.iloc[i] < sar[i]:
                bull = False
                sar[i] = ep
                ep = low.iloc[i]
                af = af_start
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_start, af_max)
        else:
            sar[i] = prev_sar - af * (prev_sar - ep)
            sar[i] = max(sar[i], high.iloc[i - 1], high.iloc[max(i - 2, 0)])
            if high.iloc[i] > sar[i]:
                bull = True
                sar[i] = ep
                ep = high.iloc[i]
                af = af_start
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_start, af_max)

    return pd.Series(sar, index=high.index, name="psar")


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> tuple:
    """Returns (ADX, +DI, -DI)."""
    tr = true_range(high, low, close)
    dm_plus = high.diff()
    dm_minus = -low.diff()

    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)

    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()
    di_plus = 100 * dm_plus.ewm(alpha=1 / period, adjust=False).mean() / atr_val
    di_minus = 100 * dm_minus.ewm(alpha=1 / period, adjust=False).mean() / atr_val

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_val, di_plus, di_minus


# ─── Momentum ─────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic_rsi(series: pd.Series, period: int = 14,
                   smooth_k: int = 3, smooth_d: int = 3):
    rsi_val = rsi(series, period)
    min_rsi = rsi_val.rolling(period, min_periods=period//2).min()
    max_rsi = rsi_val.rolling(period, min_periods=period//2).max()
    k = 100 * (rsi_val - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)
    k_smooth = k.rolling(smooth_k, min_periods=max(1, smooth_k//2)).mean()
    d_smooth = k_smooth.rolling(smooth_d, min_periods=max(1, smooth_d//2)).mean()
    return k_smooth, d_smooth


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    return 100 * (series - series.shift(period)) / series.shift(period).replace(0, np.nan)


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    return series - series.shift(period)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14) -> pd.Series:
    highest_high = high.rolling(period, min_periods=period//2).max()
    lowest_low = low.rolling(period, min_periods=period//2).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
    return wr


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(k_period, min_periods=k_period//2).min()
    highest_high = high.rolling(k_period, min_periods=k_period//2).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period, min_periods=max(1, d_period//2)).mean()
    return k, d


# ─── Volatility ───────────────────────────────────────────────────────────────

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = series.rolling(period, min_periods=period//2).mean()
    std = series.rolling(period, min_periods=period//2).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    bwidth = (upper - lower) / mid.replace(0, np.nan)
    bpct = (series - lower) / (upper - lower).replace(0, np.nan)
    return upper, mid, lower, bwidth, bpct


def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 20, atr_mult: float = 2.0):
    mid = ema(close, period)
    atr_val = atr(high, low, close, period)
    upper = mid + atr_mult * atr_val
    lower = mid - atr_mult * atr_val
    return upper, mid, lower


def historical_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(period, min_periods=period//2).std() * np.sqrt(252 * 288)  # annualized for 5m


# ─── Volume ───────────────────────────────────────────────────────────────────

def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, period: int = 20) -> pd.Series:
    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume
    return tp_vol.rolling(period, min_periods=period//2).sum() / volume.rolling(period, min_periods=period//2).sum().replace(0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume.rolling(period, min_periods=period//2).mean()


def volume_spike_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    vol_avg = volume_sma(volume, period)
    return volume / vol_avg.replace(0, np.nan)
