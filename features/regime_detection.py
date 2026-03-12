"""
Market regime detection using HMM, ADX, and volatility clustering.
Regimes: 0=trending_up, 1=trending_down, 2=sideways/choppy
"""
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logging.warning("hmmlearn not available, HMM regime detection disabled.")

from config import get_config

logger = logging.getLogger(__name__)
cfg = get_config()


class RegimeDetector:
    """
    Detects market regimes using multiple methods and returns regime labels.
    """

    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.hmm_model = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """Fit HMM on log returns and volatility."""
        if not HMM_AVAILABLE:
            logger.warning("HMM not available, skipping HMM fit.")
            return self

        log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        vol = log_ret.rolling(20, min_periods=10).std().fillna(0)
        X = np.column_stack([log_ret.values, vol.values])

        try:
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=200,
                random_state=42
            )
            self.hmm_model.fit(X)
            self._fitted = True
            logger.info(f"HMM fitted with {self.n_states} states.")
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")

        return self

    def predict_hmm_states(self, df: pd.DataFrame) -> pd.Series:
        """Predict HMM states for given DataFrame."""
        if not self._fitted or self.hmm_model is None:
            return pd.Series(np.zeros(len(df), dtype=int), index=df.index, name="hmm_state")

        log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        vol = log_ret.rolling(20, min_periods=10).std().fillna(0)
        X = np.column_stack([log_ret.values, vol.values])

        try:
            states = self.hmm_model.predict(X)
            return pd.Series(states, index=df.index, name="hmm_state")
        except Exception as e:
            logger.error(f"HMM prediction failed: {e}")
            return pd.Series(np.zeros(len(df), dtype=int), index=df.index, name="hmm_state")

    def save(self, path: str = "models/regime_detector.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = "models/regime_detector.joblib") -> "RegimeDetector":
        return joblib.load(path)


def build_regime_features(df: pd.DataFrame, detector: RegimeDetector = None,
                           prefix: str = "") -> pd.DataFrame:
    """Build all regime-related features."""
    feat = pd.DataFrame(index=df.index)
    p = f"{prefix}_" if prefix else ""

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ── ADX-based trend regime ────────────────────────────────────────────────
    from utils.indicators import adx, atr, ema
    adx_val, di_plus, di_minus = adx(high, low, close, 14)
    adx_threshold = cfg.get("regime", {}).get("adx_trend_threshold", 25)

    feat[f"{p}adx"] = adx_val
    feat[f"{p}regime_adx_trending"] = (adx_val > adx_threshold).astype(int)
    feat[f"{p}regime_adx_trend_up"] = ((adx_val > adx_threshold) & (di_plus > di_minus)).astype(int)
    feat[f"{p}regime_adx_trend_down"] = ((adx_val > adx_threshold) & (di_minus > di_plus)).astype(int)
    feat[f"{p}regime_sideways"] = (adx_val <= adx_threshold).astype(int)

    # ── Volatility regime ─────────────────────────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    roll_vol_20 = log_ret.rolling(20, min_periods=10).std()
    vol_pct_low = cfg.get("regime", {}).get("volatility_percentile_low", 33)
    vol_pct_high = cfg.get("regime", {}).get("volatility_percentile_high", 66)

    # Rolling percentile-based regime
    def vol_regime(vol_series: pd.Series, window: int = 252) -> pd.Series:
        pct_lo = vol_series.rolling(window, min_periods=window//2).quantile(vol_pct_low / 100)
        pct_hi = vol_series.rolling(window, min_periods=window//2).quantile(vol_pct_high / 100)
        regime = pd.Series(1, index=vol_series.index)  # medium
        regime[vol_series < pct_lo] = 0  # low vol
        regime[vol_series > pct_hi] = 2  # high vol
        return regime

    feat[f"{p}vol_20"] = roll_vol_20
    feat[f"{p}vol_regime"] = vol_regime(roll_vol_20)
    feat[f"{p}vol_regime_low"] = (feat[f"{p}vol_regime"] == 0).astype(int)
    feat[f"{p}vol_regime_mid"] = (feat[f"{p}vol_regime"] == 1).astype(int)
    feat[f"{p}vol_regime_high"] = (feat[f"{p}vol_regime"] == 2).astype(int)

    # Volatility ratio (current vs historical)
    feat[f"{p}vol_ratio"] = roll_vol_20 / log_ret.rolling(100, min_periods=50).std().replace(0, np.nan)

    # ── HMM states ────────────────────────────────────────────────────────────
    if detector is not None and detector._fitted:
        hmm_states = detector.predict_hmm_states(df)
        feat[f"{p}hmm_state"] = hmm_states
        # One-hot encode HMM states
        for s in range(detector.n_states):
            feat[f"{p}hmm_state_{s}"] = (hmm_states == s).astype(int)

    # ── Mean reversion vs trend ───────────────────────────────────────────────
    # Hurst exponent approximation (R/S method, simplified)
    feat[f"{p}mean_rev_score"] = _rolling_hurst(close, window=50)

    # ── Trend strength ────────────────────────────────────────────────────────
    ema_20 = ema(close, 20)
    ema_50 = ema(close, 50)
    feat[f"{p}trend_strength"] = (ema_20 - ema_50) / ema_50.replace(0, np.nan)
    feat[f"{p}trend_direction"] = np.sign(feat[f"{p}trend_strength"])

    return feat


def _rolling_hurst(series: pd.Series, window: int = 50) -> pd.Series:
    """
    Approximate rolling Hurst exponent using variance ratios.
    H ~= 0.5: random walk, H > 0.5: trending, H < 0.5: mean-reverting
    """
    def hurst_approx(x):
        if len(x) < 10:
            return 0.5
        lags = [2, 4, 8, 16]
        lags = [l for l in lags if l < len(x)]
        if len(lags) < 2:
            return 0.5
        variances = [np.var(np.diff(x, n=l)) for l in lags]
        try:
            log_lags = np.log(lags)
            log_vars = np.log([max(v, 1e-12) for v in variances])
            h = np.polyfit(log_lags, log_vars, 1)[0] / 2
            return float(np.clip(h, 0, 1))
        except Exception:
            return 0.5

    return series.rolling(window, min_periods=window//2).apply(hurst_approx, raw=True)
