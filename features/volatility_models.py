"""
Volatility clustering models: GARCH, rolling regimes, breakout signals.
"""
import logging
import warnings
import numpy as np
import pandas as pd

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logging.warning("arch package not available. GARCH features disabled.")

logger = logging.getLogger(__name__)


def build_volatility_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Build volatility clustering features."""
    feat = pd.DataFrame(index=df.index)
    p = f"{prefix}_" if prefix else ""

    close = df["close"]
    log_ret = np.log(close / close.shift(1)) * 100  # in percent

    # ── Realized volatility measures ──────────────────────────────────────────
    for window in [5, 10, 20, 50]:
        feat[f"{p}rv_{window}"] = log_ret.rolling(window, min_periods=window//2).std()

    # Parkinson volatility (uses high-low range)
    hl_ratio = np.log(df["high"] / df["low"])
    feat[f"{p}parkinson_vol_20"] = (hl_ratio ** 2 / (4 * np.log(2))).rolling(20, min_periods=10).mean() ** 0.5

    # Garman-Klass estimator
    oc_sq = np.log(close / df["open"]) ** 2
    hl_sq = 0.5 * hl_ratio ** 2
    feat[f"{p}gk_vol_20"] = (hl_sq - (2 * np.log(2) - 1) * oc_sq).rolling(20, min_periods=10).mean() ** 0.5

    # ── Volatility regimes via z-score ────────────────────────────────────────
    vol_20 = feat[f"{p}rv_20"]
    vol_mean = vol_20.rolling(100, min_periods=50).mean()
    vol_std = vol_20.rolling(100, min_periods=50).std()
    feat[f"{p}vol_zscore"] = (vol_20 - vol_mean) / vol_std.replace(0, np.nan)
    feat[f"{p}vol_expansion"] = (feat[f"{p}vol_zscore"] > 1).astype(int)
    feat[f"{p}vol_contraction"] = (feat[f"{p}vol_zscore"] < -1).astype(int)

    # ── Volatility breakout signals ───────────────────────────────────────────
    # Squeeze: BB inside KC → anticipate breakout
    from utils.indicators import bollinger_bands, keltner_channels
    bb_u, _, bb_l, bb_w, _ = bollinger_bands(close, 20, 2.0)
    kc_u, _, kc_l = keltner_channels(df["high"], df["low"], close, 20, 1.5)

    feat[f"{p}bb_kc_squeeze"] = ((bb_u < kc_u) & (bb_l > kc_l)).astype(int)
    feat[f"{p}bb_width"] = bb_w
    feat[f"{p}bb_width_pct"] = bb_w / bb_w.rolling(50, min_periods=25).mean().replace(0, np.nan)

    # ── Volatility momentum ───────────────────────────────────────────────────
    feat[f"{p}vol_momentum"] = vol_20 / feat[f"{p}rv_50"].replace(0, np.nan)
    feat[f"{p}vol_delta"] = vol_20.diff()
    feat[f"{p}vol_acceleration"] = feat[f"{p}vol_delta"].diff()

    # ── GARCH conditional volatility ──────────────────────────────────────────
    if ARCH_AVAILABLE:
        garch_vol = _compute_garch_volatility(log_ret)
        feat[f"{p}garch_vol"] = garch_vol
        feat[f"{p}garch_vol_ratio"] = (
            garch_vol / garch_vol.rolling(50, min_periods=25).mean().replace(0, np.nan)
        )
    else:
        feat[f"{p}garch_vol"] = feat[f"{p}rv_20"]
        feat[f"{p}garch_vol_ratio"] = feat[f"{p}vol_momentum"]

    # ── Range expansion ───────────────────────────────────────────────────────
    atr_5 = (df["high"] - df["low"]).rolling(5, min_periods=2).mean()
    atr_20 = (df["high"] - df["low"]).rolling(20, min_periods=10).mean()
    feat[f"{p}range_expansion"] = atr_5 / atr_20.replace(0, np.nan)

    return feat


def _compute_garch_volatility(
    log_ret: pd.Series,
    rolling_window: int = 500,
    step: int = 100
) -> pd.Series:
    """
    Rolling GARCH(1,1) conditional volatility.
    Fits on a rolling window and extracts one-step-ahead forecast.
    """
    garch_vol = pd.Series(np.nan, index=log_ret.index)
    clean = log_ret.dropna()

    for end in range(rolling_window, len(clean), step):
        start = max(0, end - rolling_window)
        window_data = clean.iloc[start:end]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arch_model(window_data, vol="Garch", p=1, q=1,
                                   mean="Zero", dist="normal")
                res = model.fit(disp="off", show_warning=False)
                cond_vol = res.conditional_volatility
                # Assign to corresponding index positions
                for i, idx in enumerate(clean.iloc[start:end].index):
                    garch_vol.loc[idx] = cond_vol.iloc[i]
        except Exception:
            pass

    return garch_vol.ffill()
