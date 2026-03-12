"""
Market microstructure features.
Computes order flow, price velocity, volume imbalance, and trade pressure signals.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_microstructure_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """
    Generate microstructure features from OHLCV data.
    In absence of tick data, these are approximated from candle structure.
    """
    feat = pd.DataFrame(index=df.index)
    p = f"{prefix}_" if prefix else ""

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    # ── Price velocity and acceleration ───────────────────────────────────────
    feat[f"{p}price_velocity"] = close.diff(1)
    feat[f"{p}price_velocity_3"] = close.diff(3)
    feat[f"{p}price_acceleration"] = feat[f"{p}price_velocity"].diff(1)
    feat[f"{p}price_jerk"] = feat[f"{p}price_acceleration"].diff(1)

    # Normalized velocity
    feat[f"{p}norm_velocity"] = feat[f"{p}price_velocity"] / close.rolling(20, min_periods=10).std().replace(0, np.nan)

    # ── Buy/Sell volume approximation (Tick rule) ──────────────────────────────
    # Up-tick bars → buy pressure; down-tick → sell pressure
    direction = np.sign(close - open_)
    feat[f"{p}buy_volume"] = volume.where(direction > 0, volume * 0.5)
    feat[f"{p}sell_volume"] = volume.where(direction < 0, volume * 0.5)
    feat[f"{p}delta_volume"] = feat[f"{p}buy_volume"] - feat[f"{p}sell_volume"]

    # ── Signed volume (more accurate with body proportion) ────────────────────
    body_pct = (close - open_).abs() / (high - low).replace(0, np.nan)
    feat[f"{p}signed_volume"] = volume * np.sign(close - open_) * body_pct

    # ── Volume imbalance ──────────────────────────────────────────────────────
    total_vol = feat[f"{p}buy_volume"] + feat[f"{p}sell_volume"]
    feat[f"{p}volume_imbalance"] = feat[f"{p}delta_volume"] / total_vol.replace(0, np.nan)

    # Rolling buy/sell ratio
    feat[f"{p}rolling_buy_sell_ratio_5"] = (
        feat[f"{p}buy_volume"].rolling(5, min_periods=2).sum() /
        feat[f"{p}sell_volume"].rolling(5, min_periods=2).sum().replace(0, np.nan)
    )
    feat[f"{p}rolling_buy_sell_ratio_10"] = (
        feat[f"{p}buy_volume"].rolling(10, min_periods=5).sum() /
        feat[f"{p}sell_volume"].rolling(10, min_periods=5).sum().replace(0, np.nan)
    )

    # ── Cumulative delta ──────────────────────────────────────────────────────
    feat[f"{p}cum_delta_10"] = feat[f"{p}delta_volume"].rolling(10, min_periods=5).sum()
    feat[f"{p}cum_delta_20"] = feat[f"{p}delta_volume"].rolling(20, min_periods=10).sum()

    # ── Candle-based order flow ────────────────────────────────────────────────
    # High-low range normalized
    feat[f"{p}range_pct"] = (high - low) / close
    feat[f"{p}close_location"] = (close - low) / (high - low).replace(0, np.nan)  # 0=bottom, 1=top

    # Rejection signals (long wicks = price rejected)
    feat[f"{p}upper_rejection"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / (high - low).replace(0, np.nan)
    feat[f"{p}lower_rejection"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / (high - low).replace(0, np.nan)

    # ── VWAP deviation signals ─────────────────────────────────────────────────
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * volume).rolling(20, min_periods=10).sum()
    cum_vol = volume.rolling(20, min_periods=10).sum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

    feat[f"{p}vwap_deviation"] = (close - vwap) / vwap.replace(0, np.nan)
    feat[f"{p}vwap_cross"] = np.sign(close - vwap)

    # ── Microstructure noise ──────────────────────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    feat[f"{p}micro_noise"] = log_ret.rolling(5, min_periods=2).std()
    feat[f"{p}autocorr_5"] = log_ret.rolling(20, min_periods=10).apply(
        lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False
    )

    # ── Trade intensity proxy ─────────────────────────────────────────────────
    feat[f"{p}trade_intensity"] = volume * feat[f"{p}range_pct"]
    feat[f"{p}trade_intensity_spike"] = (
        feat[f"{p}trade_intensity"] /
        feat[f"{p}trade_intensity"].rolling(20, min_periods=10).mean().replace(0, np.nan)
    )

    # ── Price efficiency ──────────────────────────────────────────────────────
    # Ratio of net price change to total path (efficiency = 1 means straight line)
    net_change = (close - close.shift(5)).abs()
    total_path = close.diff().abs().rolling(5, min_periods=2).sum()
    feat[f"{p}price_efficiency"] = net_change / total_path.replace(0, np.nan)

    return feat


def update_microstructure_from_trade(
    trade_buffer: list,
    window: int = 100
) -> dict:
    """
    Compute real-time microstructure features from live trade buffer.
    trade_buffer: list of dicts with keys: price, qty, is_buyer_maker
    """
    if len(trade_buffer) < 2:
        return {}

    df = pd.DataFrame(trade_buffer[-window:])
    buy_vol = df.loc[~df["is_buyer_maker"], "qty"].sum()
    sell_vol = df.loc[df["is_buyer_maker"], "qty"].sum()
    total_vol = buy_vol + sell_vol

    prices = df["price"].values
    velocity = float(prices[-1] - prices[-5]) if len(prices) >= 5 else 0.0
    acceleration = float(
        (prices[-1] - prices[-2]) - (prices[-3] - prices[-4])
    ) if len(prices) >= 4 else 0.0

    return {
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "delta_volume": buy_vol - sell_vol,
        "volume_imbalance": (buy_vol - sell_vol) / max(total_vol, 1e-8),
        "rolling_buy_sell_ratio": buy_vol / max(sell_vol, 1e-8),
        "price_velocity": velocity,
        "price_acceleration": acceleration,
    }
