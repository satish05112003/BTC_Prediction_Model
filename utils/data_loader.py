"""
Data loading and preprocessing module.
Handles CSV and Parquet ingestion, timestamp alignment, and data validation.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from config import get_config

logger = logging.getLogger(__name__)
cfg = get_config()


def load_ohlcv(timeframe: str, base_path: str = None) -> pd.DataFrame:
    data_cfg = cfg.get("data", {})
    bp = base_path or data_cfg.get("base_path", "data/")
    filename = data_cfg.get("files", {}).get(timeframe)
    if not filename:
        raise ValueError(f"No file configured for timeframe: {timeframe}")

    path = Path(bp) / filename

    # If exact path not found, try swapping extension
    if not path.exists():
        alt = path.with_suffix(".parquet" if path.suffix == ".csv" else ".csv")
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Data file not found: {path}")

    logger.info(f"Loading {timeframe} data from {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    df = _normalize_columns(df)
    df = _parse_timestamps(df)
    df = _validate_ohlcv(df)
    df = _remove_duplicates(df)
    df = df.sort_index()

    logger.info(f"Loaded {len(df):,} rows for {timeframe} | "
                f"{df.index[0]} to {df.index[-1]}")
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    aliases = {
        "open_time": "timestamp", "open price": "open",
        "high price": "high", "low price": "low",
        "close price": "close", "vol": "volume",
    }
    df = df.rename(columns=aliases)
    return df


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    ts_col = None
    for candidate in ["timestamp", "datetime", "date", "time", "open_time"]:
        if candidate in df.columns:
            ts_col = candidate
            break

    if ts_col is None:
        ts_col = df.columns[0]

    # If timestamp is already the index
    if ts_col not in df.columns:
        ts = df.index
    else:
        ts = df[ts_col]
        df = df.drop(columns=[ts_col])

    if pd.api.types.is_numeric_dtype(ts):
        if pd.Series(ts).median() > 1e12:
            ts = pd.to_datetime(ts, unit="ms", utc=True)
        else:
            ts = pd.to_datetime(ts, unit="s", utc=True)
    else:
        ts = pd.to_datetime(ts, utc=True)

    df.index = ts
    df.index.name = "timestamp"
    return df


def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    initial_len = len(df)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[(df["open"] > 0) & (df["close"] > 0) &
            (df["high"] > 0) & (df["low"] > 0) & (df["volume"] >= 0)]

    mask = df["high"] < df["low"]
    df.loc[mask, ["high", "low"]] = df.loc[mask, ["low", "high"]].values
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)

    dropped = initial_len - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped:,} invalid rows during validation.")
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    dupes = df.index.duplicated(keep="last")
    n_dupes = dupes.sum()
    if n_dupes:
        logger.warning(f"Removing {n_dupes:,} duplicate timestamps.")
        df = df[~dupes]
    return df


def align_timeframes(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start = max(df_1m.index[0], df_5m.index[0], df_1h.index[0])
    end = min(df_1m.index[-1], df_5m.index[-1], df_1h.index[-1])
    df_1m = df_1m.loc[start:end]
    df_5m = df_5m.loc[start:end]
    df_1h = df_1h.loc[start:end]
    logger.info(f"Aligned timeframes: {start} to {end}")
    return df_1m, df_5m, df_1h


def resample_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    return df_1m.resample("5min").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last",
        "volume": "sum",
    }).dropna()


def create_target(
    df: pd.DataFrame,
    forward_periods: int = 1,
    threshold_pct: float = 0.03,
) -> pd.Series:
    """
    Create binary classification target.

    Uses settlement price logic identical to Polymarket:
      settlement_price = close of CURRENT candle
      outcome = close of NEXT candle

    Target logic:
      future_return = (close[t+1] - close[t]) / close[t] * 100

      if future_return >  threshold_pct  → 1 (UP)
      if future_return < -threshold_pct  → 0 (DOWN)
      else                               → NaN (SKIP — ambiguous)

    Rows with NaN target are DROPPED from training.
    This removes noise candles and forces model to learn real moves.

    threshold_pct=0.03 means 0.03% price move required.
    For BTC at $70,000: threshold = $21 minimum move.
    This eliminates ~15-25% of candles as ambiguous.
    """
    future_close = df["close"].shift(-forward_periods)
    current_close = df["close"]
    future_return = (future_close - current_close) / current_close * 100

    target = pd.Series(np.nan, index=df.index, name="target")
    target[future_return >  threshold_pct] = 1
    target[future_return < -threshold_pct] = 0
    # NaN rows = ambiguous → dropped during training

    logger.info(
        f"Target distribution | UP: {(target==1).sum():,} | "
        f"DOWN: {(target==0).sum():,} | "
        f"SKIP: {target.isna().sum():,} ({target.isna().mean()*100:.1f}%)"
    )
    return target


def validate_data(df: pd.DataFrame, name: str):
    """
    Before training, validate:
      1. Timestamps sorted ascending
      2. No duplicate timestamps
      3. No gap > 2x expected interval
      4. No NaN in OHLCV columns
      5. high >= low always
      6. high >= open and high >= close
      7. low  <= open and low  <= close
      8. volume >= 0

    Log any violations. Fix where possible (drop dups, fill tiny gaps).
    Raise ValueError if violations > 0.1% of rows.
    """
    initial_len = len(df)
    
    # 1. Timestamps sorted
    if not df.index.is_monotonic_increasing:
        logger.warning(f"[{name}] Timestamps not sorted. Sorting now.")
        df = df.sort_index()

    # 2. Duplicate timestamps
    dupes = df.index.duplicated(keep="last")
    if dupes.sum() > 0:
        logger.warning(f"[{name}] Found {dupes.sum()} duplicate timestamps. Dropping.")
        df = df[~dupes]

    # 3. Gaps
    diffs = df.index.to_series().diff()
    median_diff = diffs.median()
    gaps = (diffs > 2 * median_diff).sum()
    if gaps > 0:
        logger.warning(f"[{name}] Found {gaps} gaps larger than 2x expected interval ({median_diff}).")

    # 4. NaN in OHLCV
    nan_mask = df[["open", "high", "low", "close", "volume"]].isna().any(axis=1)
    if nan_mask.sum() > 0:
        logger.warning(f"[{name}] Found {nan_mask.sum()} rows with NaN. Dropping.")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    # 5-8. Price and volume violations
    bad_prices = (
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"]) |
        (df["volume"] < 0)
    )
    if bad_prices.sum() > 0:
        logger.warning(f"[{name}] Found {bad_prices.sum()} rows with invalid prices/volume. Fixing.")
        mask = df["high"] < df["low"]
        df.loc[mask, ["high", "low"]] = df.loc[mask, ["low", "high"]].values
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)
        df.loc[df["volume"] < 0, "volume"] = 0

    final_len = len(df)
    dropped = initial_len - final_len
    if dropped / initial_len > 0.001:
        raise ValueError(f"[{name}] Dropped {dropped} rows ({dropped/initial_len*100:.2f}%) which exceeds 0.1% threshold.")
        
    return df



def load_all_data(base_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_1m = load_ohlcv("1m", base_path)
    df_5m = load_ohlcv("5m", base_path)
    df_1h = load_ohlcv("1h", base_path)
    return align_timeframes(df_1m, df_5m, df_1h)