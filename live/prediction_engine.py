"""
Real-time prediction engine.

Flow every 5 minutes:
  1. At exact boundary (e.g. 2:05:00) → fetch Chainlink settlement price
  2. That price = "Price to Beat" for the new window (2:05-2:10)
  3. Within 5-10 seconds → generate prediction and send to Telegram

Key fix: always uses the most recent candles available.
Parquet = historical base. Binance stream = live candles on top.
The model always predicts using the freshest data possible.
"""
import asyncio
import logging
import time
import json
import pytz
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Callable, List

from config import get_config
from live.price_stream import PriceStreamManager
from features.feature_engineering import build_features
from features.microstructure_features import (
    build_microstructure_features,
    update_microstructure_from_trade,
)
from features.regime_detection import build_regime_features
from features.volatility_models import build_volatility_features
from features.liquidity_shock_features import build_liquidity_features
from logs.prediction_logger import PredictionLogger
from live.candle_fetcher import BinanceCandleFetcher

try:
    from meta.meta_label_model import MetaLabelModel as _MetaLabelModel
except ImportError:
    _MetaLabelModel = None

logger = logging.getLogger(__name__)
cfg = get_config()
IST = pytz.timezone("Asia/Kolkata")


class _RawClassifierMetaAdapter:
    """
    Thin adapter so a raw sklearn/LightGBM classifier saved as meta_model.joblib
    can satisfy the validate_single() interface expected by the prediction engine.

    The saved LGBMClassifier was trained on OOS data where:
      - features are X_oos columns (ens_prob + whatever feature cols were in X_oos)
      - target is 1 if the ensemble was correct, 0 otherwise

    At inference we only have `primary_proba` (a scalar), so we use a single
    feature vector [primary_proba] and fall back gracefully if the model expects
    more columns.
    """

    def __init__(self, clf, min_meta_confidence: float = 0.50):
        self._clf = clf
        self._min_meta_confidence = min_meta_confidence

    def validate_single(
        self,
        ensemble,
        X_single: "pd.DataFrame",
        primary_proba: float,
    ) -> dict:
        try:
            # X_single is already aligned to the base feature_names.
            # The raw LGBMClassifier was trained on X_oos, which is exactly
            # the selected features PLUS "ens_prob" as an extra column.
            import pandas as pd
            
            if isinstance(X_single, pd.DataFrame):
                X_meta = X_single.copy()
                # LightGBM might strict-check column names and order
                X_meta["ens_prob"] = primary_proba
                
                # If LightGBM expects strict column names/order, make sure we match
                if hasattr(self._clf, "feature_name_"):
                    expected_cols = self._clf.feature_name_
                    # Add missing columns with 0 if needed (shouldn't happen ideally)
                    for col in expected_cols:
                        if col not in X_meta.columns:
                            X_meta[col] = 0.0
                    X_meta = X_meta[expected_cols]
            else:
                # Fallback if X_single is not a DataFrame
                n_expected = getattr(self._clf, "n_features_in_", 1)
                X_meta = np.zeros((1, n_expected))
                X_meta[0, -1] = primary_proba
                
            meta_proba = float(self._clf.predict_proba(X_meta)[0, 1])
        except Exception as e:
            # Fall back: treat high primary confidence as meta-validated
            meta_proba = abs(primary_proba - 0.5) * 2  # 0-1 scale

        should_trade = meta_proba >= self._min_meta_confidence
        return {"should_trade": bool(should_trade), "meta_confidence": meta_proba}

MODELS_DIR = Path("models")
CANDLE_CACHE_5M = Path("logs/live_candles_5m.json")
CANDLE_CACHE_1M = Path("logs/live_candles_1m.json")
MIN_CANDLES = 10


# ─── Candle Helpers ───────────────────────────────────────────────────────────

def _save_candles(candles: list, filepath: Path):
    try:
        filepath.parent.mkdir(exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(candles[-300:], f)
    except Exception as e:
        logger.debug(f"Could not save candles: {e}")


def _load_candles_from_cache(filepath: Path) -> list:
    try:
        if filepath.exists():
            with open(filepath, "r") as f:
                candles = json.load(f)
            logger.info(f"Cache loaded: {len(candles)} candles ({filepath.name})")
            return candles
    except Exception as e:
        logger.debug(f"Cache load failed: {e}")
    return []


def _load_candles_from_parquet(timeframe: str, n: int = 300) -> list:
    """Load last N candles from parquet for instant startup."""
    try:
        data_cfg = cfg.get("data", {})
        base_path = data_cfg.get("base_path", "data/")
        filename = data_cfg.get("files", {}).get(
            timeframe, f"btc_{timeframe}.parquet"
        )
        path = Path(base_path) / filename

        if not path.exists():
            logger.warning(f"Parquet not found: {path}")
            return []

        df = pd.read_parquet(path)
        df.columns = [c.strip().lower() for c in df.columns]

        ts_col = None
        for candidate in ["timestamp", "datetime", "date", "time", "open_time"]:
            if candidate in df.columns:
                ts_col = candidate
                break

        if ts_col:
            ts = df[ts_col]
            if pd.api.types.is_numeric_dtype(ts):
                unit = "ms" if ts.median() > 1e12 else "s"
                df.index = pd.to_datetime(ts, unit=unit, utc=True)
            else:
                df.index = pd.to_datetime(ts, utc=True)
            df = df.drop(columns=[ts_col])

        df = df.tail(n)
        candles = []
        for idx, row in df.iterrows():
            try:
                candles.append({
                    "ts": idx.timestamp(),
                    "open":   float(row.get("open", 0)),
                    "high":   float(row.get("high", 0)),
                    "low":    float(row.get("low", 0)),
                    "close":  float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)),
                })
            except Exception:
                pass

        logger.info(f"Parquet loaded: {len(candles)} x {timeframe} candles")
        return candles

    except Exception as e:
        logger.error(f"Parquet load failed ({timeframe}): {e}")
        return []


def _merge_candles(old: list, new: list) -> list:
    """Merge + deduplicate by timestamp. Newer value wins."""
    seen = {}
    for c in old + new:
        seen[c["ts"]] = c
    return sorted(seen.values(), key=lambda x: x["ts"])


def _candles_to_df(candles: list) -> Optional[pd.DataFrame]:
    if len(candles) < MIN_CANDLES:
        return None
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.set_index("timestamp")
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _align_features(
    feature_matrix: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:
    feature_matrix = feature_matrix.loc[
        :, ~feature_matrix.columns.duplicated()
    ]
    aligned = pd.DataFrame(index=feature_matrix.index)
    for col in feature_names:
        if col in feature_matrix.columns:
            try:
                series = feature_matrix[col]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                aligned[col] = pd.to_numeric(series, errors="coerce")
            except Exception:
                aligned[col] = 0.0
        else:
            aligned[col] = 0.0
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    aligned = aligned.ffill().bfill().fillna(0)
    return aligned.astype(np.float32)


def _log_candle_freshness(candles: list, label: str):
    """Log oldest and newest candle timestamps so we can verify data freshness."""
    if not candles:
        logger.warning(f"{label}: No candles in buffer!")
        return
    oldest = datetime.fromtimestamp(candles[0]["ts"], tz=timezone.utc).astimezone(IST)
    newest = datetime.fromtimestamp(candles[-1]["ts"], tz=timezone.utc).astimezone(IST)
    age_minutes = (datetime.now(timezone.utc).timestamp() - candles[-1]["ts"]) / 60
    logger.info(
        f"{label}: {len(candles)} candles | "
        f"Oldest: {oldest.strftime('%b %d %I:%M %p')} | "
        f"Newest: {newest.strftime('%b %d %I:%M %p')} IST | "
        f"Age: {age_minutes:.1f} mins ago"
    )


# ─── Signal ───────────────────────────────────────────────────────────────────

class PredictionSignal:

    def __init__(
        self,
        timestamp: datetime,
        market_open: datetime,
        market_close: datetime,
        direction: str,
        prob_up: float,
        prob_down: float,
        confidence: float,
        price_to_beat: float,
        meta_validated: bool,
        meta_confidence: float,
    ):
        self.timestamp = timestamp
        self.market_open = market_open
        self.market_close = market_close
        self.direction = direction
        self.prob_up = prob_up
        self.prob_down = prob_down
        self.confidence = confidence
        self.price = price_to_beat
        self.meta_validated = meta_validated
        self.meta_confidence = meta_confidence

    def format_signal(self) -> str:
        ist_open = self.market_open.astimezone(IST)
        ist_close = self.market_close.astimezone(IST)
        dir_label = "LONG (UP)" if self.direction == "UP" else "SHORT (DOWN)"
        dir_emoji = "GREEN" if self.direction == "UP" else "RED"
        time_fmt = "%I:%M %p"
        validated = (
            "Meta-validated OK" if self.meta_validated else "Low confidence"
        )
        return (
            f"BTC SIGNAL (5 MIN)\n\n"
            f"[{dir_emoji}] Direction: {dir_label}\n\n"
            f"Price to Beat: ${self.price:,.2f}\n\n"
            f"Probabilities\n"
            f"UP:   {self.prob_up * 100:.1f}%\n"
            f"DOWN: {self.prob_down * 100:.1f}%\n\n"
            f"Confidence: {self.confidence * 100:.1f}%\n\n"
            f"Time: {ist_open.strftime(time_fmt)}"
            f" - {ist_close.strftime(time_fmt)} (IST)\n\n"
            f"{validated}"
        )

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "market_open": self.market_open.isoformat(),
            "market_close": self.market_close.isoformat(),
            "direction": self.direction,
            "prob_up": round(self.prob_up, 4),
            "prob_down": round(self.prob_down, 4),
            "confidence": round(self.confidence, 4),
            "price": round(self.price, 2),
            "meta_validated": self.meta_validated,
            "meta_confidence": round(self.meta_confidence, 4),
        }


# ─── Engine ───────────────────────────────────────────────────────────────────

class PredictionEngine:

    def __init__(self, stream_manager: PriceStreamManager):
        self.stream_manager = stream_manager
        self.ensemble = None
        self.meta_model = None
        self.regime_detector = None
        self.feature_names: List[str] = []
        self._signal_callbacks: List[Callable] = []
        self._prediction_logger: Optional[PredictionLogger] = None
        self._loaded = False

        self._candles_5m: list = []
        self._candles_1m: list = []
        self._last_cache_save: float = 0
        self._last_window_ts: float = 0

    def load_models(self):
        try:
            self.ensemble = joblib.load(MODELS_DIR / "ensemble_model.joblib")
            raw_meta = joblib.load(MODELS_DIR / "meta_model.joblib")

            # Wrap raw classifiers that don't have a validate_single() method
            # (e.g. a plain LGBMClassifier saved directly during training).
            if hasattr(raw_meta, "validate_single"):
                self.meta_model = raw_meta
                logger.info("Meta model loaded (MetaLabelModel).")
            else:
                self.meta_model = _RawClassifierMetaAdapter(raw_meta)
                logger.info(
                    "Meta model loaded (raw classifier wrapped with "
                    "_RawClassifierMetaAdapter)."
                )

            self.regime_detector = joblib.load(
                MODELS_DIR / "regime_detector.joblib"
            )
            self.feature_names = joblib.load(
                MODELS_DIR / "feature_names.joblib"
            )
            self._prediction_logger = PredictionLogger()
            self._loaded = True
            logger.info("All models loaded successfully.")
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}. Run training first.")
            self._loaded = False

    def _load_candles_from_parquet(self):
        """
        Fallback: load candles from local parquet files.
        Used only if Binance REST API is unavailable.
        """
        logger.info("Loading startup candles from local parquet files (fallback)...")
        cached_5m = _load_candles_from_cache(CANDLE_CACHE_5M)
        cached_1m = _load_candles_from_cache(CANDLE_CACHE_1M)
        historical_5m = _load_candles_from_parquet("5m", n=300)
        historical_1m = _load_candles_from_parquet("1m", n=300)

        # Merge: historical first, cache on top (cache is more recent)
        self._candles_5m = _merge_candles(historical_5m, cached_5m)
        self._candles_1m = _merge_candles(historical_1m, cached_1m)

        _log_candle_freshness(self._candles_5m, "5m candles at startup")
        logger.info("Candle buffer ready. Predictions start immediately!")

    async def _load_startup_candles(self):
        """
        Fetch fresh candles from Binance REST API at startup.
        Falls back to parquet files if API fails.
        """
        logger.info("Fetching fresh startup candles from Binance REST API...")
        fetcher = BinanceCandleFetcher()

        try:
            candles = fetcher.fetch_all_timeframes()

            # Populate 5m buffer
            if candles.get("5m"):
                for c in candles["5m"]:
                    await self.stream_manager.candle_buffer_5m.inject_candle(c)
                logger.info(f"Loaded {len(candles['5m'])} fresh 5m candles")

            # Populate 1m buffer
            if candles.get("1m"):
                for c in candles["1m"]:
                    await self.stream_manager.candle_buffer_1m.inject_candle(c)
                logger.info(f"Loaded {len(candles['1m'])} fresh 1m candles")

            logger.info("✅ Startup candles loaded from Binance REST API (fresh data)")

        except Exception as e:
            logger.warning(f"Binance REST fetch failed: {e}")
            logger.warning("Falling back to local parquet files...")
            self._load_candles_from_parquet()

    def add_signal_callback(self, cb: Callable):
        self._signal_callbacks.append(cb)

    async def _emit_signal(self, signal: PredictionSignal):
        for cb in self._signal_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(signal)
                else:
                    cb(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

    async def _update_candle_cache(self):
        """
        Pull latest live candles from Binance buffer and merge.
        This is how today's fresh candles get into the prediction.
        """
        live_5m = await self.stream_manager.candle_buffer_5m.get_candles()
        live_1m = await self.stream_manager.candle_buffer_1m.get_candles()

        if live_5m:
            self._candles_5m = _merge_candles(self._candles_5m, live_5m)[-300:]

        if live_1m:
            self._candles_1m = _merge_candles(self._candles_1m, live_1m)[-300:]

        # Save to disk every 5 minutes
        now = time.time()
        if now - self._last_cache_save > 300:
            _save_candles(self._candles_5m, CANDLE_CACHE_5M)
            _save_candles(self._candles_1m, CANDLE_CACHE_1M)
            self._last_cache_save = now

    async def _compute_features(self) -> Optional[pd.DataFrame]:
        """
        Build all features from the candle buffer.
        Always uses the most recent candles available
        (historical + today's live candles merged together).
        """
        # Pull fresh live candles from Binance first
        await self._update_candle_cache()

        # Log exactly which candles are being used
        _log_candle_freshness(self._candles_5m, "5m prediction data")

        df_5m = _candles_to_df(self._candles_5m)
        if df_5m is None:
            logger.warning(
                f"Not enough candles: {len(self._candles_5m)}/{MIN_CANDLES}"
            )
            return None

        df_1m = _candles_to_df(self._candles_1m)

        try:
            feat_5m     = build_features(df_5m, "5m")
            feat_5m_ms  = build_microstructure_features(df_5m, "5m")
            feat_5m_vol = build_volatility_features(df_5m, "5m")
            feat_5m_liq = build_liquidity_features(df_5m, "5m")
            feat_5m_reg = build_regime_features(
                df_5m, self.regime_detector, "5m"
            )

            combined = pd.concat(
                [feat_5m, feat_5m_ms, feat_5m_vol,
                 feat_5m_liq, feat_5m_reg],
                axis=1,
            )

            # Add 1m features if available
            if df_1m is not None and len(df_1m) >= MIN_CANDLES:
                feat_1m    = build_features(df_1m, "1m")
                feat_1m_ms = build_microstructure_features(df_1m, "1m")
                feat_1m_all = pd.concat([feat_1m, feat_1m_ms], axis=1)
                feat_1m_5m  = feat_1m_all.resample("5min").last()
                for col in feat_1m_all.columns:
                    combined[col] = feat_1m_5m[col].reindex(
                        combined.index, method="ffill"
                    )

            # Add live microstructure from recent trades
            trades = await self.stream_manager.trade_buffer.get_trades(200)
            live_ms = update_microstructure_from_trade(trades)
            for k, v in live_ms.items():
                combined[f"live_{k}"] = v

            return combined

        except Exception as e:
            logger.error(f"Feature computation error: {e}", exc_info=True)
            return None

    async def _generate_and_emit(
        self,
        price_to_beat: float,
        window_open: datetime,
        window_close: datetime,
    ):
        """
        Run models and emit signal.
        Called within seconds of settlement.
        Uses freshest available candles including today's live ones.
        """
        if not self._loaded:
            return

        feature_matrix = await self._compute_features()
        if feature_matrix is None:
            logger.error("Feature matrix empty — skipping prediction.")
            return

        # Always use the very last row = most recent market state
        latest = feature_matrix.iloc[[-1]]
        X = _align_features(latest, self.feature_names)

        # Log feature row timestamp so we know how fresh it is
        latest_ts = feature_matrix.index[-1].astimezone(IST)
        logger.info(
            f"Predicting from feature row: "
            f"{latest_ts.strftime('%b %d %I:%M %p')} IST"
        )

        # Ensemble prediction
        proba = self.ensemble.predict_proba(X)
        prob_up   = float(proba[0, 1])
        prob_down = float(proba[0, 0])
        direction = "UP" if prob_up > 0.5 else "DOWN"
        confidence = max(prob_up, prob_down)

        # Meta validation
        meta_result    = self.meta_model.validate_single(
            self.ensemble, X, prob_up
        )
        meta_validated  = meta_result["should_trade"]
        meta_confidence = meta_result["meta_confidence"]

        signal = PredictionSignal(
            timestamp=datetime.now(timezone.utc),
            market_open=window_open,
            market_close=window_close,
            direction=direction,
            prob_up=prob_up,
            prob_down=prob_down,
            confidence=confidence,
            price_to_beat=price_to_beat,
            meta_validated=meta_validated,
            meta_confidence=meta_confidence,
        )

        if self._prediction_logger:
            self._prediction_logger.log_prediction(signal)

        ist_open  = window_open.astimezone(IST)
        ist_close = window_close.astimezone(IST)
        logger.info(
            f"SIGNAL: {direction} | "
            f"UP={prob_up:.3f} DOWN={prob_down:.3f} | "
            f"Meta={'OK' if meta_validated else 'SKIP'} "
            f"({meta_confidence:.3f}) | "
            f"Price to Beat=${price_to_beat:,.2f} | "
            f"Window={ist_open.strftime('%I:%M')}"
            f"-{ist_close.strftime('%I:%M %p')} IST"
        )

        await self._emit_signal(signal)

    async def run(self):
        """
        Main loop.

        Every 5 minutes exactly:
          1. Detect new settlement boundary
          2. Pull latest live candles from Binance into buffer
          3. Fetch Chainlink settlement price (8s timeout)
          4. Generate prediction using fresh candles
          5. Send to Telegram within 5-10 seconds of settlement
        """
        logger.info("Prediction engine started.")

        # NEW: Load fresh startup candles before anything else
        await self._load_startup_candles()

        # Then wait for live WebSocket price as before
        logger.info("Waiting for live price data...")
        for _ in range(60):
            price = await self.stream_manager.get_current_price()
            if price:
                logger.info(f"✅ Live price received: ${price:,.2f}")
                break
            await asyncio.sleep(1)

        logger.info("Prediction engine ready.")
        logger.info(
            "System ready. Predicting within 10 seconds of each settlement."
        )

        window_seconds = 300

        while True:
            try:
                now = time.time()

                # Which 5-min window are we currently in?
                current_window_start = (int(now) // window_seconds) * window_seconds

                # Already handled this window? Sleep until next boundary
                if current_window_start <= self._last_window_ts:
                    next_boundary = current_window_start + window_seconds
                    sleep_for = next_boundary - time.time() - 0.1
                    await asyncio.sleep(min(max(sleep_for, 0), 1.0))
                    continue

                # Are we within the first 15 seconds of a new window?
                seconds_into_window = now - current_window_start
                if seconds_into_window > 15:
                    # Missed it — mark as done and wait for next
                    self._last_window_ts = current_window_start
                    await asyncio.sleep(1)
                    continue

                # Mark this window as being processed
                self._last_window_ts = current_window_start

                window_open = datetime.fromtimestamp(
                    current_window_start, tz=timezone.utc
                )
                window_close = window_open + timedelta(seconds=window_seconds)
                ist_open  = window_open.astimezone(IST)
                ist_close = window_close.astimezone(IST)

                logger.info(
                    f"New window: "
                    f"{ist_open.strftime('%I:%M')}"
                    f"-{ist_close.strftime('%I:%M %p')} IST | "
                    f"Fetching settlement price..."
                )

                # Step 1: Fetch Chainlink settlement price
                price_to_beat = await self.stream_manager.fetch_settlement_now()

                if price_to_beat is None:
                    logger.error("No settlement price. Skipping window.")
                    continue

                logger.info(
                    f"Settlement price locked: ${price_to_beat:,.2f} | "
                    f"Generating prediction..."
                )

                # Step 2: Generate prediction using latest candles
                await self._generate_and_emit(
                    price_to_beat=price_to_beat,
                    window_open=window_open,
                    window_close=window_close,
                )

                # Sleep until just before next boundary
                next_boundary = current_window_start + window_seconds
                sleep_for = next_boundary - time.time() - 2.0
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)

            except asyncio.CancelledError:
                logger.info("Prediction engine stopping...")
                _save_candles(self._candles_5m, CANDLE_CACHE_5M)
                _save_candles(self._candles_1m, CANDLE_CACHE_1M)
                logger.info(
                    f"Saved {len(self._candles_5m)} candles. "
                    f"Safe to restart."
                )
                break
            except Exception as e:
                logger.error(f"Prediction loop error: {e}", exc_info=True)
                await asyncio.sleep(5)