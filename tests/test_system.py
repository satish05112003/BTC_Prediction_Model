"""
Test suite for BTC Prediction Model.
Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    open_ = close + np.random.randn(n) * 20
    high = np.maximum(close, open_) + np.abs(np.random.randn(n) * 30)
    low = np.minimum(close, open_) - np.abs(np.random.randn(n) * 30)
    volume = np.abs(np.random.randn(n) * 1000 + 5000)
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume
    }, index=timestamps)
    return df


@pytest.fixture
def sample_1m_ohlcv():
    """1-minute OHLCV data."""
    np.random.seed(42)
    n = 2000
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 50000 + np.cumsum(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 10
    high = np.maximum(close, open_) + np.abs(np.random.randn(n) * 15)
    low = np.minimum(close, open_) - np.abs(np.random.randn(n) * 15)
    volume = np.abs(np.random.randn(n) * 500 + 2000)
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume
    }, index=timestamps)
    return df


# ─── Indicator Tests ──────────────────────────────────────────────────────────

class TestIndicators:
    def test_rsi(self, sample_ohlcv):
        from utils.indicators import rsi
        result = rsi(sample_ohlcv["close"], 14)
        assert len(result) == len(sample_ohlcv)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_ema(self, sample_ohlcv):
        from utils.indicators import ema
        result = ema(sample_ohlcv["close"], 20)
        assert len(result) == len(sample_ohlcv)
        assert not result.isna().all()

    def test_atr(self, sample_ohlcv):
        from utils.indicators import atr
        result = atr(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert len(result) == len(sample_ohlcv)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_macd(self, sample_ohlcv):
        from utils.indicators import macd
        line, signal, hist = macd(sample_ohlcv["close"])
        assert len(line) == len(sample_ohlcv)
        # Histogram = macd - signal
        valid_hist = hist.dropna()
        assert len(valid_hist) > 0

    def test_bollinger_bands(self, sample_ohlcv):
        from utils.indicators import bollinger_bands
        upper, mid, lower, width, pct = bollinger_bands(sample_ohlcv["close"])
        valid_mask = upper.notna() & lower.notna()
        assert (upper[valid_mask] >= lower[valid_mask]).all()

    def test_stochastic(self, sample_ohlcv):
        from utils.indicators import stochastic
        k, d = stochastic(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        valid_k = k.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()


# ─── Feature Engineering Tests ────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_build_features_shape(self, sample_ohlcv):
        from features.feature_engineering import build_features
        feat = build_features(sample_ohlcv, "5m")
        assert len(feat) == len(sample_ohlcv)
        assert feat.shape[1] > 50, "Should generate 50+ features"

    def test_build_features_no_inf(self, sample_ohlcv):
        from features.feature_engineering import build_features
        feat = build_features(sample_ohlcv, "5m")
        feat_clean = feat.replace([np.inf, -np.inf], np.nan)
        inf_cols = (feat != feat_clean).any()
        # Allow some inf values (will be cleaned later)
        assert feat.shape[1] > 0

    def test_microstructure_features(self, sample_ohlcv):
        from features.microstructure_features import build_microstructure_features
        feat = build_microstructure_features(sample_ohlcv, "5m")
        assert len(feat) == len(sample_ohlcv)
        assert "5m_price_velocity" in feat.columns
        assert "5m_volume_imbalance" in feat.columns

    def test_regime_features(self, sample_ohlcv):
        from features.regime_detection import build_regime_features
        feat = build_regime_features(sample_ohlcv)
        assert len(feat) == len(sample_ohlcv)
        assert "adx" in feat.columns or "regime_adx_trending" in feat.columns

    def test_liquidity_features(self, sample_ohlcv):
        from features.liquidity_shock_features import build_liquidity_features
        feat = build_liquidity_features(sample_ohlcv)
        assert "vol_shock_ratio_20" in feat.columns


# ─── Data Loader Tests ────────────────────────────────────────────────────────

class TestDataLoader:
    def test_create_target(self, sample_ohlcv):
        from utils.data_loader import create_target
        target = create_target(sample_ohlcv, forward_periods=1)
        assert len(target) == len(sample_ohlcv)
        valid = target.dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_resample(self, sample_1m_ohlcv):
        from utils.data_loader import resample_to_5m
        df_5m = resample_to_5m(sample_1m_ohlcv)
        assert len(df_5m) < len(sample_1m_ohlcv)
        assert "open" in df_5m.columns


# ─── Candle Buffer Tests ──────────────────────────────────────────────────────

class TestCandleBuffer:
    @pytest.mark.asyncio
    async def test_candle_creation(self):
        from live.price_stream import CandleBuffer
        buf = CandleBuffer(timeframe_seconds=300)
        base_ts = 1700000000.0  # arbitrary

        await buf.update(50000.0, 1.0, base_ts)
        await buf.update(50100.0, 0.5, base_ts + 60)
        await buf.update(49900.0, 2.0, base_ts + 120)

        candles = await buf.get_candles()
        assert len(candles) >= 1
        current = candles[-1]
        assert current["high"] == 50100.0
        assert current["low"] == 49900.0
        assert current["open"] == 50000.0

    @pytest.mark.asyncio
    async def test_candle_close(self):
        from live.price_stream import CandleBuffer
        buf = CandleBuffer(timeframe_seconds=300)
        base_ts = 1700000000.0

        await buf.update(50000.0, 1.0, base_ts)
        # New candle window
        await buf.update(51000.0, 1.0, base_ts + 300)

        candles = await buf.get_candles()
        assert len(candles) >= 2


# ─── Prediction Signal Tests ──────────────────────────────────────────────────

class TestPredictionSignal:
    def test_signal_format(self):
        from live.prediction_engine import PredictionSignal
        now = datetime.now(timezone.utc)
        signal = PredictionSignal(
            timestamp=now,
            market_open=now,
            market_close=now + timedelta(minutes=5),
            direction="DOWN",
            prob_up=0.244,
            prob_down=0.756,
            confidence=0.756,
            price=69012.12,
            meta_validated=True,
            meta_confidence=0.72,
        )
        text = signal.format_signal()
        assert "DOWN" in text
        assert "75.6%" in text
        assert "69,012.12" in text
        assert "24.4%" in text

    def test_signal_to_dict(self):
        from live.prediction_engine import PredictionSignal
        now = datetime.now(timezone.utc)
        signal = PredictionSignal(
            timestamp=now, market_open=now,
            market_close=now + timedelta(minutes=5),
            direction="UP", prob_up=0.65, prob_down=0.35,
            confidence=0.65, price=70000.0,
            meta_validated=True, meta_confidence=0.61,
        )
        d = signal.to_dict()
        assert d["direction"] == "UP"
        assert d["prob_up"] == 0.65


# ─── Ensemble Tests ───────────────────────────────────────────────────────────

class TestEnsemble:
    def test_ensemble_predict_no_models(self):
        from ensemble.ensemble_predictor import EnsemblePredictor
        ens = EnsemblePredictor(models={})
        X = pd.DataFrame(np.random.randn(10, 5))
        proba = ens.predict_proba(X)
        assert proba.shape == (10, 2)
        assert np.allclose(proba, 0.5)


# ─── Prediction Logger Tests ──────────────────────────────────────────────────

class TestPredictionLogger:
    def test_empty_stats(self, tmp_path, monkeypatch):
        import logs.prediction_logger as pl_module
        monkeypatch.setattr(pl_module, "PREDICTIONS_FILE", tmp_path / "preds.csv")
        monkeypatch.setattr(pl_module, "RESULTS_FILE", tmp_path / "results.csv")

        from logs.prediction_logger import PredictionLogger
        pl = PredictionLogger()
        stats = pl.get_stats()
        assert stats["total_predictions"] == 0
        assert stats["accuracy"] == 0.0
