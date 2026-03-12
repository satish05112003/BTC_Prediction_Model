"""
Real-time price streaming.
- Binance WebSocket: continuous live BTC price + candle building
- Chainlink (Polymarket): fetched ONLY at exact 5-min settlement moment
"""

import asyncio
import json
import logging
import time
from collections import deque
from typing import Optional, Dict

import websockets

from config import get_config

try:
    import orjson
    def loads(x): return orjson.loads(x)
    def dumps(x): return orjson.dumps(x).decode()
except ImportError:
    def loads(x): return json.loads(x)
    def dumps(x): return json.dumps(x)

logger = logging.getLogger(__name__)
cfg = get_config()


class CandleBuffer:

    def __init__(self, timeframe_seconds: int = 300, buffer_size: int = 500):
        self.tf_seconds = timeframe_seconds
        self._candles = deque(maxlen=buffer_size)
        self._current: Optional[Dict] = None
        self._lock = asyncio.Lock()

    def _candle_start(self, ts: float) -> float:
        return (int(ts) // self.tf_seconds) * self.tf_seconds

    async def update(self, price: float, volume: float, ts: float):
        async with self._lock:
            candle_start = self._candle_start(ts)
            if self._current is None:
                self._current = self._new_candle(candle_start, price, volume)
                return
            if candle_start > self._current["ts"]:
                self._candles.append(dict(self._current))
                self._current = self._new_candle(candle_start, price, volume)
            else:
                c = self._current
                c["high"] = max(c["high"], price)
                c["low"] = min(c["low"], price)
                c["close"] = price
                c["volume"] += volume

    def _new_candle(self, ts: float, price: float, volume: float) -> Dict:
        return {
            "ts": ts,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume,
        }

    async def inject_candle(self, candle: dict):
        """
        Inject a pre-built OHLCV candle dict directly into the closed candles list.

        candle format: {ts, open, high, low, close, volume}

        This is used ONLY at startup to warm up the buffer with
        historical candles from Binance REST API.

        After injection, live WebSocket ticks will continue as normal.
        Injected candles are treated exactly like naturally closed candles.

        Inject at the END of the buffer (most recent = last).
        Keep max buffer size enforced (drop oldest if over limit).
        """
        async with self._lock:
            self._candles.append(candle)
            # Deque automatically drops oldest if over limit.

    async def get_candles(self, n: int = None) -> list:
        async with self._lock:
            candles = list(self._candles)
            if self._current:
                candles.append(dict(self._current))
            return candles[-n:] if n else candles

    async def get_current_price(self) -> Optional[float]:
        async with self._lock:
            return self._current["close"] if self._current else None

    async def count(self) -> int:
        async with self._lock:
            n = len(self._candles)
            if self._current:
                n += 1
            return n


class TradeBuffer:

    def __init__(self, maxsize: int = 1000):
        self._trades = deque(maxlen=maxsize)
        self._lock = asyncio.Lock()

    async def add(self, price: float, qty: float,
                  is_buyer_maker: bool, ts: float):
        async with self._lock:
            self._trades.append({
                "price": price,
                "qty": qty,
                "is_buyer_maker": is_buyer_maker,
                "ts": ts,
            })

    async def get_trades(self, n: int = 100) -> list:
        async with self._lock:
            return list(self._trades)[-n:]


class BinanceStreamer:
    """
    Continuous Binance aggTrade stream.
    Builds 5m and 1m candles from raw ticks in real time.
    """

    def __init__(self,
                 candle_buffer_5m: CandleBuffer,
                 candle_buffer_1m: CandleBuffer,
                 trade_buffer: TradeBuffer):
        self.candle_buffer_5m = candle_buffer_5m
        self.candle_buffer_1m = candle_buffer_1m
        self.trade_buffer = trade_buffer
        self._ws_url = cfg.get("binance", {}).get(
            "ws_url",
            "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
        )
        self._running = False
        self.last_price: Optional[float] = None
        self.last_update: float = 0

    async def start(self):
        self._running = True
        while self._running:
            try:
                await self._connect()
            except Exception as e:
                logger.warning(f"Binance stream error: {e}. Reconnecting in 3s...")
                await asyncio.sleep(3)

    async def _connect(self):
        async with websockets.connect(
            self._ws_url,
            ping_interval=20,
            ping_timeout=10,
            open_timeout=15,
        ) as ws:
            logger.info("Binance WebSocket connected.")
            async for msg in ws:
                if not self._running:
                    break
                try:
                    data = loads(msg)
                    if data.get("e") != "aggTrade":
                        continue
                    price = float(data["p"])
                    qty = float(data["q"])
                    maker = data["m"]
                    ts = data["T"] / 1000.0

                    self.last_price = price
                    self.last_update = time.time()

                    await self.candle_buffer_5m.update(price, qty, ts)
                    await self.candle_buffer_1m.update(price, qty, ts)
                    await self.trade_buffer.add(price, qty, maker, ts)
                except Exception:
                    pass

    def stop(self):
        self._running = False


class ChainlinkSettlementFetcher:
    """
    Fetches Polymarket Chainlink settlement price ONCE at boundary.
    Connects, reads one price, disconnects immediately.
    Not a continuous stream.
    """

    WS_URL = "wss://ws-live-data.polymarket.com"
    SUBSCRIBE_MSG = dumps({
        "action": "subscribe",
        "subscriptions": [{
            "topic": "crypto_prices_chainlink",
            "type": "*",
            "filters": '{"symbol":"btc/usd"}'
        }]
    })

    async def fetch_settlement_price(self, timeout: float = 8.0) -> Optional[float]:
        try:
            async with websockets.connect(
                self.WS_URL,
                ping_interval=None,
                ping_timeout=None,
                open_timeout=10,
                close_timeout=5,
            ) as ws:
                await ws.send(self.SUBSCRIBE_MSG)
                deadline = time.time() + timeout
                while time.time() < deadline:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    try:
                        msg = await asyncio.wait_for(
                            ws.recv(), timeout=remaining
                        )
                        data = loads(msg)
                        if data.get("topic") != "crypto_prices_chainlink":
                            continue
                        payload = data.get("payload") or {}
                        raw = payload.get("value")
                        if raw:
                            price = float(raw)
                            if price > 0:
                                logger.info(
                                    f"Chainlink settlement: ${price:,.2f}"
                                )
                                return price
                    except asyncio.TimeoutError:
                        break
                    except Exception:
                        break
        except Exception as e:
            logger.warning(f"Chainlink fetch failed: {e}")
        return None


class PriceStreamManager:

    def __init__(self):
        self.candle_buffer_5m = CandleBuffer(timeframe_seconds=300, buffer_size=500)
        self.candle_buffer_1m = CandleBuffer(timeframe_seconds=60, buffer_size=500)
        self.trade_buffer = TradeBuffer(maxsize=2000)

        self.binance_streamer = BinanceStreamer(
            self.candle_buffer_5m,
            self.candle_buffer_1m,
            self.trade_buffer,
        )

        self.chainlink_fetcher = ChainlinkSettlementFetcher()
        self.settlement_price: Optional[float] = None
        self.settlement_ts: float = 0

    async def start(self):
        logger.info("Starting Binance price stream...")
        await self.binance_streamer.start()

    async def fetch_settlement_now(self) -> Optional[float]:
        """
        Called ONCE at exact 5-min boundary.
        Tries Chainlink first, falls back to Binance price.
        """
        price = await self.chainlink_fetcher.fetch_settlement_price(timeout=8.0)
        if price:
            self.settlement_price = price
            self.settlement_ts = time.time()
            return price

        # Fallback: use live Binance price
        fallback = self.binance_streamer.last_price
        if fallback:
            logger.warning(
                f"Chainlink unavailable. Using Binance fallback: ${fallback:,.2f}"
            )
            self.settlement_price = fallback
            self.settlement_ts = time.time()
            return fallback

        return None

    async def get_current_price(self) -> Optional[float]:
        return self.binance_streamer.last_price

    def is_healthy(self) -> bool:
        return (time.time() - self.binance_streamer.last_update) < 30

    def stop(self):
        self.binance_streamer.stop()