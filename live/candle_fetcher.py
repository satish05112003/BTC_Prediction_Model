"""
Fetches recent historical candles from Binance REST API at startup.
Used to warm up CandleBuffer before live WebSocket stream takes over.
"""
import logging
import time
import requests
from typing import List, Dict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

BINANCE_REST_URL = "https://api.binance.com/api/v3"
SYMBOL = "BTCUSDT"

class BinanceCandleFetcher:

    def fetch_candles(
        self,
        interval: str,        # "5m", "1m", "1h"
        limit: int = 300,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> List[Dict]:
        """
        Fetch the latest `limit` closed candles from Binance REST API.
        """
        params = {
            "symbol": SYMBOL,
            "interval": interval,
            "limit": limit + 1
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{BINANCE_REST_URL}/klines", params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                # Drop the last unclosed candle
                closed_candles = data[:-1]
                
                # Ensure we only return `limit` candles
                results = []
                for kline in closed_candles[-limit:]:
                    ts = kline[0] / 1000.0
                    results.append({
                        "ts": ts,
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5]),
                    })
                
                if results:
                    oldest = datetime.fromtimestamp(results[0]["ts"], tz=timezone.utc)
                    newest = datetime.fromtimestamp(results[-1]["ts"], tz=timezone.utc)
                    logger.info(
                        f"Fetched {len(results)} x {interval} candles | "
                        f"oldest: {oldest.strftime('%Y-%m-%d %H:%M')} UTC | "
                        f"newest: {newest.strftime('%Y-%m-%d %H:%M')} UTC"
                    )
                return results

            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to fetch {interval} candles: {e}")
                time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

        return []

    def fetch_all_timeframes(self) -> Dict[str, List[Dict]]:
        """
        Fetch 300 candles for all three timeframes: 5m, 1m, 1h.
        """
        start_time = time.time()
        timeframes = ["5m", "1m", "1h"]
        results = {}
        
        for tf in timeframes:
            try:
                results[tf] = self.fetch_candles(interval=tf, limit=300)
            except Exception as e:
                logger.warning(f"Failed to fetch {tf} candles: {e}")
                results[tf] = []
                
        elapsed = time.time() - start_time
        logger.info(f"Startup candle fetch complete in {elapsed:.1f} seconds")
        return results
