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

# FIX 1: Use Binance public data endpoint (works on cloud servers)
BINANCE_REST_URL = "https://data-api.binance.vision/api/v3"

SYMBOL = "BTCUSDT"


class BinanceCandleFetcher:

    def fetch_candles(
        self,
        interval: str,
        limit: int = 300,
        max_retries: int = 5,      # increased retries
        retry_delay: float = 3.0,
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

                url = f"{BINANCE_REST_URL}/klines"

                response = requests.get(
                    url,
                    params=params,
                    timeout=15
                )

                response.raise_for_status()

                data = response.json()

                if not data:
                    raise RuntimeError("Empty response from Binance")

                # Drop last unclosed candle
                closed_candles = data[:-1]

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

                    oldest = datetime.fromtimestamp(
                        results[0]["ts"],
                        tz=timezone.utc
                    )

                    newest = datetime.fromtimestamp(
                        results[-1]["ts"],
                        tz=timezone.utc
                    )

                    logger.info(
                        f"Fetched {len(results)} x {interval} candles | "
                        f"oldest: {oldest.strftime('%Y-%m-%d %H:%M')} UTC | "
                        f"newest: {newest.strftime('%Y-%m-%d %H:%M')} UTC"
                    )

                return results

            except requests.HTTPError as e:

                if e.response.status_code == 451:
                    logger.error(
                        "Binance API returned 451 (region blocked). "
                        "Using public data endpoint."
                    )

                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to fetch {interval} candles: {e}"
                    )

                logger.warning(
                    f"{interval} candle fetch failed "
                    f"(attempt {attempt + 1}/{max_retries}). Retrying..."
                )

                time.sleep(retry_delay)

            except (requests.ConnectionError, requests.Timeout) as e:

                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to fetch {interval} candles: {e}"
                    )

                logger.warning(
                    f"Network error fetching {interval} candles. Retrying..."
                )

                time.sleep(retry_delay)

            except Exception as e:

                logger.error(f"Unexpected error while fetching candles: {e}")

                if attempt == max_retries - 1:
                    raise

                time.sleep(retry_delay)

        return []

    def fetch_all_timeframes(self) -> Dict[str, List[Dict]]:

        """
        Fetch candles for multiple timeframes.
        """

        start_time = time.time()

        timeframes = ["5m", "1m", "1h"]

        results = {}

        for tf in timeframes:

            try:

                candles = self.fetch_candles(interval=tf, limit=300)

                if not candles:
                    logger.warning(
                        f"No candles received for {tf}. "
                        f"Prediction engine may retry."
                    )

                results[tf] = candles

            except Exception as e:

                logger.warning(f"Failed to fetch {tf} candles: {e}")

                results[tf] = []

        elapsed = time.time() - start_time

        logger.info(
            f"Startup candle fetch complete in {elapsed:.1f} seconds"
        )

        return results
