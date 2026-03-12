import json
import os
import threading
from datetime import datetime
import pytz

class PredictionLogger:
    def __init__(self, file_path="logs/predictions.json"):
        self.file_path = file_path
        self.lock = threading.Lock()
        self.predictions = []
        self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.predictions = json.load(f)
            except Exception as e:
                print(f"Error loading predictions: {e}")
                self.predictions = []

    def _save(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.predictions, f, indent=2)
        except Exception as e:
            print(f"Error saving predictions: {e}")

    def log_prediction(self, signal) -> int:
        with self.lock:
            next_id = 1 if not self.predictions else self.predictions[-1].get("id", 0) + 1
            
            record = {
                "id": next_id,
                "timestamp": signal.timestamp.isoformat() if hasattr(signal.timestamp, "isoformat") else str(signal.timestamp),
                "direction": signal.direction,
                "prob_up": signal.prob_up,
                "prob_down": signal.prob_down,
                "confidence": signal.confidence,
                "price": signal.price,
                "market_open": signal.market_open.isoformat() if hasattr(signal.market_open, "isoformat") else str(signal.market_open),
                "market_close": signal.market_close.isoformat() if hasattr(signal.market_close, "isoformat") else str(signal.market_close),
                "meta_validated": signal.meta_validated,
                "meta_confidence": signal.meta_confidence,
                "outcome": None
            }
            self.predictions.append(record)
            self._save()
            return next_id

    def resolve_last_prediction(self, settlement_price: float):
        with self.lock:
            # Find most recent unresolved
            for i in range(len(self.predictions) - 1, -1, -1):
                if self.predictions[i]["outcome"] is None:
                    pred = self.predictions[i]
                    if not pred.get("meta_validated", True):
                        pred["outcome"] = "skip"
                    elif pred["direction"] == "UP" and settlement_price > pred["price"]:
                        pred["outcome"] = "win"
                    elif pred["direction"] == "DOWN" and settlement_price < pred["price"]:
                        pred["outcome"] = "win"
                    else:
                        pred["outcome"] = "loss"
                    
                    self._save()
                    break

    def get_all_predictions(self) -> list:
        with self.lock:
            return list(self.predictions)

    def get_today_predictions(self) -> list:
        ist = pytz.timezone("Asia/Kolkata")
        today_date = datetime.now(ist).date()
        
        result = []
        with self.lock:
            for p in self.predictions:
                try:
                    ts = datetime.fromisoformat(p["timestamp"])
                    if ts.tzinfo is None:
                        ts = pytz.utc.localize(ts)
                    local_ts = ts.astimezone(ist)
                    if local_ts.date() == today_date:
                        result.append(p)
                except Exception:
                    pass
        return result

    def get_stats(self, predictions: list) -> dict:
        total = len(predictions)
        wins = sum(1 for p in predictions if p.get("outcome") == "win")
        losses = sum(1 for p in predictions if p.get("outcome") == "loss")
        skipped = sum(1 for p in predictions if p.get("outcome") == "skip")
        
        resolved = wins + losses
        accuracy = (wins / resolved * 100) if resolved > 0 else 0.0
        
        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "skipped": skipped,
            "accuracy": accuracy
        }

    def get_last_n(self, n=5) -> list:
        with self.lock:
            return list(reversed(self.predictions[-n:]))
