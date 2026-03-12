"""
LSTM time-series model for BTC direction prediction.
"""
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LSTMModel:
    def __init__(self, params: dict = None):
        from config import get_config
        cfg = get_config()
        default_params = cfg.get("models", {}).get("lstm", {})
        self.params = params or default_params
        self.model = None
        self.scaler = None
        self.seq_len = self.params.get("sequence_length", 60)
        self.n_features = None

    def _build_model(self, n_features: int):
        import torch
        import torch.nn as nn

        class LSTMNet(nn.Module):
            def __init__(self, n_features, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    n_features, hidden_size, num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0.0
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, 2)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.dropout(out[:, -1, :])
                return self.fc(out)

        return LSTMNet(
            n_features=n_features,
            hidden_size=self.params.get("hidden_size", 128),
            num_layers=self.params.get("num_layers", 2),
            dropout=self.params.get("dropout", 0.3),
        )

    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """Convert flat feature array to sequences."""
        sequences = []
        labels = []
        for i in range(self.seq_len, len(X)):
            sequences.append(X[i - self.seq_len:i])
            if y is not None:
                labels.append(y[i])
        X_seq = np.array(sequences, dtype=np.float32)
        y_seq = np.array(labels, dtype=np.int64) if y is not None else None
        return X_seq, y_seq

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "LSTMModel":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler

        # Scale features
        self.scaler = StandardScaler()
        X_arr = self.scaler.fit_transform(
            X_train.values if hasattr(X_train, "values") else X_train
        )
        y_arr = y_train.values if hasattr(y_train, "values") else np.array(y_train)

        X_seq, y_seq = self._prepare_sequences(X_arr, y_arr)
        self.n_features = X_seq.shape[2]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(self.n_features).to(device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.get("learning_rate", 0.001)
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        dataset = TensorDataset(
            torch.tensor(X_seq), torch.tensor(y_seq)
        )
        loader = DataLoader(
            dataset,
            batch_size=self.params.get("batch_size", 64),
            shuffle=True
        )

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_arr = self.scaler.transform(
                X_val.values if hasattr(X_val, "values") else X_val
            )
            y_val_arr = y_val.values if hasattr(y_val, "values") else np.array(y_val)
            X_val_seq, y_val_seq = self._prepare_sequences(X_val_arr, y_val_arr)
            val_dataset = TensorDataset(
                torch.tensor(X_val_seq), torch.tensor(y_val_seq)
            )
            val_loader = DataLoader(val_dataset, batch_size=64)

        best_val_loss = float("inf")
        patience = self.params.get("patience", 10)
        patience_counter = 0
        best_state = None

        for epoch in range(self.params.get("epochs", 50)):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = self.model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_b, y_b in val_loader:
                        out = self.model(X_b.to(device))
                        val_loss += criterion(out, y_b.to(device)).item()
                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"LSTM early stopping at epoch {epoch + 1}")
                        break

            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

        if best_state:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self._device = device
        logger.info("LSTM training complete.")
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        X_arr = self.scaler.transform(
            X.values if hasattr(X, "values") else X
        )
        X_seq, _ = self._prepare_sequences(X_arr)

        if len(X_seq) == 0:
            return np.array([[0.5, 0.5]])

        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(X_seq).to(self._device))
            proba = F.softmax(out, dim=1).cpu().numpy()

        # Pad beginning (before seq_len) with 0.5
        n_pad = len(X) - len(proba)
        if n_pad > 0:
            pad = np.full((n_pad, 2), 0.5)
            proba = np.vstack([pad, proba])
        return proba

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: str = "models/lstm_model.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = "models/lstm_model.joblib") -> "LSTMModel":
        return joblib.load(path)
