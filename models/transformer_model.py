"""
Transformer-based time-series classifier for BTC direction prediction.
"""
import logging
import numpy as np
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class TransformerModel:
    def __init__(self, params: dict = None):
        from config import get_config
        cfg = get_config()
        default_params = cfg.get("models", {}).get("transformer", {})
        self.params = params or default_params
        self.model = None
        self.scaler = None
        self.seq_len = self.params.get("sequence_length", 60)
        self._device = None

    def _build_model(self, n_features: int):
        import torch
        import torch.nn as nn
        import math

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, dropout=0.1, max_len=500):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x):
                x = x + self.pe[:, :x.size(1)]
                return self.dropout(x)

        class TSTransformer(nn.Module):
            def __init__(self, n_features, d_model, nhead, num_layers,
                         dim_feedforward, dropout):
                super().__init__()
                self.input_proj = nn.Linear(n_features, d_model)
                self.pos_enc = PositionalEncoding(d_model, dropout)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(d_model, 2)

            def forward(self, x):
                x = self.input_proj(x)
                x = self.pos_enc(x)
                x = self.transformer(x)
                x = x.transpose(1, 2)
                x = self.pool(x).squeeze(-1)
                return self.fc(x)

        d_model = self.params.get("d_model", 64)
        # Ensure d_model is divisible by nhead
        nhead = self.params.get("nhead", 8)
        if d_model % nhead != 0:
            d_model = (d_model // nhead) * nhead

        return TSTransformer(
            n_features=n_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=self.params.get("num_encoder_layers", 3),
            dim_feedforward=self.params.get("dim_feedforward", 256),
            dropout=self.params.get("dropout", 0.1),
        )

    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray = None):
        sequences, labels = [], []
        for i in range(self.seq_len, len(X)):
            sequences.append(X[i - self.seq_len:i])
            if y is not None:
                labels.append(y[i])
        X_seq = np.array(sequences, dtype=np.float32)
        y_seq = np.array(labels, dtype=np.int64) if y is not None else None
        return X_seq, y_seq

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "TransformerModel":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X_arr = self.scaler.fit_transform(
            X_train.values if hasattr(X_train, "values") else X_train
        )
        y_arr = y_train.values if hasattr(y_train, "values") else np.array(y_train)
        X_seq, y_seq = self._prepare_sequences(X_arr, y_arr)

        n_features = X_seq.shape[2]
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(n_features).to(self._device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params.get("learning_rate", 0.0005),
            weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.params.get("epochs", 50)
        )

        loader = DataLoader(
            TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq)),
            batch_size=self.params.get("batch_size", 64),
            shuffle=True,
        )

        patience = self.params.get("patience", 10)
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        val_loader = None

        if X_val is not None and y_val is not None:
            X_v = self.scaler.transform(
                X_val.values if hasattr(X_val, "values") else X_val
            )
            y_v = y_val.values if hasattr(y_val, "values") else np.array(y_val)
            X_vs, y_vs = self._prepare_sequences(X_v, y_v)
            val_loader = DataLoader(
                TensorDataset(torch.tensor(X_vs), torch.tensor(y_vs)),
                batch_size=64
            )

        for epoch in range(self.params.get("epochs", 50)):
            self.model.train()
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self._device), y_b.to(self._device)
                optimizer.zero_grad()
                loss = criterion(self.model(X_b), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            if val_loader:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_b, y_b in val_loader:
                        val_loss += criterion(
                            self.model(X_b.to(self._device)), y_b.to(self._device)
                        ).item()
                val_loss /= len(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Transformer early stopping at epoch {epoch + 1}")
                        break

        if best_state:
            self.model.load_state_dict(best_state)
        self.model.eval()
        logger.info("Transformer training complete.")
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        X_arr = self.scaler.transform(X.values if hasattr(X, "values") else X)
        X_seq, _ = self._prepare_sequences(X_arr)

        if len(X_seq) == 0:
            return np.array([[0.5, 0.5]])

        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(X_seq).to(self._device))
            proba = F.softmax(out, dim=1).cpu().numpy()

        n_pad = len(X) - len(proba)
        if n_pad > 0:
            pad = np.full((n_pad, 2), 0.5)
            proba = np.vstack([pad, proba])
        return proba

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: str = "models/transformer_model.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = "models/transformer_model.joblib") -> "TransformerModel":
        return joblib.load(path)
