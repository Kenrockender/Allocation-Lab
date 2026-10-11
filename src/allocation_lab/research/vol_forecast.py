"""
vol_forecast.py — Volatility forecasting models for Phase 6.

Two models:
  1. EWMAVolatility   — exponentially weighted moving average baseline (no ML)
  2. GRUVolatility    — small GRU that forecasts next-period realised vol

Both share a common interface:
    model.fit(log_returns_train)          -> self
    model.predict(log_returns_history)    -> float  (annualised vol forecast)

Walk-forward training is handled by vol_walkforward.py.

Design choices (keep model small — data is limited):
  - Input  : rolling window of daily log-returns (lookback=63 days)
  - Target : next 21-day realised vol (annualised), computed from actual returns
  - Loss   : MSE on log(vol) — stabilises gradient when vol spikes
  - Layers : 1-layer GRU, hidden_size=32, followed by a linear head
  - Epochs : up to 200 with early stopping (patience=20)
  - No external data — only price-derived features
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

LOOKBACK = 63     # input window (trading days)
HORIZON  = 21     # forecast horizon (trading days ≈ 1 month)
ANN      = 252    # annualisation factor


# ------------------------------------------------------------------ #
# 1. EWMA baseline                                                     #
# ------------------------------------------------------------------ #

class EWMAVolatility:
    """RiskMetrics-style EWMA vol forecast.

    σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}
    Forecast for next period: σ_{t+1} (annualised).

    λ=0.94 is the RiskMetrics daily decay factor.
    """

    name = "ewma"

    def __init__(self, lam: float = 0.94):
        self.lam = lam
        self._var: float | None = None

    def fit(self, log_returns: pd.Series) -> "EWMAVolatility":
        """Initialise variance from the training window."""
        self._var = float(log_returns.var())
        for r in log_returns:
            self._var = self.lam * self._var + (1 - self.lam) * r ** 2
        return self

    def predict(self, log_returns: pd.Series) -> float:
        """Update EWMA with all returns in log_returns and return annualised vol."""
        if self._var is None:
            self._var = float(log_returns.var())
        var = self._var
        for r in log_returns:
            var = self.lam * var + (1 - self.lam) * r ** 2
        self._var = var
        return float(np.sqrt(var * ANN))

    def predict_series(self, log_returns: pd.Series) -> pd.Series:
        """Return a daily series of one-step-ahead annualised vol forecasts."""
        if self._var is None:
            self._var = float(log_returns.iloc[0] ** 2)
        var = self._var
        vols = []
        for r in log_returns:
            var = self.lam * var + (1 - self.lam) * r ** 2
            vols.append(np.sqrt(var * ANN))
        self._var = var
        return pd.Series(vols, index=log_returns.index, name="ewma_vol")


# ------------------------------------------------------------------ #
# 2. GRU volatility forecaster                                         #
# ------------------------------------------------------------------ #

class GRUVolatility:
    """Small GRU that predicts next-21-day realised volatility.

    Feature engineering (all computed from log returns):
      - raw log return
      - |log return|   (absolute value)
      - r²             (squared return)

    So the input shape is (batch, lookback, 3).
    Target: log(realised_vol_next_21_days) — log-scale stabilises training.
    """

    name = "gru"

    def __init__(
        self,
        lookback: int = LOOKBACK,
        horizon: int = HORIZON,
        hidden_size: int = 32,
        lr: float = 1e-3,
        max_epochs: int = 200,
        patience: int = 20,
        seed: int = 42,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self._model = None
        self._scaler_mean: float = 0.0
        self._scaler_std:  float = 1.0

    # ---- internal helpers ----------------------------------------- #

    def _make_features(self, log_returns: np.ndarray) -> np.ndarray:
        """Shape: (n_samples, lookback, 3)."""
        r = log_returns.reshape(-1)
        X = []
        for i in range(self.lookback, len(r)):
            window = r[i - self.lookback: i]
            feat = np.stack([window, np.abs(window), window ** 2], axis=1)
            X.append(feat)
        return np.array(X, dtype=np.float32)

    def _make_targets(self, log_returns: np.ndarray) -> np.ndarray:
        """Realised vol over the next `horizon` days (annualised, log-scale)."""
        r = log_returns.reshape(-1)
        y = []
        for i in range(self.lookback, len(r) - self.horizon + 1):
            rv = float(np.std(r[i: i + self.horizon]) * np.sqrt(ANN))
            y.append(np.log(rv + 1e-8))
        return np.array(y, dtype=np.float32)

    def _build_model(self) -> "torch.nn.Module":
        import torch.nn as nn

        class _GRUNet(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x):
                _, h = self.gru(x)
                return self.head(h.squeeze(0)).squeeze(-1)

        return _GRUNet(input_size=3, hidden_size=self.hidden_size)

    # ---- public API ----------------------------------------------- #

    def fit(self, log_returns: pd.Series) -> "GRUVolatility":
        """Train on log_returns. Returns self."""
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        r = log_returns.values.astype(np.float32)
        X = self._make_features(r)
        y = self._make_targets(r)

        # Align lengths (target is shorter by horizon-1)
        n = min(len(X), len(y))
        X, y = X[:n], y[:n]

        if len(X) < 30:
            # Not enough data — fall back to EWMA
            self._model = None
            return self

        # Normalise features channel-wise
        self._scaler_mean = X.mean(axis=(0, 1), keepdims=True)
        self._scaler_std  = X.std(axis=(0, 1), keepdims=True) + 1e-8
        X_norm = (X - self._scaler_mean) / self._scaler_std

        X_t = torch.tensor(X_norm).to(device)
        y_t = torch.tensor(y).to(device)

        split = int(0.85 * len(X_t))
        ds_tr = TensorDataset(X_t[:split], y_t[:split])
        ds_va = TensorDataset(X_t[split:], y_t[split:])
        dl_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)

        model = self._build_model().to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            model.train()
            for xb, yb in dl_tr:
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            # Validation
            model.eval()
            with torch.no_grad():
                xv, yv = ds_va.tensors
                val_loss = float(loss_fn(model(xv), yv))

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        self._model = model.to(device)
        self._device = device
        return self

    def predict(self, log_returns: pd.Series) -> float:
        """Forecast annualised vol for the next horizon using the last `lookback` returns."""
        if self._model is None:
            # Fallback to simple realised vol
            return float(log_returns.tail(self.lookback).std() * np.sqrt(ANN))

        import torch
        r = log_returns.values[-self.lookback:].astype(np.float32)
        if len(r) < self.lookback:
            return float(log_returns.std() * np.sqrt(ANN))

        feat = np.stack([r, np.abs(r), r ** 2], axis=1).astype(np.float32)
        feat_norm = (feat - self._scaler_mean[0]) / self._scaler_std[0]
        x = torch.tensor(feat_norm).unsqueeze(0)   # (1, lookback, 3)

        device = getattr(self, "_device", torch.device("cpu"))
        x = x.to(device)

        self._model.eval()
        with torch.no_grad():
            log_vol_pred = float(self._model(x).item())

        return float(np.exp(log_vol_pred))

    def predict_series(self, log_returns: pd.Series) -> pd.Series:
        """Return a daily series of annualised vol forecasts (rolling predict)."""
        preds = []
        dates = []
        for i in range(self.lookback, len(log_returns)):
            window = log_returns.iloc[: i + 1]
            preds.append(self.predict(window))
            dates.append(log_returns.index[i])
        return pd.Series(preds, index=dates, name="gru_vol")
