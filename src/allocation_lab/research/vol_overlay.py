"""
vol_overlay.py — Volatility-scaling overlay for the backtester.

Idea: instead of fixed weights, scale each sleeve's allocation by a
"vol target" factor — reduce exposure when forecast vol is high, increase
when it's low, capped at 1.0 (no leverage, consistent with long-only rule).

  target_vol = 0.15  (15% annualised — roughly equal-weight DCA historical vol)
  scale_i    = min(1, target_vol / forecast_vol_i)
  weights    = base_weights_i * scale_i  (then renormalise to sum ≤ 1)

Two overlay strategies are exported:
  EWMAOverlayDCA   — equal-weight base + EWMA vol scaling
  GRUOverlayDCA    — equal-weight base + GRU vol scaling

Walk-forward training:
  The GRU is retrained on every 12-month step using the prior 36 months.
  Models are cached by window index so the training only runs once per run.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from allocation_lab.backtest.portfolio import PortfolioState, SLEEVES
from allocation_lab.research.vol_forecast import EWMAVolatility, GRUVolatility, ANN

logger = logging.getLogger(__name__)

TARGET_VOL = 0.15   # 15% annualised target
SLEEVE_LIST = list(SLEEVES)


def _log_returns(sleeve_df: pd.DataFrame) -> pd.DataFrame:
    return np.log(sleeve_df / sleeve_df.shift(1)).dropna()


def _vol_scaled_weights(
    base_weights: dict[str, float],
    vol_forecasts: dict[str, float],
    target_vol: float = TARGET_VOL,
) -> dict[str, float]:
    """Scale base_weights by (target_vol / forecast_vol), cap at 1, renormalise."""
    scaled = {}
    for sleeve, w in base_weights.items():
        fvol = vol_forecasts.get(sleeve, target_vol)
        if fvol > 0:
            scale = min(1.0, target_vol / fvol)
        else:
            scale = 1.0
        scaled[sleeve] = w * scale

    total = sum(scaled.values())
    if total > 1e-8:
        return {s: v / total for s, v in scaled.items()}
    return base_weights


# ------------------------------------------------------------------ #
# EWMA overlay strategy                                               #
# ------------------------------------------------------------------ #

class EWMAOverlayDCA:
    """Equal-weight DCA with EWMA vol scaling."""

    name = "ewma_overlay_dca"

    def __init__(self, target_vol: float = TARGET_VOL, lam: float = 0.94):
        self.target_vol = target_vol
        self._ewma: dict[str, EWMAVolatility] = {}

    def signal(
        self, history: pd.DataFrame, portfolio: PortfolioState
    ) -> tuple[dict[str, float], str]:
        cols = [s for s in SLEEVE_LIST if s in history.columns]
        n = len(cols)
        base_w = {s: 1.0 / n for s in cols}

        if len(history) < 30:
            return base_w, "invest"

        lr = _log_returns(history[cols])
        vol_forecasts = {}
        for sleeve in cols:
            if sleeve not in self._ewma:
                self._ewma[sleeve] = EWMAVolatility()
                self._ewma[sleeve].fit(lr[sleeve].dropna())
            vol_forecasts[sleeve] = self._ewma[sleeve].predict(lr[sleeve].dropna().tail(1))

        weights = _vol_scaled_weights(base_w, vol_forecasts, self.target_vol)
        return weights, "invest"


# ------------------------------------------------------------------ #
# GRU overlay strategy                                                #
# ------------------------------------------------------------------ #

class GRUOverlayDCA:
    """Equal-weight DCA with GRU-forecasted vol scaling.

    Models are trained walk-forward: retrained every `retrain_months` months
    on the prior `train_months` of data.
    """

    name = "gru_overlay_dca"

    def __init__(
        self,
        target_vol: float = TARGET_VOL,
        train_months: int = 36,
        retrain_months: int = 12,
        **gru_kwargs,
    ):
        self.target_vol = target_vol
        self.train_months = train_months
        self.retrain_months = retrain_months
        self.gru_kwargs = gru_kwargs
        self._models: dict[str, GRUVolatility] = {}
        self._last_train_date: pd.Timestamp | None = None

    def _should_retrain(self, current_date: pd.Timestamp) -> bool:
        if self._last_train_date is None:
            return True
        months_elapsed = (
            (current_date.year - self._last_train_date.year) * 12
            + current_date.month - self._last_train_date.month
        )
        return months_elapsed >= self.retrain_months

    def signal(
        self, history: pd.DataFrame, portfolio: PortfolioState
    ) -> tuple[dict[str, float], str]:
        cols = [s for s in SLEEVE_LIST if s in history.columns]
        n = len(cols)
        base_w = {s: 1.0 / n for s in cols}

        min_rows = 252 + 63   # ~15 months minimum before GRU is useful
        if len(history) < min_rows:
            return base_w, "invest"

        current_date = history.index[-1]
        lr = _log_returns(history[cols])

        # Retrain if needed
        if self._should_retrain(current_date):
            train_start = current_date - pd.DateOffset(months=self.train_months)
            lr_train = lr.loc[train_start:]
            logger.debug("GRU retraining on %d rows up to %s", len(lr_train), current_date.date())
            for sleeve in cols:
                s_lr = lr_train[sleeve].dropna()
                model = GRUVolatility(**self.gru_kwargs)
                model.fit(s_lr)
                self._models[sleeve] = model
            self._last_train_date = current_date

        # Forecast vol for each sleeve
        vol_forecasts = {}
        for sleeve in cols:
            if sleeve in self._models:
                vol_forecasts[sleeve] = self._models[sleeve].predict(
                    lr[sleeve].dropna().tail(63 + 21)
                )
            else:
                # Fallback: realised vol
                vol_forecasts[sleeve] = float(
                    lr[sleeve].dropna().tail(63).std() * np.sqrt(ANN)
                )

        weights = _vol_scaled_weights(base_w, vol_forecasts, self.target_vol)
        return weights, "invest"
