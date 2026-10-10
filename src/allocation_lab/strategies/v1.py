"""
v1.py — Three Phase 3 strategies.

1. InverseVolatilityDCA  — risk-parity: weights ∝ 1/vol (63-day window).
2. MomentumDCA           — weights ∝ 12-month momentum score, long only.
3. TrendFilterDCA        — equal-weight but hold cash for a sleeve when its
                           price is below the 200-day MA; force-invest after
                           MAX_WAIT_MONTHS months (set in engine.py).

All three only buy; no selling (decisions.md #12).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from allocation_lab.backtest.portfolio import PortfolioState, SLEEVES

_SLEEVE_LIST = list(SLEEVES)
MIN_HISTORY_DAYS = 63   # need at least this many rows before producing a signal


# --------------------------------------------------------------------------- #
# 1. Inverse-volatility (risk parity)                                          #
# --------------------------------------------------------------------------- #

class InverseVolatilityDCA:
    """Weights are proportional to 1 / trailing_vol.

    Uses 63-day rolling annualised vol of log returns.
    Falls back to equal weight when history is too short.
    """

    name = "inverse_vol_dca"

    def __init__(self, vol_window: int = 63):
        self.vol_window = vol_window

    def signal(
        self, history: pd.DataFrame, portfolio: PortfolioState
    ) -> tuple[dict[str, float], str]:
        cols = [c for c in _SLEEVE_LIST if c in history.columns]

        if len(history) < self.vol_window + 1:
            # Not enough data — fall back to equal weight
            w = 1.0 / len(cols)
            return {s: w for s in cols}, "invest"

        prices = history[cols].ffill()
        log_ret = np.log(prices / prices.shift(1)).dropna()
        vol = log_ret.tail(self.vol_window).std() * np.sqrt(252)

        # Avoid divide by zero; clip minimum vol to 1%
        vol = vol.clip(lower=0.01)
        inv_vol = 1.0 / vol
        weights_raw = inv_vol / inv_vol.sum()
        weights = {s: float(weights_raw.get(s, 0.0)) for s in _SLEEVE_LIST}
        return weights, "invest"


# --------------------------------------------------------------------------- #
# 2. Momentum DCA                                                               #
# --------------------------------------------------------------------------- #

class MomentumDCA:
    """Weights are proportional to 12-1 month momentum (skip last month).

    12-1 month momentum: total return from 12 months ago to 1 month ago.
    Sleeves with negative momentum get zero weight (long-only).
    Falls back to equal weight when history is too short.
    """

    name = "momentum_dca"

    def __init__(self, lookback_days: int = 252, skip_days: int = 21):
        self.lookback_days = lookback_days   # ~12 months
        self.skip_days = skip_days           # ~1 month (skip last month)

    def signal(
        self, history: pd.DataFrame, portfolio: PortfolioState
    ) -> tuple[dict[str, float], str]:
        cols = [c for c in _SLEEVE_LIST if c in history.columns]
        min_rows = self.lookback_days + self.skip_days + 1

        if len(history) < min_rows:
            w = 1.0 / len(cols)
            return {s: w for s in cols}, "invest"

        prices = history[cols].ffill()
        # Momentum score: price at -skip_days vs price at -lookback_days
        p_recent = prices.iloc[-(self.skip_days + 1)]
        p_old    = prices.iloc[-(self.lookback_days + 1)]
        mom = (p_recent / p_old) - 1.0

        # Long-only: zero weight for negative momentum
        mom_pos = mom.clip(lower=0.0)
        total = mom_pos.sum()

        if total <= 0:
            # Everything negative — equal weight fallback
            w = 1.0 / len(cols)
            return {s: w for s in cols}, "invest"

        weights = (mom_pos / total).to_dict()
        weights = {s: float(weights.get(s, 0.0)) for s in _SLEEVE_LIST}
        return weights, "invest"


# --------------------------------------------------------------------------- #
# 3. Trend filter (200-day MA hold-cash)                                       #
# --------------------------------------------------------------------------- #

class TrendFilterDCA:
    """Equal-weight DCA but hold cash for a sleeve below its 200-day MA.

    Per decisions.md #13: buy only when price > 200-day MA.
    The engine's MAX_WAIT_MONTHS cap forces investment after 3 months.

    When a sleeve is in 'wait' mode, its weight is still returned but the
    cash_policy is 'wait' — the engine respects the per-sleeve wait counter.

    Simplification: this implementation returns 'invest' or 'wait' globally.
    Sleeves above MA get invested; sleeves below MA stay in cash (the engine
    accumulates their cash share and force-invests after MAX_WAIT_MONTHS).
    """

    name = "trend_filter_dca"

    def __init__(self, ma_window: int = 200):
        self.ma_window = ma_window

    def signal(
        self, history: pd.DataFrame, portfolio: PortfolioState
    ) -> tuple[dict[str, float], str]:
        cols = [c for c in _SLEEVE_LIST if c in history.columns]

        if len(history) < self.ma_window + 1:
            # Not enough history — invest equally
            w = 1.0 / len(cols)
            return {s: w for s in cols}, "invest"

        prices = history[cols].ffill()
        current_price = prices.iloc[-1]
        ma = prices.tail(self.ma_window).mean()
        above_ma = current_price > ma

        # Only invest in sleeves above their MA
        invest_sleeves = [s for s in cols if above_ma.get(s, False)]

        if not invest_sleeves:
            # All sleeves below MA — wait (engine will force after cap)
            w = 1.0 / len(cols)
            return {s: w for s in cols}, "wait"

        w_in = 1.0 / len(invest_sleeves)
        weights = {
            s: (w_in if s in invest_sleeves else 0.0)
            for s in _SLEEVE_LIST
        }
        return weights, "invest"
