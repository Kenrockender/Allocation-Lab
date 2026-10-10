"""
baselines.py — Fixed-weight DCA, equal-weight DCA, single-sleeve benchmark.

These are the three comparison strategies from decisions.md #7.
Each returns a (weights_dict, cash_policy) tuple from .signal().
"""
from __future__ import annotations

import pandas as pd

from allocation_lab.backtest.portfolio import PortfolioState, SLEEVES

_SLEEVE_LIST = list(SLEEVES)


class FixedWeightDCA:
    """DCA with a user-specified fixed weight per sleeve.

    Example: FixedWeightDCA({"idx_equities": 0.4, "us_equities": 0.3,
                              "gold": 0.2, "crypto": 0.1})
    """

    def __init__(self, weights: dict[str, float], name: str | None = None):
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1, got {total}"
        self.weights = weights
        self.name = name or ("fixed_weight_" + "_".join(
            f"{s[:3]}{int(w*100)}" for s, w in weights.items()
        ))

    def signal(
        self, history: pd.DataFrame, portfolio: PortfolioState
    ) -> tuple[dict[str, float], str]:
        return self.weights.copy(), "invest"


class EqualWeightDCA:
    """DCA splitting contributions equally across available sleeves."""

    name = "equal_weight_dca"

    def signal(
        self, history: pd.DataFrame, portfolio: PortfolioState
    ) -> tuple[dict[str, float], str]:
        n = len(_SLEEVE_LIST)
        weights = {s: 1.0 / n for s in _SLEEVE_LIST}
        return weights, "invest"


class SingleSleeveDCA:
    """DCA into one sleeve only (used for benchmark comparisons)."""

    def __init__(self, sleeve: str):
        assert sleeve in SLEEVES, f"Unknown sleeve: {sleeve}"
        self.sleeve = sleeve
        self.name = f"dca_{sleeve}"

    def signal(
        self, history: pd.DataFrame, portfolio: PortfolioState
    ) -> tuple[dict[str, float], str]:
        weights = {s: (1.0 if s == self.sleeve else 0.0) for s in SLEEVES}
        return weights, "invest"
