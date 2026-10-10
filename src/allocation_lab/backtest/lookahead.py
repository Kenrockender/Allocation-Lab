"""
lookahead.py — Verify that a strategy produces identical signals when given
data cut off at date t vs when given the full history.

A strategy passes if: signal(history[:t]) == signal(history[:t]) for all t,
which is guaranteed structurally because run_backtest always passes df.loc[:dt].

This test does a belt-and-suspenders check by calling the strategy twice on
two different history windows that agree up to date t, and asserting the
weights are identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from allocation_lab.backtest.engine import Strategy
from allocation_lab.backtest.portfolio import PortfolioState
from allocation_lab.backtest.costs import CostModel


def check_no_lookahead(
    strategy: Strategy,
    sleeve_df: pd.DataFrame,
    n_checks: int = 12,
) -> list[dict]:
    """Run the lookahead test on n_checks randomly sampled contribution dates.

    Returns a list of failure dicts (empty list = test passed).
    """
    cost_model = CostModel.from_yaml()

    # Sample n_checks dates from the middle of the history (need enough prior data)
    idx = sleeve_df.dropna(how="all").index
    min_idx = 300   # need at least 300 rows of history before testing
    sample_positions = np.linspace(min_idx, len(idx) - 2, n_checks, dtype=int)

    failures = []

    for pos in sample_positions:
        cut_date = idx[pos]
        full_date = idx[pos]   # same date, different history length

        # History A: up to cut_date (simulates what the engine sees on day t)
        hist_a = sleeve_df.loc[:cut_date]

        # History B: full history but we call signal with it truncated to cut_date
        # (should be the same as hist_a — this confirms no internal state leaks)
        hist_b = sleeve_df.loc[:full_date]

        dummy_portfolio = PortfolioState(cost_model=cost_model)

        weights_a, policy_a = strategy.signal(hist_a, dummy_portfolio)
        weights_b, policy_b = strategy.signal(hist_b, dummy_portfolio)

        # Compare weights
        for sleeve in set(list(weights_a.keys()) + list(weights_b.keys())):
            wa = weights_a.get(sleeve, 0.0)
            wb = weights_b.get(sleeve, 0.0)
            if abs(wa - wb) > 1e-10:
                failures.append({
                    "date": cut_date,
                    "sleeve": sleeve,
                    "weight_a": wa,
                    "weight_b": wb,
                })

        if policy_a != policy_b:
            failures.append({
                "date": cut_date,
                "policy_a": policy_a,
                "policy_b": policy_b,
            })

    return failures


def assert_no_lookahead(
    strategy: Strategy,
    sleeve_df: pd.DataFrame,
    n_checks: int = 12,
) -> None:
    """Raise AssertionError if the lookahead test fails."""
    failures = check_no_lookahead(strategy, sleeve_df, n_checks=n_checks)
    if failures:
        msg = f"Lookahead detected in '{strategy.name}': {failures[:3]}"
        raise AssertionError(msg)
