"""
walkforward.py — Walk-forward (out-of-sample) evaluation.

Splits the history into rolling train/test windows.  Each window:
  1. Train period: strategy may use this to choose parameters (not used yet
     for baselines, but the split is enforced).
  2. Test period: run the backtest on this window only, record metrics.
  3. Slide forward by step_months.

Returns a DataFrame of per-window metrics so we can see how each strategy
performs in different market regimes without cherry-picking the best window.

Default: 36-month train, 12-month test, 12-month step.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from allocation_lab.backtest.engine import run_backtest, Strategy
from allocation_lab.backtest.performance import performance_summary

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _windows(
    index: pd.DatetimeIndex,
    train_months: int,
    test_months: int,
    step_months: int,
) -> list[WalkForwardWindow]:
    """Generate non-overlapping test windows."""
    windows = []
    start = index[0]
    end = index[-1]

    while True:
        train_start = start
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

        if test_end > end:
            break

        windows.append(WalkForwardWindow(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ))
        start += pd.DateOffset(months=step_months)

    return windows


def run_walk_forward(
    strategies: list[Strategy],
    sleeve_df: pd.DataFrame,
    contribution_idr: float = 5_000_000,
    train_months: int = 36,
    test_months: int = 12,
    step_months: int = 12,
    rf_annual: float = 0.0575,
) -> pd.DataFrame:
    """Run walk-forward evaluation for all strategies.

    Returns a long-form DataFrame with columns:
      strategy, window, train_start, test_start, test_end,
      xirr_pct, cagr_twr_pct, sharpe_twr, max_dd_pct, cost_drag_pct
    """
    idx = sleeve_df.dropna(how="all").index
    windows = _windows(idx, train_months, test_months, step_months)
    logger.info("Walk-forward: %d windows (%d-month train / %d-month test)",
                len(windows), train_months, test_months)

    rows = []
    for i, w in enumerate(windows):
        # Slice sleeve_df to the test window only
        # Strategy receives history[:test_date] at each step — no lookahead
        test_slice = sleeve_df.loc[w.test_start:w.test_end]
        if len(test_slice) < 20:
            continue

        for strat in strategies:
            try:
                # Run on test window; strategy can see all history up to each date
                # because run_backtest uses df.loc[:dt] internally — but the
                # contribution loop only runs over the test_slice dates.
                result = run_backtest(
                    strat,
                    sleeve_df,           # full history for signal context
                    contribution_idr=contribution_idr,
                    start=str(w.test_start.date()),
                    end=str(w.test_end.date()),
                )
                summary = performance_summary(result, rf_annual=rf_annual)
                rows.append({
                    "strategy": strat.name,
                    "window": i,
                    "train_start": w.train_start.date(),
                    "test_start": w.test_start.date(),
                    "test_end": w.test_end.date(),
                    **{k: summary[k] for k in [
                        "xirr_pct", "cagr_twr_pct", "sharpe_twr",
                        "max_dd_pct", "cost_drag_pct", "total_return_pct",
                    ]},
                })
            except Exception as exc:
                logger.warning("Window %d strat %s failed: %s", i, strat.name, exc)

    df = pd.DataFrame(rows)
    return df


def summarise_walk_forward(wf_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate walk-forward results: mean and std of key metrics per strategy."""
    metrics = ["xirr_pct", "sharpe_twr", "max_dd_pct", "total_return_pct"]
    agg = (
        wf_df.groupby("strategy")[metrics]
        .agg(["mean", "std"])
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.sort_values("xirr_pct_mean", ascending=False)
