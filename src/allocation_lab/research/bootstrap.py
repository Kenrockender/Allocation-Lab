"""
bootstrap.py — Block bootstrap confidence intervals.

Resamples monthly return blocks to construct a distribution of outcomes for
each strategy, then reports:
  - 5th / 50th / 95th percentile of XIRR and final wealth
  - Probability that each strategy beats the equal-weight baseline

Block bootstrap (Künsch 1989): samples contiguous blocks of monthly returns
to preserve autocorrelation structure.  Block length ~3 months is typical
for financial returns.

Method:
  1. Compute monthly sleeve returns from the real data.
  2. Resample with block bootstrap to get B synthetic monthly return paths
     of the same length.
  3. Rebuild synthetic sleeve prices, run each strategy on them, record XIRR.
  4. Report percentiles and p(strategy > baseline).
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from allocation_lab.backtest.engine import run_backtest, Strategy
from allocation_lab.backtest.performance import xirr as compute_xirr

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Block bootstrap resampler                                                    #
# --------------------------------------------------------------------------- #

def _block_bootstrap_returns(
    monthly_returns: pd.DataFrame,
    n_blocks: int | None = None,
    block_size: int = 3,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Return a resampled monthly-return DataFrame of the same length.

    Parameters
    ----------
    monthly_returns : DataFrame of shape (T, n_sleeves), each cell a simple return
    block_size      : number of consecutive months per block
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(monthly_returns)
    if n_blocks is None:
        n_blocks = int(np.ceil(T / block_size))

    # Sample start indices of blocks (with replacement)
    max_start = T - block_size
    starts = rng.integers(0, max(1, max_start + 1), size=n_blocks)

    blocks = [monthly_returns.iloc[s:s + block_size] for s in starts]
    resampled = pd.concat(blocks, ignore_index=True).iloc[:T]
    return resampled


def _rebuild_prices_from_returns(
    monthly_returns: pd.DataFrame,
    start_prices: pd.Series,
    freq: str = "MS",
) -> pd.DataFrame:
    """Reconstruct a monthly price DataFrame from simple returns."""
    dates = pd.date_range(
        start="2018-01-01",
        periods=len(monthly_returns) + 1,
        freq=freq,
        name="date",
    )
    # price[t] = price[t-1] * (1 + ret[t])
    prices = pd.DataFrame(index=dates, columns=monthly_returns.columns, dtype=float)
    prices.iloc[0] = start_prices
    for i, row in monthly_returns.iterrows():
        prices.iloc[i + 1] = prices.iloc[i] * (1.0 + row)
    return prices


# --------------------------------------------------------------------------- #
# Bootstrap runner                                                              #
# --------------------------------------------------------------------------- #

def run_bootstrap(
    strategies: list[Strategy],
    sleeve_df: pd.DataFrame,
    contribution_idr: float = 5_000_000,
    n_sims: int = 500,
    block_size: int = 3,
    rf_annual: float = 0.0575,
    seed: int = 42,
) -> pd.DataFrame:
    """Run block bootstrap and return a DataFrame of simulated XIRRs.

    Columns: strategy name.  Rows: one per simulation.
    """
    rng = np.random.default_rng(seed)

    # Build monthly returns from the sleeve indices
    sleeve_cols = [c for c in ["idx_equities", "us_equities", "gold", "crypto"]
                   if c in sleeve_df.columns]
    monthly_prices = sleeve_df[sleeve_cols].resample("MS").first().dropna(how="all")
    monthly_returns = monthly_prices.pct_change().dropna()
    start_prices = monthly_prices.iloc[0]

    T = len(monthly_returns)
    logger.info("Bootstrap: %d simulations, %d monthly periods, block=%d",
                n_sims, T, block_size)

    all_xirrs: dict[str, list[float]] = {s.name: [] for s in strategies}

    for sim_idx in range(n_sims):
        if sim_idx % 100 == 0:
            logger.info("  sim %d / %d", sim_idx, n_sims)

        # Resample returns
        sim_returns = _block_bootstrap_returns(
            monthly_returns, block_size=block_size, rng=rng
        )
        # Rebuild monthly prices
        sim_prices_monthly = _rebuild_prices_from_returns(sim_returns, start_prices)

        # Upsample to daily by forward-filling within each month
        date_range = pd.date_range(
            sim_prices_monthly.index[0],
            sim_prices_monthly.index[-1] + pd.DateOffset(months=1) - pd.Timedelta(days=1),
            freq="D",
            name="date",
        )
        sim_prices_daily = sim_prices_monthly.reindex(date_range).ffill()

        for strat in strategies:
            try:
                result = run_backtest(
                    strat,
                    sim_prices_daily,
                    contribution_idr=contribution_idr,
                )
                irr = compute_xirr(result.cash_flows)
                all_xirrs[strat.name].append(irr * 100)
            except Exception:
                all_xirrs[strat.name].append(float("nan"))

    return pd.DataFrame(all_xirrs)


def bootstrap_summary(boot_df: pd.DataFrame, baseline: str = "equal_weight_dca") -> pd.DataFrame:
    """Compute percentiles and probability of beating the baseline."""
    rows = []
    baseline_xirrs = boot_df[baseline].dropna().values if baseline in boot_df.columns else None

    for col in boot_df.columns:
        vals = boot_df[col].dropna().values
        row = {
            "strategy": col,
            "xirr_p5":  float(np.percentile(vals, 5)),
            "xirr_p50": float(np.percentile(vals, 50)),
            "xirr_p95": float(np.percentile(vals, 95)),
            "xirr_mean": float(np.mean(vals)),
        }
        if baseline_xirrs is not None and col != baseline:
            row["p_beat_baseline"] = float(np.mean(vals > baseline_xirrs[:len(vals)]))
        else:
            row["p_beat_baseline"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows).set_index("strategy").sort_values("xirr_p50", ascending=False)
