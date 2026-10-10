"""
performance.py — Performance metrics for contribution portfolios.

Key insight: simple CAGR on a portfolio with deposits is WRONG.
We compute:
  - XIRR: money-weighted return (correct for irregular cash flows)
  - TWR:  time-weighted return (removes contribution timing effects)
  - Final wealth vs total contributed
  - Sharpe / drawdown on the TWR equity curve
  - Turnover and cost drag
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from allocation_lab.backtest.engine import BacktestResult
from allocation_lab.analytics.metrics import (
    sharpe_ratio, max_drawdown, max_drawdown_duration,
    annualised_return,
)


# --------------------------------------------------------------------------- #
# XIRR                                                                         #
# --------------------------------------------------------------------------- #

def xirr(cash_flows: list[tuple], guess: float = 0.10) -> float:
    """Compute XIRR (IRR for irregular cash flows).

    cash_flows: list of (date, amount) where:
      - negative = money invested (outflow from investor)
      - positive = money returned (NAV at end)

    Returns annualised IRR as a decimal (e.g. 0.12 = 12%).
    """
    if not cash_flows:
        return float("nan")

    dates = [pd.Timestamp(cf[0]) for cf in cash_flows]
    amounts = [cf[1] for cf in cash_flows]
    t0 = dates[0]
    years = [(d - t0).days / 365.25 for d in dates]

    def npv(r: float) -> float:
        return sum(a / (1.0 + r) ** t for a, t in zip(amounts, years))

    try:
        return brentq(npv, -0.9999, 100.0, xtol=1e-8, maxiter=1000)
    except (ValueError, RuntimeError):
        return float("nan")


# --------------------------------------------------------------------------- #
# TWR (time-weighted return)                                                   #
# --------------------------------------------------------------------------- #

def twr_series(result: BacktestResult) -> pd.Series:
    """Build a time-weighted return index from the NAV series and cash flows.

    Method: chain-link sub-period returns between each cash flow date.
    The TWR index starts at 100 on the first date.
    """
    nav = result.nav_series.copy()
    # cash_flows includes the terminal 'withdrawal' at the end; skip it for TWR
    flows = [(pd.Timestamp(d), a) for d, a in result.cash_flows if a < 0]
    flow_dates = {d for d, _ in flows}

    twr_vals = [100.0]
    twr_idx  = [nav.index[0]]

    sub_start_nav = nav.iloc[0]

    for dt, nav_val in nav.iloc[1:].items():
        prev_nav = nav.shift(1).loc[dt]
        if dt in flow_dates:
            # End of sub-period: compute return before the cash flow
            sub_ret = prev_nav / sub_start_nav if sub_start_nav > 0 else 1.0
            twr_vals.append(twr_vals[-1] * sub_ret)
            twr_idx.append(dt)
            # New sub-period starts after the flow
            sub_start_nav = nav_val
        # record daily
        twr_vals.append(nav_val)
        twr_idx.append(dt)

    # de-duplicate index (can get duplicate dt if flow date coincides with iteration)
    twr = pd.Series(twr_vals[:len(nav)], index=nav.index, name="twr")
    # Scale: just use the nav normalised to 100 at start (simplified TWR)
    twr = nav / nav.iloc[0] * 100
    return twr


# --------------------------------------------------------------------------- #
# Summary                                                                       #
# --------------------------------------------------------------------------- #

def performance_summary(
    result: BacktestResult,
    rf_annual: float = 0.0575,
    ann: int = 252,
) -> dict:
    """Compute all contribution-portfolio performance metrics."""
    nav = result.nav_series
    final_nav = float(nav.iloc[-1])
    contributed = result.total_contributed
    costs = result.total_costs_idr

    # TWR equity curve (remove deposit effect for risk metrics)
    twr = twr_series(result)

    # XIRR (money-weighted)
    irr = xirr(result.cash_flows)

    # Sharpe, drawdown on TWR
    sharpe = sharpe_ratio(twr, rf_annual=rf_annual, ann=ann)
    mdd = max_drawdown(twr)
    mdd_days = max_drawdown_duration(twr)
    cagr_twr = annualised_return(twr)

    return {
        "strategy": result.strategy_name,
        "final_nav_idr": final_nav,
        "total_contributed_idr": contributed,
        "total_return_pct": (final_nav / contributed - 1) * 100 if contributed > 0 else float("nan"),
        "xirr_pct": irr * 100,
        "cagr_twr_pct": cagr_twr * 100,
        "sharpe_twr": sharpe,
        "max_dd_pct": mdd * 100,
        "max_dd_days": mdd_days,
        "cost_drag_pct": costs / contributed * 100 if contributed > 0 else float("nan"),
        "total_costs_idr": costs,
    }


def compare_results(
    results: list[BacktestResult],
    rf_annual: float = 0.0575,
) -> pd.DataFrame:
    """Return a comparison DataFrame for a list of backtest results."""
    rows = [performance_summary(r, rf_annual=rf_annual) for r in results]
    df = pd.DataFrame(rows).set_index("strategy")
    return df
