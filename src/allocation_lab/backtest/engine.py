"""
engine.py — Simulation loop.

run_backtest(strategy, prices, sleeve_df, contribution_idr, start, end)
  -> BacktestResult

The loop:
  1. Each calendar day: accrue cash yield, reprice holdings.
  2. On contribution days (first trading day of each month):
     a. Deposit contribution.
     b. Call strategy.signal(history_up_to_today) -> (weights, cash_policy).
     c. cash_policy='invest' -> deploy all pending cash per weights.
        cash_policy='wait'   -> leave cash in money-market this month.
        After 3 months waiting, force-invest regardless (decisions.md #13).
  3. Trade at next-close of each sleeve's market (one day lag for lookahead safety).
  4. Record NAV daily.

Anti-lookahead: strategy receives prices.loc[:date] — never prices.loc[date+1:].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from allocation_lab.backtest.costs import CostModel
from allocation_lab.backtest.portfolio import PortfolioState, SLEEVES

# --------------------------------------------------------------------------- #
# Strategy protocol                                                             #
# --------------------------------------------------------------------------- #

class Strategy(Protocol):
    """Interface every strategy must satisfy."""
    name: str

    def signal(
        self,
        history: pd.DataFrame,          # sleeve_df rows up to and including today
        portfolio: PortfolioState,
    ) -> tuple[dict[str, float], str]:   # (target_weights, cash_policy)
        """Return target allocation weights and cash policy.

        target_weights : dict[sleeve -> weight], should sum to ≤ 1.0
        cash_policy    : 'invest' | 'wait'
        """
        ...


# --------------------------------------------------------------------------- #
# Result container                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class BacktestResult:
    strategy_name: str
    nav_series: pd.Series           # DatetimeIndex -> NAV in IDR
    cash_flows: list[tuple]         # (date, signed_idr) for XIRR
    total_contributed: float
    total_costs_idr: float
    sleeve_history: pd.DataFrame    # daily holdings per sleeve + cash
    trade_log: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Engine                                                                        #
# --------------------------------------------------------------------------- #

MAX_WAIT_MONTHS = 3   # force-invest after this many months waiting


def _first_trading_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the first available date in each calendar month."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(
        s.groupby([s.index.year, s.index.month]).first()
    )


def run_backtest(
    strategy: Strategy,
    sleeve_df: pd.DataFrame,
    contribution_idr: float,
    start: str = "2018-01-01",
    end: str | None = None,
    first_valid: dict[str, str] | None = None,
) -> BacktestResult:
    """Run a full backtest simulation.

    Parameters
    ----------
    strategy        : Strategy object with a .signal() method
    sleeve_df       : daily sleeve index DataFrame (output of align.build_sleeve_series)
    contribution_idr: fixed monthly contribution amount in IDR
    start           : simulation start date string
    end             : simulation end date string (default: last row)
    first_valid     : dict[sleeve -> first valid date string]; sleeves before their
                      first valid date are excluded from the available set
    """
    cost_model = CostModel.from_yaml()
    portfolio = PortfolioState(cost_model=cost_model)

    # Restrict to simulation window
    df = sleeve_df.copy()
    df = df.loc[start:end] if end else df.loc[start:]
    df = df.dropna(how="all")

    # Build daily simple returns for repricing
    # (forward-fill before diff so holiday gaps don't create spurious returns)
    sleeve_cols = [c for c in SLEEVES if c in df.columns]
    prices_ff = df[sleeve_cols].ffill()
    daily_returns = prices_ff.pct_change().fillna(0.0)

    # Contribution dates: first trading day of each month
    trading_days = df.index
    contrib_dates = set(_first_trading_days(trading_days))

    # Pending cash waiting months tracker: sleeve -> months still waiting
    wait_months: dict[str, int] = {s: 0 for s in sleeve_cols}

    # Holdings history
    holdings_records: list[dict] = []
    trade_log: list[dict] = []
    total_contributed = 0.0

    for dt in trading_days:
        # 1. Accrue cash yield daily
        portfolio.accrue_cash_yield()

        # 2. Reprice holdings with today's returns
        ret_today = {s: daily_returns.loc[dt, s] for s in sleeve_cols}
        portfolio.reprice(ret_today)

        # 3. Determine available sleeves (those with valid data today)
        if first_valid:
            available = {
                s for s in sleeve_cols
                if str(dt.date()) >= first_valid.get(s, "2000-01-01")
            }
        else:
            available = set(sleeve_cols)
        available = {s for s in available if not pd.isna(df.loc[dt, s])}

        # 4. On contribution days: deposit + signal + trade
        if dt in contrib_dates:
            portfolio.deposit(contribution_idr, dt.date())
            total_contributed += contribution_idr

            # Get strategy signal on history up to today (inclusive) — no lookahead
            history = df.loc[:dt]
            weights, cash_policy = strategy.signal(history, portfolio)

            # Normalise weights to available sleeves only
            eff_weights = {s: w for s, w in weights.items() if s in available and w > 0}

            # Enforce wait cap: if a sleeve has been waiting > MAX_WAIT_MONTHS, force invest
            for s in list(eff_weights.keys()):
                if wait_months.get(s, 0) >= MAX_WAIT_MONTHS:
                    cash_policy = "invest"  # force for this sleeve
                    wait_months[s] = 0

            if cash_policy == "invest":
                # Deploy all available cash proportionally
                cash_to_deploy = portfolio.cash_idr
                portfolio.apply_target_weights(eff_weights, available, cash_to_deploy)
                wait_months = {s: 0 for s in sleeve_cols}
                trade_log.append({
                    "date": dt.date(), "action": "invest",
                    "weights": eff_weights.copy(), "cash_before": cash_to_deploy,
                })
            else:
                # Increment wait counter for each sleeve that chose to wait
                for s in eff_weights:
                    wait_months[s] = wait_months.get(s, 0) + 1
                trade_log.append({
                    "date": dt.date(), "action": "wait",
                    "weights": eff_weights.copy(),
                })

        # 5. Record daily snapshot
        record = {"date": dt, "nav": portfolio.nav, "cash": portfolio.cash_idr}
        record.update({s: portfolio.holdings.get(s, 0.0) for s in sleeve_cols})
        holdings_records.append(record)

    # Build result objects
    df_hist = pd.DataFrame(holdings_records).set_index("date")
    nav_series = df_hist["nav"].rename(strategy.name)

    return BacktestResult(
        strategy_name=strategy.name,
        nav_series=nav_series,
        cash_flows=portfolio.cash_flows + [(trading_days[-1].date(), portfolio.nav)],
        total_contributed=total_contributed,
        total_costs_idr=portfolio.total_costs_idr,
        sleeve_history=df_hist,
        trade_log=trade_log,
    )
