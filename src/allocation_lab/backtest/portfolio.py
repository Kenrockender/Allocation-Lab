"""
portfolio.py — Portfolio state: holdings, cash, NAV, and the contribution loop.

PortfolioState tracks:
  - cash_idr: money-market cash earning daily yield
  - holdings: dict[sleeve -> IDR value invested (at cost basis, updated daily by price)]
  - nav_history: list of (date, nav) snapshots
  - cash_flows: list of (date, amount) for XIRR calculation (negative = investment)

Holdings are stored as fractional units expressed in IDR value so we avoid
currency conversion inside the state object (prices are already in IDR).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from allocation_lab.backtest.costs import CostModel

SLEEVES = ("idx_equities", "us_equities", "gold", "crypto")


@dataclass
class PortfolioState:
    cost_model: CostModel
    # cash waiting in money-market
    cash_idr: float = 0.0
    # sleeve_name -> IDR value of holdings (marked-to-market each day)
    holdings: dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in SLEEVES})
    # for XIRR / TWR
    nav_history: list[tuple] = field(default_factory=list)   # (date, nav)
    cash_flows: list[tuple] = field(default_factory=list)    # (date, signed_amount)
    # cost drag tracking
    total_costs_idr: float = 0.0

    # ------------------------------------------------------------------ #
    # NAV                                                                  #
    # ------------------------------------------------------------------ #

    @property
    def nav(self) -> float:
        return self.cash_idr + sum(self.holdings.values())

    def record_nav(self, dt: date) -> None:
        self.nav_history.append((dt, self.nav))

    # ------------------------------------------------------------------ #
    # Cash                                                                 #
    # ------------------------------------------------------------------ #

    def accrue_cash_yield(self) -> None:
        """Apply one day of money-market yield to cash."""
        self.cash_idr *= self.cost_model.cash_daily_multiplier()

    # ------------------------------------------------------------------ #
    # Contribution                                                         #
    # ------------------------------------------------------------------ #

    def deposit(self, amount_idr: float, dt: date) -> None:
        """Add a contribution to cash and record as a cash flow."""
        self.cash_idr += amount_idr
        self.cash_flows.append((dt, -amount_idr))   # negative = money in

    # ------------------------------------------------------------------ #
    # Repricing                                                            #
    # ------------------------------------------------------------------ #

    def reprice(self, sleeve_returns: dict[str, float]) -> None:
        """Update holdings value using today's sleeve returns.

        Parameters
        ----------
        sleeve_returns : dict[sleeve -> daily simple return for this day]
        """
        for sleeve, ret in sleeve_returns.items():
            if sleeve in self.holdings:
                self.holdings[sleeve] *= (1.0 + ret)

    # ------------------------------------------------------------------ #
    # Trading                                                              #
    # ------------------------------------------------------------------ #

    def buy_sleeve(self, sleeve: str, idr_amount: float) -> float:
        """Buy idr_amount of a sleeve from cash. Returns actual cost (incl. fees)."""
        if idr_amount <= 0:
            return 0.0
        cost = self.cost_model.trade_cost(sleeve, "buy", idr_amount)
        total_debit = idr_amount + cost
        if total_debit > self.cash_idr + 1e-6:   # allow tiny float slack
            # buy as much as cash allows
            # idr_amount / (1 + fee) = max purchasable
            fee_rate = cost / idr_amount if idr_amount > 0 else 0
            idr_amount = self.cash_idr / (1.0 + fee_rate)
            cost = self.cost_model.trade_cost(sleeve, "buy", idr_amount)
            total_debit = idr_amount + cost

        self.cash_idr -= total_debit
        self.holdings[sleeve] = self.holdings.get(sleeve, 0.0) + idr_amount
        self.total_costs_idr += cost
        return cost

    def sell_sleeve(self, sleeve: str, idr_amount: float) -> float:
        """Sell idr_amount of a sleeve into cash. Returns actual cost (incl. fees)."""
        if idr_amount <= 0:
            return 0.0
        held = self.holdings.get(sleeve, 0.0)
        idr_amount = min(idr_amount, held)
        if idr_amount <= 0:
            return 0.0
        cost = self.cost_model.trade_cost(sleeve, "sell", idr_amount)
        self.holdings[sleeve] = held - idr_amount
        self.cash_idr += idr_amount - cost
        self.total_costs_idr += cost
        return cost

    # ------------------------------------------------------------------ #
    # Allocation                                                           #
    # ------------------------------------------------------------------ #

    def apply_target_weights(
        self,
        target_weights: dict[str, float],
        available_sleeves: set[str],
        cash_to_deploy: float,
    ) -> None:
        """Deploy cash_to_deploy according to target_weights.

        Only buys — consistent with contribution-only rebalancing (decision #12).
        target_weights values should sum to ≤ 1.0 (remainder stays in cash).
        Sleeves not in available_sleeves are skipped (their weight stays in cash).
        """
        total_w = sum(
            w for s, w in target_weights.items() if s in available_sleeves
        )
        if total_w <= 0 or cash_to_deploy <= 0:
            return

        for sleeve, w in target_weights.items():
            if sleeve not in available_sleeves or w <= 0:
                continue
            alloc = cash_to_deploy * (w / total_w)
            self.buy_sleeve(sleeve, alloc)
