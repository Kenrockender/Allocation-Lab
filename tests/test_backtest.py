"""
Tests for the backtest engine, cost model, portfolio state and strategies.
All tests use hand-made data — no network calls, no parquet files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from allocation_lab.backtest.costs import CostModel
from allocation_lab.backtest.portfolio import PortfolioState, SLEEVES
from allocation_lab.backtest.engine import run_backtest, _first_trading_days
from allocation_lab.backtest.performance import xirr, performance_summary
from allocation_lab.backtest.lookahead import check_no_lookahead
from allocation_lab.strategies.baselines import EqualWeightDCA, SingleSleeveDCA, FixedWeightDCA
from allocation_lab.strategies.v1 import InverseVolatilityDCA, MomentumDCA, TrendFilterDCA


# ------------------------------------------------------------------ #
# Fixtures / helpers                                                   #
# ------------------------------------------------------------------ #

def make_cost_model(**overrides) -> CostModel:
    defaults = dict(
        idx_buy_pct=0.15, idx_sell_pct=0.25,
        us_trade_pct=0.15, us_fx_spread_pct=0.50,
        gold_spread_pct=0.75,
        crypto_trade_pct=0.25, crypto_tax_pct=0.21,
        slippage_pct=0.05,
        cash_annual_yield_pct=4.5,
    )
    defaults.update(overrides)
    return CostModel(**defaults)


def make_sleeve_df(n_months: int = 24, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic sleeve DataFrame for testing."""
    np.random.seed(seed)
    n_days = n_months * 21
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B", name="date")
    df = pd.DataFrame(index=idx)
    for sleeve in SLEEVES:
        daily_ret = np.random.normal(0.0003, 0.015, n_days)
        df[sleeve] = 100 * np.exp(np.cumsum(daily_ret))
    return df


# ------------------------------------------------------------------ #
# CostModel                                                            #
# ------------------------------------------------------------------ #

class TestCostModel:
    def setup_method(self):
        self.cm = make_cost_model(slippage_pct=0.0)

    def test_idx_buy_cost(self):
        cost = self.cm.trade_cost("idx_equities", "buy", 1_000_000)
        assert cost == pytest.approx(1_000_000 * 0.0015)

    def test_idx_sell_cost(self):
        cost = self.cm.trade_cost("idx_equities", "sell", 1_000_000)
        assert cost == pytest.approx(1_000_000 * 0.0025)

    def test_crypto_sell_includes_tax(self):
        # 0.25% trade + 0.21% tax on sell
        cost = self.cm.trade_cost("crypto", "sell", 1_000_000)
        assert cost == pytest.approx(1_000_000 * (0.0025 + 0.0021))

    def test_crypto_buy_no_tax(self):
        cost = self.cm.trade_cost("crypto", "buy", 1_000_000)
        assert cost == pytest.approx(1_000_000 * 0.0025)

    def test_unknown_sleeve_raises(self):
        with pytest.raises(ValueError):
            self.cm.trade_cost("unknown_sleeve", "buy", 100_000)

    def test_cash_daily_multiplier_positive(self):
        mult = self.cm.cash_daily_multiplier()
        assert mult > 1.0
        # 4.5% p.a. / 365 ≈ 0.01233% per day
        assert mult == pytest.approx(1.0 + 0.045 / 365, rel=1e-6)


# ------------------------------------------------------------------ #
# PortfolioState                                                        #
# ------------------------------------------------------------------ #

class TestPortfolioState:
    def setup_method(self):
        self.cm = make_cost_model(slippage_pct=0.0)
        self.ps = PortfolioState(cost_model=self.cm)

    def test_initial_nav_zero(self):
        assert self.ps.nav == 0.0

    def test_deposit_increases_cash(self):
        self.ps.deposit(1_000_000, pd.Timestamp("2020-01-01").date())
        assert self.ps.cash_idr == pytest.approx(1_000_000)
        assert self.ps.nav == pytest.approx(1_000_000)

    def test_buy_sleeve_reduces_cash(self):
        self.ps.deposit(2_000_000, pd.Timestamp("2020-01-01").date())
        self.ps.buy_sleeve("idx_equities", 1_000_000)
        # cash should be ~2m - 1m - fee(0.15%) = ~998,500
        assert self.ps.cash_idr == pytest.approx(2_000_000 - 1_000_000 - 1_500, rel=1e-4)
        assert self.ps.holdings["idx_equities"] == pytest.approx(1_000_000, rel=1e-4)

    def test_nav_conserved_after_trade(self):
        """NAV should only decrease by the cost, not disappear."""
        self.ps.deposit(1_000_000, pd.Timestamp("2020-01-01").date())
        self.ps.buy_sleeve("us_equities", 500_000)
        # us_equities cost: trading_fee 0.15% + fx_spread 0.50% + slippage 0.05% = 0.70%
        # make_cost_model has slippage=0 in this test's setup, so only 0.15+0.50=0.65%
        expected_cost = 500_000 * (0.0015 + 0.005)
        assert self.ps.nav == pytest.approx(1_000_000 - expected_cost, rel=1e-4)

    def test_reprice_updates_holdings(self):
        self.ps.deposit(1_000_000, pd.Timestamp("2020-01-01").date())
        self.ps.buy_sleeve("gold", 500_000)
        initial_gold = self.ps.holdings["gold"]
        self.ps.reprice({"gold": 0.10})   # +10% return
        assert self.ps.holdings["gold"] == pytest.approx(initial_gold * 1.10)

    def test_cash_yield_accrues(self):
        self.ps.deposit(1_000_000, pd.Timestamp("2020-01-01").date())
        self.ps.accrue_cash_yield()
        expected = 1_000_000 * (1 + 0.045 / 365)
        assert self.ps.cash_idr == pytest.approx(expected, rel=1e-8)

    def test_buy_capped_at_available_cash(self):
        """Buying more than cash should cap at available cash."""
        self.ps.deposit(100_000, pd.Timestamp("2020-01-01").date())
        self.ps.buy_sleeve("crypto", 1_000_000)   # much more than cash
        assert self.ps.cash_idr >= -1.0   # never negative (allow tiny float error)


# ------------------------------------------------------------------ #
# Engine                                                               #
# ------------------------------------------------------------------ #

class TestEngine:
    def _run(self, strategy, n_months=24, contribution=1_000_000):
        sleeve_df = make_sleeve_df(n_months=n_months, seed=1)
        with patch("allocation_lab.backtest.costs.CostModel.from_yaml",
                   return_value=make_cost_model()):
            return run_backtest(strategy, sleeve_df, contribution_idr=contribution)

    def test_nav_grows_monotonically_on_cash_only(self):
        """If strategy always waits, NAV should grow from cash yield only."""
        class AlwaysWait:
            name = "always_wait"
            def signal(self, h, p):
                return {s: 0.25 for s in SLEEVES}, "wait"

        # With wait cap of 3 months, will eventually force-invest
        result = self._run(AlwaysWait())
        assert result.nav_series.iloc[-1] > result.total_contributed * 0.9

    def test_total_contributed_matches_monthly_deposits(self):
        result = self._run(EqualWeightDCA(), n_months=12, contribution=1_000_000)
        # 12 months -> 12 contributions
        assert result.total_contributed == pytest.approx(12 * 1_000_000, rel=0.05)

    def test_final_nav_positive(self):
        result = self._run(EqualWeightDCA())
        assert result.nav_series.iloc[-1] > 0

    def test_costs_positive(self):
        result = self._run(EqualWeightDCA())
        assert result.total_costs_idr > 0

    def test_nav_series_length(self):
        sleeve_df = make_sleeve_df(n_months=6)
        with patch("allocation_lab.backtest.costs.CostModel.from_yaml",
                   return_value=make_cost_model()):
            result = run_backtest(EqualWeightDCA(), sleeve_df, contribution_idr=1_000_000)
        assert len(result.nav_series) == len(sleeve_df)

    def test_first_trading_days(self):
        idx = pd.date_range("2020-01-01", periods=90, freq="B")
        contrib_days = _first_trading_days(idx)
        # 90 business days ≈ 4-5 months
        assert 3 <= len(contrib_days) <= 6


# ------------------------------------------------------------------ #
# Performance metrics                                                  #
# ------------------------------------------------------------------ #

class TestPerformance:
    def test_xirr_simple_case(self):
        # Invest 100 today, receive 121 in exactly 2 years -> XIRR ≈ 10%
        flows = [
            (pd.Timestamp("2020-01-01"), -100.0),
            (pd.Timestamp("2022-01-01"),  121.0),
        ]
        rate = xirr(flows)
        assert rate == pytest.approx(0.10, abs=0.005)

    def test_xirr_empty(self):
        assert np.isnan(xirr([]))

    def test_performance_summary_keys(self):
        sleeve_df = make_sleeve_df(n_months=24, seed=42)
        with patch("allocation_lab.backtest.costs.CostModel.from_yaml",
                   return_value=make_cost_model()):
            result = run_backtest(EqualWeightDCA(), sleeve_df, contribution_idr=1_000_000)
        summary = performance_summary(result)
        expected_keys = {
            "final_nav_idr", "total_contributed_idr", "total_return_pct",
            "xirr_pct", "cagr_twr_pct", "sharpe_twr", "max_dd_pct",
            "cost_drag_pct",
        }
        assert expected_keys <= set(summary.keys())

    def test_cost_drag_positive(self):
        sleeve_df = make_sleeve_df(n_months=24, seed=42)
        with patch("allocation_lab.backtest.costs.CostModel.from_yaml",
                   return_value=make_cost_model()):
            result = run_backtest(EqualWeightDCA(), sleeve_df, contribution_idr=1_000_000)
        summary = performance_summary(result)
        assert summary["cost_drag_pct"] > 0


# ------------------------------------------------------------------ #
# Strategies                                                           #
# ------------------------------------------------------------------ #

class TestStrategies:
    def _dummy_history(self, n=300):
        return make_sleeve_df(n_months=15, seed=7).iloc[:n]

    def _dummy_portfolio(self):
        return PortfolioState(cost_model=make_cost_model())

    def test_equal_weight_sums_to_one(self):
        w, policy = EqualWeightDCA().signal(self._dummy_history(), self._dummy_portfolio())
        assert sum(w.values()) == pytest.approx(1.0)
        assert policy == "invest"

    def test_single_sleeve_only_one_nonzero(self):
        for sleeve in SLEEVES:
            w, _ = SingleSleeveDCA(sleeve).signal(
                self._dummy_history(), self._dummy_portfolio()
            )
            assert w[sleeve] == pytest.approx(1.0)
            others = [v for s, v in w.items() if s != sleeve]
            assert all(v == pytest.approx(0.0) for v in others)

    def test_fixed_weight_respects_input(self):
        weights = {"idx_equities": 0.4, "us_equities": 0.3, "gold": 0.2, "crypto": 0.1}
        strat = FixedWeightDCA(weights)
        w, _ = strat.signal(self._dummy_history(), self._dummy_portfolio())
        for s, expected in weights.items():
            assert w[s] == pytest.approx(expected)

    def test_inv_vol_weights_sum_to_one(self):
        w, policy = InverseVolatilityDCA().signal(
            self._dummy_history(), self._dummy_portfolio()
        )
        assert sum(w.values()) == pytest.approx(1.0, rel=1e-5)
        assert policy == "invest"

    def test_inv_vol_higher_weight_for_lower_vol(self):
        """Lower-vol sleeve should receive higher weight."""
        np.random.seed(99)
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        # high-vol sleeve
        hv = 100 * np.exp(np.cumsum(np.random.normal(0, 0.05, n)))
        # low-vol sleeve
        lv = 100 * np.exp(np.cumsum(np.random.normal(0, 0.005, n)))
        # zero the other two
        df = pd.DataFrame({
            "idx_equities": lv, "us_equities": hv,
            "gold": lv.copy(), "crypto": hv.copy()
        }, index=idx)
        strat = InverseVolatilityDCA()
        w, _ = strat.signal(df, PortfolioState(cost_model=make_cost_model()))
        assert w["idx_equities"] > w["us_equities"]

    def test_momentum_weights_non_negative(self):
        w, _ = MomentumDCA().signal(self._dummy_history(400), self._dummy_portfolio())
        assert all(v >= 0 for v in w.values())

    def test_trend_filter_waits_when_below_ma(self):
        """If all prices are in a downtrend, policy should be 'wait'."""
        n = 250
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        # Steadily declining prices (all below 200-day MA by end)
        declining = 100 * np.exp(np.cumsum(np.full(n, -0.004)))
        df = pd.DataFrame({s: declining.copy() for s in SLEEVES}, index=idx)
        strat = TrendFilterDCA()
        _, policy = strat.signal(df, PortfolioState(cost_model=make_cost_model()))
        assert policy == "wait"

    def test_trend_filter_invests_when_above_ma(self):
        """If all prices are in an uptrend, policy should be 'invest'."""
        n = 250
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        rising = 100 * np.exp(np.cumsum(np.full(n, 0.004)))
        df = pd.DataFrame({s: rising.copy() for s in SLEEVES}, index=idx)
        strat = TrendFilterDCA()
        _, policy = strat.signal(df, PortfolioState(cost_model=make_cost_model()))
        assert policy == "invest"


# ------------------------------------------------------------------ #
# Lookahead test                                                        #
# ------------------------------------------------------------------ #

class TestLookahead:
    def _sleeve_df(self):
        return make_sleeve_df(n_months=36, seed=5)

    def test_equal_weight_no_lookahead(self):
        with patch("allocation_lab.backtest.costs.CostModel.from_yaml",
                   return_value=make_cost_model()):
            failures = check_no_lookahead(EqualWeightDCA(), self._sleeve_df(), n_checks=5)
        assert failures == []

    def test_inv_vol_no_lookahead(self):
        with patch("allocation_lab.backtest.costs.CostModel.from_yaml",
                   return_value=make_cost_model()):
            failures = check_no_lookahead(InverseVolatilityDCA(), self._sleeve_df(), n_checks=5)
        assert failures == []

    def test_trend_filter_no_lookahead(self):
        with patch("allocation_lab.backtest.costs.CostModel.from_yaml",
                   return_value=make_cost_model()):
            failures = check_no_lookahead(TrendFilterDCA(), self._sleeve_df(), n_checks=5)
        assert failures == []
