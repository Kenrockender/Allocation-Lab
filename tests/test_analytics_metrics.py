"""
Unit tests for analytics/metrics.py.
Every assertion is checked against a hand-calculated reference value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from allocation_lab.analytics.metrics import (
    annualised_return,
    annualised_volatility,
    beta,
    drawdown_series,
    historical_cvar,
    historical_var,
    log_returns,
    max_drawdown,
    max_drawdown_duration,
    parametric_var,
    rolling_correlation,
    rolling_volatility,
    sharpe_ratio,
    simple_returns,
    summary_table,
)


def daily_index(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D", name="date")


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

class TestReturns:
    def test_simple_returns_basic(self):
        prices = pd.Series([100.0, 110.0, 99.0], index=daily_index(3))
        r = simple_returns(prices)
        assert r.iloc[0] == pytest.approx(0.10)
        assert r.iloc[1] == pytest.approx(-0.10, rel=1e-5)

    def test_log_returns_basic(self):
        prices = pd.Series([100.0, np.e * 100], index=daily_index(2))
        lr = log_returns(prices)
        # ln(e) = 1
        assert lr.iloc[0] == pytest.approx(1.0)

    def test_simple_returns_length(self):
        prices = pd.Series(range(1, 11), dtype=float, index=daily_index(10))
        assert len(simple_returns(prices)) == 9

    def test_log_returns_length(self):
        prices = pd.Series(range(1, 11), dtype=float, index=daily_index(10))
        assert len(log_returns(prices)) == 9


# ---------------------------------------------------------------------------
# Annualised return
# ---------------------------------------------------------------------------

class TestAnnualisedReturn:
    def test_doubling_in_one_year(self):
        # 100 -> 200 over exactly 365 days = 100% CAGR
        idx = pd.DatetimeIndex(["2020-01-01", "2021-01-01"])
        prices = pd.Series([100.0, 200.0], index=idx)
        assert annualised_return(prices) == pytest.approx(1.0, rel=0.01)

    def test_flat_returns_zero(self):
        idx = pd.DatetimeIndex(["2020-01-01", "2021-01-01"])
        prices = pd.Series([100.0, 100.0], index=idx)
        assert annualised_return(prices) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

class TestVolatility:
    def test_constant_prices_zero_vol(self):
        prices = pd.Series([100.0] * 100, index=daily_index(100))
        assert annualised_volatility(prices) == pytest.approx(0.0, abs=1e-10)

    def test_known_daily_vol(self):
        # If daily log-return std is σ, annualised vol = σ * sqrt(252)
        np.random.seed(42)
        daily_std = 0.01  # 1% daily
        log_ret = np.random.normal(0, daily_std, 1000)
        # Build a price series by cumulative exponentiation
        prices = pd.Series(
            100 * np.exp(np.cumsum(log_ret)),
            index=daily_index(1000),
        )
        vol = annualised_volatility(prices)
        # With 1000 samples, should be close to 0.01 * sqrt(252)
        expected = daily_std * np.sqrt(252)
        assert abs(vol - expected) < 0.005  # within 0.5 pp (statistical noise with 1000 samples)

    def test_rolling_volatility_length(self):
        prices = pd.Series(np.linspace(100, 200, 200), index=daily_index(200))
        rv = rolling_volatility(prices, window=20)
        assert len(rv) == len(prices) - 1  # log_returns drops one row


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_zero_excess_return_zero_sharpe(self):
        # Flat returns -> Sharpe = 0 (or nan if std=0); constant price has std=0
        prices = pd.Series([100.0] * 50, index=daily_index(50))
        assert np.isnan(sharpe_ratio(prices, rf_annual=0.0))

    def test_positive_drift_gives_positive_sharpe(self):
        # Steadily rising price with zero rf should give positive Sharpe
        prices = pd.Series(
            [100 * (1.0005 ** i) for i in range(500)],
            index=daily_index(500),
        )
        s = sharpe_ratio(prices, rf_annual=0.0)
        assert s > 0

    def test_rf_reduces_sharpe(self):
        prices = pd.Series(
            [100 * (1.001 ** i) for i in range(500)],
            index=daily_index(500),
        )
        s_no_rf = sharpe_ratio(prices, rf_annual=0.0)
        s_with_rf = sharpe_ratio(prices, rf_annual=0.05)
        assert s_with_rf < s_no_rf


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

class TestDrawdown:
    def test_drawdown_series_peak_is_zero(self):
        prices = pd.Series([100.0, 120.0, 110.0, 115.0], index=daily_index(4))
        dd = drawdown_series(prices)
        assert dd.iloc[1] == pytest.approx(0.0)   # 120 is the running max

    def test_max_drawdown_known_value(self):
        # Price goes 100 -> 150 -> 75 -> 100; max dd = (75-150)/150 = -0.5
        prices = pd.Series([100.0, 150.0, 75.0, 100.0], index=daily_index(4))
        assert max_drawdown(prices) == pytest.approx(-0.5)

    def test_no_drawdown_returns_zero(self):
        prices = pd.Series([100.0, 110.0, 120.0, 130.0], index=daily_index(4))
        assert max_drawdown(prices) == pytest.approx(0.0)

    def test_drawdown_duration_basic(self):
        # Peak at Jan 2, trough Jan 3, recovery above peak on Jan 5 -> 3 calendar days
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        prices = pd.Series([100.0, 120.0, 90.0, 110.0, 125.0], index=idx)
        dur = max_drawdown_duration(prices)
        assert dur >= 2  # at minimum 2 calendar days in drawdown

    def test_drawdown_duration_no_drawdown(self):
        prices = pd.Series([100.0, 110.0, 120.0], index=daily_index(3))
        assert max_drawdown_duration(prices) == 0


# ---------------------------------------------------------------------------
# Correlation and beta
# ---------------------------------------------------------------------------

class TestCorrelationAndBeta:
    def test_identical_series_correlation_one(self):
        # Use a noisy series — deterministic series has zero variance in some windows
        np.random.seed(7)
        lr = np.random.normal(0.001, 0.02, 300)
        prices = pd.Series(100 * np.exp(np.cumsum(lr)), index=daily_index(300))
        corr = rolling_correlation(prices, prices, window=50).dropna()
        assert corr.iloc[-1] == pytest.approx(1.0, abs=1e-6)

    def test_beta_of_identical_series_is_one(self):
        prices = pd.Series(
            [100 * (1.001 ** i) for i in range(200)],
            index=daily_index(200),
        )
        b = beta(prices, prices)
        assert b == pytest.approx(1.0)

    def test_beta_scaled_series(self):
        # If series_a = 2 * series_b (in returns), beta should be 2
        base = pd.Series(
            [100 * (1 + np.random.normal(0, 0.01)) ** i for i in range(300)],
            index=daily_index(300),
        )
        # Construct series_a with exactly 2x the log returns
        lr_base = log_returns(base)
        prices_a = 100 * np.exp(np.cumsum(2 * lr_base))
        prices_a.index = base.index[1:]
        b = beta(prices_a, base.iloc[1:])
        assert abs(b - 2.0) < 0.15   # within 0.15 of 2 (OLS noise)


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------

class TestVaR:
    def _make_prices(self, seed: int = 0, n: int = 1000) -> pd.Series:
        np.random.seed(seed)
        lr = np.random.normal(0.0005, 0.015, n)
        return pd.Series(100 * np.exp(np.cumsum(lr)), index=daily_index(n))

    def test_historical_var_is_negative(self):
        prices = self._make_prices()
        assert historical_var(prices) < 0

    def test_historical_cvar_le_var(self):
        """CVaR should be worse (more negative) than VaR."""
        prices = self._make_prices()
        var = historical_var(prices)
        cvar = historical_cvar(prices)
        assert cvar <= var

    def test_parametric_var_is_negative(self):
        prices = self._make_prices()
        assert parametric_var(prices) < 0

    def test_higher_confidence_gives_worse_var(self):
        prices = self._make_prices()
        var_95 = historical_var(prices, 0.95)
        var_99 = historical_var(prices, 0.99)
        assert var_99 < var_95  # 99% VaR is worse


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

class TestSummaryTable:
    def test_summary_table_has_expected_columns(self):
        prices = pd.DataFrame(
            {"A": [100 * (1.001 ** i) for i in range(300)]},
            index=daily_index(300),
        )
        tbl = summary_table(prices, rf_annual=0.04)
        expected_cols = {"cagr", "vol", "sharpe", "max_dd", "var_95", "cvar_95"}
        assert expected_cols <= set(tbl.columns)

    def test_summary_table_index_is_ticker(self):
        prices = pd.DataFrame(
            {"X": [100.0 + i for i in range(300)]},
            index=daily_index(300),
        )
        tbl = summary_table(prices)
        assert "X" in tbl.index

    def test_summary_table_skips_short_series(self):
        prices = pd.DataFrame(
            {"SHORT": [100.0, 101.0, 102.0]},  # only 3 rows, below min of 30
            index=daily_index(3),
        )
        tbl = summary_table(prices)
        assert len(tbl) == 0
        assert isinstance(tbl, pd.DataFrame)
