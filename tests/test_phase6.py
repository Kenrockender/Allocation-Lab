"""
Tests for Phase 6: vol_forecast.py and vol_overlay.py.
All tests use synthetic data — no files, no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from allocation_lab.research.vol_forecast import EWMAVolatility, GRUVolatility, ANN
from allocation_lab.research.vol_overlay import (
    EWMAOverlayDCA, GRUOverlayDCA, _vol_scaled_weights, TARGET_VOL
)
from allocation_lab.backtest.costs import CostModel
from allocation_lab.backtest.portfolio import PortfolioState, SLEEVES


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


def make_log_returns(n: int = 500, seed: int = 0, vol: float = 0.015) -> pd.Series:
    np.random.seed(seed)
    r = np.random.normal(0, vol, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", name="date")
    return pd.Series(r, index=idx, name="test")


def make_sleeve_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", name="date")
    df = pd.DataFrame(index=idx)
    for sleeve in SLEEVES:
        r = np.random.normal(0.0003, 0.015, n)
        df[sleeve] = 100 * np.exp(np.cumsum(r))
    return df


# ------------------------------------------------------------------ #
# EWMA                                                                 #
# ------------------------------------------------------------------ #

class TestEWMA:
    def test_fit_predict_positive(self):
        lr = make_log_returns(200)
        ewma = EWMAVolatility()
        ewma.fit(lr)
        vol = ewma.predict(lr.tail(10))
        assert vol > 0

    def test_predict_series_length(self):
        lr = make_log_returns(200)
        ewma = EWMAVolatility()
        ewma.fit(lr[:100])
        s = ewma.predict_series(lr)
        assert len(s) == len(lr)

    def test_higher_vol_input_gives_higher_forecast(self):
        """EWMA should forecast higher vol when recent returns are more volatile."""
        ewma_low  = EWMAVolatility(); ewma_low.fit(make_log_returns(100, vol=0.005))
        ewma_high = EWMAVolatility(); ewma_high.fit(make_log_returns(100, vol=0.050))
        # fresh predict on same 1-return series
        r = pd.Series([0.02])
        assert ewma_high.predict(r) > ewma_low.predict(r)

    def test_annualised_scale(self):
        """Vol forecast should be in annualised scale (~0.01*sqrt(252) ≈ 0.16 for 1% daily)."""
        lr = make_log_returns(500, vol=0.01)
        ewma = EWMAVolatility()
        ewma.fit(lr)
        vol = ewma.predict(lr.tail(50))
        # Should be roughly 0.01 * sqrt(252) ≈ 0.159 ± 50%
        assert 0.05 < vol < 0.50


# ------------------------------------------------------------------ #
# GRU                                                                  #
# ------------------------------------------------------------------ #

class TestGRU:
    def test_fit_predict_positive(self):
        lr = make_log_returns(400, seed=1)
        gru = GRUVolatility(max_epochs=5, patience=3, seed=0)
        gru.fit(lr[:300])
        vol = gru.predict(lr)
        assert vol > 0

    def test_fallback_on_short_history(self):
        """GRU with <30 samples should fall back gracefully (no crash)."""
        lr = make_log_returns(20)
        gru = GRUVolatility(max_epochs=5, patience=3, seed=0)
        gru.fit(lr)
        vol = gru.predict(lr)
        assert vol > 0

    def test_predict_series_length(self):
        lr = make_log_returns(400, seed=2)
        gru = GRUVolatility(max_epochs=5, patience=3, seed=0)
        gru.fit(lr[:300])
        s = gru.predict_series(lr)
        assert len(s) > 0
        assert (s > 0).all()


# ------------------------------------------------------------------ #
# Vol scaling helper                                                   #
# ------------------------------------------------------------------ #

class TestVolScaling:
    def test_scale_down_when_vol_high(self):
        base = {"idx_equities": 0.25, "us_equities": 0.25,
                "gold": 0.25, "crypto": 0.25}
        # All sleeves have 30% vol — above 15% target → scale down
        forecasts = {s: 0.30 for s in base}
        w = _vol_scaled_weights(base, forecasts, target_vol=0.15)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
        # Each raw scale = 15/30 = 0.5 → equal weights preserved after renorm
        for s in base:
            assert w[s] == pytest.approx(0.25, abs=1e-6)

    def test_no_scale_when_vol_low(self):
        """When all vols < target, scale = 1.0 and weights are unchanged."""
        base = {"idx_equities": 0.40, "us_equities": 0.60}
        forecasts = {s: 0.05 for s in base}   # 5% vol < 15% target
        w = _vol_scaled_weights(base, forecasts, target_vol=0.15)
        # scale = min(1, 0.15/0.05) = 1 → weights sum to 1 after renorm
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_asymmetric_scaling_changes_weights(self):
        """High-vol sleeve should get lower weight after scaling."""
        base = {"low_vol": 0.5, "high_vol": 0.5}
        forecasts = {"low_vol": 0.10, "high_vol": 0.30}
        w = _vol_scaled_weights(base, forecasts, target_vol=0.15)
        assert w["low_vol"] > w["high_vol"]

    def test_weights_sum_to_one(self):
        base = {s: 0.25 for s in SLEEVES}
        forecasts = {s: np.random.uniform(0.05, 0.50) for s in SLEEVES}
        w = _vol_scaled_weights(base, forecasts)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------ #
# Overlay strategies                                                   #
# ------------------------------------------------------------------ #

class TestOverlayStrategies:
    def _portfolio(self):
        return PortfolioState(cost_model=make_cost_model())

    def test_ewma_overlay_returns_valid_weights(self):
        sleeve_df = make_sleeve_df(n=300, seed=3)
        strat = EWMAOverlayDCA()
        w, policy = strat.signal(sleeve_df, self._portfolio())
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-4)
        assert policy == "invest"
        assert all(v >= 0 for v in w.values())

    def test_ewma_overlay_short_history_fallback(self):
        sleeve_df = make_sleeve_df(n=10, seed=4)
        strat = EWMAOverlayDCA()
        w, policy = strat.signal(sleeve_df, self._portfolio())
        assert policy == "invest"

    def test_gru_overlay_returns_valid_weights(self):
        sleeve_df = make_sleeve_df(n=400, seed=5)
        strat = GRUOverlayDCA(max_epochs=5, patience=3, seed=0)
        w, policy = strat.signal(sleeve_df, self._portfolio())
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-4)
        assert policy == "invest"

    def test_overlay_runs_in_backtest(self):
        """Full backtest with EWMA overlay should complete without error."""
        from allocation_lab.backtest.engine import run_backtest
        sleeve_df = make_sleeve_df(n=500, seed=6)
        strat = EWMAOverlayDCA()
        with patch("allocation_lab.backtest.costs.CostModel.from_yaml",
                   return_value=make_cost_model()):
            result = run_backtest(strat, sleeve_df, contribution_idr=1_000_000)
        assert result.nav_series.iloc[-1] > 0
        assert result.total_costs_idr > 0

    def test_gru_overlay_runs_in_backtest(self):
        from allocation_lab.backtest.engine import run_backtest
        sleeve_df = make_sleeve_df(n=500, seed=7)
        strat = GRUOverlayDCA(max_epochs=5, patience=3, seed=0)
        with patch("allocation_lab.backtest.costs.CostModel.from_yaml",
                   return_value=make_cost_model()):
            result = run_backtest(strat, sleeve_df, contribution_idr=1_000_000)
        assert result.nav_series.iloc[-1] > 0
