"""
Tests for the data alignment and FX conversion logic.
Uses small hand-made DataFrames — no network calls, no files on disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from allocation_lab.data.align import build_sleeve_series
from allocation_lab.data.quality import check_ticker, _longest_run


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_prices(data: dict[str, list[float]], start: str = "2020-01-01") -> pd.DataFrame:
    """Build a price DataFrame with a daily DatetimeIndex."""
    n = max(len(v) for v in data.values())
    idx = pd.date_range(start, periods=n, freq="D", name="date")
    df = pd.DataFrame(data, index=idx)
    return df


# ---------------------------------------------------------------------------
# FX conversion
# ---------------------------------------------------------------------------

class TestFxConversion:
    def test_usd_to_idr_simple(self):
        """Multiplying a USD series by the FX rate should give IDR values."""
        usd_prices = pd.Series([10.0, 20.0, 30.0], name="SPY")
        fx_rates = pd.Series([15000.0, 15100.0, 15200.0], name="USDIDR")
        idr = usd_prices * fx_rates
        assert idr[0] == pytest.approx(150_000.0)
        assert idr[1] == pytest.approx(302_000.0)
        assert idr[2] == pytest.approx(456_000.0)

    def test_usd_to_idr_preserves_returns(self):
        """When FX is flat, USD returns == IDR returns."""
        usd = pd.Series([100.0, 105.0, 110.0])
        fx = pd.Series([15000.0, 15000.0, 15000.0])
        idr = usd * fx
        usd_ret = usd.pct_change().dropna()
        idr_ret = idr.pct_change().dropna()
        pd.testing.assert_series_equal(usd_ret, idr_ret)


# ---------------------------------------------------------------------------
# Calendar alignment / forward-fill
# ---------------------------------------------------------------------------

class TestAlignment:
    def test_ffill_fills_weekend_gaps(self):
        """A Friday close should fill Saturday and Sunday."""
        dates = pd.date_range("2020-01-03", periods=7, freq="D", name="date")  # Fri..Thu
        prices = pd.Series([100.0, np.nan, np.nan, 101.0, 102.0, 103.0, 104.0], index=dates)
        filled = prices.ffill(limit=5)
        # Sat and Sun (index 1,2) should be filled from Friday
        assert filled.iloc[1] == pytest.approx(100.0)
        assert filled.iloc[2] == pytest.approx(100.0)
        # Monday restores the real value
        assert filled.iloc[3] == pytest.approx(101.0)

    def test_ffill_does_not_exceed_limit(self):
        """Forward-fill must not propagate beyond MAX_FFILL_DAYS (5)."""
        prices = pd.Series([100.0] + [np.nan] * 10)
        filled = prices.ffill(limit=5)
        # Positions 1-5 get filled; 6-10 remain NaN
        assert filled.iloc[5] == pytest.approx(100.0)
        assert np.isnan(filled.iloc[6])

    def test_aligned_index_is_union_of_calendars(self):
        """After reindex to full calendar, every date in the range has a row."""
        dates_a = pd.date_range("2020-01-01", periods=5, freq="B", name="date")
        dates_b = pd.date_range("2020-01-01", periods=7, freq="D", name="date")
        s_a = pd.Series(range(5), index=dates_a, dtype=float)
        s_b = pd.Series(range(7), index=dates_b, dtype=float)
        combined = pd.concat([s_a, s_b], axis=1, sort=True)
        full_idx = pd.date_range(combined.index.min(), combined.index.max(), freq="D")
        reindexed = combined.reindex(full_idx)
        assert len(reindexed) == len(full_idx)


# ---------------------------------------------------------------------------
# Sleeve index construction
# ---------------------------------------------------------------------------

class TestSleeveIndex:
    def test_single_member_sleeve_starts_at_100(self):
        """A sleeve with one member should start at 100."""
        # Build a minimal price_df that matches what align.py expects
        prices = make_prices({"SPY": [50.0, 52.0, 54.0, 53.0, 55.0]})
        # Patch universe to point to this ticker
        import unittest.mock as mock
        universe = {
            "sleeves": {
                "idx_equities": {"members": [], "weighting": "equal", "rebalance": "monthly"},
                "us_equities": {"members": ["SPY"], "currency": "USD"},
                "gold": {"members": [], "currency": "USD"},
                "crypto": {"members": [], "currency": "USD"},
            },
            "benchmarks": {"reference_index": "^JKSE"},
        }
        with mock.patch("allocation_lab.data.align._load_universe", return_value=universe):
            sleeve_df = build_sleeve_series(prices)
        assert sleeve_df["us_equities"].iloc[0] == pytest.approx(100.0)

    def test_equal_weight_sleeve_averages_members(self):
        """Equal-weight sleeve with two members should average normalised levels."""
        # Two tickers with identical price paths -> sleeve == either normalised series
        prices = make_prices({"A": [100.0, 110.0, 120.0], "B": [200.0, 220.0, 240.0]})
        import unittest.mock as mock
        universe = {
            "sleeves": {
                "idx_equities": {"members": ["A", "B"], "weighting": "equal", "rebalance": "monthly"},
                "us_equities": {"members": [], "currency": "USD"},
                "gold": {"members": [], "currency": "USD"},
                "crypto": {"members": [], "currency": "USD"},
            },
            "benchmarks": {"reference_index": "^JKSE"},
        }
        with mock.patch("allocation_lab.data.align._load_universe", return_value=universe):
            sleeve_df = build_sleeve_series(prices)
        # Both normalise identically, so sleeve equals a single-member normalisation
        expected = pd.Series([100.0, 110.0, 120.0])
        actual = sleeve_df["idx_equities"].reset_index(drop=True)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_sleeve_handles_missing_member_gracefully(self):
        """Sleeve should only use members that exist in the price_df; missing ones are skipped."""
        prices = make_prices({"SPY": [100.0, 105.0, 110.0]})
        import unittest.mock as mock
        universe = {
            "sleeves": {
                "idx_equities": {"members": [], "weighting": "equal", "rebalance": "monthly"},
                "us_equities": {"members": ["SPY", "MISSING_TICKER"], "currency": "USD"},
                "gold": {"members": [], "currency": "USD"},
                "crypto": {"members": [], "currency": "USD"},
            },
            "benchmarks": {"reference_index": "^JKSE"},
        }
        with mock.patch("allocation_lab.data.align._load_universe", return_value=universe):
            sleeve_df = build_sleeve_series(prices)
        # us_equities should work with just SPY (MISSING_TICKER is skipped)
        assert sleeve_df["us_equities"].iloc[0] == pytest.approx(100.0)
        # idx_equities has no members at all -> NaN series
        assert sleeve_df["idx_equities"].isna().all()


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

class TestQualityChecks:
    def test_longest_run_all_false(self):
        mask = pd.Series([False, False, False])
        assert _longest_run(mask) == 0

    def test_longest_run_basic(self):
        mask = pd.Series([True, True, False, True, True, True, False])
        assert _longest_run(mask) == 3

    def test_check_ticker_clean_series(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="D", name="date")
        prices = pd.Series(np.linspace(100, 200, 100), index=idx, name="TEST")
        result = check_ticker(prices)
        assert result["n_missing"] == 0
        assert result["n_zero_or_negative"] == 0
        assert result["longest_stale"] == 0
        assert result["n_large_moves"] == 0

    def test_check_ticker_detects_gap(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="D", name="date")
        data = [100.0] * 5 + [np.nan] * 12 + [105.0] * 3
        prices = pd.Series(data, index=idx, name="GAPPY")
        result = check_ticker(prices)
        assert result["longest_gap"] == 12

    def test_check_ticker_detects_large_move(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D", name="date")
        prices = pd.Series([100.0, 100.0, 100.0, 60.0, 100.0], index=idx, name="CRASH")
        result = check_ticker(prices)
        assert result["n_large_moves"] >= 1

    def test_check_ticker_detects_stale(self):
        idx = pd.date_range("2020-01-01", periods=30, freq="D", name="date")
        # 20 days of identical prices — above the 15-day threshold
        prices = pd.Series([100.0] * 30, index=idx, name="FROZEN")
        result = check_ticker(prices)
        assert result["longest_stale"] >= 20
