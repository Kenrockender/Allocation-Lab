"""
metrics.py — Core return and risk metric functions.

All functions are pure: they accept pandas Series/DataFrames and return
scalars or Series. No file I/O here.

Conventions
-----------
- prices   : pd.Series of price levels (IDR), DatetimeIndex, daily
- returns  : pd.Series of simple (not log) daily returns, already computed
- rf       : annual risk-free rate as a decimal (e.g. 0.0575 for 5.75%)
- ann      : annualisation factor; default 252 (trading days per year)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def simple_returns(prices: pd.Series) -> pd.Series:
    """Daily simple returns: (P_t / P_{t-1}) - 1."""
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def annualised_return(prices: pd.Series, ann: int = 252) -> float:
    """Compound annual growth rate from first to last price.

    Uses calendar days so that a series with gaps (crypto) isn't overstated.
    """
    total_days = (prices.index[-1] - prices.index[0]).days
    if total_days <= 0:
        return float("nan")
    total_return = prices.iloc[-1] / prices.iloc[0]
    return float(total_return ** (365 / total_days) - 1)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def annualised_volatility(prices: pd.Series, ann: int = 252) -> float:
    """Annualised volatility of log returns."""
    lr = log_returns(prices)
    if len(lr) < 2:
        return float("nan")
    return float(lr.std() * np.sqrt(ann))


def rolling_volatility(
    prices: pd.Series,
    window: int = 63,   # ~quarter
    ann: int = 252,
) -> pd.Series:
    """Rolling annualised volatility of log returns."""
    lr = log_returns(prices)
    return (lr.rolling(window).std() * np.sqrt(ann)).rename(f"vol_{window}d")


# ---------------------------------------------------------------------------
# Risk-adjusted return
# ---------------------------------------------------------------------------

def sharpe_ratio(
    prices: pd.Series,
    rf_annual: float = 0.0,
    ann: int = 252,
) -> float:
    """Annualised Sharpe ratio: (annualised return - rf) / annualised vol.

    Uses log returns for both numerator and denominator for consistency.
    """
    lr = log_returns(prices)
    if len(lr) < 2:
        return float("nan")
    rf_daily = rf_annual / ann
    excess = lr - rf_daily
    if excess.std() == 0:
        return float("nan")
    return float(excess.mean() / excess.std() * np.sqrt(ann))


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def drawdown_series(prices: pd.Series) -> pd.Series:
    """Drawdown at each point: (price - running_max) / running_max."""
    running_max = prices.cummax()
    return (prices - running_max) / running_max


def max_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown (negative number, e.g. -0.35 for -35%)."""
    dd = drawdown_series(prices)
    return float(dd.min())


def max_drawdown_duration(prices: pd.Series) -> int:
    """Length in calendar days of the longest drawdown period (peak to recovery)."""
    dd = drawdown_series(prices)
    in_dd = dd < 0

    if not in_dd.any():
        return 0

    max_dur = 0
    start = None
    for date, flag in in_dd.items():
        if flag and start is None:
            start = date
        elif not flag and start is not None:
            dur = (date - start).days
            max_dur = max(max_dur, dur)
            start = None

    # If still in drawdown at end of series
    if start is not None:
        dur = (in_dd.index[-1] - start).days
        max_dur = max(max_dur, dur)

    return max_dur


# ---------------------------------------------------------------------------
# Correlation and beta
# ---------------------------------------------------------------------------

def rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling Pearson correlation of log returns."""
    lr_a = log_returns(series_a)
    lr_b = log_returns(series_b)
    aligned = pd.concat([lr_a, lr_b], axis=1).dropna()
    a_col, b_col = aligned.columns
    return aligned[a_col].rolling(window).corr(aligned[b_col]).rename(
        f"corr_{a_col}_{b_col}_{window}d"
    )


def beta(prices: pd.Series, benchmark_prices: pd.Series) -> float:
    """OLS beta of the series vs a benchmark (on log returns)."""
    lr = log_returns(prices)
    lr_bm = log_returns(benchmark_prices)
    aligned = pd.concat([lr, lr_bm], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1])


# ---------------------------------------------------------------------------
# Value at Risk / Conditional Value at Risk
# ---------------------------------------------------------------------------

def historical_var(
    prices: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Historical (empirical) VaR at a given confidence level.

    Returns a negative number: e.g. -0.025 means "lose at least 2.5% on
    the worst (1-confidence)% of days".
    """
    lr = log_returns(prices)
    return float(np.percentile(lr, (1 - confidence) * 100))


def historical_cvar(
    prices: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Historical CVaR (Expected Shortfall): mean of returns below the VaR threshold."""
    lr = log_returns(prices)
    var = historical_var(prices, confidence)
    tail = lr[lr <= var]
    if tail.empty:
        return var
    return float(tail.mean())


def parametric_var(
    prices: pd.Series,
    confidence: float = 0.95,
    ann: int = 252,
) -> float:
    """Parametric (normal) daily VaR."""
    from scipy.stats import norm
    lr = log_returns(prices)
    mu = lr.mean()
    sigma = lr.std()
    return float(norm.ppf(1 - confidence, loc=mu, scale=sigma))


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summary_table(
    price_df: pd.DataFrame,
    rf_annual: float = 0.0,
    benchmark_col: str | None = None,
    ann: int = 252,
) -> pd.DataFrame:
    """Compute a risk/return summary for every column in price_df.

    Returns a DataFrame with tickers as index and metrics as columns.
    """
    rows = []
    bm_prices = price_df[benchmark_col] if benchmark_col and benchmark_col in price_df.columns else None

    for col in price_df.columns:
        prices = price_df[col].dropna()
        if len(prices) < 30:
            continue

        row: dict = {"ticker": col}
        row["cagr"] = annualised_return(prices)
        row["vol"] = annualised_volatility(prices, ann=ann)
        row["sharpe"] = sharpe_ratio(prices, rf_annual=rf_annual, ann=ann)
        row["max_dd"] = max_drawdown(prices)
        row["max_dd_days"] = max_drawdown_duration(prices)
        row["var_95"] = historical_var(prices)
        row["cvar_95"] = historical_cvar(prices)
        row["beta_vs_bm"] = (
            beta(prices, bm_prices) if bm_prices is not None else float("nan")
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("ticker")
    return df
