"""
align.py — Load raw parquet files, convert USD assets to IDR, and produce
a single aligned daily price table for the full universe.

The output is:  data/processed/prices_idr.parquet
Columns:        one per ticker, values are IDR-denominated adjusted close prices.
Index:          DatetimeIndex named 'date', daily frequency (business days).

The alignment rule (from decisions.md #11):
  Evaluate on the contribution date using the LAST AVAILABLE price.
  USD assets are converted using the USD/IDR rate on the same date
  (or the last available rate if the FX market is also closed).

Missing values:
  IDX and US tickers trade Mon-Fri on their respective calendars.
  Crypto trades 7 days a week.
  After forward-filling each series independently (up to 5 days),
  we reindex to the union of all trading days and then forward-fill
  again so every row has a value.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]  # repo root
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CONFIG_DIR = ROOT / "config"

MAX_FFILL_DAYS = 5  # forward-fill limit before we treat a price as stale  # forward-fill limit before we treat a price as stale


def _load_universe() -> dict:
    with open(CONFIG_DIR / "universe.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_raw(ticker: str) -> pd.Series | None:
    """Read the adjusted close from a raw parquet file."""
    safe_name = ticker.replace("/", "_")
    path = RAW_DIR / f"{safe_name}.parquet"
    if not path.exists():
        logger.warning("Raw file not found for %s: %s", ticker, path)
        return None

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # yfinance uses 'Close' when auto_adjust=True
    if "Close" in df.columns:
        return df["Close"].rename(ticker)

    logger.warning("No 'Close' column in %s (columns: %s)", path.name, list(df.columns))
    return None


def load_fx_series() -> pd.Series:
    """Return USD/IDR close prices (IDR per 1 USD)."""
    universe = _load_universe()
    fx_ticker = universe["fx"]["usd_idr"]
    series = _read_raw(fx_ticker)
    if series is None:
        raise FileNotFoundError(
            f"USD/IDR raw file missing. Run `python -m allocation_lab.data.fetch` first."
        )
    return series.rename("USDIDR")


def build_price_table(start: str = "2018-01-01", end: str | None = None) -> pd.DataFrame:
    """Build the aligned IDR-denominated price table.

    Parameters
    ----------
    start : str
        First date to include (backtest start from decisions.md).
    end : str | None
        Last date to include (default: latest available).

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (daily business days), columns: ticker strings.
        Values: adjusted close prices in IDR.
    """
    universe = _load_universe()
    sleeves = universe["sleeves"]

    # Gather all tickers (everything except FX and benchmark which we handle separately)
    usd_tickers: list[str] = (
        sleeves["us_equities"]["members"]
        + sleeves["gold"]["members"]
        + sleeves["crypto"]["members"]
    )
    idr_tickers: list[str] = sleeves["idx_equities"]["members"]
    benchmark_ticker: str = universe["benchmarks"]["reference_index"]

    # Load FX series
    fx = load_fx_series()
    fx = fx.ffill(limit=MAX_FFILL_DAYS)

    # Load all price series
    all_series: dict[str, pd.Series] = {}

    for ticker in idr_tickers + usd_tickers + [benchmark_ticker]:
        s = _read_raw(ticker)
        if s is not None:
            all_series[ticker] = s

    if not all_series:
        raise RuntimeError("No raw price files found. Run fetch.py first.")

    # Build a combined DataFrame on the union of all dates
    raw_df = pd.DataFrame(all_series)
    raw_df.index = pd.to_datetime(raw_df.index)
    raw_df = raw_df.sort_index()

    # Forward-fill each series independently (handles local holidays and weekends)
    raw_df = raw_df.ffill(limit=MAX_FFILL_DAYS)

    # Reindex to full calendar range, then forward-fill again
    full_idx = pd.date_range(
        start=raw_df.index.min(),
        end=raw_df.index.max(),
        freq="D",
    )
    raw_df = raw_df.reindex(full_idx)
    raw_df.index.name = "date"
    raw_df = raw_df.ffill(limit=MAX_FFILL_DAYS)

    # Align FX to the same index
    fx_aligned = fx.reindex(full_idx).ffill(limit=MAX_FFILL_DAYS)

    # Convert USD-denominated tickers to IDR
    for ticker in usd_tickers:
        if ticker in raw_df.columns:
            raw_df[ticker] = raw_df[ticker] * fx_aligned

    # The benchmark (^JKSE) is already in IDR
    # Trim to the requested date range
    if start:
        raw_df = raw_df.loc[start:]
    if end:
        raw_df = raw_df.loc[:end]

    # Drop rows where everything is NaN (pure weekend with no data)
    raw_df = raw_df.dropna(how="all")

    return raw_df


def build_sleeve_series(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute equal-weight sleeve total-return indices from the price table.

    Returns a DataFrame with columns: idx_equities, us_equities, gold, crypto, benchmark.
    Each column is an IDR price index starting at 100 on its first valid date.
    """
    universe = _load_universe()
    sleeves = universe["sleeves"]

    def _sleeve_index(tickers: list[str], name: str) -> pd.Series:
        available = [t for t in tickers if t in price_df.columns]
        if not available:
            # Sleeve has no data yet (e.g. missing raw file or empty member list).
            # Return an all-NaN series so callers can decide what to do.
            logger.warning("No tickers available for sleeve '%s': %s — returning NaN series", name, tickers)
            return pd.Series(np.nan, index=price_df.index, name=name)
        # Equal-weight: average the price level of each normalised member
        # We normalise each member to start at 1.0 on its first valid date
        sub = price_df[available].copy()
        normalised = sub / sub.bfill().iloc[0]
        sleeve = normalised.mean(axis=1)
        # Convert to index starting at 100
        first_valid = sleeve.first_valid_index()
        sleeve = sleeve / sleeve.loc[first_valid] * 100
        return sleeve.rename(name)

    idx_sleeve = _sleeve_index(sleeves["idx_equities"]["members"], "idx_equities")
    us_sleeve = _sleeve_index(sleeves["us_equities"]["members"], "us_equities")
    gold_sleeve = _sleeve_index(sleeves["gold"]["members"], "gold")
    crypto_sleeve = _sleeve_index(sleeves["crypto"]["members"], "crypto")

    benchmark_ticker = universe["benchmarks"]["reference_index"]
    if benchmark_ticker in price_df.columns:
        bench = price_df[benchmark_ticker].copy()
        first_valid = bench.first_valid_index()
        bench = bench / bench.loc[first_valid] * 100
        bench = bench.rename("benchmark_ihsg")
    else:
        bench = pd.Series(dtype=float, name="benchmark_ihsg")

    sleeve_df = pd.concat(
        [idx_sleeve, us_sleeve, gold_sleeve, crypto_sleeve, bench], axis=1
    )
    return sleeve_df


def save_processed(
    price_df: pd.DataFrame,
    sleeve_df: pd.DataFrame,
) -> None:
    """Save processed tables to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    price_path = PROCESSED_DIR / "prices_idr.parquet"
    sleeve_path = PROCESSED_DIR / "sleeve_indices.parquet"

    price_df.to_parquet(price_path)
    sleeve_df.to_parquet(sleeve_path)

    logger.info("Saved prices_idr.parquet (%d rows x %d cols)", *price_df.shape)
    logger.info("Saved sleeve_indices.parquet (%d rows x %d cols)", *sleeve_df.shape)


def run(start: str = "2018-01-01", end: str | None = None) -> None:
    """Full pipeline: load raw -> align -> convert -> save."""
    logger.info("Building aligned price table ...")
    price_df = build_price_table(start=start, end=end)
    logger.info("Price table: %d rows x %d cols", *price_df.shape)

    logger.info("Building sleeve indices ...")
    sleeve_df = build_sleeve_series(price_df)
    logger.info("Sleeve indices: %d rows x %d cols", *sleeve_df.shape)

    save_processed(price_df, sleeve_df)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Align raw prices and build sleeve indices.")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    run(start=args.start, end=args.end)
