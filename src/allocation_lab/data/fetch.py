"""
fetch.py — Download raw price data from Yahoo Finance and save as parquet.

Run via CLI:
    python -m allocation_lab.data.fetch          # fetch everything
    python -m allocation_lab.data.fetch --dry-run

Each ticker gets its own file:  data/raw/<TICKER>.parquet
The file contains OHLCV columns plus Dividends and Stock Splits from yfinance.
Adjusted close (used for returns) is the 'Close' column when auto_adjust=True.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]  # repo root
RAW_DIR = ROOT / "data" / "raw"
CONFIG_DIR = ROOT / "config"


def _load_universe() -> dict:
    with open(CONFIG_DIR / "universe.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _all_tickers(universe: dict) -> list[str]:
    """Return every raw ticker that needs to be fetched."""
    tickers: list[str] = []
    sleeves = universe["sleeves"]

    tickers += sleeves["idx_equities"]["members"]
    tickers += sleeves["us_equities"]["members"]
    tickers += sleeves["gold"]["members"]
    tickers += sleeves["crypto"]["members"]
    tickers.append(universe["fx"]["usd_idr"])
    tickers.append(universe["benchmarks"]["reference_index"])
    return tickers


def fetch_ticker(
    ticker: str,
    start: str = "2017-01-01",
    end: str | None = None,
    dry_run: bool = False,
) -> pd.DataFrame | None:
    """Download a single ticker and save to data/raw/<ticker>.parquet.

    Returns the DataFrame, or None on failure.
    """
    import yfinance as yf  # imported here so the module is importable without it installed

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitise the ticker for use as a filename (e.g. BTC-USD -> BTC-USD, ^JKSE -> ^JKSE)
    safe_name = ticker.replace("/", "_")
    out_path = RAW_DIR / f"{safe_name}.parquet"

    if dry_run:
        logger.info("[dry-run] would fetch %s -> %s", ticker, out_path)
        return None

    logger.info("Fetching %s ...", ticker)
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,   # Close = adjusted close; dividends folded in
            progress=False,
            timeout=30,
        )
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", ticker, exc)
        return None

    if df.empty:
        logger.warning("No data returned for %s", ticker)
        return None

    # yfinance >=0.2.31 returns MultiIndex columns when downloading a single ticker;
    # flatten them to simple strings.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.index.name = "date"
    df.to_parquet(out_path)
    logger.info("  saved %d rows -> %s", len(df), out_path)
    return df


def fetch_all(
    start: str = "2017-01-01",
    end: str | None = None,
    dry_run: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch every ticker in the universe config.

    Returns a dict of {ticker: DataFrame}.
    """
    universe = _load_universe()
    tickers = _all_tickers(universe)

    results: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in tickers:
        df = fetch_ticker(ticker, start=start, end=end, dry_run=dry_run)
        if df is not None:
            results[ticker] = df
        elif not dry_run:
            failed.append(ticker)

    if failed:
        logger.warning("Failed tickers: %s", failed)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch raw price data from Yahoo Finance.")
    parser.add_argument("--start", default="2017-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), default=today")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be fetched without downloading")
    parser.add_argument("--ticker", default=None, help="Fetch a single ticker instead of all")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.ticker:
        fetch_ticker(args.ticker, start=args.start, end=args.end, dry_run=args.dry_run)
    else:
        fetch_all(start=args.start, end=args.end, dry_run=args.dry_run)
