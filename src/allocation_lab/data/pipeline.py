"""
pipeline.py — One-command data build: fetch -> align -> quality check.

Usage:
    python -m allocation_lab.data.pipeline
    python -m allocation_lab.data.pipeline --start 2018-01-01 --skip-fetch
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def run(
    start: str = "2018-01-01",
    end: str | None = None,
    skip_fetch: bool = False,
    dry_run: bool = False,
) -> None:
    from allocation_lab.data import fetch, align, quality

    if not skip_fetch:
        logger.info("=== Step 1/3: Fetch raw data ===")
        fetch.fetch_all(start=start, end=end, dry_run=dry_run)
    else:
        logger.info("=== Step 1/3: Skipping fetch (--skip-fetch) ===")

    if dry_run:
        logger.info("Dry run: stopping after fetch step.")
        return

    logger.info("=== Step 2/3: Align and convert ===")
    align.run(start=start, end=end)

    logger.info("=== Step 3/3: Quality checks ===")
    quality.run(save_report=True)

    logger.info("=== Done. Data pipeline complete. ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full data pipeline: fetch -> align -> quality.")
    parser.add_argument("--start", default="2018-01-01", help="Backtest start date")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip downloading; use existing raw files")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be fetched")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    run(
        start=args.start,
        end=args.end,
        skip_fetch=args.skip_fetch,
        dry_run=args.dry_run,
    )
