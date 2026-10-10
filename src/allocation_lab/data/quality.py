"""
quality.py — Data-quality checks on the aligned price table.

Checks performed:
  1. Gap detection: runs of NaN longer than MAX_GAP_DAYS
  2. Stale prices: runs of identical non-zero prices longer than MAX_STALE_DAYS
  3. Zero or negative prices
  4. Large single-day moves (absolute log return > LARGE_MOVE_THRESHOLD)
  5. First valid date per ticker (so the backtester knows when each series starts)

Outputs a plain-text quality report and returns a dict with results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]  # repo root
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "data"

MAX_GAP_DAYS = 10       # flag gaps longer than this
MAX_STALE_DAYS = 15     # IDX can have ~10 day holiday closures (Lebaran etc); flag longer runs
LARGE_MOVE_THRESHOLD = 0.30  # flag |log-return| > 30%


def _longest_run(mask: pd.Series) -> int:
    """Return the length of the longest True-run in a boolean Series."""
    if not mask.any():
        return 0
    groups = (mask != mask.shift()).cumsum()
    return int(mask.groupby(groups).sum().max())


def check_ticker(series: pd.Series) -> dict:
    """Run quality checks on a single price series."""
    result: dict = {
        "ticker": series.name,
        "n_rows": len(series),
        "first_valid": None,
        "last_valid": None,
        "n_missing": int(series.isna().sum()),
        "longest_gap": 0,
        "n_zero_or_negative": 0,
        "n_stale_runs": 0,
        "longest_stale": 0,
        "n_large_moves": 0,
        "large_move_dates": [],
    }

    valid = series.dropna()
    if valid.empty:
        logger.warning("%s: entirely empty", series.name)
        return result

    result["first_valid"] = str(valid.index[0].date())
    result["last_valid"] = str(valid.index[-1].date())

    # 1. Gaps
    result["longest_gap"] = _longest_run(series.isna())

    # 2. Zero / negative
    result["n_zero_or_negative"] = int((valid <= 0).sum())

    # 3. Stale prices (identical consecutive values)
    stale_mask = (valid.diff() == 0)
    result["longest_stale"] = _longest_run(stale_mask)
    # Count separate stale runs
    runs = stale_mask.ne(stale_mask.shift()).cumsum()
    result["n_stale_runs"] = int((stale_mask.groupby(runs).sum() >= MAX_STALE_DAYS).sum())

    # 4. Large moves
    log_ret = np.log(valid / valid.shift(1)).dropna()
    large = log_ret[log_ret.abs() > LARGE_MOVE_THRESHOLD]
    result["n_large_moves"] = len(large)
    result["large_move_dates"] = [str(d.date()) for d in large.index[:10]]  # first 10 only

    return result


def run_quality_checks(price_df: pd.DataFrame) -> dict[str, dict]:
    """Run quality checks on all tickers in the price table."""
    results = {}
    for col in price_df.columns:
        results[col] = check_ticker(price_df[col])
    return results


def format_report(results: dict[str, dict]) -> str:
    """Format quality check results as a human-readable text report."""
    lines = [
        "=" * 70,
        "DATA QUALITY REPORT",
        "=" * 70,
        "",
    ]

    issues_found = False
    for ticker, r in results.items():
        ticker_lines = []
        flags = []

        if r["n_missing"] > 0 and r["longest_gap"] > MAX_GAP_DAYS:
            flags.append(f"LONG GAP: {r['longest_gap']} days")
        if r["n_zero_or_negative"] > 0:
            flags.append(f"ZERO/NEG PRICES: {r['n_zero_or_negative']}")
        if r["longest_stale"] >= MAX_STALE_DAYS:
            flags.append(f"STALE: {r['longest_stale']} identical days in a row")
        if r["n_large_moves"] > 0:
            flags.append(f"LARGE MOVES (>{LARGE_MOVE_THRESHOLD*100:.0f}%): {r['n_large_moves']}")

        status = "OK" if not flags else "WARN"
        if flags:
            issues_found = True

        ticker_lines.append(
            f"{ticker:<20} {status:>4}  "
            f"rows={r['n_rows']}  "
            f"missing={r['n_missing']}  "
            f"first={r['first_valid']}  "
            f"last={r['last_valid']}"
        )
        for flag in flags:
            ticker_lines.append(f"  {'':20} !! {flag}")
            if "LARGE MOVES" in flag and r["large_move_dates"]:
                ticker_lines.append(f"  {'':20}    dates: {', '.join(r['large_move_dates'][:5])}")

        lines.extend(ticker_lines)

    lines.append("")
    lines.append("=" * 70)
    lines.append("SUMMARY: " + ("Issues found — review WARNs above." if issues_found else "All checks passed."))
    lines.append("=" * 70)
    return "\n".join(lines)


def run(save_report: bool = True) -> dict[str, dict]:
    """Load processed prices and run quality checks."""
    price_path = PROCESSED_DIR / "prices_idr.parquet"
    if not price_path.exists():
        raise FileNotFoundError(
            "prices_idr.parquet not found. Run align.py first."
        )

    price_df = pd.read_parquet(price_path)
    results = run_quality_checks(price_df)
    report_text = format_report(results)

    print(report_text)

    if save_report:
        report_path = REPORTS_DIR / "quality_report.txt"
        report_path.write_text(report_text, encoding="utf-8")
        logger.info("Saved quality report -> %s", report_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data quality checks.")
    parser.add_argument("--no-save", action="store_true", help="Don't save report to disk")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    run(save_report=not args.no_save)
