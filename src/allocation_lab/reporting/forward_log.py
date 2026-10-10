"""
forward_log.py — Append this month's suggested weights to forward_log.csv,
then fill in realized outcomes later.

Log format (weights only — no IDR amounts, for privacy):
  date, strategy, idx_equities, us_equities, gold, crypto,
  above_ma_sleeves, realized_xirr_pct (filled in later)

Usage:
    python -m allocation_lab.reporting.forward_log append
    python -m allocation_lab.reporting.forward_log show
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pandas as pd

from allocation_lab.reporting.monthly_report import build_report

ROOT = Path(__file__).resolve().parents[3]
LOG_PATH = ROOT / "forward_log.csv"

FIELDNAMES = [
    "date", "strategy",
    "idx_equities", "us_equities", "gold", "crypto",
    "above_ma_sleeves", "realized_xirr_pct",
]


def append_entry(contribution_idr: float | None = None) -> None:
    """Append today's weights to the forward log."""
    report = build_report(contribution_idr=contribution_idr)
    report_date = report["report_date"]

    # Check if we already have an entry for this date
    if LOG_PATH.exists():
        existing = pd.read_csv(LOG_PATH)
        if report_date in existing["date"].astype(str).values:
            print(f"Entry for {report_date} already exists — skipping.")
            return

    new_rows = []
    for strat_name, weights in report["weights"].items():
        row = {
            "date": report_date,
            "strategy": strat_name,
            "idx_equities": weights.get("idx_equities", 0),
            "us_equities":  weights.get("us_equities", 0),
            "gold":         weights.get("gold", 0),
            "crypto":       weights.get("crypto", 0),
            "above_ma_sleeves": "|".join(report["above_ma_sleeves"]),
            "realized_xirr_pct": "",   # filled in manually later
        }
        new_rows.append(row)

    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"Appended {len(new_rows)} rows to {LOG_PATH}")
    for r in new_rows:
        print(f"  {r['date']}  {r['strategy']:<20}  "
              f"idx={float(r['idx_equities']):.1%}  "
              f"us={float(r['us_equities']):.1%}  "
              f"gold={float(r['gold']):.1%}  "
              f"crypto={float(r['crypto']):.1%}")


def show_log() -> None:
    if not LOG_PATH.exists():
        print("No forward log yet. Run: python -m allocation_lab.reporting.forward_log append")
        return
    df = pd.read_csv(LOG_PATH)
    print(df.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["append", "show"])
    parser.add_argument("--contribution", type=float, default=None)
    args = parser.parse_args()

    if args.action == "append":
        append_entry(contribution_idr=args.contribution)
    else:
        show_log()
