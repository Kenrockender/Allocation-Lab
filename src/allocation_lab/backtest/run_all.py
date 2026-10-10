"""
run_all.py — Run every strategy and print a comparison table.

Usage:
    python -m allocation_lab.backtest.run_all
    python -m allocation_lab.backtest.run_all --contribution 5000000
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger(__name__)


def main(contribution_idr: float = 5_000_000, start: str = "2018-01-01") -> pd.DataFrame:
    from allocation_lab.backtest.engine import run_backtest
    from allocation_lab.backtest.performance import compare_results
    from allocation_lab.backtest.lookahead import assert_no_lookahead
    from allocation_lab.strategies.baselines import (
        EqualWeightDCA, SingleSleeveDCA, FixedWeightDCA,
    )
    from allocation_lab.strategies.v1 import (
        InverseVolatilityDCA, MomentumDCA, TrendFilterDCA,
    )

    sleeve_df = pd.read_parquet(ROOT / "data/processed/sleeve_indices.parquet")

    strategies = [
        EqualWeightDCA(),
        SingleSleeveDCA("idx_equities"),
        SingleSleeveDCA("us_equities"),
        FixedWeightDCA(
            {"idx_equities": 0.30, "us_equities": 0.30, "gold": 0.20, "crypto": 0.20},
            name="fixed_30_30_20_20",
        ),
        InverseVolatilityDCA(),
        MomentumDCA(),
        TrendFilterDCA(),
    ]

    # Lookahead test — fail fast if any strategy has a bug
    logger.info("Running lookahead tests ...")
    for strat in strategies:
        assert_no_lookahead(strat, sleeve_df)
        logger.info("  %s: OK", strat.name)

    # Run backtests
    logger.info("Running backtests (contribution=IDR %.0f/month) ...", contribution_idr)
    results = []
    for strat in strategies:
        logger.info("  %s ...", strat.name)
        r = run_backtest(strat, sleeve_df, contribution_idr=contribution_idr, start=start)
        results.append(r)

    # Comparison table
    table = compare_results(results)
    print("\n=== BACKTEST COMPARISON TABLE ===")
    print(table.round(2).to_string())
    print()

    # Save NAV curves
    nav_df = pd.concat([r.nav_series for r in results], axis=1)
    nav_path = ROOT / "data/processed/nav_curves.parquet"
    nav_df.to_parquet(nav_path)
    logger.info("NAV curves saved to %s", nav_path)

    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contribution", type=float, default=5_000_000,
                        help="Monthly contribution in IDR (default: 5,000,000)")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main(contribution_idr=args.contribution, start=args.start)
