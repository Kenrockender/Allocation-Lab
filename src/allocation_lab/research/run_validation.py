"""
run_validation.py — Phase 4 validation runner.

Runs walk-forward, bootstrap, and sensitivity checks, prints results,
and saves them to data/processed/validation_*.parquet.

Usage:
    python -m allocation_lab.research.run_validation
    python -m allocation_lab.research.run_validation --bootstrap-sims 200
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"
logger = logging.getLogger(__name__)


def main(bootstrap_sims: int = 500) -> None:
    from allocation_lab.strategies.baselines import EqualWeightDCA, SingleSleeveDCA, FixedWeightDCA
    from allocation_lab.strategies.v1 import InverseVolatilityDCA, MomentumDCA, TrendFilterDCA
    from allocation_lab.research.walkforward import run_walk_forward, summarise_walk_forward
    from allocation_lab.research.bootstrap import run_bootstrap, bootstrap_summary
    from allocation_lab.research.sensitivity import run_sensitivity, print_sensitivity

    sleeve_df = pd.read_parquet(PROCESSED / "sleeve_indices.parquet")

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

    # ------------------------------------------------------------------ #
    # 1. Walk-forward                                                      #
    # ------------------------------------------------------------------ #
    logger.info("=== Walk-forward evaluation ===")
    wf_df = run_walk_forward(strategies, sleeve_df)
    wf_summary = summarise_walk_forward(wf_df)
    print("\n=== WALK-FORWARD SUMMARY (mean ± std of XIRR across windows) ===")
    print(wf_summary.round(2).to_string())
    wf_df.to_parquet(PROCESSED / "validation_walkforward.parquet", index=False)

    # ------------------------------------------------------------------ #
    # 2. Bootstrap                                                         #
    # ------------------------------------------------------------------ #
    logger.info("=== Block bootstrap (n=%d) ===", bootstrap_sims)
    boot_df = run_bootstrap(strategies, sleeve_df, n_sims=bootstrap_sims)
    boot_summary = bootstrap_summary(boot_df)
    print("\n=== BOOTSTRAP CONFIDENCE INTERVALS (XIRR %) ===")
    print(boot_summary.round(2).to_string())
    boot_df.to_parquet(PROCESSED / "validation_bootstrap.parquet", index=False)

    # ------------------------------------------------------------------ #
    # 3. Sensitivity                                                       #
    # ------------------------------------------------------------------ #
    logger.info("=== Sensitivity checks ===")
    sens = run_sensitivity(sleeve_df)
    print_sensitivity(sens)
    sens["sub_periods"].to_parquet(PROCESSED / "validation_subperiods.parquet", index=False)

    logger.info("All validation results saved to %s", PROCESSED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-sims", type=int, default=500)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main(bootstrap_sims=args.bootstrap_sims)
