"""
run_phase4.py — Run each Phase 4 check independently so they can be timed.

Usage:
    python -m allocation_lab.research.run_phase4 wf       # walk-forward only
    python -m allocation_lab.research.run_phase4 boot     # bootstrap only
    python -m allocation_lab.research.run_phase4 sens     # sensitivity only
    python -m allocation_lab.research.run_phase4 all      # everything
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"


def _strategies():
    from allocation_lab.strategies.baselines import EqualWeightDCA, SingleSleeveDCA, FixedWeightDCA
    from allocation_lab.strategies.v1 import InverseVolatilityDCA, MomentumDCA, TrendFilterDCA
    return [
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


def run_wf(sleeve_df):
    from allocation_lab.research.walkforward import run_walk_forward, summarise_walk_forward
    logging.info("Walk-forward ...")
    wf_df = run_walk_forward(_strategies(), sleeve_df)
    wf_summary = summarise_walk_forward(wf_df)
    print("\n=== WALK-FORWARD SUMMARY (mean ± std across windows) ===")
    print(wf_summary.round(2).to_string())
    wf_df.to_parquet(PROCESSED / "validation_walkforward.parquet", index=False)
    return wf_df, wf_summary


def run_boot(sleeve_df, n_sims=100):
    from allocation_lab.research.bootstrap import run_bootstrap, bootstrap_summary
    logging.info("Bootstrap n=%d ...", n_sims)
    boot_df = run_bootstrap(_strategies(), sleeve_df, n_sims=n_sims)
    summary = bootstrap_summary(boot_df)
    print("\n=== BOOTSTRAP CI (XIRR %) ===")
    print(summary.round(2).to_string())
    boot_df.to_parquet(PROCESSED / "validation_bootstrap.parquet", index=False)
    return boot_df, summary


def run_sens(sleeve_df):
    from allocation_lab.research.sensitivity import run_sensitivity, print_sensitivity
    logging.info("Sensitivity ...")
    sens = run_sensitivity(sleeve_df)
    print_sensitivity(sens)
    sens["sub_periods"].to_parquet(PROCESSED / "validation_subperiods.parquet", index=False)
    return sens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["wf", "boot", "sens", "all"])
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    sleeve_df = pd.read_parquet(PROCESSED / "sleeve_indices.parquet")

    if args.task in ("wf", "all"):
        run_wf(sleeve_df)
    if args.task in ("boot", "all"):
        run_boot(sleeve_df, n_sims=args.sims)
    if args.task in ("sens", "all"):
        run_sens(sleeve_df)
