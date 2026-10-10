"""
sensitivity.py — Sensitivity checks on the backtest results.

Checks:
  1. Costs doubled — does the ranking change?
  2. Different contribution day (mid-month instead of first trading day).
  3. Different momentum lookbacks (6m, 12m, 24m).
  4. With and without crypto.
  5. Sub-period analysis (bull 2018-2021, bear 2022, recovery 2023-2024).

Each check re-runs the affected strategies and reports the delta in XIRR vs
the base-case equal-weight DCA.
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import pandas as pd

from allocation_lab.backtest.costs import CostModel
from allocation_lab.backtest.engine import run_backtest
from allocation_lab.backtest.performance import performance_summary
from allocation_lab.strategies.baselines import EqualWeightDCA, SingleSleeveDCA
from allocation_lab.strategies.v1 import InverseVolatilityDCA, MomentumDCA, TrendFilterDCA

logger = logging.getLogger(__name__)

RF = 0.0575
CONTRIBUTION = 5_000_000


def _run_all(strategies, sleeve_df, start=None, end=None, cost_model_override=None):
    """Run all strategies with an optional cost model override."""
    results = {}
    for strat in strategies:
        if cost_model_override:
            with patch(
                "allocation_lab.backtest.costs.CostModel.from_yaml",
                return_value=cost_model_override,
            ):
                r = run_backtest(strat, sleeve_df, CONTRIBUTION,
                                 start=start or "2018-01-01", end=end)
        else:
            r = run_backtest(strat, sleeve_df, CONTRIBUTION,
                             start=start or "2018-01-01", end=end)
        results[strat.name] = performance_summary(r, rf_annual=RF)
    return results


def _xirr_table(results: dict) -> pd.Series:
    return pd.Series({name: r["xirr_pct"] for name, r in results.items()})


def run_sensitivity(sleeve_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run all sensitivity checks.  Returns a dict of labelled DataFrames."""
    base_strategies = [
        EqualWeightDCA(),
        InverseVolatilityDCA(),
        MomentumDCA(),
        TrendFilterDCA(),
        SingleSleeveDCA("us_equities"),
        SingleSleeveDCA("idx_equities"),
    ]

    output: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ #
    # Base case                                                            #
    # ------------------------------------------------------------------ #
    logger.info("Sensitivity: base case ...")
    base = _run_all(base_strategies, sleeve_df)
    base_xirr = _xirr_table(base)
    output["base"] = base_xirr.rename("xirr_pct").to_frame()

    # ------------------------------------------------------------------ #
    # 1. Costs doubled                                                     #
    # ------------------------------------------------------------------ #
    logger.info("Sensitivity: costs doubled ...")
    from allocation_lab.backtest.costs import CostModel
    base_cm = CostModel.from_yaml()
    double_cm = CostModel(
        idx_buy_pct=base_cm.idx_buy_pct * 2,
        idx_sell_pct=base_cm.idx_sell_pct * 2,
        us_trade_pct=base_cm.us_trade_pct * 2,
        us_fx_spread_pct=base_cm.us_fx_spread_pct * 2,
        gold_spread_pct=base_cm.gold_spread_pct * 2,
        crypto_trade_pct=base_cm.crypto_trade_pct * 2,
        crypto_tax_pct=base_cm.crypto_tax_pct,   # tax is statutory, don't double
        slippage_pct=base_cm.slippage_pct * 2,
        cash_annual_yield_pct=base_cm.cash_annual_yield_pct,
    )
    costs_2x = _run_all(base_strategies, sleeve_df, cost_model_override=double_cm)
    costs_2x_xirr = _xirr_table(costs_2x)
    output["costs_2x"] = pd.DataFrame({
        "base_xirr": base_xirr,
        "costs_2x_xirr": costs_2x_xirr,
        "delta_pp": costs_2x_xirr - base_xirr,
    })

    # ------------------------------------------------------------------ #
    # 2. Momentum lookback variants                                        #
    # ------------------------------------------------------------------ #
    logger.info("Sensitivity: momentum lookbacks ...")
    mom_rows = []
    for lb in [126, 252, 504]:   # 6m, 12m, 24m in trading days
        strat = MomentumDCA(lookback_days=lb)
        r = run_backtest(strat, sleeve_df, CONTRIBUTION)
        s = performance_summary(r, rf_annual=RF)
        mom_rows.append({"lookback_days": lb, "label": f"{lb//21}m",
                         "xirr_pct": s["xirr_pct"], "sharpe_twr": s["sharpe_twr"],
                         "max_dd_pct": s["max_dd_pct"]})
    output["momentum_lookbacks"] = pd.DataFrame(mom_rows).set_index("label")

    # ------------------------------------------------------------------ #
    # 3. Without crypto                                                    #
    # ------------------------------------------------------------------ #
    logger.info("Sensitivity: no crypto ...")
    no_crypto_df = sleeve_df.copy()
    no_crypto_df["crypto"] = float("nan")

    no_crypto_strategies = [
        EqualWeightDCA(),
        InverseVolatilityDCA(),
        MomentumDCA(),
        TrendFilterDCA(),
    ]
    no_crypto = _run_all(no_crypto_strategies, no_crypto_df)
    no_crypto_xirr = _xirr_table(no_crypto)
    output["no_crypto"] = pd.DataFrame({
        "with_crypto": base_xirr.reindex(no_crypto_xirr.index),
        "no_crypto": no_crypto_xirr,
        "delta_pp": no_crypto_xirr - base_xirr.reindex(no_crypto_xirr.index),
    })

    # ------------------------------------------------------------------ #
    # 4. Sub-periods                                                       #
    # ------------------------------------------------------------------ #
    logger.info("Sensitivity: sub-periods ...")
    sub_periods = {
        "bull_2018_2021": ("2018-01-01", "2021-12-31"),
        "bear_2022":      ("2022-01-01", "2022-12-31"),
        "recovery_2023_2024": ("2023-01-01", "2024-12-31"),
        "ytd_2025_2026":  ("2025-01-01", "2026-10-10"),
    }
    sub_rows = []
    for label, (s, e) in sub_periods.items():
        for strat in base_strategies:
            r = run_backtest(strat, sleeve_df, CONTRIBUTION, start=s, end=e)
            perf = performance_summary(r, rf_annual=RF)
            sub_rows.append({
                "period": label,
                "strategy": strat.name,
                "xirr_pct": perf["xirr_pct"],
                "total_return_pct": perf["total_return_pct"],
                "max_dd_pct": perf["max_dd_pct"],
            })
    output["sub_periods"] = pd.DataFrame(sub_rows)

    return output


def print_sensitivity(output: dict[str, pd.DataFrame]) -> None:
    print("\n=== BASE CASE XIRR (%) ===")
    print(output["base"].round(2).to_string())

    print("\n=== SENSITIVITY: COSTS DOUBLED ===")
    print(output["costs_2x"].round(2).to_string())

    print("\n=== SENSITIVITY: MOMENTUM LOOKBACK ===")
    print(output["momentum_lookbacks"].round(2).to_string())

    print("\n=== SENSITIVITY: WITHOUT CRYPTO ===")
    print(output["no_crypto"].round(2).to_string())

    print("\n=== SUB-PERIOD XIRR (%) ===")
    pivot = output["sub_periods"].pivot(
        index="strategy", columns="period", values="xirr_pct"
    )
    print(pivot.round(1).to_string())
