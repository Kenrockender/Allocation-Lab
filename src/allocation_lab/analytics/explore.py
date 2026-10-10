"""
explore.py — Compute and print exploration results for findings.md.
Run: python -m allocation_lab.analytics.explore
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from allocation_lab.analytics.metrics import (
    summary_table, log_returns, drawdown_series,
    annualised_return, max_drawdown
)

RF = 0.0575  # BI Rate

def run():
    sleeves = pd.read_parquet(ROOT / "data/processed/sleeve_indices.parquet")
    cols = ["idx_equities", "us_equities", "gold", "crypto"]

    # --- Risk/return table ---
    tbl = summary_table(sleeves[cols], rf_annual=RF)
    tbl["cagr_pct"]   = tbl["cagr"]   * 100
    tbl["vol_pct"]    = tbl["vol"]     * 100
    tbl["max_dd_pct"] = tbl["max_dd"]  * 100
    tbl["var95_pct"]  = tbl["var_95"]  * 100
    tbl["cvar95_pct"] = tbl["cvar_95"] * 100
    print("=== RISK/RETURN TABLE (rf=5.75%) ===")
    print(tbl[["cagr_pct","vol_pct","sharpe","max_dd_pct","max_dd_days","var95_pct","cvar95_pct"]].round(2).to_string())

    # --- Correlation matrices ---
    lr = sleeves[cols].apply(log_returns)
    print("\n=== PAIRWISE CORRELATION (full period) ===")
    print(lr.corr().round(3).to_string())
    print("\n=== PAIRWISE CORRELATION (COVID crash Feb-Apr 2020) ===")
    print(lr.loc["2020-02-01":"2020-04-30"].corr().round(3).to_string())

    # --- Rolling vol summary ---
    print("\n=== ROLLING 63-DAY ANNUALISED VOL (current vs peak) ===")
    for col in cols:
        s = sleeves[col].dropna()
        rv = log_returns(s).rolling(63).std() * np.sqrt(252)
        peak_date = rv.idxmax()
        print(f"  {col:<20} current={rv.iloc[-1]*100:.1f}%  peak={rv.max()*100:.1f}% ({peak_date.date()})")

    # --- Sub-period returns ---
    periods = {
        "Bull 2018-2021":        ("2018-01-01", "2021-12-31"),
        "Bear 2022":             ("2022-01-01", "2022-12-31"),
        "Recovery 2023-2024":    ("2023-01-01", "2024-12-31"),
        "YTD 2025-2026":         ("2025-01-01", "2026-10-10"),
    }
    print("\n=== SUB-PERIOD TOTAL RETURNS (%) ===")
    col_w = 16
    header = f"{'Period':<25}" + "".join(f"{c:>{col_w}}" for c in cols)
    print(header)
    for label, (s, e) in periods.items():
        row = f"{label:<25}"
        for col in cols:
            series = sleeves[col].dropna().loc[s:e]
            if len(series) < 2:
                row += f"{'N/A':>{col_w}}"
            else:
                ret = (series.iloc[-1] / series.iloc[0] - 1) * 100
                row += f"{ret:>{col_w-1}.1f}%"
        print(row)

    # --- Drawdown during key events ---
    events = {
        "COVID crash (Feb 20 - Mar 23 2020)": ("2020-02-20", "2020-03-23"),
        "Crypto bear (Nov 2021 - Dec 2022)":  ("2021-11-08", "2022-12-31"),
        "2018 selloff (Jan - Dec 2018)":       ("2018-01-01", "2018-12-31"),
    }
    print("\n=== DRAWDOWN DURING KEY EVENTS (%) ===")
    header2 = f"{'Event':<40}" + "".join(f"{c:>{col_w}}" for c in cols)
    print(header2)
    for label, (s, e) in events.items():
        row = f"{label:<40}"
        for col in cols:
            series = sleeves[col].dropna().loc[s:e]
            if len(series) < 2:
                row += f"{'N/A':>{col_w}}"
            else:
                ret = (series.iloc[-1] / series.iloc[0] - 1) * 100
                row += f"{ret:>{col_w-1}.1f}%"
        print(row)

if __name__ == "__main__":
    run()
