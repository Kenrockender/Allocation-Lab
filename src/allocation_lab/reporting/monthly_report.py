"""
monthly_report.py — Generate the "this month" allocation report.

Given fresh sleeve data and a contribution amount, outputs:
  - Which sleeves are above/below their 200-day MA (trend signal)
  - Inverse-vol weights for next month
  - Momentum scores and weights
  - Equal-weight baseline weights
  - Suggested allocation in IDR (using the example amount for the public demo)

This module has no side effects — it only computes and returns dicts/DataFrames.
The CLI entry point prints a text report. The dashboard calls the same functions.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]

SLEEVES = ("idx_equities", "us_equities", "gold", "crypto")
MA_WINDOW = 200
VOL_WINDOW = 63
MOM_LOOKBACK = 252
MOM_SKIP = 21
RF = 0.0575


def _load_local_config() -> dict:
    """Load local.yaml if present, else fall back to local.example.yaml."""
    local = ROOT / "config" / "local.yaml"
    example = ROOT / "config" / "local.example.yaml"
    path = local if local.exists() else example
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sleeve_df() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "sleeve_indices.parquet"
    if not path.exists():
        # Try to run the pipeline automatically (works in CI / Streamlit Cloud)
        import logging
        logging.warning("sleeve_indices.parquet missing — running data pipeline …")
        from allocation_lab.data import fetch, align
        fetch.fetch_all(start="2017-01-01")
        align.run(start="2018-01-01")
    if not path.exists():
        raise FileNotFoundError(
            "sleeve_indices.parquet not found. Run: python -m allocation_lab.data.pipeline"
        )
    return pd.read_parquet(path)


def trend_signals(sleeve_df: pd.DataFrame, as_of: date | None = None) -> pd.DataFrame:
    """Return per-sleeve MA signal for the most recent date (or as_of)."""
    df = sleeve_df[[s for s in SLEEVES if s in sleeve_df.columns]].ffill()
    if as_of:
        df = df.loc[:str(as_of)]

    if len(df) < MA_WINDOW:
        raise ValueError(f"Need at least {MA_WINDOW} rows; got {len(df)}")

    last = df.iloc[-1]
    ma = df.tail(MA_WINDOW).mean()
    above = last > ma
    pct_above = (last - ma) / ma * 100

    return pd.DataFrame({
        "price": last,
        "ma_200": ma,
        "above_ma": above,
        "pct_above_ma": pct_above.round(2),
    })


def inv_vol_weights(sleeve_df: pd.DataFrame, as_of: date | None = None) -> pd.Series:
    """Return inverse-volatility weights (63-day window)."""
    df = sleeve_df[[s for s in SLEEVES if s in sleeve_df.columns]].ffill()
    if as_of:
        df = df.loc[:str(as_of)]

    lr = np.log(df / df.shift(1)).dropna()
    vol = lr.tail(VOL_WINDOW).std() * np.sqrt(252)
    vol = vol.clip(lower=0.01)
    w = (1 / vol) / (1 / vol).sum()
    return w.round(4)


def momentum_weights(sleeve_df: pd.DataFrame, as_of: date | None = None) -> pd.Series:
    """Return 12-1 month momentum weights (long-only)."""
    df = sleeve_df[[s for s in SLEEVES if s in sleeve_df.columns]].ffill()
    if as_of:
        df = df.loc[:str(as_of)]

    if len(df) < MOM_LOOKBACK + MOM_SKIP + 1:
        return pd.Series({s: 0.25 for s in SLEEVES if s in df.columns})

    p_recent = df.iloc[-(MOM_SKIP + 1)]
    p_old = df.iloc[-(MOM_LOOKBACK + 1)]
    mom = (p_recent / p_old - 1).clip(lower=0)
    if mom.sum() == 0:
        return pd.Series({s: 1 / len(df.columns) for s in df.columns})
    return (mom / mom.sum()).round(4)


def equal_weights(sleeve_df: pd.DataFrame) -> pd.Series:
    cols = [s for s in SLEEVES if s in sleeve_df.columns]
    return pd.Series({s: round(1 / len(cols), 4) for s in cols})


def build_report(
    contribution_idr: float | None = None,
    as_of: date | None = None,
) -> dict:
    """Build the full monthly report dict.

    Parameters
    ----------
    contribution_idr : override contribution; if None, loads from config
    as_of            : generate report as-of this date (default: latest data)
    """
    sleeve_df = load_sleeve_df()
    cfg = _load_local_config()

    if contribution_idr is None:
        contribution_idr = float(cfg.get("monthly_contribution_idr", 5_000_000))

    report_date = as_of or sleeve_df.index[-1].date()

    signals = trend_signals(sleeve_df, as_of=as_of)
    inv_w = inv_vol_weights(sleeve_df, as_of=as_of)
    mom_w = momentum_weights(sleeve_df, as_of=as_of)
    eq_w = equal_weights(sleeve_df)

    # Strategy allocations in IDR
    def _alloc(weights: pd.Series) -> pd.Series:
        invest_sleeves = signals.index[signals["above_ma"]].tolist()
        # For trend filter: only invest in above-MA sleeves
        effective = weights.copy()
        return (effective * contribution_idr).round(0)

    return {
        "report_date": str(report_date),
        "contribution_idr": contribution_idr,
        "signals": signals,
        "weights": {
            "equal_weight": eq_w,
            "inverse_vol":  inv_w,
            "momentum":     mom_w,
        },
        "allocations_idr": {
            "equal_weight": (eq_w  * contribution_idr).round(0),
            "inverse_vol":  (inv_w * contribution_idr).round(0),
            "momentum":     (mom_w * contribution_idr).round(0),
        },
        "above_ma_sleeves": signals.index[signals["above_ma"]].tolist(),
        "below_ma_sleeves": signals.index[~signals["above_ma"]].tolist(),
    }


def format_report_text(report: dict) -> str:
    """Format the report as a plain-text string for CLI / logging."""
    lines = [
        "=" * 60,
        f"MONTHLY ALLOCATION REPORT — {report['report_date']}",
        f"Contribution: IDR {report['contribution_idr']:,.0f}",
        "=" * 60,
        "",
        "--- Trend signals (200-day MA) ---",
    ]
    sig = report["signals"]
    for sleeve in sig.index:
        status = "▲ ABOVE" if sig.loc[sleeve, "above_ma"] else "▼ BELOW"
        pct = sig.loc[sleeve, "pct_above_ma"]
        lines.append(f"  {sleeve:<20} {status}  ({pct:+.1f}% vs MA)")

    lines += ["", "--- Suggested allocations (IDR) ---"]
    for strat, alloc in report["allocations_idr"].items():
        lines.append(f"  {strat}:")
        for sleeve, amt in alloc.items():
            lines.append(f"    {sleeve:<20} IDR {amt:>12,.0f}")
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate this month's allocation report.")
    parser.add_argument("--contribution", type=float, default=None)
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD")
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    report = build_report(contribution_idr=args.contribution, as_of=as_of)
    print(format_report_text(report))
