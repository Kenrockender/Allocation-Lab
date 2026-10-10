# Allocation Lab

**How should a fixed monthly contribution be split across IDX stocks, the S&P 500, gold
and crypto — and when should cash sit in a money-market fund instead of being deployed?**
Over 8.75 years (2018–2026), contributing IDR 5 million/month to an equal-weight DCA across
the four asset classes turned IDR 530 million of contributions into approximately
**IDR 1.6 billion (XIRR ≈ 24%)** — vs IDR 720 million for an IDX-only strategy.
Bootstrap 90% CI on the equal-weight XIRR: **8.8% to 52%** — the wide interval reflects
genuine uncertainty, not sloppy methodology.

> **Disclaimer:** Research project. Nothing here is financial advice.
> Backtest results do not predict future returns.

---

## What this is

A multi-asset allocation research tool for an IDR-denominated retail investor with a fixed
monthly contribution. Four sleeves: IDX equities (equal-weight top-20 basket), US equities
(SPY), gold (GLD), crypto (BTC+ETH equal-weight). All returns in IDR.

Two outcomes:

1. A CV project with rigorous, honest results and a live demo (in progress).
2. A personal decision-support tool — not an oracle.

---

## Results (v0.1 full-period backtest)

Contribution: IDR 5,000,000/month · Period: Jan 2018 – Oct 2026 · Risk-free: BI Rate 5.75%

| Strategy | Final NAV | Total Return | XIRR | TWR Sharpe | Max DD |
|---|---:|---:|---:|---:|---:|
| equal_weight_dca | IDR 1.60B | +202% | **24.3%** | 1.02 | -50.9% |
| momentum_dca | IDR 1.52B | +186% | 23.1% | 0.97 | -55.5% |
| fixed_30_30_20_20 | IDR 1.48B | +180% | 22.7% | 1.06 | -45.7% |
| trend_filter_dca | IDR 1.36B | +157% | 20.8% | 1.09 | -40.6% |
| dca_us_equities | IDR 1.33B | +152% | 20.3% | 1.17 | -22.0% |
| inverse_vol_dca | IDR 1.30B | +146% | 19.8% | 1.17 | -26.2% |
| **dca_idx_equities** | IDR 0.72B | +36% | **6.9%** | 1.03 | -34.8% |

**Key caveat:** a significant share of multi-asset outperformance is driven by crypto's
2018–2021 and 2023–2024 bull runs. Without crypto, the XIRR advantage shrinks by 7–14 pp.
Sub-period analysis shows every strategy lost heavily in the 2022 bear market.

Walk-forward XIRR standard deviation across five 12-month test windows: **12–18 pp**.
The spread between strategies is smaller than the window-to-window noise — no strategy
reliably dominates another out-of-sample.

---

## Repository structure

```
allocation-lab/
  config/
    universe.yaml          # sleeves and tickers (IDX basket frozen 2026-10-10)
    costs.yaml             # broker fees, tax, spreads (sourced 2026-10-10)
    local.example.yaml     # copy to local.yaml (git-ignored) for real amounts
  src/allocation_lab/
    data/                  # fetch, align, FX conversion, quality checks
    analytics/             # returns, Sharpe, drawdown, VaR/CVaR, beta
    backtest/              # engine, cost model, portfolio state, XIRR/TWR
    strategies/            # baselines and v1 signal strategies
    research/              # walk-forward, block bootstrap, sensitivity
    reporting/             # (Phase 5)
  app/                     # (Phase 5 — Streamlit dashboard)
  tests/                   # 79 unit and integration tests
  docs/
    decisions.md           # design decisions the code depends on
    experiments.md         # every variant run (17 so far)
    findings.md            # observations and results
  data/                    # git-ignored
    raw/                   # one parquet per ticker
    processed/             # aligned price table, sleeve indices, NAV curves
    external/              # BI rate CSV (hand-maintained)
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -e ".[dev]"
pytest                         # 79 tests
```

## Build the data

```bash
python -m allocation_lab.data.pipeline
```

Downloads ~25 tickers from Yahoo Finance, aligns calendars, converts to IDR,
runs quality checks. Takes ~2 minutes on a first run.

## Run the backtest

```bash
python -m allocation_lab.backtest.run_all
```

Runs all 7 strategies, lookahead tests, and prints the comparison table.
Saves NAV curves to `data/processed/nav_curves.parquet`.

## Run validation

```bash
python -m allocation_lab.research.run_phase4 wf     # walk-forward
python -m allocation_lab.research.run_phase4 sens   # sensitivity checks
python -m allocation_lab.research.run_phase4 boot --sims 100  # bootstrap
```

---

## Limitations

- **Survivorship bias:** IDX basket is frozen on today's top-20 large caps. Backtesting on
  this list flatters past IDX returns. The date is recorded in `config/universe.yaml`.
- **Short crypto history:** only ~8 years of BTC/ETH data, dominated by two extreme bull
  runs. The backtest results are highly sensitive to whether crypto is included.
- **No lot-size rounding:** IDX stocks trade in lots of 100 shares. The engine uses
  fractional units — real execution would differ for small contributions.
- **No withholding tax on US dividends:** a small drag (~0.3% p.a.) not modelled in v1.
- **Backtest ≠ forecast:** the 2018–2026 period contains unusually large crypto returns that
  are unlikely to repeat at the same magnitude.
- **17 variants tested:** the more variants run, the more the best one looks like luck.
  Walk-forward and bootstrap intervals are wide — treat the headline XIRR as a rough
  order-of-magnitude, not a precise estimate.

---

## Progress

| Phase | Focus | Status |
|---|---|---|
| 0 | Setup and decisions | ✓ Done |
| 1 | Data layer | ✓ Done |
| 2 | Analytics | ✓ Done |
| 3 | Backtester | ✓ Done |
| 4 | Validation and findings | ✓ Done — **v0.1** |
| 5 | Dashboard and automation | Not started |
| 6 | Deep learning module (optional) | Not started |

See [plan.md](plan.md) for detailed task lists and [docs/findings.md](docs/findings.md)
for full results.
