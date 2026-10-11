# Allocation Lab: Project Plan

Working name, rename freely.

## Goal

Build a multi-asset allocation research tool (IDX equities, US equities, gold, crypto) that answers one practical question: **how should a fixed monthly contribution be deployed, and when should cash waiting in a money-market fund move into equities?**

Two outcomes:

1. A CV project with rigorous, honest results and a live demo.
2. A decision-support tool for personal investing (not an oracle, not financial advice).

## How to use this plan

- Work phase by phase. Every phase ends with something that runs and can be committed, so the repo is in good shape if you stop after any phase.
- Tasks have a size tag: **S** = up to 1h, **M** = 1-3h, **L** = 3h+ (needs a proper block of time).
  - Busy week: pick one S task from the current phase.
  - Free week: finish an M or L task, or a whole phase.
- Hour estimates are rough. Phases 0-5 are about 25-50h in total. Phase 6 is optional.
- Don't start a phase until the previous phase's "Done when" is true.
- The project is CV-ready after Phase 4 and complete after Phase 5.

## Scope

In scope: daily data for four sleeves (an IDX stock basket, an S&P 500 ETF, gold and crypto) converted to IDR, each sleeve being one return series so no asset class is swamped by another; monthly contributions; long-only strategies; walk-forward validation; a dashboard and a monthly report.

Out of scope (keeps the project finishable): live trading or broker integration, intraday data, leverage and shorting, options, paid data sources, tuning parameters on the full sample.

## Progress

| Phase | Focus | Rough effort | Status |
|---|---|---|---|
| 0 | Setup and decisions | 1-2h | Done ✓ |
| 1 | Data layer | 4-8h | Done ✓ |
| 2 | Analytics | 3-5h | Done ✓ |
| 3 | Backtester | 6-12h | Done ✓ |
| 4 | Validation and findings (CV-ready) | 5-10h | Done ✓ |
| 5 | Dashboard and automation | 6-12h | Done ✓ |
| 6 | Deep learning module (optional) | 8-15h | Not started |

---

## Phase 0: Setup and decisions (1-2h)

Goal: the repo exists and the key decisions are written down before any code depends on them.

- [x] **S** Create the repo and Python environment (pyproject, pytest, a linter). Add a `.gitignore` that excludes `data/` and `config/local.yaml`.
- [x] **S** Create the folder structure below and a README stub.
- [x] **M** Confirm `docs/decisions.md` (drafted with proposed defaults, change what does not fit): base currency (IDR), the four sleeves, the IDX ticker list (plus the date it was frozen), benchmarks, risk-free source (BI rate), contribution frequency (monthly), backtest start date, and the signal definition for the "hold cash until signal" strategy (for example price vs its 200-day average).
- [x] **M** Fill `config/costs.yaml` with current fees and tax for each sleeve, the IDR/USD conversion spread, the gold spread and an assumed cash yield. Look these up from the broker or the exchange. They change, so keep them in config, not in code. The template exists with every value empty.
- [x] **S** Put example numbers in `config/local.example.yaml`. Real contribution sizes and holdings live only in the git-ignored `config/local.yaml`.

**Done when:** `pytest` runs (even with one dummy test) and `docs/decisions.md` has no blanks.

```
allocation-lab/
  config/
    universe.yaml          # sleeves and tickers, base currency, benchmark
    costs.yaml             # fees, tax, FX and gold spreads, cash yield
    local.example.yaml     # example contribution size; copy to local.yaml (git-ignored)
  src/allocation_lab/
    data/                  # fetch, clean, align, FX
    analytics/             # returns and risk metrics
    backtest/              # engine, costs, portfolio state, performance metrics
    strategies/            # one file per strategy
    research/              # walk-forward, bootstrap, sensitivity
    reporting/             # monthly report
  app/                     # dashboard
  tests/
  notebooks/
  docs/
    decisions.md
    experiments.md
    findings.md
  data/                    # git-ignored
```

---

## Phase 1: Data layer (4-8h)

Goal: one clean, IDR-denominated daily price table for the whole universe, rebuilt by a single command.

- [x] **M** Fetch IDX stocks (`.JK` tickers), the S&P 500 ETF, gold, BTC, ETH and USD/IDR with yfinance. Save raw files to `data/raw/` as parquet.
- [x] **M** Use adjusted prices so dividends and splits are in the returns. Compare a few tickers against a second source to catch bad adjustments.
- [x] **M** Align calendars: IDX holidays, US holidays and 24/7 crypto. Rule: evaluate on the contribution date using the last available price.
- [x] **S** Convert USD-priced assets to IDR with the USD/IDR series.
- [x] **M** Build the sleeve series: monthly-rebalanced equal-weight baskets for IDX equities and crypto, a single series each for US equities and gold. Every sleeve ends up as one IDR total-return index.
- [x] **S** Store each sleeve's first valid date. Histories start at different dates (crypto is short), so strategies must only hold sleeves that exist at that date.
- [x] **M** Data-quality checks: gaps, stale prices, zero volume, daily moves beyond a threshold. Output a quality report.
- [x] **M** Tests for alignment and FX conversion using small hand-made data frames.
- [x] **S** One command (CLI or `make data`) rebuilds everything.

**Done when:** one command produces the IDR price table, the four sleeve series and a quality report, and the tests pass.

Note: do not commit raw downloaded price data to a public repo. Commit the fetch script and publish derived results.

---

## Phase 2: Analytics (3-5h)

Goal: trusted metric functions and a first look at the data. Sleeve-level only, no contributions yet.

- [x] **M** Implement simple and log returns, annualized volatility, Sharpe (excess over the BI rate), max drawdown and drawdown duration, rolling correlation, beta vs the benchmark, and historical and parametric VaR/CVaR.
- [x] **M** Unit tests: check each metric against a small hand-calculated example.
- [x] **M** Exploration notebook: risk/return table per sleeve, correlation heatmap, drawdown chart, rolling volatility. Write 5-10 observations in `docs/findings.md`.

**Done when:** a report command produces the tables and charts, the tests pass, and `docs/findings.md` has the first observations.

---

## Phase 3: Backtester (6-12h)

Goal: an engine that simulates monthly contributions with realistic costs and no lookahead.

- [x] **L** Portfolio state and simulation loop: cash, holdings, monthly contribution, execution at the next available price after the signal, fractional units for every sleeve. Only allow sleeves that exist at that date.
- [x] **M** Cost model driven by `config/costs.yaml`: fees, tax, FX and gold spreads, optional slippage. Cash earns the assumed yield.
- [x] **S** Strategy interface: a function that takes history up to date t and returns target weights, plus a cash policy (invest now or wait).
- [x] **M** Baselines: fixed-weight DCA, equal-weight DCA, and a benchmark with the same contributions.
- [x] **M** Strategies v1: inverse-volatility (risk parity) weights, momentum tilt, hold-in-money-market-until-signal.
- [x] **M** Performance metrics for contribution flows: final wealth vs total contributed, XIRR, turnover, cost drag. Compute Sharpe and drawdown on a time-weighted return series with deposits removed. Plain CAGR on a series with deposits is wrong.
- [x] **M** Lookahead test: a strategy given data cut off at date t must make the same decision as when it is given the full history.

**Done when:** one command runs all strategies over the full history and outputs a comparison table and equity curves, and the lookahead test passes.

---

## Phase 4: Validation and findings (5-10h)

Goal: turn backtest numbers into results that can be defended.

- [x] **M** Walk-forward evaluation: choose parameters (momentum lookback, volatility window) on a training window, evaluate on the next window only, then roll forward.
- [x] **M** Block bootstrap of monthly returns to get confidence intervals on the difference vs the baseline (XIRR, final wealth, max drawdown).
- [x] **M** Sensitivity checks: costs doubled, different contribution day, different lookbacks, with and without crypto.
- [x] **S** Keep a log of every variant tried in `docs/experiments.md`. The more variants tried, the more likely the best one is luck, so report how many were tested.
- [x] **M** Split results by sub-period (for example crypto bull vs bear years) and note where each strategy fails.
- [x] **L** Write `docs/findings.md` and README v1: the question, data, method, results table with uncertainty, and limitations (survivorship bias in the frozen IDX list, no lot-size limits, short crypto history, backtest is not a forecast).

**Done when:** the README has a headline finding with its uncertainty and a limitations section. Tag `v0.1`.

**Checkpoint A: the project is CV-ready here.**

---

## Phase 5: Dashboard and automation (6-12h)

Goal: a live demo plus a monthly routine that runs by itself.

- [x] **L** Streamlit dashboard: asset stats, strategy comparison, equity curves, drawdowns, parameters.
- [x] **M** "This month" report: given the contribution amount (from `config/local.yaml`) and fresh data, output the suggested allocation under the chosen strategy, with the weights and signals behind it. The public demo uses example numbers.
- [x] **M** Scheduled data refresh and report generation with a GitHub Actions cron. Scheduled workflows on inactive public repos can get auto-disabled, so check for that. AWS EventBridge + Lambda is the alternative if you want to practice cloud.
- [x] **M** Forward-test log: each month append the strategy's suggested weights and signals to `forward_log.csv`, then fill in the realized outcome later. Log weights, not rupiah amounts.
- [x] **S** Deploy the demo (Streamlit Community Cloud or another free host) and link it from the README and the portfolio site.

**Done when:** the live URL works, the monthly job runs without manual steps, and the forward log has its first entry. Tag `v0.2`.

**Checkpoint B: live demo and forward-test log.**

---

## Phase 6: Deep learning module (optional, 8-15h)

Goal: test whether a learned volatility forecast or regime detector improves risk control.

- [x] **M** Baseline first: EWMA volatility forecast. Define the metric (forecast error vs realized volatility, or drawdown reduction when exposure is scaled by the forecast).
- [x] **L** Model: a small LSTM/GRU or Transformer for next-period volatility, or a Hidden Markov Model for regimes. Train with walk-forward splits only. Keep it small, the data is limited.
- [x] **M** Plug it into the backtester as a risk-scaling overlay and compare against the EWMA overlay.
- [x] **S** Document the result honestly, including if it does not beat the baseline.

**Done when:** a comparison table is in `docs/findings.md`. Tag `v0.3`.

---

## Ongoing (after Phase 5)

- Monthly (15-30 min): check the job ran, read the report, update the forward log.
- Every few months: rerun the data checks and validation, and update `docs/findings.md`.

## Pitfalls checklist

- **Lookahead bias:** signals use data up to t, trades happen after t. Keep the leakage test passing.
- **Survivorship bias:** the IDX basket is frozen today, which flatters past results. Freeze it early, document the date, and say so in the README. The US sleeve is an index ETF, which avoids the problem.
- **Overfitting:** tune only on training windows and log every experiment.
- **Bad data:** verify adjustments and gaps before trusting any result.
- **Costs and rules:** exchange and broker rules change. Keep them in config and re-check them.
- **Currency:** evaluate everything in IDR.
- **Contributions:** use money-weighted metrics for deposits.
- **Privacy:** no real holdings or contribution sizes in the public repo.
- **Scope creep:** anything outside the Scope section waits until Phase 5 is done.

## CV packaging

- After Checkpoint A: add the repo to the CV with one bullet using this formula: what was built, the method, one quantified finding, what was deployed.
- After Checkpoint B: add the live demo link and mention the forward-test log, which few student projects have.
- Keep the top of the README to three lines: the question, the headline result, and the demo link.
