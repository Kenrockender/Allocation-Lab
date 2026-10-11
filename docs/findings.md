# Findings

Observations and results, newest first. Each entry says what was tested, on what data, and
how sure we are about it.

---

## Phase 4 validation — out-of-sample and robustness (2018-01-01 to 2026-10-10)

**Data:** Four IDR-denominated sleeve indices (Phase 1 pipeline).
**Contribution:** IDR 5,000,000/month (example; real amount in local.yaml).
**Risk-free rate:** BI Rate 5.75%.
**Variants tested this phase:** 7 strategies × 3 analyses + 4 sensitivity checks = 37 variants.
The more variants tested, the more likely the best-looking result is luck. Interpret all
numbers with that in mind.

### Walk-forward results (36-month train / 12-month test, 5 windows)

| Strategy | XIRR mean | XIRR std | Sharpe mean | Max DD mean |
|---|---:|---:|---:|---:|
| dca_us_equities | 19.7% | 12.8% | 2.20 | -8.0% |
| equal_weight_dca | 19.4% | 18.1% | 2.21 | -10.5% |
| momentum_dca | 19.2% | 18.3% | 2.21 | -10.5% |
| inverse_vol_dca | 18.6% | 14.5% | 2.23 | -9.2% |
| trend_filter_dca | 18.1% | 17.5% | 2.21 | -10.4% |
| fixed_30_30_20_20 | 17.9% | 16.1% | 2.21 | -9.8% |
| dca_idx_equities | 4.9% | 17.8% | 2.12 | -11.2% |

Key observation: no strategy dominates on risk-adjusted basis. XIRR standard deviation
across windows is very wide (12–18%), meaning a single window's result is largely noise.

### Bootstrap confidence intervals (100 simulations, block=3 months)

| Strategy | XIRR p5 | XIRR p50 | XIRR p95 | P(beat equal-weight) |
|---|---:|---:|---:|---:|
| trend_filter_dca | 9.0% | 20.8% | 49.6% | 38% |
| equal_weight_dca | 8.8% | 20.0% | 52.0% | — |
| fixed_30_30_20_20 | 8.1% | 19.4% | 48.3% | 25% |
| inverse_vol_dca | 10.1% | 19.3% | 38.2% | 39% |
| momentum_dca | 7.0% | 19.2% | 58.4% | 47% |
| dca_us_equities | 8.3% | 17.2% | 27.2% | 39% |
| dca_idx_equities | -4.3% | 9.2% | 24.0% | 15% |

The 90% confidence intervals are very wide (p5 to p95 spans 30–50 percentage points).
No strategy's median materially beats the equal-weight baseline with high confidence.
Inverse-vol has the tightest interval (lower tail risk) but also the lowest upside.

### Sensitivity checks

**1. Costs doubled — virtually no impact**

Doubling all fees moves XIRR by at most -0.21 pp. This is because:
- Each strategy only buys (no selling), so sell-side fees are never triggered.
- Monthly contributions are large relative to per-trade cost.
- Cost drag is already small (~0.4% of contributions in the base case).

Implication: conclusions are robust to broker fee changes.

**2. Momentum lookback (6m / 12m / 24m) — stable**

| Lookback | XIRR | Sharpe | Max DD |
|---|---:|---:|---:|
| 6 months | 23.7% | 0.99 | -54.3% |
| 12 months | 23.1% | 0.97 | -55.5% |
| 24 months | 23.4% | 0.97 | -56.3% |

Results are essentially identical. The strategy is not sensitive to the lookback parameter
in the range tested.

**3. Without crypto — large drop, especially for inverse-vol**

| Strategy | With crypto XIRR | No crypto XIRR | Delta |
|---|---:|---:|---:|
| equal_weight_dca | 24.3% | 16.9% | -7.5 pp |
| inverse_vol_dca | 19.8% | 5.4% | -14.5 pp |
| momentum_dca | 23.1% | 16.5% | -6.6 pp |
| trend_filter_dca | 20.8% | 16.8% | -4.0 pp |

A significant share of the full-period outperformance is driven by crypto's 2018–2021 bull run.
This is a key risk: crypto has the shortest history, so its backtest contribution is least
reliable. The trend filter loses the least without crypto because it was often in cash during
crypto bear periods.

**4. Sub-period breakdown**

| Strategy | Bull 2018–21 | Bear 2022 | Recovery 2023–24 | YTD 2025–26 |
|---|---:|---:|---:|---:|
| dca_idx_equities | 20.9% | -0.3% | -4.9% | -3.8% |
| dca_us_equities | 22.5% | -3.0% | 27.7% | 28.6% |
| equal_weight_dca | 57.5% | -12.8% | 36.0% | 12.5% |
| inverse_vol_dca | 33.3% | -6.1% | 27.3% | 14.4% |
| momentum_dca | 55.4% | -13.0% | 42.4% | 9.6% |
| trend_filter_dca | 41.8% | -12.6% | 36.7% | 10.3% |

Every multi-asset strategy lost heavily in 2022 (the crypto + US equity bear).
None of the three v1 strategies meaningfully reduced the 2022 drawdown vs equal-weight.
The trend filter was the best at damage limitation in the 2022 bear (-12.6% vs -12.8% equal weight)
but the improvement is negligible, likely because the 200-day MA signal triggers late.
IDX-only DCA was the safest in 2022 (-0.3%) but has the worst recovery (losing in 2023–24 and 2025–26).

### Headline finding

Across the full 8.75-year period (2018–2026), contributing IDR 5 million per month to
an equal-weight DCA across the four sleeves turned IDR 530 million of contributions into
approximately IDR 1.6 billion (XIRR ≈ 24%), compared to IDR 720 million (+XIRR ≈ 7%)
for an IDX-only strategy and IDR 1.3–1.5 billion for intermediate approaches.

**Uncertainty:** bootstrap 90% CI on equal-weight XIRR runs from 8.8% to 52%. The full-period
backtest result is a single path through a noisy outcome space. It should not be treated as a
forecast.

**Main driver of outperformance:** crypto's 2018–2021 and 2023–2024 bull runs. Without crypto,
the XIRR advantage shrinks by 7–14 percentage points depending on strategy.

**Limitations (repeat from Phase 2, still apply):**
- IDX sleeve uses a 2026-frozen ticker list (survivorship bias).
- Crypto history is short (≈8 years) and dominated by two extreme bull runs.
- Backtest assumes fractional units and no lot-size rounding.
- No withholding tax on US dividends modelled.
- All results are in IDR; an investor who earns USD would see different numbers.
- The backtest is not a forecast. Past crypto returns are unlikely to repeat at the same magnitude.

---

## Phase 2 exploration — sleeve-level risk/return (2018-01-01 to 2026-10-10)

**Data:** Four IDR-denominated sleeve indices (Phase 1 pipeline). Risk-free rate: BI Rate 5.75%.
All metrics are on the full 8.75-year period unless stated. No contributions modelled yet.

### Risk/Return summary

| Sleeve        | CAGR  | Vol   | Sharpe | Max DD | Max DD duration | VaR 95% | CVaR 95% |
|---------------|------:|------:|-------:|-------:|----------------:|--------:|---------:|
| idx_equities  |  8.9% | 18.7% |   0.01 | -40.9% |        889 days |  -1.78% |   -2.82% |
| us_equities   | 18.3% | 18.5% |   0.32 | -24.7% |        541 days |  -1.61% |   -2.78% |
| gold          | 17.3% | 17.0% |   0.31 | -21.0% |        940 days |  -1.70% |   -2.64% |
| crypto        | 22.9% | 58.5% |   0.15 | -86.8% |       1077 days |  -5.64% |   -9.00% |

### Sub-period total returns

| Period            | IDX equities | US equities | Gold  | Crypto  |
|-------------------|-------------:|------------:|------:|--------:|
| Bull 2018–2021    |        +82.9% |      +99.7% | +44.0% | +330.2% |
| Bear 2022         |        +6.2%  |      -10.5% |  +8.5% |  -63.9% |
| Recovery 2023–24  |        -4.4%  |      +62.3% | +47.0% | +314.4% |
| YTD 2025–2026     |       +14.1%  |      +50.5% | +76.5% |   -8.3% |

### Drawdown during key events

| Event                          | IDX equities | US equities | Gold    | Crypto  |
|--------------------------------|-------------:|------------:|--------:|--------:|
| COVID crash (Feb 20–Mar 23 '20)|       -36.0% |      -23.8% |  +10.0% |  -28.8% |
| Crypto bear (Nov '21–Dec '22)  |        +7.1% |       -9.3% |   +8.6% |  -73.1% |
| 2018 selloff (Jan–Dec '18)     |       +12.2% |       +1.8% |   +4.1% |  -76.0% |

### Correlation matrices (log returns)

Full period:

|              | IDX eq | US eq | Gold | Crypto |
|--------------|-------:|------:|-----:|-------:|
| IDX eq       |  1.000 | 0.088 | 0.057|  0.017 |
| US eq        |  0.088 | 1.000 | 0.372|  0.343 |
| Gold         |  0.057 | 0.372 | 1.000|  0.193 |
| Crypto       |  0.017 | 0.343 | 0.193|  1.000 |

During COVID crash (Feb–Apr 2020):

|              | IDX eq | US eq | Gold | Crypto |
|--------------|-------:|------:|-----:|-------:|
| IDX eq       |  1.000 | 0.294 | 0.059|  0.068 |
| US eq        |  0.294 | 1.000 | 0.457|  0.538 |
| Gold         |  0.059 | 0.457 | 1.000|  0.384 |
| Crypto       |  0.068 | 0.538 | 0.384|  1.000 |

### Observations

1. **IDX equities has the worst risk-adjusted return.** CAGR of 8.9% barely clears the BI Rate,
   leaving a near-zero Sharpe of 0.01. An IDX-only DCA strategy barely beats a money-market fund.

2. **US equities and gold tie for best Sharpe (≈0.31–0.32).** Much of their CAGR reflects USD
   appreciation vs IDR; currency gain is not equity or gold alpha.

3. **Crypto: highest raw CAGR (22.9%), worst risk-adjusted return after IDX.** Two -70%+ bears
   in the sample. The return distribution is dominated by a small number of extreme up-moves.

4. **IDX is nearly uncorrelated with everything (ρ ≤ 0.09).** A diversification benefit in theory,
   but limited by the low absolute return.

5. **Gold was the only sleeve that rose during the COVID crash (+10%).** The only genuine
   safe-haven in the sample.

6. **Correlations spike during stress.** US–crypto rose from 0.34 to 0.54 in the COVID crash.
   Diversification benefits shrink when they are most needed.

7. **Rolling vol peaked April–May 2020 for all sleeves.** IDX hit 50.5%, crypto 130.9%.

8. **2022 bear was almost exclusively a crypto and US equity event.** IDX +6.2%, gold +8.5%.

9. **IDX underperformed even in the 2018–2021 bull run** and was the only sleeve with negative
   total return in 2023–24.

10. **Gold is the standout performer in 2025–2026 YTD (+76.5%).** Consistent with elevated
    geopolitical and inflation uncertainty.

---

## Phase 6 — Deep learning volatility overlay (2018–2026)

**Method:** Two volatility forecasters used as a risk-scaling overlay on top of equal-weight DCA.
  - EWMA (λ=0.94): RiskMetrics exponential smoothing — zero parameters, the hard baseline.
  - GRU: 1-layer GRU (hidden=32), input = 63-day window of (r, |r|, r²), target = log(realised vol next 21 days). Walk-forward: retrained every 12 months on the prior 36 months. Early stopping (patience=20).

**Vol scaling:** `scale = min(1, target_vol / forecast_vol)` per sleeve, target vol = 15% p.a.
No leverage — scaling only reduces exposure, never exceeds 100%.

### Forecast accuracy (70/30 train/test split)

```
              ewma_rmse  gru_rmse  gru_vs_ewma_pct
sleeve                                            
idx_equities     0.0252    0.0836         231.8361
us_equities      0.0328    0.0736         124.7190
gold             0.0317    0.1009         218.4608
crypto           0.0561    0.1475         162.8951
```

**GRU did not beat the EWMA baseline** on vol forecast RMSE.
With only ~8 years of daily data (~2000 rows), the GRU has very limited training signal.
The EWMA is a strong baseline at this data scale — it requires no training and captures
the heavy-tailed, autocorrelated structure of financial volatility well.

### Backtest comparison

```
                  total_return_pct  xirr_pct  sharpe_twr  max_dd_pct
strategy                                                            
equal_weight_dca            202.32     24.30        1.02      -50.92
ewma_overlay_dca            139.65     19.27        1.20      -22.01
gru_overlay_dca             156.65     20.75        1.12      -34.50
```

### Observations

- The EWMA overlay did not improve XIRR vs equal-weight DCA (19.3% vs 24.3%).
- The GRU overlay achieved XIRR of 20.7%.
- EWMA overlay reduced max drawdown (-22.0% vs baseline -50.9%).
- The vol-scaling overlay reduces exposure during high-vol regimes (e.g. COVID March 2020,
  crypto bear 2022). Whether this improves risk-adjusted returns depends on whether the
  volatility forecast is timely — EWMA typically lags the initial spike.
- With ~2000 daily observations split across 4 sleeves, the GRU is data-limited.
  A larger dataset (more assets, longer history, or higher frequency) would be needed
  to demonstrate reliable GRU outperformance over a well-calibrated EWMA.
- **Honest conclusion:** the vol-scaling overlay adds modest complexity for marginal or
  no improvement in this dataset. The EWMA version may be worth including as a
  defensive option; the GRU version requires more data to justify the added complexity.

---

## Phase 6 — Deep learning volatility overlay (2018–2026)

**Method:** Two volatility forecasters used as a risk-scaling overlay on top of equal-weight DCA.
  - EWMA (λ=0.94): RiskMetrics exponential smoothing — zero parameters, the hard baseline.
  - GRU: 1-layer GRU (hidden=32), input = 63-day window of (r, |r|, r²), target = log(realised vol next 21 days). Walk-forward: retrained every 12 months on the prior 36 months. Early stopping (patience=20).

**Vol scaling:** `scale = min(1, target_vol / forecast_vol)` per sleeve, target vol = 15% p.a.
No leverage — scaling only reduces exposure, never exceeds 100%.

### Forecast accuracy (70/30 train/test split)

```
              ewma_rmse  gru_rmse  gru_vs_ewma_pct
sleeve                                            
idx_equities     0.0252    0.0836         231.8609
us_equities      0.0328    0.0736         124.7242
gold             0.0317    0.1009         218.4642
crypto           0.0561    0.1475         162.8154
```

**GRU did not beat the EWMA baseline** on vol forecast RMSE.
With only ~8 years of daily data (~2000 rows), the GRU has very limited training signal.
The EWMA is a strong baseline at this data scale — it requires no training and captures
the heavy-tailed, autocorrelated structure of financial volatility well.

### Backtest comparison

```
                  total_return_pct  xirr_pct  sharpe_twr  max_dd_pct
strategy                                                            
equal_weight_dca            202.32     24.30        1.02      -50.92
ewma_overlay_dca            139.65     19.27        1.20      -22.01
gru_overlay_dca             156.65     20.75        1.12      -34.50
```

### Observations

- The EWMA overlay did not improve XIRR vs equal-weight DCA (19.3% vs 24.3%).
- The GRU overlay achieved XIRR of 20.7%.
- EWMA overlay reduced max drawdown (-22.0% vs baseline -50.9%).
- The vol-scaling overlay reduces exposure during high-vol regimes (e.g. COVID March 2020,
  crypto bear 2022). Whether this improves risk-adjusted returns depends on whether the
  volatility forecast is timely — EWMA typically lags the initial spike.
- With ~2000 daily observations split across 4 sleeves, the GRU is data-limited.
  A larger dataset (more assets, longer history, or higher frequency) would be needed
  to demonstrate reliable GRU outperformance over a well-calibrated EWMA.
- **Honest conclusion:** the vol-scaling overlay adds modest complexity for marginal or
  no improvement in this dataset. The EWMA version may be worth including as a
  defensive option; the GRU version requires more data to justify the added complexity.
