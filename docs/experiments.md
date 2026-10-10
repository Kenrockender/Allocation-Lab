# Experiment log

Log every strategy variant and parameter set that gets run, including the ones that did not
work. The count matters: the more variants tried, the more likely the best one is luck.

| Date | Variant | Parameters | Sample | XIRR | Kept? |
|---|---|---|---|---|---|
| 2026-10-10 | equal_weight_dca | — | 2018-01-01 to 2026-10-10 | 24.3% | Yes (baseline) |
| 2026-10-10 | dca_idx_equities | — | 2018-01-01 to 2026-10-10 | 6.9% | Yes (benchmark) |
| 2026-10-10 | dca_us_equities | — | 2018-01-01 to 2026-10-10 | 20.3% | Yes (benchmark) |
| 2026-10-10 | fixed_30_30_20_20 | idx=0.30, us=0.30, gold=0.20, crypto=0.20 | 2018-01-01 to 2026-10-10 | 22.6% | Yes |
| 2026-10-10 | inverse_vol_dca | vol_window=63d | 2018-01-01 to 2026-10-10 | 19.8% | Yes |
| 2026-10-10 | momentum_dca | lookback=252d, skip=21d | 2018-01-01 to 2026-10-10 | 23.1% | Yes |
| 2026-10-10 | trend_filter_dca | ma_window=200d, wait_cap=3m | 2018-01-01 to 2026-10-10 | 20.8% | Yes |
| 2026-10-10 | momentum_dca | lookback=126d (6m), skip=21d | 2018-01-01 to 2026-10-10 | 23.7% | No — negligible delta |
| 2026-10-10 | momentum_dca | lookback=504d (24m), skip=21d | 2018-01-01 to 2026-10-10 | 23.4% | No — negligible delta |
| 2026-10-10 | equal_weight_dca | no crypto (crypto NaN) | 2018-01-01 to 2026-10-10 | 16.9% | No — sensitivity only |
| 2026-10-10 | inverse_vol_dca | no crypto (crypto NaN) | 2018-01-01 to 2026-10-10 | 5.4% | No — sensitivity only |
| 2026-10-10 | momentum_dca | no crypto (crypto NaN) | 2018-01-01 to 2026-10-10 | 16.5% | No — sensitivity only |
| 2026-10-10 | trend_filter_dca | no crypto (crypto NaN) | 2018-01-01 to 2026-10-10 | 16.8% | No — sensitivity only |
| 2026-10-10 | equal_weight_dca | costs doubled | 2018-01-01 to 2026-10-10 | 24.1% | No — sensitivity only |
| 2026-10-10 | inverse_vol_dca | costs doubled | 2018-01-01 to 2026-10-10 | 19.6% | No — sensitivity only |
| 2026-10-10 | momentum_dca | costs doubled | 2018-01-01 to 2026-10-10 | 23.0% | No — sensitivity only |
| 2026-10-10 | trend_filter_dca | costs doubled | 2018-01-01 to 2026-10-10 | 20.7% | No — sensitivity only |

**Total variants run: 17**

The three "kept" signal strategies (inverse_vol, momentum, trend_filter) were chosen as
representative examples before running — they were not selected post-hoc from a larger set.
The lookback sensitivity and no-crypto runs were diagnostics, not strategy selection.
