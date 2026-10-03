# Decisions

Each row is a choice the code will depend on. Every row starts as **Proposed** or **Open**. Change the choice if it does not fit, then set the status to **Confirmed**. Phase 0 is done when no row is Proposed or Open and `config/costs.yaml` is filled in.

| # | Decision | Choice | Why | Status |
|---|---|---|---|---|
| 1 | Base currency | IDR | Returns are judged in the investor's own currency. USD-priced sleeves get converted. | Proposed |
| 2 | Universe structure | Four sleeves, each one return series: IDX equities, US equities, gold, crypto | One sleeve per asset class keeps stocks from swamping gold and crypto in equal-weight and risk-parity schemes. Stock-level work can be a later extension. | Proposed |
| 3 | IDX equities sleeve | Equal-weight basket of a fixed ticker list in `config/universe.yaml`, rebalanced monthly, adjusted prices (dividends included), list frozen on a recorded date | Picking today's large caps and testing backwards flatters results (survivorship bias), so the list is frozen and the date documented. The sleeve is treated like an index fund: fractional units, no lot-size rounding in v1. | Open: needs the ticker list |
| 4 | US equities sleeve | SPY (S&P 500 ETF), adjusted prices, converted to IDR | An index ETF avoids the survivorship problem of picking constituents. It is USD-priced, so rupiah moves matter. | Proposed |
| 5 | Gold sleeve | GLD, converted to IDR | Clean USD price series. It is a proxy for the gold price, so local dealer spreads go in `costs.yaml`. | Proposed |
| 6 | Crypto sleeve | Equal-weight BTC-USD and ETH-USD, rebalanced monthly, converted to IDR | One sleeve for the asset class, so crypto is not double-counted in equal-weight baselines. These have the longest and most liquid histories. | Proposed |
| 7 | Benchmarks | (a) equal-weight DCA across the four sleeves, (b) DCA into the IDX sleeve only, (c) DCA into the US sleeve only. IHSG (`^JKSE`) is a reference line only | IHSG is a price index without dividends, so it understates equity returns. | Proposed |
| 8 | Risk-free rate | BI 7-Day Reverse Repo Rate, entered by hand into `data/external/bi_rate.csv` | Local policy rate for an IDR investor. | Proposed |
| 9 | Backtest start | 2018-01-01 | ETH data begins in late 2017, and the window starts after the 2016 change in the BI policy rate definition. It still covers the 2018 and 2022 crypto bear markets, the 2020 crash and the 2021 bull run. | Proposed |
| 10 | Contributions | Monthly, first trading day of the month | Matches the project question. Keep it a parameter in case the real contribution day differs. | Proposed |
| 11 | Execution | Signal on the close of the contribution day, trade at the next close when that sleeve's market is open | Avoids lookahead. Daily closes only, no intraday data. | Proposed |
| 12 | Rebalancing | Contribution-only in v1: new money goes to underweight sleeves, no selling | Avoids sell fees and tax, and matches how a monthly saver actually invests. A sell-rebalance variant can be tested in Phase 4. | Proposed |
| 13 | "Hold cash until signal" | Per sleeve: buy only when the sleeve's index is above its 200-day average. Otherwise that share of the contribution waits in cash (earning the assumed yield) for at most 3 months, then is invested anyway | Simple, standard trend filter. The wait cap stops the strategy from sitting in cash forever. | Proposed |
| 14 | Costs and cash yield | Filled in `config/costs.yaml` from current broker, exchange, platform and fund pages | Rules and fees change and differ per provider, so they are looked up, not guessed. | Open: needs lookup |

## Not modeled in v1

- Lot-size rounding and auto-rejection price limits for single IDX stocks. The IDX sleeve stands in for an index fund or ETF. Revisit if a single-stock variant is added.
- Withholding tax on US dividends, a small drag on the US sleeve. Add it as a Phase 4 sensitivity check.
- Taxes beyond the fee and tax rates entered in `config/costs.yaml`.
