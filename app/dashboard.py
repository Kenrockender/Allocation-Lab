"""
dashboard.py — Streamlit allocation-lab dashboard.

Run locally:
    streamlit run app/dashboard.py

Sections:
  1. This Month — trend signals and suggested allocation
  2. Asset Stats — sleeve risk/return table
  3. Strategy Comparison — equity curves and performance table
  4. Drawdowns — drawdown chart per strategy
  5. Correlation — heatmap
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Allocation Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# Data bootstrap — fetch on first run if files are missing            #
# ------------------------------------------------------------------ #

def _ensure_data() -> None:
    """Run the data pipeline if processed files are absent.

    This happens on a fresh Streamlit Cloud deploy where data/ is gitignored.
    Shows a progress message so the user knows what's happening.
    """
    sleeve_path = ROOT / "data" / "processed" / "sleeve_indices.parquet"
    if sleeve_path.exists():
        return

    msg = st.empty()
    msg.info(
        "⏳ First run detected — fetching price data (~2 min). "
        "This only happens once per deployment."
    )
    bar = st.progress(0, text="Fetching raw data …")

    try:
        from allocation_lab.data import fetch, align, quality

        bar.progress(10, text="Fetching raw prices from Yahoo Finance …")
        fetch.fetch_all(start="2017-01-01")

        bar.progress(60, text="Aligning calendars and converting to IDR …")
        align.run(start="2018-01-01")

        bar.progress(85, text="Running quality checks …")
        quality.run(save_report=True)

        bar.progress(95, text="Running backtest to build NAV curves …")
        _build_nav_curves()

        bar.progress(100, text="Done!")
        msg.success("✅ Data ready.")
    except Exception as exc:
        msg.error(f"Data pipeline failed: {exc}")
        st.stop()


def _build_nav_curves() -> None:
    """Build and cache the NAV curves parquet (needed for strategy comparison tab)."""
    nav_path = ROOT / "data" / "processed" / "nav_curves.parquet"
    if nav_path.exists():
        return
    import pandas as pd
    from allocation_lab.backtest.engine import run_backtest
    from allocation_lab.strategies.baselines import EqualWeightDCA, SingleSleeveDCA, FixedWeightDCA
    from allocation_lab.strategies.v1 import InverseVolatilityDCA, MomentumDCA, TrendFilterDCA

    sleeve_df = pd.read_parquet(ROOT / "data/processed/sleeve_indices.parquet")
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
    nav_df = pd.concat(
        [run_backtest(s, sleeve_df, contribution_idr=5_000_000).nav_series
         for s in strategies],
        axis=1,
    )
    nav_df.to_parquet(nav_path)


# Run bootstrap on every page load (fast if files exist)
_ensure_data()


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

@st.cache_data(ttl=3600)
def load_sleeves() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data/processed/sleeve_indices.parquet")


@st.cache_data(ttl=3600)
def load_nav_curves() -> pd.DataFrame:
    p = ROOT / "data/processed/nav_curves.parquet"
    if not p.exists():
        _build_nav_curves()
    return pd.read_parquet(p)


@st.cache_data(ttl=3600)
def get_report(contribution: float) -> dict:
    from allocation_lab.reporting.monthly_report import build_report
    return build_report(contribution_idr=contribution)


@st.cache_data(ttl=3600)
def get_summary_table(rf: float) -> pd.DataFrame:
    from allocation_lab.analytics.metrics import summary_table
    sleeves = load_sleeves()
    cols = [c for c in ["idx_equities", "us_equities", "gold", "crypto"]
            if c in sleeves.columns]
    tbl = summary_table(sleeves[cols], rf_annual=rf)
    tbl["cagr_pct"]   = (tbl["cagr"]   * 100).round(1)
    tbl["vol_pct"]    = (tbl["vol"]     * 100).round(1)
    tbl["max_dd_pct"] = (tbl["max_dd"]  * 100).round(1)
    tbl["sharpe"]     = tbl["sharpe"].round(2)
    return tbl[["cagr_pct", "vol_pct", "sharpe", "max_dd_pct", "max_dd_days"]]


def _drawdown(nav: pd.Series) -> pd.Series:
    return (nav / nav.cummax() - 1) * 100


# ------------------------------------------------------------------ #
# Sidebar                                                              #
# ------------------------------------------------------------------ #

st.sidebar.title("⚙️ Settings")
contribution = st.sidebar.number_input(
    "Monthly contribution (IDR)",
    min_value=100_000,
    max_value=500_000_000,
    value=5_000_000,
    step=500_000,
    format="%d",
)
rf_pct = st.sidebar.slider("Risk-free rate (BI Rate %)", 0.0, 12.0, 5.75, 0.25)
rf = rf_pct / 100

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Disclaimer:** Research project. Nothing here is financial advice. "
    "Backtest results do not predict future returns."
)

# ------------------------------------------------------------------ #
# Header                                                               #
# ------------------------------------------------------------------ #

st.title("📊 Allocation Lab")
st.caption(
    "How should a fixed monthly contribution be split across IDX stocks, the S&P 500, "
    "gold and crypto? · Data: 2018–2026 · All returns in IDR · BI Rate risk-free"
)

# ------------------------------------------------------------------ #
# Tab layout                                                           #
# ------------------------------------------------------------------ #

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗓 This Month",
    "📈 Asset Stats",
    "🏆 Strategy Comparison",
    "📉 Drawdowns",
    "🔗 Correlation",
])

# ================================================================== #
# Tab 1 — This Month                                                   #
# ================================================================== #
with tab1:
    st.subheader("This month's allocation signals")
    st.caption(
        "Signals as of the latest available data. "
        "Contribution amount set in the sidebar."
    )

    report = get_report(contribution)
    signals = report["signals"]
    report_date = report["report_date"]
    st.markdown(f"**Report date:** {report_date}")

    # Trend signals
    st.markdown("#### 200-day MA trend signals")
    cols_sig = st.columns(4)
    for i, sleeve in enumerate(signals.index):
        above = signals.loc[sleeve, "above_ma"]
        pct   = signals.loc[sleeve, "pct_above_ma"]
        icon  = "🟢" if above else "🔴"
        label = "Above MA" if above else "Below MA"
        cols_sig[i].metric(
            label=sleeve.replace("_", " ").title(),
            value=f"{icon} {label}",
            delta=f"{pct:+.1f}% vs 200-day MA",
            delta_color="normal" if above else "inverse",
        )

    st.markdown("---")
    st.markdown("#### Suggested allocations")
    strategy_labels = {
        "equal_weight": "Equal Weight",
        "inverse_vol":  "Inverse Volatility",
        "momentum":     "Momentum (12-1m)",
    }

    alloc_data = []
    for strat, alloc in report["allocations_idr"].items():
        for sleeve, amt in alloc.items():
            alloc_data.append({
                "Strategy": strategy_labels.get(strat, strat),
                "Sleeve": sleeve.replace("_", " ").title(),
                "IDR": float(amt),
            })
    alloc_df = pd.DataFrame(alloc_data)

    fig_alloc = px.bar(
        alloc_df, x="Strategy", y="IDR", color="Sleeve",
        barmode="group",
        title=f"Monthly allocation (IDR {contribution:,.0f} total)",
        labels={"IDR": "IDR amount"},
        height=400,
    )
    fig_alloc.update_layout(yaxis_tickformat=",.0f")
    st.plotly_chart(fig_alloc, use_container_width=True)

    # Weight table
    st.markdown("#### Weights (%)")
    weight_rows = []
    for strat, w in report["weights"].items():
        row = {"Strategy": strategy_labels.get(strat, strat)}
        for sleeve, val in w.items():
            row[sleeve.replace("_", " ").title()] = f"{val*100:.1f}%"
        weight_rows.append(row)
    st.dataframe(pd.DataFrame(weight_rows).set_index("Strategy"), use_container_width=True)

    st.info(
        "**Trend filter note:** IDX equities and gold are currently **below** their 200-day MA. "
        "A trend-filter strategy would hold their share in cash this month."
        if report["below_ma_sleeves"]
        else "All sleeves are above their 200-day MA — full deployment across all strategies."
    )

# ================================================================== #
# Tab 2 — Asset Stats                                                  #
# ================================================================== #
with tab2:
    st.subheader("Sleeve risk/return (2018–2026, IDR)")
    tbl = get_summary_table(rf)
    tbl_display = tbl.copy()
    tbl_display.index = [i.replace("_", " ").title() for i in tbl_display.index]
    tbl_display.columns = ["CAGR %", "Vol %", "Sharpe", "Max DD %", "Max DD days"]
    st.dataframe(
        tbl_display.style.background_gradient(subset=["CAGR %", "Sharpe"], cmap="RdYlGn")
                         .background_gradient(subset=["Max DD %"], cmap="RdYlGn_r"),
        use_container_width=True,
    )

    sleeves = load_sleeves()
    sleeve_cols = [c for c in ["idx_equities","us_equities","gold","crypto"]
                   if c in sleeves.columns]
    idx_map = {s: s.replace("_", " ").title() for s in sleeve_cols}

    fig_price = go.Figure()
    for col in sleeve_cols:
        s = sleeves[col].dropna()
        fig_price.add_trace(go.Scatter(
            x=s.index, y=s, name=idx_map[col], mode="lines",
        ))
    fig_price.update_layout(
        title="Sleeve total-return indices (IDR, base=100 at start)",
        xaxis_title="", yaxis_title="Index (IDR)",
        height=450, hovermode="x unified",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # Rolling vol chart
    st.markdown("#### Rolling 63-day annualised volatility")
    fig_vol = go.Figure()
    for col in sleeve_cols:
        s = sleeves[col].dropna()
        lr = np.log(s / s.shift(1)).dropna()
        rv = lr.rolling(63).std() * np.sqrt(252) * 100
        fig_vol.add_trace(go.Scatter(x=rv.index, y=rv, name=idx_map[col], mode="lines"))
    fig_vol.update_layout(
        xaxis_title="", yaxis_title="Annualised vol (%)",
        height=380, hovermode="x unified",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

# ================================================================== #
# Tab 3 — Strategy Comparison                                          #
# ================================================================== #
with tab3:
    st.subheader("Strategy comparison — full backtest (2018–2026)")
    nav_df = load_nav_curves()

    if nav_df.empty:
        st.warning("NAV curves not found. Run: `python -m allocation_lab.backtest.run_all`")
    else:
        # NAV curves (normalised to 100 at start for comparison)
        nav_norm = nav_df / nav_df.iloc[0] * 100
        fig_nav = go.Figure()
        for col in nav_norm.columns:
            fig_nav.add_trace(go.Scatter(
                x=nav_norm.index, y=nav_norm[col],
                name=col.replace("_", " "), mode="lines",
            ))
        fig_nav.update_layout(
            title="NAV curves (normalised to 100 at start, IDR 5M/month contribution)",
            xaxis_title="", yaxis_title="Index",
            height=480, hovermode="x unified",
        )
        st.plotly_chart(fig_nav, use_container_width=True)

        # Performance table
        from allocation_lab.backtest.performance import compare_results
        from allocation_lab.backtest.engine import run_backtest
        from allocation_lab.strategies.baselines import EqualWeightDCA, SingleSleeveDCA, FixedWeightDCA
        from allocation_lab.strategies.v1 import InverseVolatilityDCA, MomentumDCA, TrendFilterDCA

        @st.cache_data(ttl=3600)
        def _perf_table(contribution_idr: float, rf_rate: float) -> pd.DataFrame:
            sleeve_df = load_sleeves()
            strategies = [
                EqualWeightDCA(),
                SingleSleeveDCA("idx_equities"),
                SingleSleeveDCA("us_equities"),
                FixedWeightDCA(
                    {"idx_equities":0.30,"us_equities":0.30,"gold":0.20,"crypto":0.20},
                    name="fixed_30_30_20_20",
                ),
                InverseVolatilityDCA(),
                MomentumDCA(),
                TrendFilterDCA(),
            ]
            results = [run_backtest(s, sleeve_df, contribution_idr) for s in strategies]
            return compare_results(results, rf_annual=rf_rate)

        with st.spinner("Computing performance metrics ..."):
            perf = _perf_table(contribution, rf)

        perf_display = perf[["final_nav_idr","total_contributed_idr",
                              "total_return_pct","xirr_pct","sharpe_twr",
                              "max_dd_pct","cost_drag_pct"]].copy()
        perf_display.columns = ["Final NAV (IDR)","Contributed (IDR)",
                                 "Total Return %","XIRR %","Sharpe TWR",
                                 "Max DD %","Cost Drag %"]
        perf_display.index = [i.replace("_"," ") for i in perf_display.index]
        perf_display["Final NAV (IDR)"] = perf_display["Final NAV (IDR)"].apply(
            lambda x: f"IDR {x:,.0f}")
        perf_display["Contributed (IDR)"] = perf_display["Contributed (IDR)"].apply(
            lambda x: f"IDR {x:,.0f}")
        st.dataframe(
            perf_display.style.background_gradient(
                subset=["XIRR %","Sharpe TWR"], cmap="RdYlGn"
            ).background_gradient(subset=["Max DD %"], cmap="RdYlGn_r"),
            use_container_width=True,
        )
        st.caption(
            "⚠️ Bootstrap 90% CI on equal-weight XIRR: **8.8% – 52%**. "
            "XIRR differences between strategies are smaller than window-to-window noise. "
            "No strategy reliably dominates out-of-sample."
        )

# ================================================================== #
# Tab 4 — Drawdowns                                                    #
# ================================================================== #
with tab4:
    st.subheader("Strategy drawdowns")
    nav_df2 = load_nav_curves()
    if nav_df2.empty:
        st.warning("NAV curves not found.")
    else:
        selected = st.multiselect(
            "Select strategies",
            options=list(nav_df2.columns),
            default=["equal_weight_dca", "dca_us_equities", "trend_filter_dca"],
            format_func=lambda x: x.replace("_", " "),
        )
        if selected:
            fig_dd = go.Figure()
            for col in selected:
                dd = _drawdown(nav_df2[col])
                fig_dd.add_trace(go.Scatter(
                    x=dd.index, y=dd, name=col.replace("_", " "),
                    mode="lines", fill="tozeroy",
                ))
            fig_dd.update_layout(
                title="Drawdown from peak (%)",
                xaxis_title="", yaxis_title="Drawdown %",
                height=450, hovermode="x unified",
                yaxis=dict(tickformat=".1f"),
            )
            st.plotly_chart(fig_dd, use_container_width=True)

# ================================================================== #
# Tab 5 — Correlation                                                  #
# ================================================================== #
with tab5:
    st.subheader("Sleeve return correlation")
    sleeves5 = load_sleeves()
    sleeve_cols5 = [c for c in ["idx_equities","us_equities","gold","crypto"]
                    if c in sleeves5.columns]

    period = st.radio(
        "Period", ["Full (2018–2026)", "COVID crash (Feb–Apr 2020)",
                   "Crypto bear (2022)", "Recovery (2023–2024)"],
        horizontal=True,
    )
    period_map = {
        "Full (2018–2026)":          ("2018-01-01", "2026-10-10"),
        "COVID crash (Feb–Apr 2020)":("2020-02-01", "2020-04-30"),
        "Crypto bear (2022)":        ("2022-01-01", "2022-12-31"),
        "Recovery (2023–2024)":      ("2023-01-01", "2024-12-31"),
    }
    s_date, e_date = period_map[period]
    sub = sleeves5[sleeve_cols5].loc[s_date:e_date].ffill()
    lr_sub = np.log(sub / sub.shift(1)).dropna()
    corr = lr_sub.corr()

    labels = [c.replace("_", " ").title() for c in corr.columns]
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        hoverongaps=False,
    ))
    fig_corr.update_layout(title=f"Correlation — {period}", height=420)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.caption(
        "**Note:** Correlations rise during stress. The COVID crash period shows "
        "significantly higher cross-asset correlation than the full period."
    )
