"""Streamlit dashboard for NSE Quant.

Run with:
    streamlit run nse_quant/dashboard/app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nse_quant.backtest import Backtester
from nse_quant.data import DataFetcher, load_fundamentals, load_universe
from nse_quant.indicators import add_all_indicators, classify_regime
from nse_quant.risk import RiskManager, stress_test
from nse_quant.strategies import STRATEGY_REGISTRY

st.set_page_config(page_title="NSE Quant", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def _fetcher() -> DataFetcher:
    return DataFetcher(use_live=False)


@st.cache_data(show_spinner=False)
def _prices(tickers: tuple[str, ...] | None, start: str | None, end: str | None) -> pd.DataFrame:
    f = _fetcher()
    return f.get_many(tickers=list(tickers) if tickers else None, start=start, end=end, column="adj_close")


@st.cache_data(show_spinner=False)
def _fundamentals() -> pd.DataFrame:
    return load_fundamentals()


@st.cache_data(show_spinner=False)
def _universe() -> pd.DataFrame:
    return load_universe()


def _signal_heatmap(weights: pd.DataFrame) -> go.Figure:
    recent = weights.tail(63)
    fig = px.imshow(
        recent.T,
        labels={"x": "Date", "y": "Ticker", "color": "Weight"},
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    fig.update_layout(height=420, margin={"l": 20, "r": 20, "t": 30, "b": 20})
    return fig


def _equity_plot(equity: pd.Series, title: str = "Equity curve") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="equity"))
    fig.update_layout(title=title, height=360, margin={"l": 20, "r": 20, "t": 40, "b": 20})
    return fig


def render_overview(prices: pd.DataFrame, universe: pd.DataFrame) -> None:
    st.subheader("Market overview")
    cols = st.columns(4)
    if prices.empty:
        st.warning("No price data available")
        return
    last = prices.iloc[-1]
    prev = prices.iloc[-2] if len(prices) > 1 else last
    chg = (last / prev - 1) * 100
    top = chg.nlargest(3)
    bot = chg.nsmallest(3)
    cols[0].metric("Tickers tracked", len(prices.columns))
    cols[1].metric("Start", str(prices.index[0].date()))
    cols[2].metric("End", str(prices.index[-1].date()))
    cols[3].metric("Avg 1D change", f"{chg.mean():.2f}%")

    st.markdown("**Top movers (last bar)**")
    top_df = pd.DataFrame({"ticker": top.index, "change_%": top.values.round(2)})
    bot_df = pd.DataFrame({"ticker": bot.index, "change_%": bot.values.round(2)})
    c1, c2 = st.columns(2)
    c1.dataframe(top_df, hide_index=True, use_container_width=True)
    c2.dataframe(bot_df, hide_index=True, use_container_width=True)

    st.markdown("**Sector performance (lookback 63d)**")
    sector_map = universe.set_index("ticker")["sector"].to_dict()
    ret = prices.pct_change().tail(63)
    sector_ret: dict[str, float] = {}
    for sector in set(sector_map.values()):
        members = [t for t, s in sector_map.items() if s == sector and t in ret.columns]
        if members:
            sector_ret[sector] = float((1 + ret[members].mean(axis=1)).prod() - 1)
    sec_df = pd.DataFrame({"sector": list(sector_ret.keys()), "return_%": [v * 100 for v in sector_ret.values()]})
    fig = px.bar(sec_df.sort_values("return_%"), x="return_%", y="sector", orientation="h")
    fig.update_layout(height=380, margin={"l": 20, "r": 20, "t": 20, "b": 20})
    st.plotly_chart(fig, use_container_width=True)


def render_indicators(ticker: str) -> None:
    st.subheader(f"Indicators — {ticker}")
    fetcher = _fetcher()
    df = fetcher.get(ticker).tail(500)
    ind = add_all_indicators(df)
    regime = classify_regime(df)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="OHLC"))
    if "sma_50" in ind.columns:
        fig.add_trace(go.Scatter(x=ind.index, y=ind["sma_50"], mode="lines", name="SMA 50"))
    if "sma_200" in ind.columns:
        fig.add_trace(go.Scatter(x=ind.index, y=ind["sma_200"], mode="lines", name="SMA 200"))
    fig.update_layout(height=480, xaxis_rangeslider_visible=False, margin={"l": 20, "r": 20, "t": 20, "b": 20})
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("RSI 14", f"{ind['rsi_14'].iloc[-1]:.2f}" if "rsi_14" in ind.columns else "n/a")
    c2.metric("MACD hist", f"{ind['macd_hist'].iloc[-1]:.3f}" if "macd_hist" in ind.columns else "n/a")
    c3.metric("Regime", str(regime.iloc[-1]) if len(regime) else "unknown")

    with st.expander("All indicator values (last bar)"):
        last_row = ind.iloc[-1].drop(labels=["open", "high", "low", "close", "volume"], errors="ignore")
        st.dataframe(last_row.to_frame("value"), use_container_width=True)


def render_backtest(prices: pd.DataFrame, fundamentals: pd.DataFrame, universe: pd.DataFrame) -> None:
    st.subheader("Backtest")
    strat_name = st.selectbox("Strategy", sorted(STRATEGY_REGISTRY.keys()))
    run = st.button("Run backtest")
    if not run:
        st.info("Pick a strategy and click **Run backtest**.")
        return
    with st.spinner(f"Running {strat_name}…"):
        strat = STRATEGY_REGISTRY[strat_name]()
        result = strat.generate(prices, fundamentals=fundamentals, universe=universe)
        bt = Backtester().run(result.weights, prices)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", f"{bt.metrics['cagr']*100:.2f}%")
    c2.metric("Vol", f"{bt.metrics['volatility']*100:.2f}%")
    c3.metric("Sharpe", f"{bt.metrics['sharpe']:.2f}")
    c4.metric("Max DD", f"{bt.metrics['max_drawdown']*100:.2f}%")
    st.plotly_chart(_equity_plot(bt.equity_curve, f"{strat_name} equity"), use_container_width=True)
    st.plotly_chart(_signal_heatmap(result.weights), use_container_width=True)

    with st.expander("Risk metrics"):
        st.json({k: round(v, 4) for k, v in bt.metrics.items()})
        st.markdown("**Stress test**")
        st.dataframe(stress_test(bt.returns), hide_index=True, use_container_width=True)


def render_portfolio(prices: pd.DataFrame, universe: pd.DataFrame) -> None:
    st.subheader("Portfolio builder")
    st.write(
        "Upload your current positions (CSV with `ticker,shares` columns) or pick from the universe."
    )
    uploaded = st.file_uploader("Upload positions CSV", type=["csv"])
    if uploaded is not None:
        pos = pd.read_csv(uploaded)
    else:
        st.markdown("**Quick-build demo portfolio**")
        default = universe.head(5)["ticker"].tolist()
        picks = st.multiselect("Tickers", options=universe["ticker"].tolist(), default=default)
        if not picks:
            return
        shares = st.number_input("Shares per holding", min_value=1, value=100)
        pos = pd.DataFrame({"ticker": picks, "shares": shares})
    st.dataframe(pos, hide_index=True, use_container_width=True)

    last_prices = prices.iloc[-1]
    pos = pos[pos["ticker"].isin(last_prices.index)]
    if pos.empty:
        st.warning("None of the tickers are available in the loaded price history.")
        return
    pos["price"] = pos["ticker"].map(last_prices)
    pos["value"] = pos["price"] * pos["shares"]
    total = pos["value"].sum()
    pos["weight"] = pos["value"] / total if total else 0.0
    st.metric("Portfolio value (KES)", f"{total:,.2f}")
    st.dataframe(pos, hide_index=True, use_container_width=True)

    ret = prices[pos["ticker"].tolist()].pct_change().dropna()
    weights = pos.set_index("ticker")["weight"]
    port_ret = (ret * weights).sum(axis=1)
    equity = 1_000_000 * (1 + port_ret).cumprod()
    st.plotly_chart(_equity_plot(equity, "Portfolio equity (hypothetical 1M KES)"), use_container_width=True)

    mgr = RiskManager()
    st.json({k: round(v, 4) for k, v in mgr.evaluate(port_ret).items()})


def main() -> None:
    st.title("NSE Quant")
    st.caption(
        "Quantitative analysis for the Nairobi Securities Exchange. Runs on bundled sample data by default; "
        "toggle live mode to attempt yfinance fetches."
    )
    universe = _universe()
    fundamentals = _fundamentals()

    with st.sidebar:
        st.header("Data")
        tickers = st.multiselect(
            "Tickers",
            options=universe["ticker"].tolist(),
            default=universe["ticker"].tolist()[:10],
        )
        start = st.text_input("Start (YYYY-MM-DD)", "2022-01-01")
        end = st.text_input("End (YYYY-MM-DD)", "")
        end = end or None
        st.caption("Sample data covers 2020–2025 business days.")

    prices = _prices(tuple(tickers), start, end)

    tab_overview, tab_ind, tab_bt, tab_port = st.tabs(
        ["Overview", "Indicators", "Backtest", "Portfolio"]
    )
    with tab_overview:
        render_overview(prices, universe)
    with tab_ind:
        if tickers:
            pick = st.selectbox("Ticker", tickers)
            render_indicators(pick)
    with tab_bt:
        render_backtest(prices, fundamentals, universe)
    with tab_port:
        render_portfolio(prices, universe)


if __name__ == "__main__":
    main()
