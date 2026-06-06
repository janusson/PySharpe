"""Backtesting tab for the PySharpe Streamlit dashboard."""

from __future__ import annotations

import pandas as pd

from pysharpe import metrics

try:
    import streamlit as st  # type: ignore[import]
except ImportError:  # pragma: no cover - only needed in Streamlit context
    st = None  # type: ignore[assignment]

try:
    import plotly.express as px  # type: ignore[import]
    import plotly.graph_objects as go  # type: ignore[import]
except ImportError:  # pragma: no cover - optional visualization dependency
    go = None  # type: ignore[assignment]
    px = None  # type: ignore[assignment]

from pysharpe.analysis.backtest_engine import HistoricalBacktester
from pysharpe.analysis.benchmarks import CANADIAN_BENCHMARKS
from pysharpe.data.fetcher import YFinancePriceFetcher

# Map UI strategy labels to pandas frequency aliases accepted by HistoricalBacktester
STRATEGY_FREQ: dict[str, str] = {
    "Monthly (calendar)": "ME",
    "Quarterly (calendar)": "QE",
    "Annual (calendar)": "YE",
}


def parse_weight_input(text: str) -> dict[str, float]:
    """Parse a weight specification string into a normalised weight dict.

    The expected format is ``"TICKER=weight, TICKER=weight"`` where weights can be
    any positive float.  The returned dict sums to 1.0.

    Args:
        text: Comma-separated ``key=value`` pairs, e.g. ``"A=0.6, B=0.4"`` or
            ``"A=3, B=1"``.

    Returns:
        Normalised mapping of ticker → weight fraction.

    Raises:
        ValueError: If the string cannot be parsed, contains fewer than 2 entries,
            or any value is not a positive number.
    """
    entries = [entry.strip() for entry in text.split(",") if entry.strip()]
    if len(entries) < 2:
        raise ValueError(
            f"At least 2 ticker=weight pairs are required; got {len(entries)}. "
            "Expected format: 'A=0.6, B=0.4'"
        )

    raw: dict[str, float] = {}
    for entry in entries:
        parts = entry.split("=")
        if len(parts) != 2:
            raise ValueError(
                f"Could not parse entry {entry!r}. Expected format: 'TICKER=weight'."
            )
        key = parts[0].strip().upper()
        if not key:
            raise ValueError(f"Empty ticker symbol in entry {entry!r}.")
        try:
            value = float(parts[1].strip())
        except ValueError as exc:
            raise ValueError(
                f"Could not convert {parts[1].strip()!r} to float in entry {entry!r}."
            ) from exc
        if value < 0:
            raise ValueError(f"Weight for {key!r} must be non-negative; got {value}.")
        raw[key] = value

    total = sum(raw.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")

    return {k: v / total for k, v in raw.items()}


def render_backtest_tab(prices: pd.DataFrame) -> None:
    """Render the backtesting section within the Streamlit app."""
    st.subheader("Portfolio Backtesting")

    if prices.empty:
        st.info("Load price data from the sidebar to run a backtest.")
        return

    col1, col2 = st.columns(2)

    with col1:
        strategy = st.selectbox(
            "Rebalancing strategy",
            [
                "No rebalancing",
                "Monthly (calendar)",
                "Quarterly (calendar)",
                "Annual (calendar)",
                "Drift band (absolute)",
                "Drift band (relative)",
            ],
        )
        initial_capital = st.number_input(
            "Initial capital ($)", value=10_000.0, min_value=100.0, step=500.0
        )

    with col2:
        fee_per_trade = st.number_input(
            "Fee per trade ($)", value=0.0, min_value=0.0, step=1.0
        )
        slippage_pct = (
            st.slider("Slippage (%)", 0.0, 1.0, 0.0, step=0.05) / 100.0
        )

    # Drift band controls — only shown when relevant
    abs_band = None
    rel_band = None
    if strategy == "Drift band (absolute)":
        abs_band = st.slider(
            "Absolute drift threshold", 0.01, 0.30, 0.05, step=0.01
        )
    elif strategy == "Drift band (relative)":
        rel_band = st.slider(
            "Relative drift threshold", 0.05, 1.0, 0.20, step=0.05
        )

    st.markdown(
        "**Portfolio weights** (format: `TICKER=weight, TICKER=weight`)"
    )
    default_weights = ", ".join(
        f"{col}={1 / len(prices.columns):.2f}" for col in prices.columns
    )
    weight_text = st.text_input("Weights", value=default_weights)

    # Benchmark
    bench_col1, bench_col2 = st.columns(2)
    with bench_col1:
        run_benchmark = st.checkbox("Compare to benchmark", value=True)
    with bench_col2:
        benchmark_ticker = (
            st.selectbox(
                "Benchmark",
                list(CANADIAN_BENCHMARKS.keys()),
                format_func=lambda t: CANADIAN_BENCHMARKS[t],
            )
            if run_benchmark
            else None
        )

    if st.button("Run Backtest", type="primary"):
        _run_and_display(
            prices=prices,
            weight_text=weight_text,
            strategy=strategy,
            initial_capital=initial_capital,
            fee_per_trade=fee_per_trade,
            slippage_pct=slippage_pct,
            abs_band=abs_band,
            rel_band=rel_band,
            benchmark_ticker=benchmark_ticker if run_benchmark else None,
        )


def _run_and_display(
    prices: pd.DataFrame,
    weight_text: str,
    strategy: str,
    initial_capital: float,
    fee_per_trade: float,
    slippage_pct: float,
    abs_band: float | None,
    rel_band: float | None,
    benchmark_ticker: str | None,
) -> None:
    """Execute backtest and render results into the Streamlit UI."""
    try:
        weights = parse_weight_input(weight_text)
    except ValueError as exc:
        st.error(f"Invalid weights: {exc}")
        return

    # Validate tickers
    missing = set(weights) - set(prices.columns)
    if missing:
        st.error(f"Tickers not in price data: {sorted(missing)}")
        return

    # Filter prices to only tickers in weights
    prices_filtered = prices[list(weights.keys())].dropna()

    backtester = HistoricalBacktester(
        prices=prices_filtered,
        target_weights=weights,
        initial_capital=initial_capital,
        rebalance_freq=STRATEGY_FREQ.get(strategy),
        abs_band=abs_band,
        rel_band=rel_band,
        fee_per_trade=fee_per_trade,
        slippage_pct=slippage_pct,
    )

    with st.spinner("Running backtest..."):
        result = backtester.run()

    port_returns = result.portfolio_value.pct_change().dropna()
    cagr_val = metrics.cagr(result.portfolio_value)
    mdd_val = metrics.maximum_drawdown(result.portfolio_value)
    sharpe_val = metrics.sharpe_ratio(port_returns)
    n_rebalances = len(result.rebalance_events)

    # Performance summary
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CAGR", f"{cagr_val:.2%}")
    m2.metric("Max Drawdown", f"{mdd_val:.2%}")
    m3.metric("Sharpe Ratio", f"{sharpe_val:.2f}")
    m4.metric("Rebalances", str(n_rebalances))

    # Normalise portfolio index to tz-naive so Plotly can mix it with the
    # benchmark trace (also tz-naive) and vline timestamps without errors.
    port_index = result.portfolio_value.index
    if hasattr(port_index, "tz") and port_index.tz is not None:
        port_index = port_index.tz_localize(None)

    # Equity curve
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=port_index,
            y=result.portfolio_value.values,
            name="Portfolio",
            line=dict(color="#2196F3"),
        )
    )

    # Benchmark overlay
    if benchmark_ticker:
        try:
            fetcher = YFinancePriceFetcher()
            start = prices_filtered.index.min().strftime("%Y-%m-%d")
            end = prices_filtered.index.max().strftime("%Y-%m-%d")
            bm_data = fetcher.fetch_history(
                benchmark_ticker,
                period="max",
                interval="1d",
                start=start,
                end=end,
            )
            if not bm_data.empty:
                bm_close = bm_data["Close"].dropna()
                bm_close.index = (
                    bm_close.index.tz_localize(None)
                    if bm_close.index.tz
                    else bm_close.index
                )
                if bm_close.empty or bm_close.iloc[0] == 0:
                    raise ValueError("Benchmark first price is zero or missing.")
                bm_norm = initial_capital * (bm_close / bm_close.iloc[0])
                fig.add_trace(
                    go.Scatter(
                        x=bm_norm.index,
                        y=bm_norm.values,
                        name=f"Benchmark ({benchmark_ticker})",
                        line=dict(color="#FF9800", dash="dash"),
                    )
                )
        except Exception:
            st.warning(
                f"Could not fetch benchmark data for {benchmark_ticker}."
            )

    # Rebalance markers (strip tz to match the tz-naive portfolio index)
    for dt in result.rebalance_events:
        dt_naive = dt.tz_localize(None) if dt.tzinfo else dt
        fig.add_vline(
            x=dt_naive,
            line_width=1,
            line_dash="dot",
            line_color="rgba(150,150,150,0.5)",
        )

    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickformat="$,.0f",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Weight drift
    if not result.historical_weights.empty:
        st.subheader("Weight Drift Over Time")
        fig2 = px.area(
            result.historical_weights,
            title="Asset Weight Allocation Over Time",
            labels={"value": "Weight", "index": "Date", "variable": "Ticker"},
        )
        fig2.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    # Download
    st.download_button(
        "Download backtest results (CSV)",
        data=result.portfolio_value.to_frame("value").to_csv().encode("utf-8"),
        file_name="backtest_portfolio_values.csv",
        mime="text/csv",
    )
