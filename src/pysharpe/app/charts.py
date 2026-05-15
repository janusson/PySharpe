"""Chart rendering helpers for the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from pysharpe.analysis import CANADIAN_BENCHMARKS, fetch_benchmark_metrics
from pysharpe.app.analytics import MetricResults
from pysharpe.optimization.models import (
    OptimisationPerformance,
    OptimisationResult,
    PortfolioWeights,
)
from pysharpe.optimization.sharpe_optimizer import SharpeOptimizer
from pysharpe.visualization import (
    generate_efficient_frontier,
    plot_portfolio_comparison,
)

try:
    import altair as alt
except ImportError:  # pragma: no cover - visual dependency
    alt = None


def _prepare_weight_chart_data(series: pd.Series) -> pd.DataFrame:
    """Shape weight allocations for charting, keeping only positive weights."""

    if series.empty:
        return pd.DataFrame(columns=["Ticker", "Weight"])

    positive_weights = series[series > 0]
    if positive_weights.empty:
        return pd.DataFrame(columns=["Ticker", "Weight"])

    chart_frame = pd.DataFrame(
        {
            "Ticker": [str(idx) for idx in positive_weights.index],
            "Weight": positive_weights.values,
        }
    )
    return chart_frame.reset_index(drop=True)


def plot_cumulative_returns(price_frame: pd.DataFrame) -> None:
    """Render a cumulative returns line chart for the provided price data."""

    if price_frame.empty:
        st.info("No price data available to plot cumulative returns.")
        return

    returns = price_frame.pct_change().dropna(how="all")
    if returns.empty:
        st.info("Not enough price history to compute cumulative returns.")
        return

    cumulative = (1 + returns).cumprod()
    try:
        st.line_chart(cumulative, height=320, use_container_width=True)
    except Exception as exc:  # pragma: no cover - visual guardrail
        st.warning(f"Unable to render returns chart: {exc}")


def plot_weights(weights: PortfolioWeights) -> None:
    """Display a bar chart of portfolio weights."""

    if not weights or not weights.allocations:
        st.info("No positive weights to display.")
        return
    series = pd.Series(weights.allocations).sort_values(ascending=False)
    series = series[series > 0]
    if series.empty:
        st.info("No positive weights to display.")
        return

    bar_col, pie_col = st.columns(2)

    with bar_col:
        try:
            st.bar_chart(series, height=320)
        except Exception as exc:  # pragma: no cover
            st.warning(f"Unable to render weights chart: {exc}")

    chart_data = _prepare_weight_chart_data(series)
    with pie_col:
        if chart_data.empty:
            st.info("No positive weights available for pie chart.")
        elif alt is None:
            st.info("Install the 'altair' package to view the allocation pie chart.")
        else:
            try:
                donut_chart = (
                    alt.Chart(chart_data)
                    .mark_arc(innerRadius=60)
                    .encode(
                        theta=alt.Theta(field="Weight", type="quantitative"),
                        color=alt.Color(field="Ticker", type="nominal"),
                        tooltip=[
                            alt.Tooltip("Ticker:N", title="Ticker"),
                            alt.Tooltip("Weight:Q", title="Weight", format=".2%"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(donut_chart, use_container_width=True)
            except Exception as exc:  # pragma: no cover
                st.warning(f"Unable to render allocation pie chart: {exc}")


def render_metrics_table(metrics_result: MetricResults) -> pd.DataFrame:
    """Show metric summary table and return the raw DataFrame."""

    summary = pd.DataFrame(
        {
            "expected_return": metrics_result.expected,
            "annual_volatility": metrics_result.volatility,
            "sharpe_ratio": metrics_result.sharpe,
        }
    )
    st.dataframe(
        summary.style.format(
            {
                "expected_return": "{:.4f}",
                "annual_volatility": "{:.4f}",
                "sharpe_ratio": "{:.4f}",
            }
        )
    )
    return summary


def render_frontier_comparison(
    price_data: pd.DataFrame, custom_weights: dict[str, float]
) -> None:
    """Render the Efficient Frontier overlay plot and comparison table."""

    if price_data.empty or len(price_data.columns) < 2:
        return

    try:
        # Prepare Custom Mix Performance
        optimizer = SharpeOptimizer(price_data)
        assets = optimizer.assets
        user_weights_array = np.array(
            [custom_weights.get(asset, 0.0) for asset in assets]
        )

        user_ret, user_vol, user_sharpe = optimizer.calculate_portfolio_performance(
            user_weights_array
        )
        user_perf = OptimisationPerformance(
            expected_return=user_ret,
            volatility=user_vol,
            sharpe_ratio=user_sharpe,
            start_date=str(price_data.index.min().date()),
            end_date=str(price_data.index.max().date()),
        )
        user_port = OptimisationResult(
            name="Custom Mix",
            weights=PortfolioWeights(custom_weights),
            performance=user_perf,
        )

        # Prepare Optimized Portfolio
        opt_result = optimizer.optimize()

        # Fetch Benchmarks
        start_str = str(price_data.index.min().date())
        end_str = str(price_data.index.max().date())
        benchmarks_df = fetch_benchmark_metrics(
            list(CANADIAN_BENCHMARKS.keys()), start_date=start_str, end_date=end_str
        )

        # Generate Frontier Points
        frontier_rets, frontier_vols = generate_efficient_frontier(price_data)

        # Plot
        fig = plot_portfolio_comparison(
            frontier_returns=frontier_rets,
            frontier_vols=frontier_vols,
            user_portfolio=user_port,
            optimized_portfolio=opt_result,
            benchmarks_df=benchmarks_df,
            prices=price_data,
        )
        st.pyplot(fig)

        # Markdown Table Comparison
        st.markdown("### Performance Comparison")

        comp_data = []
        comp_data.append(
            {
                "Portfolio": "Custom Mix",
                "Expected Return": f"{user_perf.expected_return:.2%}",
                "Volatility": f"{user_perf.volatility:.2%}",
                "Sharpe Ratio": f"{user_perf.sharpe_ratio:.2f}",
            }
        )
        comp_data.append(
            {
                "Portfolio": "PySharpe Optimized",
                "Expected Return": f"{opt_result.performance.expected_return:.2%}",
                "Volatility": f"{opt_result.performance.volatility:.2%}",
                "Sharpe Ratio": f"{opt_result.performance.sharpe_ratio:.2f}",
            }
        )

        for _, row in benchmarks_df.iterrows():
            comp_data.append(
                {
                    "Portfolio": f"Benchmark: {row['Ticker']}",
                    "Expected Return": f"{row['Annualized Return']:.2%}",
                    "Volatility": f"{row['Annualized Volatility']:.2%}",
                    "Sharpe Ratio": f"{row['Sharpe Ratio']:.2f}",
                }
            )

        st.table(pd.DataFrame(comp_data))

    except Exception as e:
        st.error(f"Error generating Efficient Frontier plot: {e}")


__all__ = [
    "plot_cumulative_returns",
    "plot_weights",
    "render_metrics_table",
    "render_frontier_comparison",
]
