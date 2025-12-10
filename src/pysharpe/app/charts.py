"""Chart rendering helpers for the Streamlit app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pysharpe.app.analytics import MetricResults
from pysharpe.optimization import PortfolioWeights

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


__all__ = [
    "plot_cumulative_returns",
    "plot_weights",
    "render_metrics_table",
]
