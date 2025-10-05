"""Streamlit interface for PySharpe analytics."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier

from pysharpe import metrics
from pysharpe.optimization import PortfolioWeights
from pysharpe.visualization import simulate_dca


@dataclass
class MetricResults:
    returns: pd.DataFrame
    expected: pd.Series
    volatility: pd.Series
    sharpe: pd.Series


@st.cache_data(show_spinner=False)
def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for given tickers and flatten columns safely."""

    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        group_by="ticker",
        auto_adjust=False,
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            df = df["Adj Close"]
        elif "Close" in df.columns.get_level_values(0):
            df = df["Close"]
        else:
            df.columns = ["_".join(col).strip() for col in df.columns.values]
            df = df.select_dtypes("number")
    else:
        df.columns = pd.io.parsers.ParserBase({"names": df.columns})._maybe_dedup_names(df.columns)
        df = df.select_dtypes("number")

    if df.empty:
        return pd.DataFrame()

    df = df.dropna(how="all", axis=1)
    if df.empty:
        return df

    df = df.ffill().bfill()
    df.columns = pd.Index([str(c).strip() for c in df.columns], name="Ticker")
    return df


@st.cache_data(show_spinner=False)
def gather_metadata(tickers: list[str]) -> pd.DataFrame:
    """Fetch ticker metadata (name, exchange, currency)."""

    records: dict[str, dict[str, str]] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
        records[ticker] = {
            "name": info.get("shortName", "Unknown"),
            "exchange": info.get("exchange", "Unknown"),
            "currency": info.get("currency", "N/A"),
        } if info else {
            "name": "Lookup failed",
            "exchange": "-",
            "currency": "-",
        }

    if not records:
        return pd.DataFrame(columns=["name", "exchange", "currency"])
    metadata = pd.DataFrame.from_dict(records, orient="index")
    metadata.index.name = "Ticker"
    return metadata


@st.cache_data(show_spinner=False)
def load_preview_data(tickers: list[str], end_date: dt.date) -> pd.DataFrame:
    """Return a 60-day preview containing close prices and volumes."""

    if not tickers:
        return pd.DataFrame()

    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    start_ts = end_ts - pd.Timedelta(days=90)
    raw = yf.download(
        tickers,
        start=start_ts.to_pydatetime(),
        end=end_ts.to_pydatetime(),
        progress=False,
        group_by="ticker",
        auto_adjust=False,
    )

    if raw.empty:
        return pd.DataFrame()

    closes: pd.DataFrame
    volumes: pd.DataFrame | None = None

    if isinstance(raw.columns, pd.MultiIndex):
        levels = raw.columns.get_level_values(0)
        if "Adj Close" in levels:
            closes = raw["Adj Close"]
        elif "Close" in levels:
            closes = raw["Close"]
        else:
            closes = raw.select_dtypes("number")
        if "Volume" in levels:
            volumes = raw["Volume"]
    else:
        closes = raw.select_dtypes("number")
        volume_cols = [col for col in raw.columns if "volume" in str(col).lower()]
        volumes = raw[volume_cols] if volume_cols else None

    if isinstance(closes, pd.Series):
        closes = closes.to_frame(name=str(tickers[0]))
    closes = closes.dropna(how="all", axis=1)
    closes = closes.ffill().bfill()
    closes.columns = [str(col).strip() for col in closes.columns]

    combined = closes
    if volumes is not None:
        if isinstance(volumes, pd.Series):
            volumes = volumes.to_frame(name=str(tickers[0]))
        volumes = volumes.dropna(how="all", axis=1)
        if not volumes.empty:
            volumes = volumes.ffill().bfill()
            volumes.columns = [f"{col} Volume" for col in volumes.columns]
            combined = pd.concat([closes, volumes], axis=1)

    return combined.tail(60)


def compute_metrics(price_frame: pd.DataFrame) -> MetricResults:
    """Compute metrics using PySharpe helpers and ensure aligned indices."""

    if price_frame.empty:
        raise ValueError("Price data is empty; please adjust tickers, dates, or upload a richer dataset.")

    returns = price_frame.pct_change().dropna(how="all")
    expected = metrics.expected_return(returns)
    volatility = metrics.annualize_volatility(returns)
    sharpe = metrics.sharpe_ratio(returns)

    # Align indices explicitly for downstream joins
    idx = expected.index
    expected = expected.loc[idx]
    volatility = volatility.loc[idx]
    sharpe = sharpe.loc[idx]

    return MetricResults(returns=returns, expected=expected, volatility=volatility, sharpe=sharpe)


def optimise_weights(metrics_result: MetricResults) -> PortfolioWeights | None:
    """Optimise portfolio weights using EfficientFrontier."""

    if metrics_result.returns.empty:
        return None

    mu = metrics_result.expected
    cov = metrics_result.returns.cov() * 252

    try:
        frontier = EfficientFrontier(mu, cov)
        weights = frontier.max_sharpe()
        cleaned = frontier.clean_weights()
    except Exception as exc:  # pragma: no cover - shown in UI
        st.warning(f"Optimisation failed: {exc}")
        return None

    return PortfolioWeights(cleaned)


def plot_cumulative_returns(returns: pd.DataFrame) -> None:
    if returns.empty:
        st.info("No return series available to plot.")
        return
    cumulative = (1 + returns).cumprod()
    try:
        st.line_chart(cumulative, height=320)
    except Exception as exc:  # pragma: no cover - visual guardrail
        st.warning(f"Unable to render returns chart: {exc}")


def plot_weights(weights: PortfolioWeights) -> None:
    if not weights or not weights.allocations:
        st.info("No positive weights to display.")
        return
    series = pd.Series(weights.allocations).sort_values(ascending=False)
    try:
        st.bar_chart(series, height=320)
    except Exception as exc:  # pragma: no cover
        st.warning(f"Unable to render weights chart: {exc}")


def render_metrics_table(metrics_result: MetricResults) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "expected_return": metrics_result.expected,
            "annual_volatility": metrics_result.volatility,
            "sharpe_ratio": metrics_result.sharpe,
        }
    )
    st.dataframe(summary.style.format({
        "expected_return": "{:.4f}",
        "annual_volatility": "{:.4f}",
        "sharpe_ratio": "{:.4f}",
    }))
    return summary


def render_dca_projection(
    months: int,
    initial: float,
    monthly: float,
    rate: float,
) -> pd.DataFrame:
    projection = simulate_dca(
        months=months,
        initial_investment=initial,
        monthly_contribution=monthly,
        annual_return_rate=rate,
    )
    df = pd.DataFrame(
        {
            "Months": projection.months,
            "Balance": projection.balances,
            "Contributions": projection.contributions,
        }
    )
    st.line_chart(df.set_index("Months"), height=320)
    st.metric("Final Balance", f"${projection.final_balance():,.2f}")
    st.metric("Total Contributions", f"${projection.final_contribution():,.2f}")
    return df


def sidebar_controls() -> dict[str, object]:
    st.sidebar.header("Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

    today = pd.Timestamp.today().normalize()
    default_start = (today - pd.Timedelta(days=365)).date()
    default_end = today.date()

    start_date = st.sidebar.date_input("Start", default_start)
    end_date = st.sidebar.date_input("End", default_end)
    if end_date < start_date:
        st.sidebar.error("End date must be on or after the start date.")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        df = df.sort_index().dropna(how="all")
        df = df.loc[str(start_date):str(end_date)] if not df.empty else df
        tickers = tuple(str(col) for col in df.columns)
        data_source = "upload"
        preview = df.tail(60)
        metadata = pd.DataFrame(
            {
                ticker: {"name": "Provided via CSV", "exchange": "-", "currency": "-"}
                for ticker in tickers
            }
        ).T if tickers else pd.DataFrame()
        if not metadata.empty:
            metadata.index.name = "Ticker"
    else:
        tickers_raw = st.sidebar.text_input("Tickers (comma-separated)", value="AAPL,MSFT,GOOGL")
        tickers_list = [ticker.strip().upper() for ticker in tickers_raw.split(",") if ticker.strip()]
        tickers = tuple(sorted(set(tickers_list)))
        df = load_prices(
            list(tickers),
            start=start_date.isoformat(),
            end=(end_date + dt.timedelta(days=1)).isoformat(),
        )
        data_source = "download"
        preview = load_preview_data(list(tickers), end_date)
        metadata = gather_metadata(list(tickers)) if tickers else pd.DataFrame()
        if not metadata.empty:
            metadata.index.name = "Ticker"

    st.sidebar.header("DCA Settings")
    dca_initial = st.sidebar.number_input("Initial Investment", min_value=0.0, value=1000.0, step=100.0)
    dca_monthly = st.sidebar.number_input("Monthly Contribution", min_value=0.0, value=250.0, step=25.0)
    dca_months = st.sidebar.slider("Months", min_value=12, max_value=600, value=240, step=12)
    dca_rate = st.sidebar.slider("Annual Return Rate", min_value=-0.5, max_value=0.5, value=0.08, step=0.01)

    return {
        "data": df,
        "tickers": tickers,
        "preview": preview,
        "metadata": metadata,
        "start": start_date,
        "end": end_date,
        "source": data_source,
        "dca_initial": dca_initial,
        "dca_monthly": dca_monthly,
        "dca_months": dca_months,
        "dca_rate": dca_rate,
    }


def main() -> None:
    st.set_page_config(page_title="PySharpe Analytics", layout="wide")
    controls = sidebar_controls()

    st.title("PySharpe Interactive Dashboard")
    st.write(
        "Download market data, evaluate portfolio metrics, run optimisations, and simulate dollar-cost averaging from a single interface."
    )

    prices: pd.DataFrame = controls["data"]

    if prices.empty:
        st.warning("No valid price data available for the selected tickers or file. Please adjust your inputs.")
        return

    st.subheader("Price Preview")
    preview = controls.get("preview", pd.DataFrame())
    if preview.empty:
        preview = prices.tail(60)
    if preview.columns.duplicated().any():
        preview = preview.loc[:, ~preview.columns.duplicated()]
    st.dataframe(preview.tail(60).style.format("{:.2f}"))

    st.subheader("Ticker Verification")
    metadata = controls.get("metadata", pd.DataFrame())
    if metadata.empty:
        st.info("No ticker metadata available for the current selection.")
    else:
        st.dataframe(metadata)

    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    weights_placeholder = st.empty()

    if st.button("Compute Metrics", type="primary"):
        metrics_result = compute_metrics(prices)
        with metrics_placeholder.container():
            summary = render_metrics_table(metrics_result)
            st.download_button(
                "Download Metrics CSV",
                data=summary.to_csv().encode("utf-8"),
                file_name="pysharpe_metrics.csv",
                mime="text/csv",
            )
        with chart_placeholder.container():
            st.subheader("Cumulative Returns")
            plot_cumulative_returns(metrics_result.returns)
        weights_placeholder.empty()

    if st.button("Optimise Portfolio"):
        metrics_result = compute_metrics(prices)
        weights = optimise_weights(metrics_result)
        if weights:
            with weights_placeholder.container():
                st.subheader("Portfolio Weights")
                plot_weights(weights)
                weight_series = pd.Series(weights.allocations, name="weight")
                st.download_button(
                    "Download Weights CSV",
                    data=weight_series.to_csv().encode("utf-8"),
                    file_name="pysharpe_weights.csv",
                    mime="text/csv",
                )

    st.subheader("Dollar-Cost Averaging Simulation")
    dca_df = render_dca_projection(
        controls["dca_months"],
        controls["dca_initial"],
        controls["dca_monthly"],
        controls["dca_rate"],
    )
    st.download_button(
        "Download DCA Projection CSV",
        data=dca_df.to_csv(index=False).encode("utf-8"),
        file_name="pysharpe_dca_projection.csv",
        mime="text/csv",
    )


if __name__ == "__main__":  # pragma: no cover - manual execution only
    main()
