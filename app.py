"""Streamlit interface for PySharpe analytics."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf  # noqa: F401 - re-exported for tests

try:
    import altair as alt  # noqa: F401 - re-exported for tests
except Exception:  # pragma: no cover - optional dependency
    alt = None  # type: ignore[assignment]

from pypfopt import EfficientFrontier  # noqa: F401 - re-exported for tests

import pysharpe.app.charts as _charts
import pysharpe.app.data as _data
import pysharpe.app.dca as _dca
from pysharpe.app.analytics import (
    MetricResults,  # noqa: F401 - re-exported for tests
    compute_metrics,
)
from pysharpe.app.data import (
    _STREAMLIT_SERVICE,  # noqa: F401 - test visibility
    SETTINGS,
    PortfolioData,
    _clean_numeric_frame,  # noqa: F401 - re-exported for tests
    _load_collated_from_disk,  # noqa: F401 - re-exported for tests
    _make_portfolio_name,  # noqa: F401 - re-exported for tests
    _resolve_field_frame,  # noqa: F401 - re-exported for tests
    gather_metadata,
    load_preview_data,
    select_price_data,
)
from pysharpe.app.dca import render_dca_projection as _render_dca_projection
from pysharpe.optimization import PortfolioWeights
from pysharpe.visualization import simulate_dca  # noqa: F401 - re-exported for tests

_prepare_weight_chart_data = _charts._prepare_weight_chart_data  # noqa: F401

_select_price_data = select_price_data


def load_prices(tickers: list[str], start: str, end: str):
    """Load prices while honouring monkeypatched cache loaders."""

    return _data.load_prices(tickers, start, end, _loader=_load_collated_from_disk)


def plot_cumulative_returns(price_frame: pd.DataFrame) -> None:
    """Proxy to chart helper ensuring the patched Streamlit stub is used."""

    _charts.st = st
    _charts.alt = alt
    _charts.plot_cumulative_returns(price_frame)


def plot_weights(weights) -> None:
    """Proxy to chart helper ensuring the patched Streamlit stub is used."""

    _charts.st = st
    _charts.alt = alt
    _charts.plot_weights(weights)


def render_metrics_table(metrics_result):
    """Proxy to chart helper ensuring the patched Streamlit stub is used."""

    _charts.st = st
    return _charts.render_metrics_table(metrics_result)


def optimise_weights(
    metrics_result: MetricResults, on_warning=None
) -> PortfolioWeights | None:
    """Optimise portfolio weights using the (monkeypatchable) frontier class."""

    if metrics_result.returns.empty:
        return None

    mu = metrics_result.expected
    cov = metrics_result.returns.cov() * 252

    try:
        frontier = EfficientFrontier(mu, cov)
        frontier.max_sharpe()
        cleaned = frontier.clean_weights()
    except Exception as exc:  # pragma: no cover - surfaced in UI/tests
        if on_warning:
            on_warning(f"Optimisation failed: {exc}")
        return None

    return PortfolioWeights(cleaned)


def render_dca_projection(months: int, initial: float, monthly: float, rate: float):
    """Render the DCA projection while honouring patched Streamlit hooks."""

    _dca.st = st
    _dca.simulate_dca = simulate_dca
    return _render_dca_projection(months, initial, monthly, rate)


@st.cache_data(show_spinner=False)
def _build_metadata_from_upload(tickers: tuple[str, ...]) -> pd.DataFrame:
    metadata = pd.DataFrame(
        {
            ticker: {"name": "Provided via CSV", "exchange": "-", "currency": "-"}
            for ticker in tickers
        }
    ).T
    metadata.index.name = "Ticker"
    return metadata


def sidebar_controls() -> dict[str, object]:
    """Compose sidebar widgets and return the resulting state."""

    if "dca_rate_default" not in st.session_state:
        st.session_state["dca_rate_default"] = 0.08
    if "dca_rate_override" not in st.session_state:
        st.session_state["dca_rate_override"] = False
    if "dca_rate_value" not in st.session_state:
        st.session_state["dca_rate_value"] = st.session_state["dca_rate_default"]
    if "dca_rate_pending_reset" not in st.session_state:
        st.session_state["dca_rate_pending_reset"] = False

    st.sidebar.header("Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

    today = pd.Timestamp.today().normalize()
    default_start = (today - pd.Timedelta(days=365)).date()
    default_end = today.date()

    start_date = st.sidebar.date_input("Start", default_start)
    end_date = st.sidebar.date_input("End", default_end)
    if end_date < start_date:
        st.sidebar.error("End date must be on or after the start date.")

    download_summary: dict[str, object] | None = None

    if uploaded_file is not None:
        price_frame = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        price_frame = price_frame.sort_index().dropna(how="all")
        if not price_frame.empty:
            price_frame = price_frame.loc[str(start_date) : str(end_date)]
        tickers = tuple(str(col) for col in price_frame.columns)
        if not tickers:
            st.warning("Uploaded file contains no ticker columns.")
            st.stop()
        data_source = "upload"
        preview = price_frame.tail(60)
        metadata = _build_metadata_from_upload(tickers)
        portfolio_data = PortfolioData(
            tickers=tickers,
            prices=price_frame,
            collated=price_frame,
            price_history_dir=SETTINGS.price_history_dir,
            collated_path=None,
            start=price_frame.index.min() if not price_frame.empty else None,
            end=price_frame.index.max() if not price_frame.empty else None,
            warnings=(),
            used_cache=False,
        )
    else:
        tickers_raw = st.sidebar.text_input(
            "Tickers (comma-separated)", value="AAPL,MSFT,GOOGL"
        )
        tickers_list = [
            ticker.strip().upper()
            for ticker in tickers_raw.split(",")
            if ticker.strip()
        ]
        tickers = tuple(sorted(set(tickers_list)))
        if not tickers:
            st.warning("Please enter at least one valid ticker symbol.")
            st.stop()
        resolved_tickers = list(tickers)
        try:
            portfolio_data = load_prices(
                resolved_tickers,
                start=start_date.isoformat(),
                end=(end_date + dt.timedelta(days=1)).isoformat(),
            )
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        data_source = "download"
        price_frame = portfolio_data.prices
        if price_frame.empty:
            st.error(
                "No valid price data retrieved. "
                "Please adjust the tickers or date range and try again."
            )
            st.stop()
        preview = load_preview_data(resolved_tickers, end_date)
        metadata = (
            gather_metadata(resolved_tickers) if resolved_tickers else pd.DataFrame()
        )
        if not metadata.empty:
            metadata.index.name = "Ticker"
        download_summary = {
            "tickers": tickers,
            "price_history_dir": str(portfolio_data.price_history_dir),
            "collated_path": str(portfolio_data.collated_path)
            if portfolio_data.collated_path
            else None,
            "start": portfolio_data.start.isoformat()
            if isinstance(portfolio_data.start, pd.Timestamp)
            else None,
            "end": portfolio_data.end.isoformat()
            if isinstance(portfolio_data.end, pd.Timestamp)
            else None,
            "rows": portfolio_data.prices.shape[0],
            "columns": portfolio_data.prices.shape[1],
            "warnings": portfolio_data.warnings,
            "used_cache": portfolio_data.used_cache,
        }

    numeric_df = (
        price_frame.select_dtypes("number") if not price_frame.empty else pd.DataFrame()
    )
    price_data = select_price_data(numeric_df)

    st.sidebar.header("DCA Settings")
    dca_initial = st.sidebar.number_input(
        "Initial Investment", min_value=0.0, value=1000.0, step=100.0
    )
    dca_monthly = st.sidebar.number_input(
        "Monthly Contribution", min_value=0.0, value=250.0, step=25.0
    )
    dca_months = st.sidebar.slider(
        "Months", min_value=12, max_value=600, value=240, step=12
    )
    pending_reset = st.session_state.get("dca_rate_pending_reset", False)
    rate_overridden = st.session_state.get("dca_rate_override", False)
    if pending_reset and not rate_overridden:
        st.session_state.pop("dca_rate_slider", None)
        st.session_state["dca_rate_pending_reset"] = False

    default_rate = float(st.session_state.get("dca_rate_default", 0.08))
    dca_rate = st.sidebar.slider(
        "Annual Return Rate",
        min_value=-0.5,
        max_value=0.5,
        value=default_rate,
        step=0.01,
        key="dca_rate_slider",
    )
    st.session_state["dca_rate_value"] = float(dca_rate)
    st.session_state["dca_rate_override"] = not np.isclose(
        st.session_state["dca_rate_value"],
        st.session_state.get("dca_rate_default", default_rate),
    )

    return {
        "data": price_frame,
        "portfolio_data": portfolio_data,
        "price_data": price_data,
        "tickers": tickers,
        "preview": preview,
        "metadata": metadata,
        "start": start_date,
        "end": end_date,
        "source": data_source,
        "dca_initial": dca_initial,
        "dca_monthly": dca_monthly,
        "dca_months": dca_months,
        "dca_rate": float(st.session_state["dca_rate_value"]),
        "download_summary": download_summary,
    }


def main() -> None:
    """Launch the Streamlit dashboard."""

    st.set_page_config(page_title="PySharpe Analytics", layout="wide")
    controls = sidebar_controls()

    st.title("PySharpe Interactive Dashboard")
    st.write(
        "Download market data, evaluate portfolio metrics, run optimisations, and "
        "simulate dollar-cost averaging from a single interface."
    )

    prices: pd.DataFrame = controls["data"]
    price_data: pd.DataFrame = controls.get("price_data", pd.DataFrame())
    portfolio_data: PortfolioData | None = controls.get("portfolio_data")  # type: ignore[assignment]
    download_summary = controls.get("download_summary")

    if prices.empty:
        st.warning(
            "No valid price data available for the selected tickers or file. Please "
            "adjust your inputs."
        )
        return

    if download_summary:
        tickers_display = ", ".join(download_summary["tickers"])
        if download_summary.get("used_cache"):
            st.info(f"Using cached price data for: {tickers_display}")
        else:
            st.success(
                f"Downloaded {len(download_summary['tickers'])} tickers: "
                f"{tickers_display}"
            )
        st.caption(
            f"Price history directory: `{download_summary['price_history_dir']}`"
        )
        if download_summary.get("collated_path"):
            st.caption(f"Collated CSV: `{download_summary['collated_path']}`")
        for warning_message in download_summary.get("warnings", ()) or ():
            st.warning(warning_message)
        stats_frame = pd.DataFrame(
            {
                "start": [download_summary.get("start")],
                "end": [download_summary.get("end")],
                "rows": [download_summary.get("rows")],
                "tickers": [len(download_summary["tickers"])],
            },
            index=["Portfolio"],
        )
        st.dataframe(stats_frame)
        if portfolio_data and not portfolio_data.collated.empty:
            st.subheader("Collated Portfolio Preview")
            st.dataframe(portfolio_data.collated.head().style.format("{:.2f}"))

    st.subheader("Price Preview")
    preview = controls.get("preview", pd.DataFrame())
    if preview.empty:
        preview = prices.tail(60)
    preview = preview.tail(60)
    price_columns = set(price_data.columns)
    preview_columns = [
        column
        for column in preview.columns
        if (column in price_columns)
        or ("close" in str(column).lower())
        or ("volume" in str(column).lower())
    ]
    if preview_columns:
        preview = preview.loc[:, preview_columns]
    if preview.columns.duplicated().any():
        preview = preview.loc[:, ~preview.columns.duplicated()]
    st.dataframe(preview.style.format("{:.2f}"))

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
        if price_data.empty or price_data.select_dtypes("number").empty:
            st.warning("No suitable Close/Adj Close series found for analytics.")
        else:
            metrics_result = compute_metrics(price_data)
            mean_expected = (
                float(metrics_result.expected.mean())
                if not metrics_result.expected.empty
                else np.nan
            )
            if not np.isnan(mean_expected):
                st.session_state["dca_rate_default"] = mean_expected
                if not st.session_state.get("dca_rate_override", False):
                    st.session_state["dca_rate_value"] = mean_expected
                    st.session_state["dca_rate_pending_reset"] = True
                else:
                    st.session_state["dca_rate_pending_reset"] = False
            else:
                st.session_state["dca_rate_pending_reset"] = False
            controls["dca_rate"] = float(
                st.session_state.get(
                    "dca_rate_value", st.session_state.get("dca_rate_default", 0.08)
                )
            )
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
                plot_cumulative_returns(price_data)
            weights_placeholder.empty()

    if st.button("Optimise Portfolio"):
        if price_data.empty or price_data.select_dtypes("number").empty:
            st.warning("Cannot optimise without Close/Adj Close price history.")
        else:
            metrics_result = compute_metrics(price_data)
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
        float(st.session_state.get("dca_rate_value", controls["dca_rate"])),
    )
    st.download_button(
        "Download DCA Projection CSV",
        data=dca_df.to_csv(index=False).encode("utf-8"),
        file_name="pysharpe_dca_projection.csv",
        mime="text/csv",
    )


if __name__ == "__main__":  # pragma: no cover - manual execution only
    main()
