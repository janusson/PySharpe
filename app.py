"""Streamlit interface for PySharpe analytics."""

from __future__ import annotations

import datetime as dt
from typing import cast

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf  # noqa: F401 - re-exported for tests

try:
    import altair as alt  # noqa: F401 - re-exported for tests
except Exception:  # pragma: no cover - optional dependency
    alt = None  # type: ignore[assignment]

import pysharpe.app.charts as _charts
import pysharpe.app.data as _data
import pysharpe.app.dca as _dca
from pysharpe.app.analytics import (
    MetricResults,  # noqa: F401 - re-exported for tests
    compute_metrics,
)
from pysharpe.app.backtest import render_backtest_tab
from pysharpe.app.charts import (
    render_frontier_plot,
    render_performance_comparison,
    run_full_analysis,
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
from pysharpe.app.rebalance_ui import render_execution_tab
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

    start_date = st.sidebar.date_input("Start", default_start)  # type: ignore[arg-type]
    end_date = st.sidebar.date_input("End", default_end)  # type: ignore[arg-type]
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
            start=price_frame.index.min() if not price_frame.empty else None,  # type: ignore[arg-type]
            end=price_frame.index.max() if not price_frame.empty else None,  # type: ignore[arg-type]
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

    st.sidebar.header("Custom Portfolio Weights")
    custom_weights = {}
    if not price_data.empty:
        raw_weights = {}
        assets = list(price_data.columns)
        num_assets = len(assets)
        default_weight = 1.0 / num_assets if num_assets > 0 else 0.0
        for asset in assets:
            raw_weights[asset] = st.sidebar.slider(
                f"Weight: {asset}", 0.0, 1.0, default_weight, step=0.01
            )

        total = sum(raw_weights.values())
        if total > 0:
            custom_weights = {k: v / total for k, v in raw_weights.items()}
        else:
            custom_weights = {k: default_weight for k in assets}

        st.sidebar.caption("Normalized custom weights:")
        for k, v in custom_weights.items():
            st.sidebar.text(f"{k}: {v:.2%}")

    return {
        "data": price_frame,
        "portfolio_data": portfolio_data,
        "price_data": price_data,
        "custom_weights": custom_weights,
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

    prices = cast(pd.DataFrame, controls["data"])
    price_data = cast(pd.DataFrame, controls.get("price_data", pd.DataFrame()))
    portfolio_data = cast("PortfolioData | None", controls.get("portfolio_data"))
    download_summary = cast(
        "dict[str, object] | None", controls.get("download_summary")
    )

    if prices.empty:
        st.warning(
            "No valid price data available for the selected tickers or file. Please "
            "adjust your inputs."
        )
        return

    # ===================================================================
    # Tab layout -- exactly four tabs
    # ===================================================================
    tab_overview, tab_frontier, tab_dca, tab_debug = st.tabs(
        [
            "\U0001f4ca Overview",
            "\U0001f4c8 Efficient Frontier",
            "\U0001f4b0 DCA Simulation",
            "\U0001f6e0\ufe0f Raw Data & Logs",
        ]
    )

    # ===================================================================
    # Tab 4: Raw Data & Logs (all diagnostic output lives here)
    # ===================================================================
    with tab_debug:
        st.caption(
            "Diagnostic information, file paths, and unfiltered data tables. "
            "Use this tab to inspect raw downloads or troubleshoot issues."
        )

        if download_summary:
            with st.expander("Download Summary", expanded=False):
                ds_tickers = cast("tuple[str, ...]", download_summary["tickers"])
                tickers_display = ", ".join(ds_tickers)
                if download_summary.get("used_cache"):
                    st.info(f"Using cached price data for: {tickers_display}")
                else:
                    st.success(
                        f"Downloaded {len(ds_tickers)} tickers: {tickers_display}"
                    )
                st.caption(
                    f"Price history directory: `{download_summary['price_history_dir']}`"
                )
                if download_summary.get("collated_path"):
                    st.caption(f"Collated CSV: `{download_summary['collated_path']}`")
                ds_warnings = cast("tuple", download_summary.get("warnings", ()) or ())
                for warning_message in ds_warnings:
                    st.warning(warning_message)
                stats_frame = pd.DataFrame(
                    {
                        "start": [download_summary.get("start")],
                        "end": [download_summary.get("end")],
                        "rows": [download_summary.get("rows")],
                        "tickers": [len(ds_tickers)],
                    },
                    index=["Portfolio"],
                )
                st.dataframe(stats_frame)
                if portfolio_data and not portfolio_data.collated.empty:
                    st.markdown("**Collated Portfolio Preview**")
                    st.dataframe(portfolio_data.collated.head().style.format("{:.2f}"))

        with st.expander("Price Preview (last 60 rows)", expanded=False):
            preview = cast(pd.DataFrame, controls.get("preview", pd.DataFrame()))
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

        with st.expander("Ticker Metadata", expanded=False):
            metadata = cast(pd.DataFrame, controls.get("metadata", pd.DataFrame()))
            if metadata.empty:
                st.info("No ticker metadata available for the current selection.")
            else:
                st.dataframe(metadata)

    # ===================================================================
    # Tab 1: Overview (charts, metrics, weights, performance comparison)
    # ===================================================================
    with tab_overview:
        # -- Unified Run Analytics & Optimize button -------------------------
        run_col, _ = st.columns([1, 1])
        with run_col:
            if st.button(
                "Run Analytics & Optimization", type="primary", use_container_width=True
            ):
                if price_data.empty or price_data.select_dtypes("number").empty:
                    st.warning(
                        "No suitable Close/Adj Close series found for analytics."
                    )
                else:
                    with st.spinner(
                        "Computing metrics, running optimisation, and generating frontier..."
                    ):
                        # 1) Metrics
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
                                "dca_rate_value",
                                st.session_state.get("dca_rate_default", 0.08),
                            )
                        )

                        # 2) Optimisation + frontier + benchmarks (one shot)
                        try:
                            cached = run_full_analysis(
                                price_data,
                                cast(
                                    "dict[str, float]",
                                    controls.get("custom_weights", {}),
                                ),
                            )
                        except RuntimeError as exc:
                            st.error(f"Optimization Failed: {str(exc)}")
                            st.stop()

                        # 3) Store unified results in session state
                        cached["metrics_result"] = metrics_result
                        st.session_state["opt_results"] = cached

        # -- Conditional rendering (reads strictly from session state) --------
        cached = st.session_state.get("opt_results")
        if cached is not None:
            metrics_result = cached["metrics_result"]
            st.subheader("Portfolio Metrics")
            summary = render_metrics_table(metrics_result)
            st.download_button(
                "Download Metrics CSV",
                data=summary.to_csv().encode("utf-8"),
                file_name="pysharpe_metrics.csv",
                mime="text/csv",
            )
            st.subheader("Cumulative Returns")
            plot_cumulative_returns(price_data)

            opt_result = cached["opt_result"]
            weights = opt_result.weights
            if weights and weights.allocations:
                st.subheader("Portfolio Weights")
                plot_weights(weights)
                weight_series = pd.Series(weights.allocations, name="weight")
                st.download_button(
                    "Download Weights CSV",
                    data=weight_series.to_csv().encode("utf-8"),
                    file_name="pysharpe_weights.csv",
                    mime="text/csv",
                )

            # -- Performance Comparison table (from cache) --------------------
            if not price_data.empty and len(price_data.columns) >= 2:
                st.subheader("Performance Comparison")
                render_performance_comparison(
                    price_data,
                    cast("dict[str, float]", controls.get("custom_weights", {})),
                    cached=cached,
                )

        # -- Backtest (preserved from previous layout) -----------------------
        with st.expander("Portfolio Backtesting", expanded=False):
            render_backtest_tab(prices)

        # -- Execute / Rebalancing (preserved from previous layout) -----------
        with st.expander("Execution & Rebalancing", expanded=False):
            render_execution_tab(
                price_data,
                default_cash=cast(float, controls.get("dca_monthly", 1000.0)),
            )

    # ===================================================================
    # Tab 2: Efficient Frontier
    # ===================================================================
    with tab_frontier:
        st.subheader("Efficient Frontier")
        if not price_data.empty and len(price_data.columns) >= 2:
            cached = st.session_state.get("opt_results")
            if cached is not None:
                render_frontier_plot(
                    price_data,
                    cast("dict[str, float]", controls.get("custom_weights", {})),
                    cached=cached,
                )
            else:
                st.info(
                    "Click 'Run Analytics & Optimization' in the Overview tab "
                    "to generate the efficient frontier."
                )
        else:
            st.info(
                "Add at least 2 tickers with price history to view the "
                "efficient frontier."
            )

    # ===================================================================
    # Tab 3: DCA Simulation (independent of optimization — slider-only)
    # ===================================================================
    with tab_dca:
        st.subheader("Dollar-Cost Averaging Simulation")
        dca_df = render_dca_projection(
            cast(int, controls["dca_months"]),
            cast(float, controls["dca_initial"]),
            cast(float, controls["dca_monthly"]),
            float(
                st.session_state.get(
                    "dca_rate_value", cast(float, controls["dca_rate"])
                )
            ),
        )
        st.download_button(
            "Download DCA Projection CSV",
            data=dca_df.to_csv(index=False).encode("utf-8"),
            file_name="pysharpe_dca_projection.csv",
            mime="text/csv",
        )


if __name__ == "__main__":  # pragma: no cover - manual execution only
    main()
