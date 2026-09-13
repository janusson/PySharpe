"""Streamlit interface for PySharpe analytics.

Layout follows a quantitative workflow:

- Sidebar: global inputs (tickers, date range, portfolio weights).
- Tab 1: Portfolio Metrics & Comparison (vs. the 1/N equal-weight baseline).
- Tab 2: Efficient Frontier (optimizer output and optimal weights).
- Tab 3: DCA Simulation (fully decoupled from the optimization pipeline).
- Tab 4: Raw Data & Logs (download summaries and diagnostic tables).
"""

from __future__ import annotations

from typing import cast

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
    EQUAL_WEIGHT_NAME,
    equal_weight_allocation,
    plot_weight_pies,
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
    compute_overlapping_date_range,
    gather_metadata,
    load_preview_data,
    select_price_data,
)
from pysharpe.app.dca import render_dca_projection as _render_dca_projection
from pysharpe.app.rebalance_ui import render_execution_tab
from pysharpe.visualization import simulate_dca  # noqa: F401 - re-exported for tests

#: Default sidebar tickers — broad-market, CAD-denominated index ETFs.
DEFAULT_TICKERS = "VFV.TO, VCN.TO, VDY.TO, QQC.TO"

_SS_DATA = "data_state"
_SS_ANALYSIS = "analysis_state"
_SS_WEIGHTS = "custom_weights"

_prepare_weight_chart_data = _charts._prepare_weight_chart_data  # noqa: F401

_select_price_data = select_price_data

# ---------------------------------------------------------------------------
# Compatibility proxies (kept so tests can patch the Streamlit stub)
# ---------------------------------------------------------------------------


def load_prices(
    tickers: list[str], start: str | None = None, end: str | None = None
) -> PortfolioData:
    """Load prices while honouring monkeypatched cache loaders.

    Pass ``None`` for both boundaries to fetch the full available history
    (``period="max"``) and derive the UI date boundaries afterwards.
    """

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


def render_metrics_table(metrics_result, portfolios=None):
    """Proxy to chart helper ensuring the patched Streamlit stub is used."""

    _charts.st = st
    _charts.alt = alt
    return _charts.render_metrics_table(metrics_result, portfolios=portfolios)


def render_dca_projection(months: int, initial: float, monthly: float, rate: float):
    """Render the DCA projection while honouring patched Streamlit hooks."""

    _dca.st = st
    _dca.simulate_dca = simulate_dca
    return _render_dca_projection(months, initial, monthly, rate)


def _sync_chart_stub() -> None:
    """Point chart/DCA helpers at the active Streamlit module (or test stub)."""

    _charts.st = st
    _charts.alt = alt
    _dca.st = st
    _dca.simulate_dca = simulate_dca


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


def _parse_tickers(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated ticker string into a sorted, deduplicated tuple."""

    return tuple(sorted({t.strip().upper() for t in raw.split(",") if t.strip()}))


def _normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    """Normalize raw slider weights to sum to 1 (1/N fallback on zero sum)."""

    total = sum(float(value) for value in raw.values())
    if total > 0:
        return {ticker: float(value) / total for ticker, value in raw.items()}
    return equal_weight_allocation(list(raw))


def _slice_prices(full: pd.DataFrame, start: object, end: object) -> pd.DataFrame:
    """Slice a full-history price frame to the inclusive [start, end] window."""

    if full.empty:
        return full
    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None
    out = full
    if start_ts is not None:
        out = out.loc[out.index >= start_ts]
    if end_ts is not None:
        out = out.loc[out.index <= end_ts]
    return out


def _analysis_signature(controls: dict[str, object]) -> tuple[object, ...]:
    """Build the cache signature for the expensive analysis pipeline."""

    weights = cast("dict[str, float]", controls.get("custom_weights", {}) or {})
    return (
        controls.get("ticker_sig", ""),
        str(controls.get("start", "")),
        str(controls.get("end", "")),
        tuple(sorted(weights.items())),
    )


def _get_cached_analysis(controls: dict[str, object]) -> dict | None:
    """Return cached analysis results when the input signature still matches."""

    entry = st.session_state.get(_SS_ANALYSIS)
    if entry is None or entry.get("sig") != _analysis_signature(controls):
        return None
    return cast(dict, entry["results"])


def _ensure_analysis(controls: dict[str, object]) -> dict | None:
    """Run metrics + optimisation once and cache the results by signature."""

    price_data = cast(pd.DataFrame, controls.get("price_data", pd.DataFrame()))
    if price_data.empty or len(price_data.columns) < 2:
        st.warning("Add at least 2 tickers with price history to run analytics.")
        return None

    cached = _get_cached_analysis(controls)
    if cached is not None:
        return cached

    try:
        metrics_result = compute_metrics(price_data)
    except ValueError as exc:
        st.error(str(exc))
        return None

    custom_weights = cast("dict[str, float]", controls.get("custom_weights", {}) or {})
    with st.spinner(
        "Computing metrics, running optimisation, and generating frontier..."
    ):
        try:
            cached = run_full_analysis(price_data, custom_weights)
        except RuntimeError as exc:
            st.error(f"Optimization Failed: {exc}")
            st.stop()
    cached["metrics_result"] = metrics_result
    st.session_state[_SS_ANALYSIS] = {
        "sig": _analysis_signature(controls),
        "results": cached,
    }
    return cached


# ---------------------------------------------------------------------------
# Data-state builders (fetch-first with dynamic date boundaries)
# ---------------------------------------------------------------------------


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


def _build_upload_state(uploaded_file) -> tuple[str, dict[str, object]]:
    """Build the session data state from an uploaded CSV price file."""

    price_frame = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    price_frame = price_frame.sort_index().dropna(how="all")
    tickers = tuple(str(col) for col in price_frame.columns)
    if not tickers:
        st.warning("Uploaded file contains no ticker columns.")
        st.stop()
    overlap = compute_overlapping_date_range(price_frame)
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
    sig = ",".join(tickers)
    return sig, {
        "ticker_sig": sig,
        "source": "upload",
        "tickers": tickers,
        "portfolio_data": portfolio_data,
        "full_prices": price_frame,
        "overlap": overlap,
        "preview": price_frame.tail(60),
        "metadata": _build_metadata_from_upload(tickers),
        "download_summary": None,
    }


def _build_download_state(tickers: tuple[str, ...]) -> tuple[str, dict[str, object]]:
    """Fetch full price history and derive the overlapping date range.

    The full history (``period="max"``) is downloaded first so the maximum
    overlapping window across all tickers can be computed and used as the
    default UI date boundaries.
    """

    with st.spinner(f"Downloading full price history for {', '.join(tickers)}..."):
        try:
            portfolio_data = load_prices(list(tickers), None, None)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
    price_frame = portfolio_data.prices
    if price_frame.empty:
        st.error(
            "No valid price data retrieved. Please adjust the tickers and try again."
        )
        st.stop()

    overlap = compute_overlapping_date_range(price_frame)
    preview_end = overlap[1].date() if overlap else pd.Timestamp.today().date()
    preview = load_preview_data(list(tickers), preview_end)
    metadata = gather_metadata(list(tickers))
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
        "overlap_start": overlap[0].date().isoformat() if overlap else None,
        "overlap_end": overlap[1].date().isoformat() if overlap else None,
    }
    sig = ",".join(tickers)
    return sig, {
        "ticker_sig": sig,
        "source": "download",
        "tickers": tickers,
        "portfolio_data": portfolio_data,
        "full_prices": price_frame,
        "overlap": overlap,
        "preview": preview,
        "metadata": metadata,
        "download_summary": download_summary,
    }


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------


def sidebar_controls() -> dict[str, object]:
    """Compose global sidebar widgets (tickers, dates, weights).

    The price history is fetched first; the maximum overlapping date range
    across the selected tickers becomes the default date boundaries.  Custom
    weights default to a strict 1/N allocation and reset whenever the ticker
    set changes (widget keys are derived from the ticker signature so no
    stale slider state can leak between selections).
    """

    st.sidebar.header("Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

    state = st.session_state.get(_SS_DATA)

    if uploaded_file is not None:
        sig, state = _build_upload_state(uploaded_file)
        st.session_state[_SS_DATA] = state
    else:
        tickers_raw = st.sidebar.text_input(
            "Tickers (comma-separated)", value=DEFAULT_TICKERS
        )
        tickers = _parse_tickers(tickers_raw)
        if not tickers:
            st.warning("Please enter at least one valid ticker symbol.")
            st.stop()
        sig = ",".join(tickers)
        if (
            state is None
            or state.get("ticker_sig") != sig
            or state.get("source") != "download"
        ):
            sig, state = _build_download_state(tickers)
            st.session_state[_SS_DATA] = state

    full_prices = cast(pd.DataFrame, state["full_prices"])
    tickers = cast("tuple[str, ...]", state["tickers"])
    overlap = cast("tuple[pd.Timestamp, pd.Timestamp] | None", state.get("overlap"))

    # -- Date range: defaults are the maximum overlapping range -------------
    st.sidebar.header("Date Range")
    default_start = (
        overlap[0] if overlap else pd.Timestamp.today() - pd.Timedelta(days=365)
    )
    default_end = overlap[1] if overlap else pd.Timestamp.today()
    st.sidebar.caption("Defaults are the maximum overlapping range across tickers.")
    start_date = st.sidebar.date_input(  # type: ignore[arg-type]
        "Start", default_start.date(), key=f"start_{sig}"
    )
    end_date = st.sidebar.date_input(  # type: ignore[arg-type]
        "End", default_end.date(), key=f"end_{sig}"
    )
    if end_date < start_date:
        st.sidebar.error("End date must be on or after the start date.")

    price_frame = _slice_prices(full_prices, start_date, end_date)
    if price_frame.empty:
        st.sidebar.warning(
            "The selected date range contains no data; showing the full "
            "overlapping range instead."
        )
        price_frame = full_prices

    numeric_df = (
        price_frame.select_dtypes("number") if not price_frame.empty else pd.DataFrame()
    )
    price_data = select_price_data(numeric_df)

    # -- Portfolio weights: strict 1/N defaults, reset on ticker changes ----
    st.sidebar.header("Portfolio Weights")
    assets = list(price_data.columns)
    equal_weights = equal_weight_allocation(assets)
    stored_weights = cast(
        "dict[str, float]",
        st.session_state.get(_SS_WEIGHTS, dict(equal_weights)),
    )
    if not stored_weights or set(stored_weights) != set(assets):
        stored_weights = dict(equal_weights)
    raw_weights: dict[str, float] = {}
    for asset in assets:
        default_weight = float(stored_weights.get(asset, equal_weights.get(asset, 0.0)))
        raw_weights[asset] = st.sidebar.slider(
            f"Weight: {asset}",
            0.0,
            1.0,
            default_weight,
            step=0.01,
            key=f"weight_{sig}_{asset}",
        )
    custom_weights = _normalize_weights(raw_weights)
    st.session_state[_SS_WEIGHTS] = custom_weights
    st.sidebar.caption("Weights are normalized to sum to 100%.")
    for ticker, weight in custom_weights.items():
        st.sidebar.text(f"{ticker}: {weight:.2%}")

    return {
        "data": price_frame,
        "portfolio_data": state["portfolio_data"],
        "price_data": price_data,
        "custom_weights": custom_weights,
        "equal_weights": equal_weights,
        "tickers": tickers,
        "preview": state.get("preview", pd.DataFrame()),
        "metadata": state.get("metadata", pd.DataFrame()),
        "start": start_date,
        "end": end_date,
        "overlap": overlap,
        "source": state["source"],
        "download_summary": state.get("download_summary"),
        "ticker_sig": sig,
    }


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _render_raw_data_tab(controls: dict[str, object]) -> None:
    """Tab 4: Raw Data & Logs — download summaries and diagnostic tables."""

    st.caption(
        "Diagnostic information, file paths, and unfiltered data tables. "
        "Use this tab to inspect raw downloads or troubleshoot issues."
    )
    overlap = cast("tuple[pd.Timestamp, pd.Timestamp] | None", controls.get("overlap"))
    if overlap:
        st.success(
            f"Overlapping date range across tickers: "
            f"{overlap[0].date()} → {overlap[1].date()}"
        )

    download_summary = cast(
        "dict[str, object] | None", controls.get("download_summary")
    )
    if download_summary:
        with st.expander("Download Summary", expanded=False):
            ds_tickers = cast("tuple[str, ...]", download_summary["tickers"])
            tickers_display = ", ".join(ds_tickers)
            if download_summary.get("used_cache"):
                st.info(f"Using cached price data for: {tickers_display}")
            else:
                st.success(f"Downloaded {len(ds_tickers)} tickers: {tickers_display}")
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
                    "overlap_start": [download_summary.get("overlap_start")],
                    "overlap_end": [download_summary.get("overlap_end")],
                    "rows": [download_summary.get("rows")],
                    "tickers": [len(ds_tickers)],
                },
                index=["Portfolio"],
            )
            st.dataframe(stats_frame)
            portfolio_data = cast(
                "PortfolioData | None", controls.get("portfolio_data")
            )
            if portfolio_data and not portfolio_data.collated.empty:
                st.markdown("**Collated Portfolio Preview**")
                st.dataframe(portfolio_data.collated.head().style.format("{:.2f}"))

    with st.expander("Price Preview (last 60 rows)", expanded=False):
        price_data = cast(pd.DataFrame, controls.get("price_data", pd.DataFrame()))
        preview = cast(pd.DataFrame, controls.get("preview", pd.DataFrame()))
        if preview.empty:
            preview = cast(pd.DataFrame, controls["data"]).tail(60)
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


def _render_metrics_tab(controls: dict[str, object]) -> None:
    """Tab 1: Portfolio Metrics & Comparison vs. the 1/N baseline."""

    price_data = cast(pd.DataFrame, controls["price_data"])
    custom_weights = cast("dict[str, float]", controls.get("custom_weights", {}) or {})

    run_col, _ = st.columns([1, 1])
    with run_col:
        if st.button("Run Analytics & Optimization", type="primary", width="stretch"):
            _ensure_analysis(controls)

    cached = _get_cached_analysis(controls)
    if cached is None:
        st.info(
            "Click 'Run Analytics & Optimization' to compute portfolio metrics "
            "and compare against the equal-weight (1/N) baseline."
        )
    else:
        metrics_result = cached["metrics_result"]
        portfolios = {
            EQUAL_WEIGHT_NAME: cached.get("equal_port"),
            "Custom Mix": cached.get("user_port"),
            "PySharpe Optimized": cached.get("opt_result"),
        }
        st.subheader("Portfolio Metrics")
        summary = render_metrics_table(metrics_result, portfolios=portfolios)
        st.download_button(
            "Download Metrics CSV",
            data=summary.to_csv().encode("utf-8"),
            file_name="pysharpe_metrics.csv",
            mime="text/csv",
        )

        st.subheader("Cumulative Returns")
        plot_cumulative_returns(price_data)

        st.subheader("Portfolio Weights")
        plot_weight_pies(
            {
                EQUAL_WEIGHT_NAME: dict(
                    equal_weight_allocation(list(price_data.columns))
                ),
                "Custom Mix": dict(custom_weights),
                "PySharpe Optimized": dict(cached["opt_result"].weights.allocations),
            }
        )

        st.subheader("Performance Comparison")
        render_performance_comparison(price_data, custom_weights, cached=cached)

    # Preserved secondary workflows (collapsed by default).
    with st.expander("Portfolio Backtesting", expanded=False):
        render_backtest_tab(cast(pd.DataFrame, controls["data"]))
    with st.expander("Execution & Rebalancing", expanded=False):
        render_execution_tab(price_data, default_cash=250.0)


def _render_frontier_tab(controls: dict[str, object]) -> None:
    """Tab 2: Efficient Frontier — optimizer output and optimal weights."""

    price_data = cast(pd.DataFrame, controls["price_data"])
    custom_weights = cast("dict[str, float]", controls.get("custom_weights", {}) or {})

    st.subheader("Efficient Frontier")
    if price_data.empty or len(price_data.columns) < 2:
        st.info(
            "Add at least 2 tickers with price history to view the efficient frontier."
        )
        return

    run_col, _ = st.columns([1, 1])
    with run_col:
        if st.button("Run Optimization", type="primary", width="stretch"):
            _ensure_analysis(controls)

    cached = _get_cached_analysis(controls)
    if cached is None:
        st.info(
            "Click 'Run Optimization' to solve for the maximum-Sharpe "
            "portfolio and render the frontier."
        )
        return

    render_frontier_plot(price_data, custom_weights, cached=cached)

    opt_result = cached["opt_result"]
    st.subheader("Optimal Weights")
    plot_weights(opt_result.weights)
    weight_series = pd.Series(opt_result.weights.allocations, name="weight")
    st.dataframe(
        pd.DataFrame({"Ticker": weight_series.index, "Weight": weight_series.values})
    )
    st.download_button(
        "Download Weights CSV",
        data=weight_series.to_csv().encode("utf-8"),
        file_name="pysharpe_weights.csv",
        mime="text/csv",
    )


def _render_dca_tab() -> None:
    """Tab 3: DCA Simulation — fully decoupled from the optimization view."""

    st.subheader("Dollar-Cost Averaging Simulation")
    st.caption(
        "Independent of the portfolio analytics above: configure a fixed "
        "contribution plan and project the balance path."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        dca_initial = st.number_input(
            "Initial Investment",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            key="dca_initial",
        )
    with col2:
        dca_monthly = st.number_input(
            "Monthly Contribution",
            min_value=0.0,
            value=250.0,
            step=25.0,
            key="dca_monthly",
        )
    with col3:
        dca_months = st.slider(
            "Months",
            min_value=12,
            max_value=600,
            value=240,
            step=12,
            key="dca_months",
        )
    rate_col, _ = st.columns([1, 1])
    with rate_col:
        dca_rate = st.slider(
            "Annual Return Rate",
            min_value=-0.5,
            max_value=0.5,
            value=0.08,
            step=0.01,
            key="dca_rate",
        )
    dca_df = render_dca_projection(
        int(dca_months), float(dca_initial), float(dca_monthly), float(dca_rate)
    )
    st.download_button(
        "Download DCA Projection CSV",
        data=dca_df.to_csv(index=False).encode("utf-8"),
        file_name="pysharpe_dca_projection.csv",
        mime="text/csv",
    )


def main() -> None:
    """Launch the Streamlit dashboard."""

    st.set_page_config(page_title="PySharpe Analytics", layout="wide")

    # Render the header before the first data fetch so the page is never
    # blank while the full price history downloads.
    st.title("PySharpe Interactive Dashboard")
    st.write(
        "Professional portfolio analytics for broad-market CAD ETFs: metrics, "
        "mean-variance optimization, DCA simulation, and raw diagnostics."
    )

    _sync_chart_stub()
    controls = sidebar_controls()

    price_data = cast(pd.DataFrame, controls.get("price_data", pd.DataFrame()))
    prices = cast(pd.DataFrame, controls["data"])
    if prices.empty or price_data.empty:
        st.warning(
            "No valid price data available for the selected tickers or file. "
            "Please adjust your inputs."
        )
        return

    # ===================================================================
    # Tab layout -- exactly four tabs
    # ===================================================================
    tab_metrics, tab_frontier, tab_dca, tab_raw = st.tabs(
        [
            "\U0001f4ca Portfolio Metrics & Comparison",
            "\U0001f4c8 Efficient Frontier",
            "\U0001f4b0 DCA Simulation",
            "\U0001f6e0\ufe0f Raw Data & Logs",
        ]
    )

    with tab_raw:
        _render_raw_data_tab(controls)
    with tab_metrics:
        _render_metrics_tab(controls)
    with tab_frontier:
        _render_frontier_tab(controls)
    with tab_dca:
        _render_dca_tab()


if __name__ == "__main__":
    main()
