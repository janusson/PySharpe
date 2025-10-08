"""Streamlit interface for PySharpe analytics."""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier

from pysharpe import metrics
from pysharpe.config import get_settings
from pysharpe.data.collation import CollationService
from pysharpe.data.fetcher import YFinancePriceFetcher
from pysharpe.optimization import PortfolioWeights
from pysharpe.visualization import simulate_dca

try:
    import altair as alt
except ImportError:  # pragma: no cover - visual dependency
    alt = None


@dataclass
class MetricResults:
    """Container holding the output of portfolio metric calculations."""

    returns: pd.DataFrame
    expected: pd.Series
    volatility: pd.Series
    sharpe: pd.Series


@dataclass
class PortfolioData:
    """Container describing a cached portfolio download."""

    tickers: tuple[str, ...]
    prices: pd.DataFrame
    collated: pd.DataFrame
    price_history_dir: Path
    collated_path: Path | None
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    used_cache: bool = False


def _deduplicate_columns(columns: Iterable[object]) -> pd.Index:
    """Return a deduplicated, stringified index preserving order."""

    seen: dict[str, int] = {}
    deduped: list[str] = []
    for column in columns:
        label = str(column).strip()
        suffix = seen.get(label, 0)
        if suffix:
            deduped.append(f"{label}.{suffix}")
        else:
            deduped.append(label)
        seen[label] = suffix + 1
    return pd.Index(deduped)


def _clean_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Forward/backward fill numeric data and normalise labels."""

    if frame.empty:
        return frame
    cleaned = frame.dropna(how="all", axis=1)
    if cleaned.empty:
        return cleaned
    cleaned = cleaned.ffill().bfill()
    cleaned.columns = _deduplicate_columns(cleaned.columns)
    return cleaned


def _resolve_field_frame(raw_data: pd.DataFrame, field_priority: Iterable[str]) -> pd.DataFrame:
    """Extract the first matching field from yfinance output."""

    if raw_data.empty:
        return pd.DataFrame()

    if isinstance(raw_data.columns, pd.MultiIndex):
        level0 = raw_data.columns.get_level_values(0)
        for field in field_priority:
            if field in level0:
                extracted = raw_data[field]
                break
        else:
            extracted = raw_data.select_dtypes("number")
    else:
        extracted = raw_data.select_dtypes("number")
        matching = [
            col
            for col in extracted.columns
            if any(field.lower() in str(col).lower() for field in field_priority)
        ]
        if matching:
            extracted = extracted.loc[:, matching]

    if isinstance(extracted, pd.Series):
        extracted = extracted.to_frame()
    extracted.columns = _deduplicate_columns(extracted.columns)
    return extracted


def _select_price_data(numeric_df: pd.DataFrame) -> pd.DataFrame:
    """Prefer close columns but gracefully fall back to other numeric data."""

    if numeric_df.empty:
        return pd.DataFrame(index=numeric_df.index)

    close_columns = [col for col in numeric_df.columns if "close" in str(col).lower()]
    if close_columns:
        return numeric_df.loc[:, close_columns].copy()

    fallback_cols = [col for col in numeric_df.columns if "volume" not in str(col).lower()]
    if fallback_cols:
        return numeric_df.loc[:, fallback_cols].copy()
    return pd.DataFrame(index=numeric_df.index)


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


def _normalize_datetime_index(obj: pd.DataFrame | pd.Series) -> None:
    """Ensure datetime indices are timezone-naive for consistent slicing."""

    index = obj.index
    if not isinstance(index, pd.DatetimeIndex):
        converted = pd.to_datetime(index, errors="coerce")
        if converted.isna().all():
            return
        mask = ~converted.isna()
        if not mask.all():
            obj.drop(index[~mask], inplace=True)
            converted = converted[mask]
        obj.index = converted
        index = obj.index
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        obj.index = index.tz_localize(None)


def _make_portfolio_name(tickers: Iterable[str]) -> str:
    """Generate a stable portfolio name for cached downloads."""

    joined = "_".join(tickers)
    slug = "".join(ch for ch in joined if ch.isalnum() or ch == "_")
    if not slug:
        slug = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    if len(slug) > 48:
        slug = slug[:48]
    return f"streamlit_{slug}"


@st.cache_data(show_spinner=False)
def _load_collated_from_disk(path: str) -> pd.DataFrame | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    frame = pd.read_csv(csv_path, parse_dates=True, index_col="Date")
    _normalize_datetime_index(frame)
    return frame


SETTINGS = get_settings()
_STREAMLIT_FETCHER = YFinancePriceFetcher()
_STREAMLIT_SERVICE = CollationService(
    _STREAMLIT_FETCHER,
    settings=SETTINGS,
    price_history_dir=SETTINGS.price_history_dir,
    export_dir=SETTINGS.export_dir,
)

LOGGER = logging.getLogger("pysharpe.app")
if not LOGGER.handlers:
    SETTINGS.log_dir.mkdir(parents=True, exist_ok=True)
    log_file = SETTINGS.log_dir / "pysharpe_app.log"
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)


@st.cache_data(show_spinner=False)
def load_prices(tickers: list[str], start: str, end: str) -> PortfolioData:
    """Download adjusted close prices for given tickers and flatten columns safely."""

    tickers_tuple = tuple(tickers)
    if not tickers_tuple:
        empty = pd.DataFrame()
        return PortfolioData(
            tickers=tickers_tuple,
            prices=empty,
            collated=empty,
            price_history_dir=SETTINGS.price_history_dir,
            collated_path=None,
            start=None,
            end=None,
            warnings=(),
            used_cache=False,
        )

    portfolio_name = _make_portfolio_name(tickers_tuple)
    collated_path = _STREAMLIT_SERVICE.export_dir / f"{portfolio_name}_collated.csv"
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None

    cached_collated = _load_collated_from_disk(str(collated_path)) if collated_path.exists() else None
    if cached_collated is not None:
        available_columns = [ticker for ticker in tickers_tuple if ticker in cached_collated.columns]
        if len(available_columns) == len(tickers_tuple):
            combined = cached_collated.loc[:, available_columns].sort_index()
            if start_ts is not None:
                combined = combined.loc[start_ts:]
            if end_ts is not None:
                combined = combined.loc[:end_ts]
            combined = combined.ffill().bfill()
            if combined.empty:
                raise RuntimeError(
                    "Cached price data does not cover the selected date range. Please adjust the dates or refresh the cache."
                )
            combined.columns = pd.Index([str(col) for col in combined.columns], name="Ticker")
            start_idx = combined.index.min() if not combined.empty else None
            end_idx = combined.index.max() if not combined.empty else None
            LOGGER.info("Using cached collated data for tickers: %s", ", ".join(available_columns))
            return PortfolioData(
                tickers=tickers_tuple,
                prices=combined,
                collated=cached_collated,
                price_history_dir=SETTINGS.price_history_dir,
                collated_path=collated_path,
                start=start_idx,
                end=end_idx,
                warnings=(),
                used_cache=True,
            )

    try:
        downloads = _STREAMLIT_SERVICE.download_portfolio_prices(
            tickers_tuple,
            period="max",
            interval="1d",
            start=start,
            end=end,
        )
    except Exception as exc:  # pragma: no cover - network failures
        LOGGER.exception("Download failed for tickers: %s", ", ".join(tickers_tuple))
        raise RuntimeError(f"Failed to download price data: {exc}") from exc

    valid_frames: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    for ticker in tickers_tuple:
        frame = downloads.get(ticker)
        if frame is None or frame.empty:
            warnings.append(f"No data returned for {ticker}.")
            continue
        valid_frames[ticker] = frame

    if not valid_frames:
        LOGGER.error("No valid price data retrieved for tickers: %s", ", ".join(tickers_tuple))
        raise RuntimeError("No valid price data retrieved for the selected tickers.")

    price_series: list[pd.Series] = []
    ordered_valid_tickers = tuple(valid_frames.keys())
    for ticker in ordered_valid_tickers:
        history = valid_frames[ticker]
        closes = _resolve_field_frame(history, ("Adj Close", "Close"))
        closes = _clean_numeric_frame(closes)
        if closes.empty:
            warnings.append(f"No adjusted close series available for {ticker}.")
            continue
        series = closes.iloc[:, 0].copy()
        series.name = ticker
        _normalize_datetime_index(series)
        price_series.append(series)

    if not price_series:
        LOGGER.error("Adjusted close data missing after processing tickers: %s", ", ".join(tickers_tuple))
        raise RuntimeError("No usable adjusted close price series were retrieved.")

    combined = pd.concat(price_series, axis=1).sort_index()
    combined.index.name = "Date"
    _normalize_datetime_index(combined)

    if start_ts is not None:
        combined = combined.loc[start_ts:]
    if end_ts is not None:
        combined = combined.loc[:end_ts]

    combined = combined.ffill().bfill()
    if combined.empty:
        LOGGER.error("Filtered price frame is empty after applying date range for tickers: %s", ", ".join(ordered_valid_tickers))
        raise RuntimeError(
            "Price data is empty after applying the selected date range. Please choose a broader window or refresh the tickers."
        )
    combined.columns = pd.Index([str(col) for col in combined.columns], name="Ticker")

    try:
        collated_df = _STREAMLIT_SERVICE.collate_portfolio(portfolio_name, ordered_valid_tickers)
    except Exception as exc:
        LOGGER.exception("Collation failed for tickers: %s", ", ".join(tickers_tuple))
        raise RuntimeError(f"Error collating portfolio files: {exc}") from exc

    _normalize_datetime_index(collated_df)
    if collated_df.empty:
        LOGGER.error("Collated portfolio is empty for tickers: %s", ", ".join(ordered_valid_tickers))
        raise RuntimeError("Collated portfolio is empty. Please retry the download or adjust the selected tickers.")
    collated_view = collated_df.loc[combined.index.min(): combined.index.max()] if not combined.empty else collated_df

    if not collated_path.exists():
        LOGGER.warning("Expected collated file missing at %s", collated_path)
        warnings.append(f"Collated CSV was not found at {collated_path}.")

    start_idx = combined.index.min() if not combined.empty else None
    end_idx = combined.index.max() if not combined.empty else None

    LOGGER.info(
        "Downloaded and collated tickers: %s (rows=%s, cols=%s)",
        ", ".join(ordered_valid_tickers),
        combined.shape[0],
        combined.shape[1],
    )
    if warnings:
        LOGGER.warning("Download warnings: %s", "; ".join(warnings))

    return PortfolioData(
        tickers=ordered_valid_tickers,
        prices=combined,
        collated=collated_view,
        price_history_dir=SETTINGS.price_history_dir,
        collated_path=collated_path if collated_path.exists() else None,
        start=start_idx,
        end=end_idx,
        warnings=tuple(warnings),
        used_cache=False,
    )


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
    raw_download = yf.download(
        tickers,
        start=start_ts.to_pydatetime(),
        end=end_ts.to_pydatetime(),
        progress=False,
        group_by="ticker",
        auto_adjust=False,
    )

    if raw_download.empty:
        return pd.DataFrame()

    closes = _clean_numeric_frame(_resolve_field_frame(raw_download, ("Adj Close", "Close")))
    volumes = _clean_numeric_frame(_resolve_field_frame(raw_download, ("Volume",)))

    combined = closes
    if not volumes.empty:
        volumes.columns = [f"{col} Volume" for col in volumes.columns]
        combined = pd.concat([closes, volumes], axis=1)

    _normalize_datetime_index(combined)
    return combined.tail(60)


def compute_metrics(price_frame: pd.DataFrame) -> MetricResults:
    """Compute metrics using PySharpe helpers and ensure aligned indices."""

    if price_frame.empty:
        raise ValueError("Price data is empty; please adjust tickers, dates, or upload a richer dataset.")

    returns = price_frame.pct_change().dropna(how="all")
    if returns.empty:
        raise ValueError("Insufficient price history to compute portfolio metrics.")

    column_index = returns.columns
    # Reindex derived metrics to maintain consistent column ordering.
    expected = metrics.expected_return(returns)
    volatility = metrics.annualize_volatility(returns)
    sharpe = metrics.sharpe_ratio(returns)

    expected = expected.reindex(column_index)
    volatility = volatility.reindex(column_index)
    sharpe = sharpe.reindex(column_index)

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
    """Simulate and plot a dollar-cost averaging projection."""

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
        price_frame = price_frame.loc[str(start_date):str(end_date)] if not price_frame.empty else price_frame
        tickers = tuple(str(col) for col in price_frame.columns)
        if not tickers:
            st.warning("Uploaded file contains no ticker columns.")
            st.stop()
        data_source = "upload"
        preview = price_frame.tail(60)
        metadata = pd.DataFrame(
            {
                ticker: {"name": "Provided via CSV", "exchange": "-", "currency": "-"}
                for ticker in tickers
            }
        ).T if tickers else pd.DataFrame()
        if not metadata.empty:
            metadata.index.name = "Ticker"
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
        download_summary: dict[str, object] | None = None
    else:
        tickers_raw = st.sidebar.text_input("Tickers (comma-separated)", value="AAPL,MSFT,GOOGL")
        tickers_list = [ticker.strip().upper() for ticker in tickers_raw.split(",") if ticker.strip()]
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
            st.error("No valid price data retrieved. Please adjust the tickers or date range and try again.")
            st.stop()
        preview = load_preview_data(resolved_tickers, end_date)
        metadata = gather_metadata(resolved_tickers) if resolved_tickers else pd.DataFrame()
        if not metadata.empty:
            metadata.index.name = "Ticker"
        download_summary = {
            "tickers": tickers,
            "price_history_dir": str(portfolio_data.price_history_dir),
            "collated_path": str(portfolio_data.collated_path) if portfolio_data.collated_path else None,
            "start": portfolio_data.start.isoformat() if isinstance(portfolio_data.start, pd.Timestamp) else None,
            "end": portfolio_data.end.isoformat() if isinstance(portfolio_data.end, pd.Timestamp) else None,
            "rows": portfolio_data.prices.shape[0],
            "columns": portfolio_data.prices.shape[1],
            "warnings": portfolio_data.warnings,
            "used_cache": portfolio_data.used_cache,
        }

    # Normalise uploaded or downloaded data into numeric-only DataFrame.
    numeric_df = price_frame.select_dtypes("number") if not price_frame.empty else pd.DataFrame()
    price_data = _select_price_data(numeric_df)

    st.sidebar.header("DCA Settings")
    dca_initial = st.sidebar.number_input("Initial Investment", min_value=0.0, value=1000.0, step=100.0)
    dca_monthly = st.sidebar.number_input("Monthly Contribution", min_value=0.0, value=250.0, step=25.0)
    dca_months = st.sidebar.slider("Months", min_value=12, max_value=600, value=240, step=12)
    if st.session_state.get("dca_rate_pending_reset", False) and not st.session_state.get("dca_rate_override", False):
        # Drop the slider's stored value before rebuilding it so the updated default takes effect without
        # reassigning the widget key, avoiding StreamlitAPIException.
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
        "Download market data, evaluate portfolio metrics, run optimisations, and simulate dollar-cost averaging from a single interface."
    )

    prices: pd.DataFrame = controls["data"]
    price_data: pd.DataFrame = controls.get("price_data", pd.DataFrame())
    portfolio_data: PortfolioData | None = controls.get("portfolio_data")  # type: ignore[assignment]
    download_summary = controls.get("download_summary")

    if prices.empty:
        st.warning("No valid price data available for the selected tickers or file. Please adjust your inputs.")
        return

    if download_summary:
        tickers_display = ", ".join(download_summary["tickers"])
        if download_summary.get("used_cache"):
            st.info(f"Using cached price data for: {tickers_display}")
        else:
            st.success(f"Downloaded {len(download_summary['tickers'])} tickers: {tickers_display}")
        st.caption(f"Price history directory: `{download_summary['price_history_dir']}`")
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
            mean_expected = float(metrics_result.expected.mean()) if not metrics_result.expected.empty else np.nan
            if not np.isnan(mean_expected):
                st.session_state["dca_rate_default"] = mean_expected
                if not st.session_state.get("dca_rate_override", False):
                    st.session_state["dca_rate_value"] = mean_expected
                    st.session_state["dca_rate_pending_reset"] = True
                else:
                    st.session_state["dca_rate_pending_reset"] = False
            else:
                st.session_state["dca_rate_pending_reset"] = False
            controls["dca_rate"] = float(st.session_state.get("dca_rate_value", st.session_state.get("dca_rate_default", 0.08)))
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
