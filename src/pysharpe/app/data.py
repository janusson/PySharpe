"""Data loading and preparation helpers for the Streamlit app."""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

from pysharpe.config import get_settings
from pysharpe.data.collation import CollationService
from pysharpe.data.fetcher import YFinancePriceFetcher

LOGGER = logging.getLogger("pysharpe.app")

SETTINGS = get_settings()
_STREAMLIT_FETCHER = YFinancePriceFetcher()
_STREAMLIT_SERVICE = CollationService(
    _STREAMLIT_FETCHER,
    settings=SETTINGS,
    price_history_dir=SETTINGS.price_history_dir,
    export_dir=SETTINGS.export_dir,
)

if not LOGGER.handlers:
    SETTINGS.log_dir.mkdir(parents=True, exist_ok=True)
    log_file = SETTINGS.log_dir / "pysharpe_app.log"
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)


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


def _resolve_field_frame(
    raw_data: pd.DataFrame, field_priority: Iterable[str]
) -> pd.DataFrame:
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


def select_price_data(numeric_df: pd.DataFrame) -> pd.DataFrame:
    """Prefer close columns but gracefully fall back to other numeric data."""

    if numeric_df.empty:
        return pd.DataFrame(index=numeric_df.index)

    close_columns = [col for col in numeric_df.columns if "close" in str(col).lower()]
    if close_columns:
        return numeric_df.loc[:, close_columns].copy()

    fallback_cols = [
        col for col in numeric_df.columns if "volume" not in str(col).lower()
    ]
    if fallback_cols:
        return numeric_df.loc[:, fallback_cols].copy()
    return pd.DataFrame(index=numeric_df.index)


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
    try:
        frame = pd.read_csv(csv_path, parse_dates=True, index_col="Date")
    except ValueError:
        frame = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    _normalize_datetime_index(frame)
    return frame


@st.cache_data(show_spinner=False)
def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    _loader: Callable[[str], pd.DataFrame | None] = _load_collated_from_disk,
) -> PortfolioData:
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

    cached_collated = _loader(str(collated_path)) if collated_path.exists() else None
    if cached_collated is not None:
        available_columns = [
            ticker for ticker in tickers_tuple if ticker in cached_collated.columns
        ]
        if len(available_columns) == len(tickers_tuple):
            combined = cached_collated.loc[:, available_columns].sort_index()
            if start_ts is not None:
                combined = combined.loc[start_ts:]
            if end_ts is not None:
                combined = combined.loc[:end_ts]
            combined = combined.ffill().bfill()
            if combined.empty:
                raise RuntimeError(
                    "Cached price data does not cover the selected date range. "
                    "Please adjust the dates or refresh the cache."
                )
            combined.columns = pd.Index(
                [str(col) for col in combined.columns], name="Ticker"
            )
            start_idx = combined.index.min() if not combined.empty else None
            end_idx = combined.index.max() if not combined.empty else None
            LOGGER.info(
                "Using cached collated data for tickers: %s",
                ", ".join(available_columns),
            )
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
        LOGGER.error(
            "No valid price data retrieved for tickers: %s",
            ", ".join(tickers_tuple),
        )
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
        LOGGER.error(
            "Adjusted close data missing after processing tickers: %s",
            ", ".join(tickers_tuple),
        )
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
        LOGGER.error(
            "Filtered price frame is empty after applying date range for tickers: %s",
            ", ".join(ordered_valid_tickers),
        )
        raise RuntimeError(
            "Price data is empty after applying the selected date range. "
            "Please choose a broader window or refresh the tickers."
        )
    combined.columns = pd.Index([str(col) for col in combined.columns], name="Ticker")

    try:
        collated_df = _STREAMLIT_SERVICE.collate_portfolio(
            portfolio_name, ordered_valid_tickers
        )
    except Exception as exc:
        LOGGER.exception(
            "Collation failed for tickers: %s", ", ".join(tickers_tuple)
        )
        raise RuntimeError(f"Error collating portfolio files: {exc}") from exc

    _normalize_datetime_index(collated_df)
    if collated_df.empty:
        LOGGER.error(
            "Collated portfolio is empty for tickers: %s",
            ", ".join(ordered_valid_tickers),
        )
        raise RuntimeError(
            "Collated portfolio is empty. Please retry the download or adjust the "
            "selected tickers."
        )
    collated_view = (
        collated_df.loc[combined.index.min() : combined.index.max()]
        if not combined.empty
        else collated_df
    )

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
        records[ticker] = (
            {
                "name": info.get("shortName", "Unknown"),
                "exchange": info.get("exchange", "Unknown"),
                "currency": info.get("currency", "N/A"),
            }
            if info
            else {
                "name": "Lookup failed",
                "exchange": "-",
                "currency": "-",
            }
        )

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

    closes = _clean_numeric_frame(
        _resolve_field_frame(raw_download, ("Adj Close", "Close"))
    )
    volumes = _clean_numeric_frame(_resolve_field_frame(raw_download, ("Volume",)))

    combined = closes
    if not volumes.empty:
        volumes.columns = [f"{col} Volume" for col in volumes.columns]
        combined = pd.concat([closes, volumes], axis=1)

    _normalize_datetime_index(combined)
    return combined.tail(60)


__all__ = [
    "PortfolioData",
    "gather_metadata",
    "load_preview_data",
    "load_prices",
    "SETTINGS",
    "select_price_data",
]
