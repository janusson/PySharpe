"""Combine fetched price histories into collated portfolio datasets."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from pandas.errors import EmptyDataError

from pysharpe.config import PySharpeSettings, get_settings

from .fetcher import PriceFetcher, PriceHistoryError

logger = logging.getLogger(__name__)


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Read a CSV file containing price history records."""

    try:
        frame = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Price history file not found: {csv_path}") from exc
    except EmptyDataError as exc:
        raise ValueError(f"Price history file is empty: {csv_path}") from exc
    return frame


def parse_records(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalise raw price history into a single-ticker dataframe."""

    if raw.empty:
        return pd.DataFrame(columns=[ticker])

    required = {"Date", "Close"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Unexpected columns for {ticker}: missing {sorted(missing)}")

    parsed_dates: list[pd.Timestamp | pd.NaT] = []
    for value in raw["Date"]:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            parsed_dates.append(pd.NaT)
            continue
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert(None)
        parsed_dates.append(ts)

    timestamps = pd.Series(parsed_dates, index=raw.index, dtype="datetime64[ns]")
    closes = pd.to_numeric(raw["Close"], errors="coerce")
    mask = (~timestamps.isna()) & (~closes.isna())

    if not mask.any():
        raise ValueError(f"Unable to parse dates for {ticker}")

    valid_dates = timestamps.loc[mask]
    if valid_dates.dt.tz is not None:
        valid_dates = valid_dates.dt.tz_convert(None)

    index = valid_dates.dt.strftime("%Y-%m-%d")
    series = pd.Series(closes.loc[mask].to_numpy(copy=False), index=index, name=ticker)
    series.index.name = "Date"
    frame = series.to_frame()
    frame = frame[~frame.index.duplicated(keep="first")]
    return frame.sort_index()


class CollationService:
    """Coordinate price downloads, CSV normalisation, and metadata capture.

    Example:
        >>> from pysharpe.data.fetcher import PriceFetcher
        >>> class _InMemoryFetcher(PriceFetcher):
        ...     def fetch_history(self, ticker, *, period, interval, start, end):
        ...         import pandas as pd
        ...         return pd.DataFrame({"Close": [1.0, 1.1]}, index=[0, 1])
        >>> service = CollationService(_InMemoryFetcher())
        >>> isinstance(service, CollationService)
        True
    """

    def __init__(
        self,
        fetcher: PriceFetcher,
        settings: PySharpeSettings | None = None,
        *,
        price_history_dir: Path | None = None,
        export_dir: Path | None = None,
    ) -> None:
        self.fetcher = fetcher
        self.settings = settings or get_settings()
        self.price_history_dir = Path(price_history_dir or self.settings.price_history_dir)
        self.export_dir = Path(export_dir or self.settings.export_dir)
        self.settings.ensure_directories()
        self.price_history_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def download_portfolio_prices(
        self,
        tickers: Iterable[str],
        *,
        period: str,
        interval: str,
        start: str | None,
        end: str | None,
    ) -> dict[str, pd.DataFrame]:
        """Download and cache price histories for ``tickers``.

        Args:
            tickers: Iterable of ticker symbols (deduplicated internally).
            period: Rolling window requested when dates are absent.
            interval: Sampling interval passed to the fetcher.
            start: Optional ISO start date.
            end: Optional ISO end date.

        Returns:
            Mapping of ticker to the downloaded DataFrame.

        Example:
            >>> from pysharpe.data.fetcher import PriceFetcher
            >>> class _Fetcher(PriceFetcher):
            ...     def fetch_history(self, ticker, **kwargs):
            ...         import pandas as pd
            ...         return pd.DataFrame({"Close": [1.0]}, index=[0])
            >>> service = CollationService(_Fetcher())
            >>> service.download_portfolio_prices(["AAA"], period="1y", interval="1d", start=None, end=None).keys()
            dict_keys(['AAA'])
        """

        results: dict[str, pd.DataFrame] = {}
        for ticker in sorted(set(tickers)):
            try:
                frame = self.fetcher.fetch_history(
                    ticker,
                    period=period,
                    interval=interval,
                    start=start,
                    end=end,
                )
            except PriceHistoryError as exc:
                logger.error("Skipping %s: %s", ticker, exc)
                continue

            csv_path = self.price_history_dir / f"{ticker}_hist.csv"
            frame.to_csv(csv_path)
            results[ticker] = frame
        return results

    def _load_price_frame(self, ticker: str) -> pd.DataFrame | None:
        """Load and sanitise a single ticker CSV for downstream concatenation."""
        # Change: Documented helper to clarify sanitisation expectations.
        csv_path = self.price_history_dir / f"{ticker}_hist.csv"
        if not csv_path.exists():
            logger.warning("Price history missing for %s", ticker)
            return None

        try:
            raw = load_raw(csv_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Failed to load price history for %s: %s", ticker, exc)
            return None

        try:
            frame = parse_records(raw, ticker)
        except ValueError as exc:
            logger.error("Invalid price history for %s: %s", ticker, exc)
            return None

        if frame.empty:
            logger.warning("No valid price points retained for %s", ticker)
            return None

        return frame

    def collate_portfolio(
        self,
        name: str,
        tickers: Sequence[str],
    ) -> pd.DataFrame:
        """Collate individual ticker CSVs into a portfolio-wide DataFrame.

        Args:
            name: Portfolio name used for artefact filenames.
            tickers: Ordered tickers to collate.

        Returns:
            DataFrame indexed by date containing one column per included ticker.

        Example:
            >>> from pysharpe.data.fetcher import PriceFetcher
            >>> class _Fetcher(PriceFetcher):
            ...     def fetch_history(self, ticker, **kwargs):
            ...         import pandas as pd
            ...         return pd.DataFrame({"Date": ["2024-01-01"], "Close": [1.0]})
            >>> service = CollationService(_Fetcher())
            >>> service.download_portfolio_prices(["AAA"], period="1y", interval="1d", start=None, end=None)
            {'AAA': ...}
        """

        frames: list[pd.DataFrame] = []
        included: list[str] = []
        for ticker in tickers:
            frame = self._load_price_frame(ticker)
            if frame is None:
                continue
            frames.append(frame)
            included.append(ticker)

        if not frames:
            logger.warning("No price data collated for %s", name)
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1).sort_index()
        cleaned = combined.loc[:, combined.notna().any()]
        included = list(cleaned.columns)
        # Change: Derived included tickers directly from surviving columns to
        # avoid redundant membership checks per performance audit.

        output_path = self.export_dir / f"{name}_collated.csv"
        cleaned.to_csv(output_path)
        self._write_metadata(name, tickers, included)
        return cleaned

    def process_portfolio(
        self,
        name: str,
        tickers: Sequence[str],
        *,
        period: str,
        interval: str,
        start: str | None,
        end: str | None,
    ) -> pd.DataFrame:
        """Download, collate, and persist artefacts for a single portfolio.

        Args:
            name: Portfolio name.
            tickers: Iterable of tickers to fetch.
            period: Rolling window requested when dates are absent.
            interval: Sampling interval for price data.
            start: Optional ISO date limiting the window.
            end: Optional ISO end date.

        Returns:
            Collated price history. Empty if no valid tickers were available.

        Example:
            >>> from pysharpe.data.fetcher import PriceFetcher
            >>> class _Fetcher(PriceFetcher):
            ...     def fetch_history(self, ticker, **kwargs):
            ...         import pandas as pd
            ...         return pd.DataFrame({"Date": ["2024-01-01"], "Close": [1.0]})
            >>> service = CollationService(_Fetcher())
            >>> service.process_portfolio("demo", ("AAA",), period="1y", interval="1d", start=None, end=None).empty
            False
        """

        if not tickers:
            logger.warning("No tickers found for portfolio: %s", name)
            return pd.DataFrame()

        self.download_portfolio_prices(
            tickers,
            period=period,
            interval=interval,
            start=start,
            end=end,
        )
        return self.collate_portfolio(name, tickers)

    def _write_metadata(
        self,
        name: str,
        requested: Sequence[str],
        included: Sequence[str],
    ) -> None:
        metadata_path = self.export_dir / f"{name}_metadata.json"
        requested_set = set(requested)
        included_set = set(included)
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        payload = {
            "name": name,
            "requested_tickers": list(requested),
            "included_tickers": list(included),
            "dropped_tickers": sorted(requested_set - included_set),
            "generated_at": timestamp,
            "artifact_version": self.settings.artifact_version,
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
