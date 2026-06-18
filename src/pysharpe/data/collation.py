"""Combine fetched price histories into collated portfolio datasets."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from pysharpe.config import PySharpeSettings, get_settings

from .fetcher import (
    DuckDBCachedPriceFetcher,
    PriceFetcher,
    PriceHistoryError,
    YFinancePriceFetcher,
)
from .linkage import HistoryLinker

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
        self.settings = settings or get_settings()

        self.price_history_dir = Path(
            price_history_dir or self.settings.price_history_dir
        )
        self.export_dir = Path(export_dir or self.settings.export_dir)
        self.settings.ensure_directories()
        self.price_history_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Only the live yfinance fetcher needs the DuckDB write-through cache.
        # Custom fetchers (test stubs, etc.) are used directly to preserve isolation.
        if isinstance(fetcher, DuckDBCachedPriceFetcher):
            self.fetcher = fetcher
        elif isinstance(fetcher, YFinancePriceFetcher):
            db_path = str(self.settings.cache_dir / "pysharpe_cache.db")
            self.fetcher = DuckDBCachedPriceFetcher(fetcher, db_path=db_path)
        else:
            self.fetcher = fetcher

        self.crypto_tickers = {"BTC-USD", "ETH-USD"}  # Define known 24/7 tickers

    def _is_24_7_asset(self, ticker: str) -> bool:
        return ticker in self.crypto_tickers

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
                # Check if ticker has a proxy configuration
                if ticker in self.settings.proxy_map:
                    proxy_config = self.settings.proxy_map[ticker]
                    # We pass the full proxy config to a temporary proxy map.
                    # proxy_map setting format: {"VFV.TO": {"proxy": "VOO", "fx_adjust": true, "start_date": "2010-01-01"}}
                    proxy_name = proxy_config.get("proxy")
                    fx_adjust = proxy_config.get("fx_adjust", False)
                    start_date = proxy_config.get(
                        "start_date", "1900-01-01"
                    )  # Fallback to very early date if missing

                    linker = HistoryLinker(
                        proxy_map={ticker: proxy_name},
                        fx_adjust=fx_adjust,
                        fetcher=self.fetcher,
                    )

                    # HistoryLinker returns a Series named ticker
                    stitched_series = linker.get_stitched_series(ticker, start_date)

                    # Convert to dataframe to match fetch_history return type expected by this method
                    frame = stitched_series.to_frame(name="Close")
                    frame.index.name = "Date"

                else:
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

        # Separate 24/7 assets and equity assets
        equity_frames = []
        crypto_frames = []
        for ticker, frame in zip(included, frames):
            if self._is_24_7_asset(ticker):
                crypto_frames.append(frame)
            else:
                equity_frames.append(frame)

        # Create a master equity calendar
        equity_calendar = pd.DatetimeIndex([])
        if equity_frames:
            all_equity_dates = [df.index for df in equity_frames]
            equity_calendar = pd.DatetimeIndex(sorted(set().union(*all_equity_dates)))

        # Reindex crypto assets to the equity calendar using ffill
        processed_frames = equity_frames.copy()
        if crypto_frames and not equity_calendar.empty:
            for frame in crypto_frames:
                ticker = frame.columns[0]  # Assuming single column DataFrames here
                logger.info("Aligning 24/7 asset %s to equity calendar.", ticker)
                aligned_frame = frame.reindex(equity_calendar, method="ffill")
                processed_frames.append(aligned_frame)
        elif crypto_frames and equity_calendar.empty:
            # If only crypto assets exist, no equity calendar to align to, just keep them as is
            processed_frames.extend(crypto_frames)

        combined = pd.concat(processed_frames, axis=1).sort_index()
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
