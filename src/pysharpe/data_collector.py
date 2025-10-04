"""Utilities for downloading and collating portfolio price data."""

from __future__ import annotations

import json
import logging
from functools import cached_property
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

import pandas as pd

from pysharpe.config import get_settings
from pysharpe.data import (
    CollationService,
    PortfolioDownloadWorkflow,
    PortfolioRepository,
    PriceFetcher,
    PriceHistoryError,
    YFinancePriceFetcher,
    read_tickers,
)
from pysharpe.logging_utils import configure_logging

try:  # pragma: no cover - lazy import for optional dependency
    import yfinance as yf
except ImportError:  # pragma: no cover - surfaced via helper
    yf = None  # type: ignore[assignment]


_SETTINGS = get_settings()
# Change: Removed unused DATA_DIR constant during redundancy cleanup.
PORTFOLIO_DIR = _SETTINGS.portfolio_dir
PRICE_HISTORY_DIR = Path(_SETTINGS.price_history_dir)
EXPORT_DIR = Path(_SETTINGS.export_dir)
INFO_DIR = _SETTINGS.info_dir
LOG_DIR = _SETTINGS.log_dir

logger = logging.getLogger(__name__)


def _as_path(value: Path | str) -> Path:
    """Return a consistently expanded ``Path`` for filesystem arguments."""
    # Change: Normalised helper now documented for clarity after style audit.
    return Path(value).expanduser()


def _build_collation_service(
    *,
    price_history_dir: Path | str,
    export_dir: Path | str | None = None,
    fetcher: PriceFetcher | None = None,
) -> CollationService:
    """Construct a :class:`CollationService` with shared settings defaults."""
    # Change: Added docstring to explain helper responsibility per style pass.
    return CollationService(
        fetcher or YFinancePriceFetcher(),
        _SETTINGS,
        price_history_dir=_as_path(price_history_dir),
        export_dir=_as_path(export_dir or EXPORT_DIR),
    )


def _build_download_workflow(
    *,
    portfolio_dir: Path | str,
    price_history_dir: Path | str,
    export_dir: Path | str,
    fetcher: PriceFetcher | None = None,
) -> PortfolioDownloadWorkflow:
    """Initialise the workflow that orchestrates download + collation."""
    # Change: Documented helper factory to aid readability.
    return PortfolioDownloadWorkflow(
        settings=_SETTINGS,
        portfolio_dir=_as_path(portfolio_dir),
        price_history_dir=_as_path(price_history_dir),
        export_dir=_as_path(export_dir),
        fetcher=fetcher,
    )


def setup_logging(log_dir: Path = LOG_DIR, level: Optional[str] = None) -> Path:
    """Configure basic file logging under ``log_dir``.

    Args:
        log_dir: Directory where the log file should be written.
        level: Optional logging level string.

    Returns:
        Path to the created log file.

    Example:
        >>> from pysharpe.data_collector import setup_logging
        >>> setup_logging(level="INFO").suffix
        '.log'
    """

    return configure_logging(log_dir=log_dir, level=level)


def get_csv_file_paths(directory: Path | None = None) -> list[Path]:
    """Return portfolio CSV files in ``directory`` (defaults to configured dir).

    Args:
        directory: Optional override for the portfolio directory.

    Returns:
        Sorted list of CSV paths discovered.

    Example:
        >>> from pysharpe.data_collector import get_csv_file_paths
        >>> isinstance(get_csv_file_paths(), list)
        True
    """

    root = _as_path(directory or PORTFOLIO_DIR)
    repo = PortfolioRepository(_SETTINGS, directory=root)
    return [definition.path for definition in repo.list_portfolios()]


def read_tickers_from_file(path: Path) -> Set[str]:
    """Read tickers from a CSV-style file containing one symbol per line.

    Args:
        path: File to parse.

    Returns:
        Set of unique tickers located in the file.

    Example:
        >>> from pathlib import Path
        >>> from pysharpe.data_collector import read_tickers_from_file
        >>> file_path = Path('portfolio.csv')
        >>> _ = file_path.write_text('AAPL\nMSFT\n', encoding='utf-8')
        >>> read_tickers_from_file(file_path)
        {'AAPL', 'MSFT'}
        >>> file_path.unlink()
    """

    return set(read_tickers(Path(path)))


class PortfolioTickerReader:
    """Legacy-compatible wrapper around :class:`PortfolioRepository`.

    Example:
        >>> from pysharpe.data_collector import PortfolioTickerReader
        >>> reader = PortfolioTickerReader()
        >>> isinstance(reader.portfolio_tickers, dict)
        True
    """

    def __init__(self, directory: Path = PORTFOLIO_DIR) -> None:
        self.directory = _as_path(directory)
        self.repo = PortfolioRepository(_SETTINGS, directory=self.directory)
        self.portfolio_tickers: dict[str, Set[str]] = {}
        self.refresh()

    def refresh(self) -> None:
        """Reload the in-memory mapping of portfolio tickers."""

        self.repo.refresh()
        self.portfolio_tickers = {
            definition.name: set(definition.tickers) for definition in self.repo.list_portfolios()
        }

    def get_portfolio_tickers(self, portfolio_name: str) -> Set[str]:
        """Return the tickers tracked for ``portfolio_name``.

        Args:
            portfolio_name: Portfolio identifier.

        Returns:
            Set of tickers for the portfolio, or an empty set if unknown.

        Example:
            >>> reader = PortfolioTickerReader()
            >>> reader.get_portfolio_tickers('nonexistent')
            set()
        """

        return self.portfolio_tickers.get(portfolio_name, set())


def download_portfolio_prices(
    tickers: Iterable[str],
    *,
    price_history_dir: Path | str = PRICE_HISTORY_DIR,
    export_dir: Path | str | None = None,
    period: str = "max",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    fetcher: PriceFetcher | None = None,
) -> dict[str, pd.DataFrame]:
    """Download price histories for ``tickers`` and write each to CSV.

    Args:
        tickers: Iterable of tickers (duplicates are ignored).
        price_history_dir: Directory for raw per-ticker CSVs.
        export_dir: Directory for collated artefacts (defaults to configured export directory).
        period: Rolling window used when explicit dates are not provided.
        interval: Sampling frequency for the download.
        start: Optional ISO start date.
        end: Optional ISO end date.
        fetcher: Optional custom :class:`PriceFetcher` implementation.

    Returns:
        Mapping of ticker symbol to the downloaded DataFrame.

    Example:
        >>> from pysharpe.data_collector import download_portfolio_prices
        >>> download_portfolio_prices(['AAPL'], period='1y', interval='1d', start=None, end=None)  # doctest: +SKIP
        {'AAPL': ...}
    """

    service = _build_collation_service(
        price_history_dir=price_history_dir,
        export_dir=export_dir or EXPORT_DIR,
        fetcher=fetcher,
    )
    return service.download_portfolio_prices(
        tickers,
        period=period,
        interval=interval,
        start=start,
        end=end,
    )


def collate_prices(
    portfolio_name: str,
    price_history_dir: Path | str,
    tickers: Sequence[str],
    *,
    export_dir: Path | str = EXPORT_DIR,
) -> pd.DataFrame:
    """Combine downloaded price histories into a single CSV per portfolio.

    Args:
        portfolio_name: Name used for artefact filenames.
        price_history_dir: Directory containing per-ticker CSVs.
        tickers: Tickers to include in the portfolio.
        export_dir: Directory where the collated CSV should be saved.

    Returns:
        Collated price history as a DataFrame.

    Example:
        >>> from pysharpe.data_collector import collate_prices
        >>> collate_prices('demo', 'tests/data', ['AAPL'])  # doctest: +SKIP
        ...
    """

    service = _build_collation_service(
        price_history_dir=price_history_dir,
        export_dir=export_dir,
    )
    return service.collate_portfolio(portfolio_name, list(tickers))


def process_portfolio(
    portfolio_file: Path,
    *,
    price_history_dir: Path | str = PRICE_HISTORY_DIR,
    export_dir: Path | str = EXPORT_DIR,
    period: str = "max",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    fetcher: PriceFetcher | None = None,
) -> pd.DataFrame:
    """Download and collate prices for the portfolio described in ``portfolio_file``.

    Args:
        portfolio_file: Path to a newline-delimited ticker list.
        price_history_dir: Directory for raw price CSVs.
        export_dir: Directory for collated artefacts.
        period: Rolling window used when explicit dates are not provided.
        interval: Sampling frequency for the download.
        start: Optional ISO start date.
        end: Optional ISO end date.
        fetcher: Optional custom :class:`PriceFetcher` implementation.

    Returns:
        Collated price DataFrame (empty when no tickers are found).

    Example:
        >>> from pysharpe.data_collector import process_portfolio
        >>> from pathlib import Path
        >>> csv_path = Path('demo.csv')
        >>> _ = csv_path.write_text('AAPL', encoding='utf-8')
        >>> process_portfolio(csv_path, period='1y', interval='1d')  # doctest: +SKIP
        ...
        >>> csv_path.unlink()
    """

    portfolio_file = Path(portfolio_file)
    tickers = sorted(read_tickers_from_file(portfolio_file))
    if not tickers:
        logger.warning("No tickers found for portfolio: %s", portfolio_file.stem)
        return pd.DataFrame()

    service = _build_collation_service(
        price_history_dir=price_history_dir,
        export_dir=export_dir,
        fetcher=fetcher,
    )
    return service.process_portfolio(
        portfolio_file.stem,
        tickers,
        period=period,
        interval=interval,
        start=start,
        end=end,
    )


def process_all_portfolios(
    portfolio_dir: Path | str = PORTFOLIO_DIR,
    *,
    price_history_dir: Path | str = PRICE_HISTORY_DIR,
    export_dir: Path | str = EXPORT_DIR,
    period: str = "max",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    fetcher: PriceFetcher | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the download/collation workflow for every portfolio file in ``portfolio_dir``.

    Args:
        portfolio_dir: Directory containing portfolio CSV definitions.
        price_history_dir: Directory for raw price CSVs.
        export_dir: Directory for collated artefacts.
        period: Rolling window used when explicit dates are absent.
        interval: Sampling frequency for the download.
        start: Optional ISO start date.
        end: Optional ISO end date.
        fetcher: Optional custom :class:`PriceFetcher` implementation.

    Returns:
        Mapping of portfolio name to the collated DataFrame for successful runs.

    Example:
        >>> from pysharpe.data_collector import process_all_portfolios
        >>> process_all_portfolios(portfolio_dir='tests/data')  # doctest: +SKIP
        ...
    """

    workflow = _build_download_workflow(
        portfolio_dir=portfolio_dir,
        price_history_dir=price_history_dir,
        export_dir=export_dir,
        fetcher=fetcher,
    )

    results = workflow.process_portfolios(
        names=None,
        period=period,
        interval=interval,
        start=start,
        end=end,
    )
    return results


def _ensure_yfinance():  # pragma: no cover - compatibility shim
    if yf is None:
        raise RuntimeError("yfinance must be installed to download market data.")
    return yf


class SecurityDataCollector:
    """Wrapper around yfinance helpers for single-ticker information.

    Example:
        >>> from pysharpe.data_collector import SecurityDataCollector
        >>> collector = SecurityDataCollector('AAPL')  # doctest: +SKIP
        >>> collector.get_company_name()  # doctest: +SKIP
        'Apple Inc.'
    """

    def __init__(self, ticker: str):
        if not ticker:
            raise ValueError("Ticker symbol must be provided")
        self.ticker = ticker
        self._yf = _ensure_yfinance().Ticker(ticker)

    @cached_property
    def _info(self) -> dict:
        payload = getattr(self._yf, "info", {}) or {}
        return dict(payload)

    def get_company_name(self) -> str:
        """Return the preferred company display name.

        Example:
            >>> SecurityDataCollector('AAPL').get_company_name()  # doctest: +SKIP
            'Apple Inc.'
        """

        info = self._info
        return info.get("shortName") or info.get("longName") or self.ticker

    def get_company_info(self) -> dict:
        """Return the raw info payload from yfinance.

        Example:
            >>> SecurityDataCollector('AAPL').get_company_info()  # doctest: +SKIP
            {...}
        """

        return dict(self._info)

    def download_info(self, destination: Path = INFO_DIR) -> Path:
        """Persist the info payload to ``destination`` as JSON.

        Example:
            >>> SecurityDataCollector('AAPL').download_info()  # doctest: +SKIP
            PosixPath('...')
        """

        info = self.get_company_info()
        destination = _as_path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        file_path = destination / f"{self.get_company_name().replace(' ', '_')}_summary.json"
        file_path.write_text(json.dumps(info, indent=4), encoding="utf-8")
        return file_path

    def get_news(self) -> list:
        """Return the news entries attached to the ticker."""

        payload = getattr(self._yf, "news", None) or []
        return list(payload)

    def get_options(self) -> list:
        """Return the available option expiration dates."""

        payload = getattr(self._yf, "options", None) or []
        return list(payload)

    def get_earnings_dates(self) -> pd.DataFrame:
        """Return the earnings date DataFrame (empty if unavailable)."""

        data = getattr(self._yf, "earnings_dates", pd.DataFrame())
        return data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()

    def get_recommendations(self) -> dict:
        """Return analyst recommendation data grouped by source."""

        return {
            "recommendations": getattr(self._yf, "recommendations", None),
            "summary": getattr(self._yf, "recommendations_summary", None),
            "upgrades_downgrades": getattr(self._yf, "upgrades_downgrades", None),
        }

    def get_holders(self) -> dict:
        """Return major, institutional, mutual fund, and insider holdings."""

        return {
            "major_holders": getattr(self._yf, "major_holders", None),
            "institutional_holders": getattr(self._yf, "institutional_holders", None),
            "mutualfund_holders": getattr(self._yf, "mutualfund_holders", None),
            "insider_purchases": getattr(self._yf, "insider_purchases", None),
            "insider_roster_holders": getattr(self._yf, "insider_roster_holders", None),
        }

    def get_financials(self) -> pd.DataFrame:
        """Return concatenated financial statements when available."""

        sections = {
            "Income Statement": "financials",
            "Quarterly Income Statement": "quarterly_financials",
            "Balance Sheet": "balance_sheet",
            "Quarterly Balance Sheet": "quarterly_balance_sheet",
            "Cash Flow": "cashflow",
            "Quarterly Cash Flow": "quarterly_cashflow",
        }

        frames = {}
        for label, attr in sections.items():
            value = getattr(self._yf, attr, None)
            if isinstance(value, pd.DataFrame) and not value.empty:
                frames[label] = value

        return pd.concat(frames, axis=1) if frames else pd.DataFrame()

    def get_actions(self) -> dict:
        """Return corporate action history (splits, dividends, etc.)."""

        return {
            "actions": getattr(self._yf, "actions", None),
            "dividends": getattr(self._yf, "dividends", None),
            "splits": getattr(self._yf, "splits", None),
            "capital_gains": getattr(self._yf, "capital_gains", None),
            "shares_history": self._yf.get_shares_full() if hasattr(self._yf, "get_shares_full") else None,
        }

    def get_summary(self) -> dict:
        """Return a curated subset of company fundamentals."""

        info = self.get_company_info()
        keys = [
            "longName",
            "country",
            "industry",
            "sector",
            "overallRisk",
            "dividendYield",
            "previousClose",
            "payoutRatio",
            "currency",
            "forwardPE",
            "volume",
            "marketCap",
            "priceToBook",
            "forwardEps",
            "pegRatio",
            "symbol",
            "currentPrice",
            "recommendationMean",
            "debtToEquity",
            "revenuePerShare",
            "returnOnAssets",
            "returnOnEquity",
            "freeCashflow",
            "operatingCashflow",
            "earningsGrowth",
            "revenueGrowth",
            "grossMargins",
            "ebitdaMargins",
            "operatingMargins",
            "financialCurrency",
            "trailingPegRatio",
        ]
        return {key: info.get(key) for key in keys}

    def download_price_history(
        self,
        destination: Path = PRICE_HISTORY_DIR,
        *,
        period: str = "max",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Download historical prices for the security and persist them.

        Example:
            >>> SecurityDataCollector('AAPL').download_price_history()  # doctest: +SKIP
            ...
        """

        fetcher = YFinancePriceFetcher()
        frame = fetcher.fetch_history(
            self.ticker,
            period=period,
            interval=interval,
            start=start,
            end=end,
        )
        destination = _as_path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / f"{self.ticker}_hist.csv"
        frame.to_csv(output)
        return frame


def main() -> None:  # pragma: no cover - script entry point
    setup_logging()
    csv_files = get_csv_file_paths(PORTFOLIO_DIR)
    for portfolio_path in csv_files:
        tickers = read_tickers_from_file(portfolio_path)
        logger.info("Tickers for %s: %s", portfolio_path.stem, ", ".join(sorted(tickers)))
        process_portfolio(portfolio_path)


if __name__ == "__main__":  # pragma: no cover
    main()
