"""Utilities for downloading and collating portfolio price data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

import pandas as pd

from pysharpe.config import get_settings
from pysharpe.data import (
    CollationService,
    PortfolioDownloadWorkflow,
    PortfolioRepository,
    YFinancePriceFetcher,
    read_tickers,
)
from pysharpe.logging_utils import configure_logging

try:  # pragma: no cover - lazy import for optional dependency
    import yfinance as yf
except ImportError:  # pragma: no cover - surfaced via helper
    yf = None  # type: ignore[assignment]


_SETTINGS = get_settings()
DATA_DIR = _SETTINGS.data_dir
PORTFOLIO_DIR = _SETTINGS.portfolio_dir
PRICE_HISTORY_DIR = Path(_SETTINGS.price_history_dir)
EXPORT_DIR = Path(_SETTINGS.export_dir)
INFO_DIR = _SETTINGS.info_dir
LOG_DIR = _SETTINGS.log_dir

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path = LOG_DIR, level: Optional[str] = None) -> Path:
    """Configure basic file logging under *log_dir*."""

    return configure_logging(log_dir=log_dir, level=level)


def get_csv_file_paths(directory: Path | None = None) -> list[Path]:
    """Return portfolio CSV files in *directory*. Defaults to configured dir."""

    repo = PortfolioRepository(_SETTINGS, directory=directory)
    return [definition.path for definition in repo.list_portfolios()]


def read_tickers_from_file(path: Path) -> Set[str]:
    """Read tickers from a CSV-style file containing one symbol per line."""

    return set(read_tickers(Path(path)))


class PortfolioTickerReader:
    """Legacy-compatible wrapper around :class:`PortfolioRepository`."""

    def __init__(self, directory: Path = PORTFOLIO_DIR) -> None:
        self.directory = Path(directory)
        self.repo = PortfolioRepository(_SETTINGS, directory=self.directory)
        self.portfolio_tickers: dict[str, Set[str]] = {}
        self.refresh()

    def refresh(self) -> None:
        self.repo = PortfolioRepository(_SETTINGS, directory=self.directory)
        self.portfolio_tickers = {
            definition.name: set(definition.tickers) for definition in self.repo.list_portfolios()
        }

    def get_portfolio_tickers(self, portfolio_name: str) -> Set[str]:
        return self.portfolio_tickers.get(portfolio_name, set())


def _collation_service(
    *,
    price_history_dir: Path,
    export_dir: Optional[Path] = None,
) -> CollationService:
    return CollationService(
        YFinancePriceFetcher(),
        _SETTINGS,
        price_history_dir=price_history_dir,
        export_dir=export_dir or Path(_SETTINGS.export_dir),
    )


def download_portfolio_prices(
    tickers: Iterable[str],
    *,
    price_history_dir: Path = PRICE_HISTORY_DIR,
    period: str = "max",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Download price histories for *tickers* and write each to CSV."""

    service = _collation_service(price_history_dir=Path(price_history_dir))
    return service.download_portfolio_prices(
        tickers,
        period=period,
        interval=interval,
        start=start,
        end=end,
    )


def collate_prices(
    portfolio_name: str,
    price_history_dir: Path,
    tickers: Sequence[str],
    *,
    export_dir: Path = EXPORT_DIR,
) -> pd.DataFrame:
    """Combine downloaded price histories into a single CSV per portfolio."""

    service = _collation_service(
        price_history_dir=Path(price_history_dir),
        export_dir=Path(export_dir),
    )
    return service.collate_portfolio(portfolio_name, list(tickers))


def process_portfolio(
    portfolio_file: Path,
    *,
    price_history_dir: Path = PRICE_HISTORY_DIR,
    export_dir: Path = EXPORT_DIR,
    period: str = "max",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download and collate prices for the portfolio described in *portfolio_file*."""

    portfolio_file = Path(portfolio_file)
    tickers = sorted(read_tickers_from_file(portfolio_file))
    if not tickers:
        logger.warning("No tickers found for portfolio: %s", portfolio_file.stem)
        return pd.DataFrame()

    service = _collation_service(
        price_history_dir=Path(price_history_dir),
        export_dir=Path(export_dir),
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
    portfolio_dir: Path = PORTFOLIO_DIR,
    *,
    price_history_dir: Path = PRICE_HISTORY_DIR,
    export_dir: Path = EXPORT_DIR,
    period: str = "max",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Run the download/collation workflow for every portfolio file."""

    workflow = PortfolioDownloadWorkflow(
        settings=_SETTINGS,
        portfolio_dir=Path(portfolio_dir),
        price_history_dir=Path(price_history_dir),
        export_dir=Path(export_dir),
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
    """Wrapper around yfinance helpers for single-ticker information."""

    def __init__(self, ticker: str):
        if not ticker:
            raise ValueError("Ticker symbol must be provided")
        self.ticker = ticker
        self._yf = _ensure_yfinance().Ticker(ticker)

    def get_company_name(self) -> str:
        try:
            return self._yf.info["shortName"]
        except KeyError:
            return self._yf.info.get("longName", self.ticker)

    def get_company_info(self) -> dict:
        return dict(self._yf.info)

    def download_info(self, destination: Path = INFO_DIR) -> Path:
        info = self.get_company_info()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        file_path = destination / f"{self.get_company_name().replace(' ', '_')}_summary.json"
        file_path.write_text(json.dumps(info, indent=4), encoding="utf-8")
        return file_path

    def get_news(self) -> list:
        return list(self._yf.news or [])

    def get_options(self) -> list:
        return list(self._yf.options or [])

    def get_earnings_dates(self) -> pd.DataFrame:
        data = getattr(self._yf, "earnings_dates", pd.DataFrame())
        return data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()

    def get_recommendations(self) -> dict:
        return {
            "recommendations": getattr(self._yf, "recommendations", None),
            "summary": getattr(self._yf, "recommendations_summary", None),
            "upgrades_downgrades": getattr(self._yf, "upgrades_downgrades", None),
        }

    def get_holders(self) -> dict:
        return {
            "major_holders": getattr(self._yf, "major_holders", None),
            "institutional_holders": getattr(self._yf, "institutional_holders", None),
            "mutualfund_holders": getattr(self._yf, "mutualfund_holders", None),
            "insider_purchases": getattr(self._yf, "insider_purchases", None),
            "insider_roster_holders": getattr(self._yf, "insider_roster_holders", None),
        }

    def get_financials(self) -> pd.DataFrame:
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
        return {
            "actions": getattr(self._yf, "actions", None),
            "dividends": getattr(self._yf, "dividends", None),
            "splits": getattr(self._yf, "splits", None),
            "capital_gains": getattr(self._yf, "capital_gains", None),
            "shares_history": self._yf.get_shares_full() if hasattr(self._yf, "get_shares_full") else None,
        }

    def get_summary(self) -> dict:
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
        fetcher = YFinancePriceFetcher()
        frame = fetcher.fetch_history(
            self.ticker,
            period=period,
            interval=interval,
            start=start,
            end=end,
        )
        destination = Path(destination)
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
