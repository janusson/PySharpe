"""High level helpers for working with local portfolio definitions."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import pandas as pd

from .fetch import fetch_price_history

logger = logging.getLogger(__name__)


def _find_project_root(start: Optional[Path] = None) -> Optional[Path]:
    """Return the repository root if it can be discovered."""

    search_root = (start or Path(__file__).resolve().parent).resolve()
    for candidate in [search_root, *search_root.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").is_dir():
            return candidate
    return None


def _resolve_default_data_root() -> Path:
    """Pick a default ``data`` directory mimicking the legacy scripts."""

    cwd_data = Path.cwd() / "data"
    if cwd_data.exists():
        return cwd_data

    project_root = _find_project_root()
    if project_root is not None:
        return project_root / "data"

    return cwd_data


DEFAULT_DATA_ROOT = _resolve_default_data_root()
DEFAULT_PORTFOLIO_DIR = DEFAULT_DATA_ROOT / "portfolio"
DEFAULT_PRICE_HISTORY_DIR = DEFAULT_DATA_ROOT / "price_hist"
DEFAULT_INFO_DIR = DEFAULT_DATA_ROOT / "info"
DEFAULT_EXPORT_DIR = DEFAULT_DATA_ROOT / "exports"

__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_PORTFOLIO_DIR",
    "DEFAULT_PRICE_HISTORY_DIR",
    "DEFAULT_EXPORT_DIR",
    "PortfolioTickerReader",
    "collate_prices",
    "download_portfolio_prices",
    "get_csv_file_paths",
    "process_portfolio",
    "process_all_portfolios",
    "read_tickers_from_file",
    "SecurityDataCollector",
    "save_collated_prices",
]


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def _prepare_ticker_set(tickers: Iterable[str]) -> List[str]:
    cleaned = {
        _normalize_ticker(symbol)
        for symbol in tickers
        if symbol and _normalize_ticker(symbol)
    }
    return sorted(cleaned)


def read_tickers_from_file(path: Path) -> Set[str]:
    """Return a set of tickers read from a CSV-style file.

    Each non-empty line is treated as a ticker symbol. Leading and trailing
    whitespace is stripped and symbols are normalised to upper-case. Lines that
    start with ``#`` are ignored so files can contain simple comments.
    """

    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        logger.error("Portfolio file not found: %s", path)
        return set()

    tickers: Set[str] = set()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tickers.add(_normalize_ticker(line))

    return tickers


def get_csv_file_paths(directory: Path) -> List[Path]:
    """Return all CSV files located directly under *directory*."""

    directory = Path(directory)
    if not directory.exists():
        logger.info("Portfolio directory does not exist: %s", directory)
        return []

    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory: {directory!s}")

    return sorted(path for path in directory.iterdir() if path.suffix.lower() == ".csv")


class PortfolioTickerReader:
    """Load portfolio tickers from CSV files located in a directory."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        if directory is None:
            directory = DEFAULT_PORTFOLIO_DIR
        self.directory = Path(directory) if directory is not None else None
        self.portfolio_tickers: Dict[str, Set[str]] = {}
        if self.directory is not None:
            self.refresh()

    def refresh(self) -> None:
        """Reload the portfolio definitions from disk."""

        if self.directory is None:
            self.portfolio_tickers.clear()
            return

        self.portfolio_tickers.clear()
        for csv_path in get_csv_file_paths(self.directory):
            tickers = read_tickers_from_file(csv_path)
            if not tickers:
                logger.warning("No tickers found in portfolio file: %s", csv_path)
                continue
            portfolio_name = csv_path.stem
            self.portfolio_tickers[portfolio_name] = tickers

    def get_portfolio_tickers(self, portfolio_name: str) -> Set[str]:
        """Return the tickers defined for *portfolio_name*."""

        if portfolio_name not in self.portfolio_tickers and self.directory is not None:
            # Attempt a targeted reload in case the portfolio was added after instantiation.
            csv_path = self.directory / f"{portfolio_name}.csv"
            if csv_path.exists():
                self.portfolio_tickers[portfolio_name] = read_tickers_from_file(csv_path)

        return set(self.portfolio_tickers.get(portfolio_name, set()))


def download_portfolio_prices(
    tickers: Iterable[str],
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    interval: str = "1d",
    destination: Optional[Path] = None,
) -> pd.DataFrame:
    """Download price history for the supplied *tickers*.

    When *destination* is provided, one CSV file per ticker is written to that
    directory (created automatically if it does not already exist). The
    downloaded prices are always returned as a DataFrame indexed by date.
    """

    symbols = _prepare_ticker_set(tickers)
    if not symbols:
        raise ValueError("At least one ticker symbol must be provided.")

    prices = fetch_price_history(symbols, start=start, end=end, interval=interval)
    prices = prices.loc[:, symbols].sort_index()

    if destination is not None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        for symbol in symbols:
            if symbol not in prices.columns:
                logger.warning("Downloaded prices missing expected symbol %s", symbol)
                continue

            symbol_history = prices[[symbol]].dropna(how="all")
            if symbol_history.empty:
                logger.debug("Skipping %s because no price history was returned", symbol)
                continue

            csv_frame = symbol_history.rename(columns={symbol: "Close"})
            csv_frame.index.name = "Date"
            csv_frame.to_csv(
                destination / f"{symbol}_hist.csv",
                index_label="Date",
            )

    return prices


def collate_prices(
    portfolio_name: str,
    csv_files: Sequence[Path],
    portfolio_tickers: Iterable[str],
) -> pd.DataFrame:
    """Collate closing prices for *portfolio_tickers* from *csv_files*."""

    tickers = {_normalize_ticker(ticker) for ticker in portfolio_tickers}
    if not tickers:
        logger.warning("No tickers provided for portfolio '%s'", portfolio_name)
        return pd.DataFrame()

    frames = []
    for csv_path in csv_files:
        path = Path(csv_path)
        if not path.exists():
            logger.debug("Skipping missing price file: %s", path)
            continue
        ticker = _normalize_ticker(path.stem.split("_")[0])
        if ticker not in tickers:
            continue
        try:
            df = pd.read_csv(path, usecols=["Date", "Close"])
        except ValueError:
            logger.error("Price file missing required columns: %s", path)
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["Date"]).set_index("Date")
        frames.append(df.rename(columns={"Close": ticker}))

    if not frames:
        logger.warning("No matching price files found for portfolio '%s'", portfolio_name)
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1).sort_index()
    return combined.dropna(axis=1, how="all")


def process_portfolio(
    portfolio_file: Path,
    *,
    price_history_dir: Optional[Path] = DEFAULT_PRICE_HISTORY_DIR,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    interval: str = "1d",
) -> Set[str]:
    """Read tickers from *portfolio_file* and optionally download their prices."""

    portfolio_file = Path(portfolio_file)
    portfolio_name = portfolio_file.stem
    tickers = read_tickers_from_file(portfolio_file)
    if not tickers:
        logger.warning("No tickers found for portfolio '%s'", portfolio_name)
        return set()

    if price_history_dir is not None:
        download_portfolio_prices(
            tickers,
            start=start,
            end=end,
            interval=interval,
            destination=price_history_dir,
        )

    return tickers


def save_collated_prices(
    portfolio_name: str,
    prices: pd.DataFrame,
    directory: Path,
) -> Path:
    """Persist *prices* for *portfolio_name* in the legacy collated layout."""

    if prices.empty:
        raise ValueError("Cannot save an empty price table.")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{portfolio_name}_collated.csv"
    prices.to_csv(path, index=True)
    return path


def process_all_portfolios(
    portfolio_dir: Path | str = DEFAULT_PORTFOLIO_DIR,
    *,
    price_history_dir: Path | str = DEFAULT_PRICE_HISTORY_DIR,
    collated_dir: Optional[Path | str] = DEFAULT_EXPORT_DIR,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Run the original data collection workflow for every portfolio file."""

    portfolio_dir = Path(portfolio_dir)
    price_history_dir = Path(price_history_dir)
    collated_path = Path(collated_dir) if collated_dir is not None else None

    results: Dict[str, pd.DataFrame] = {}

    for portfolio_file in get_csv_file_paths(portfolio_dir):
        portfolio_name = portfolio_file.stem
        tickers = process_portfolio(
            portfolio_file,
            price_history_dir=price_history_dir,
            start=start,
            end=end,
            interval=interval,
        )

        if not tickers:
            logger.warning("Skipping %s because no tickers were found", portfolio_name)
            continue

        csv_files = [price_history_dir / f"{ticker}_hist.csv" for ticker in tickers]
        prices = collate_prices(portfolio_name, csv_files, tickers)
        if prices.empty:
            logger.warning("Skipping %s because no price data could be collated", portfolio_name)
            continue

        if collated_path is not None:
            save_collated_prices(portfolio_name, prices, collated_path)

        results[portfolio_name] = prices

    return results


class SecurityDataCollector:
    """Convenience wrapper for retrieving security level information."""

    def __init__(self, ticker: str) -> None:
        if not ticker:
            raise ValueError("Ticker symbol must be provided.")
        self.ticker = _normalize_ticker(ticker)
        self._yf_ticker = None
        self._company_info: Optional[dict] = None

    def _ensure_ticker(self):
        if self._yf_ticker is None:
            try:
                import yfinance as yf
            except ImportError as exc:  # pragma: no cover - only hit when dependency missing
                raise RuntimeError(
                    "yfinance is required to use SecurityDataCollector"
                ) from exc
            self._yf_ticker = yf.Ticker(self.ticker)
        return self._yf_ticker

    def _safe_attribute(self, attribute: str, default, *, call: bool = False):
        ticker = self._ensure_ticker()
        try:
            value = getattr(ticker, attribute)
            if call and callable(value):
                value = value()
        except Exception as exc:  # pragma: no cover - network errors / API quirks
            logger.error("Error retrieving %s for %s: %s", attribute, self.ticker, exc)
            return default
        return value if value is not None else default

    def get_company_info(self) -> dict:
        if self._company_info is None:
            info = self._safe_attribute("info", {})
            self._company_info = dict(info) if isinstance(info, Mapping) else {}
        return dict(self._company_info)

    def get_company_name(self) -> str:
        info = self.get_company_info()
        return info.get("shortName") or info.get("longName") or self.ticker

    def download_info(self, destination: Optional[Path] = None) -> Path:
        info = self.get_company_info()
        if not info:
            raise ValueError(f"No company information available for {self.ticker}.")

        destination = Path(destination or DEFAULT_INFO_DIR)
        destination.mkdir(parents=True, exist_ok=True)
        file_name = f"{self.get_company_name().replace(' ', '_')}_summary.json"
        output_path = destination / file_name
        output_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        return output_path

    def get_news(self) -> List[dict]:
        news = self._safe_attribute("news", [])
        return list(news) if news else []

    def get_options(self) -> Sequence[str]:
        options = self._safe_attribute("options", ())
        return tuple(options) if options else ()

    def get_earnings_dates(self) -> pd.DataFrame:
        data = self._safe_attribute("earnings_dates", pd.DataFrame())
        return data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()

    def get_recommendations(self) -> Dict[str, pd.DataFrame]:
        return {
            "recommendations": self._convert_to_dataframe(
                self._safe_attribute("recommendations", pd.DataFrame())
            ),
            "summary": self._convert_to_dataframe(
                self._safe_attribute("recommendations_summary", pd.DataFrame())
            ),
            "upgrades_downgrades": self._convert_to_dataframe(
                self._safe_attribute("upgrades_downgrades", pd.DataFrame())
            ),
        }

    def get_holders(self) -> Dict[str, pd.DataFrame]:
        return {
            "major_holders": self._convert_to_dataframe(
                self._safe_attribute("major_holders", pd.DataFrame())
            ),
            "institutional_holders": self._convert_to_dataframe(
                self._safe_attribute("institutional_holders", pd.DataFrame())
            ),
            "mutualfund_holders": self._convert_to_dataframe(
                self._safe_attribute("mutualfund_holders", pd.DataFrame())
            ),
            "insider_purchases": self._convert_to_dataframe(
                self._safe_attribute("insider_purchases", pd.DataFrame())
            ),
            "insider_roster_holders": self._convert_to_dataframe(
                self._safe_attribute("insider_roster_holders", pd.DataFrame())
            ),
        }

    def get_financials(self) -> pd.DataFrame:
        sections = {
            "income_statement": "financials",
            "quarterly_income_statement": "quarterly_financials",
            "balance_sheet": "balance_sheet",
            "quarterly_balance_sheet": "quarterly_balance_sheet",
            "cash_flow": "cashflow",
            "quarterly_cash_flow": "quarterly_cashflow",
        }

        frames = {}
        for label, attribute in sections.items():
            frame = self._convert_to_dataframe(self._safe_attribute(attribute, pd.DataFrame()))
            if not frame.empty:
                frames[label] = frame

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, axis=1)

    def get_actions(self) -> Dict[str, pd.DataFrame]:
        return {
            "actions": self._convert_to_dataframe(
                self._safe_attribute("actions", pd.DataFrame())
            ),
            "dividends": self._convert_to_dataframe(
                self._safe_attribute("dividends", pd.DataFrame())
            ),
            "splits": self._convert_to_dataframe(
                self._safe_attribute("splits", pd.DataFrame())
            ),
            "capital_gains": self._convert_to_dataframe(
                self._safe_attribute("capital_gains", pd.Series())
            ),
            "shares_history": self._convert_to_dataframe(
                self._safe_attribute("get_shares_full", pd.DataFrame(), call=True)
            ),
        }

    def get_summary(self) -> Dict[str, Optional[object]]:
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

        info = self.get_company_info()
        return {key: info.get(key) for key in keys}

    def download_price_hist(
        self,
        destination: Optional[Path] = None,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download the price history for the security and persist it if requested."""

        return download_portfolio_prices(
            {self.ticker},
            start=start,
            end=end,
            interval=interval,
            destination=destination or DEFAULT_PRICE_HISTORY_DIR,
        )

    @staticmethod
    def _convert_to_dataframe(value) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, pd.Series):
            return value.to_frame()
        return pd.DataFrame()
