"""Utilities for downloading and collating portfolio price data.

This module recreates the behaviour of the original standalone
``data_collector.py`` script while organising the code into importable
functions. The defaults mirror the legacy workflow: portfolios are defined in
``./data/portfolio`` and downloads land under ``./data/price_hist`` with
collated outputs in ``./data/exports``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import pandas as pd

try:  # Lazy import so tests can stub yfinance easily.
    import yfinance as yf
except ImportError:  # pragma: no cover - surfaced at call sites
    yf = None  # type: ignore[assignment]


DATA_DIR = Path("data")
PORTFOLIO_DIR = DATA_DIR / "portfolio"
PRICE_HISTORY_DIR = DATA_DIR / "price_hist"
EXPORT_DIR = DATA_DIR / "exports"
INFO_DIR = DATA_DIR / "info"
LOG_DIR = Path("logs")

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path = LOG_DIR) -> None:
    """Configure basic logging to a dated file under *log_dir*."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = datetime.now().strftime("%Y-%m-%d_log.log")
    log_path = log_dir / log_name
    logging.basicConfig(
        level=logging.INFO,
        filename=str(log_path),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.info("Logging initialised: %s", log_path)


def _ensure_yfinance() -> "yf":  # type: ignore[name-defined]
    if yf is None:
        raise RuntimeError(
            "yfinance must be installed to download market data."
        )
    return yf


def get_csv_file_paths(directory: Path) -> List[Path]:
    """Return CSV files in *directory*, mirroring the original script."""

    directory = Path(directory)
    if not directory.exists():
        logger.error("Portfolio directory not found: %s", directory)
        return []

    return sorted(path for path in directory.iterdir() if path.suffix.lower() == ".csv")


def read_tickers_from_file(path: Path) -> Set[str]:
    """Read tickers from a CSV-style file containing one symbol per line."""

    tickers: Set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if cleaned and not cleaned.startswith("#"):
            tickers.add(cleaned)
    return tickers


class PortfolioTickerReader:
    """Load portfolio definitions from the legacy ``data/portfolio`` directory."""

    def __init__(self, directory: Path = PORTFOLIO_DIR) -> None:
        self.directory = Path(directory)
        self.portfolio_tickers: dict[str, Set[str]] = {}
        self.refresh()

    def refresh(self) -> None:
        self.portfolio_tickers.clear()

        for csv_path in get_csv_file_paths(self.directory):
            portfolio_name = csv_path.stem
            try:
                tickers = read_tickers_from_file(csv_path)
            except FileNotFoundError:
                logger.error("Portfolio file missing: %s", csv_path)
                continue

            if not tickers:
                logger.warning("No tickers found in %s", csv_path)
                continue
            self.portfolio_tickers[portfolio_name] = tickers

    def get_portfolio_tickers(self, portfolio_name: str) -> Set[str]:
        return self.portfolio_tickers.get(portfolio_name, set())


def download_portfolio_prices(
    tickers: Iterable[str],
    *,
    price_history_dir: Path = PRICE_HISTORY_DIR,
    period: str = "max",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Download price histories for *tickers* and write each to CSV.

    Behaviour mirrors the original implementation: each ticker is downloaded
    individually using ``yfinance.Ticker.history`` and persisted under
    ``data/price_hist`` (or ``price_history_dir`` when supplied).
    """

    yf_module = _ensure_yfinance()
    price_history_dir = Path(price_history_dir)
    price_history_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}

    for ticker in sorted(set(tickers)):
        logger.info("Downloading price history for: %s", ticker)
        try:
            history = yf_module.Ticker(ticker).history(period=period, interval=interval)
        except Exception as exc:  # pragma: no cover - API/network errors
            logger.error("Error downloading %s: %s", ticker, exc)
            continue

        if history.empty:
            logger.warning("No data returned for %s", ticker)
            continue

        csv_path = price_history_dir / f"{ticker}_hist.csv"
        history.to_csv(csv_path)
        results[ticker] = history

    return results


def collate_prices(
    portfolio_name: str,
    price_history_dir: Path,
    tickers: Sequence[str],
    *,
    export_dir: Path = EXPORT_DIR,
) -> pd.DataFrame:
    """Combine downloaded price histories into a single CSV per portfolio."""

    frames = []
    price_history_dir = Path(price_history_dir)

    for ticker in tickers:
        csv_path = price_history_dir / f"{ticker}_hist.csv"
        if not csv_path.exists():
            logger.warning("Price history missing for %s", ticker)
            continue

        df = pd.read_csv(csv_path)
        if "Date" not in df.columns or "Close" not in df.columns:
            logger.error("Unexpected columns in %s", csv_path)
            continue

        timestamps = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        if timestamps.isna().all():
            logger.error("Unable to parse dates for %s", csv_path)
            continue

        df = df.loc[~timestamps.isna()].copy()
        df["Date"] = timestamps.loc[~timestamps.isna()].dt.tz_convert(None).dt.date.astype(str)
        df.set_index("Date", inplace=True)
        frames.append(df[["Close"]].rename(columns={"Close": ticker}))

    if not frames:
        logger.warning("No price data collated for %s", portfolio_name)
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    cleaned = combined.dropna(axis=1, how="all")

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    output_path = export_dir / f"{portfolio_name}_collated.csv"
    cleaned.to_csv(output_path)

    return cleaned


def process_portfolio(
    portfolio_file: Path,
    *,
    price_history_dir: Path = PRICE_HISTORY_DIR,
    export_dir: Path = EXPORT_DIR,
    period: str = "max",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download and collate prices for the portfolio described in *portfolio_file*."""

    portfolio_file = Path(portfolio_file)
    portfolio_name = portfolio_file.stem
    tickers = read_tickers_from_file(portfolio_file)

    if not tickers:
        logger.warning("No tickers found for portfolio: %s", portfolio_name)
        return pd.DataFrame()

    download_portfolio_prices(
        tickers,
        price_history_dir=price_history_dir,
        period=period,
        interval=interval,
    )

    return collate_prices(
        portfolio_name,
        price_history_dir=price_history_dir,
        tickers=sorted(tickers),
        export_dir=export_dir,
    )


def process_all_portfolios(
    portfolio_dir: Path = PORTFOLIO_DIR,
    *,
    price_history_dir: Path = PRICE_HISTORY_DIR,
    export_dir: Path = EXPORT_DIR,
    period: str = "max",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Run the full download/collation workflow for every portfolio file."""

    results: dict[str, pd.DataFrame] = {}

    for csv_path in get_csv_file_paths(Path(portfolio_dir)):
        logger.info("Processing portfolio: %s", csv_path)
        frame = process_portfolio(
            csv_path,
            price_history_dir=price_history_dir,
            export_dir=export_dir,
            period=period,
            interval=interval,
        )
        if not frame.empty:
            results[csv_path.stem] = frame

    return results


class SecurityDataCollector:
    """Wrapper around ``yfinance`` helpers for single-ticker information."""

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
    ) -> pd.DataFrame:
        data = yf.Ticker(self.ticker).history(period=period, interval=interval)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / f"{self.ticker}_hist.csv"
        data.to_csv(output)
        return data


def main() -> None:  # pragma: no cover - script entry point
    setup_logging()

    csv_files = get_csv_file_paths(PORTFOLIO_DIR)
    for portfolio in csv_files:
        tickers = read_tickers_from_file(portfolio)
        logger.info("Tickers for %s: %s", portfolio.stem, ", ".join(sorted(tickers)))
        process_portfolio(portfolio)


if __name__ == "__main__":  # pragma: no cover
    main()
