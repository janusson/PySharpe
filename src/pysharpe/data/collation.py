"""Combine fetched price histories into collated portfolio datasets."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from pysharpe.config import PySharpeSettings, get_settings

from .fetcher import PriceFetcher, PriceHistoryError

logger = logging.getLogger(__name__)


class CollationService:
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

    def collate_portfolio(
        self,
        name: str,
        tickers: Sequence[str],
    ) -> pd.DataFrame:
        frames = []
        for ticker in tickers:
            csv_path = self.price_history_dir / f"{ticker}_hist.csv"
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
            logger.warning("No price data collated for %s", name)
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1)
        cleaned = combined.dropna(axis=1, how="all")

        output_path = self.export_dir / f"{name}_collated.csv"
        cleaned.to_csv(output_path)
        self._write_metadata(name, tickers)
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

    def _write_metadata(self, name: str, tickers: Sequence[str]) -> None:
        metadata_path = self.export_dir / f"{name}_metadata.json"
        payload = {
            "name": name,
            "tickers": list(tickers),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "artifact_version": self.settings.artifact_version,
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
