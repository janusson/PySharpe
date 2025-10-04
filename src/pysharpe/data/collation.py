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
    """Coordinate price downloads, CSV normalisation, and metadata capture."""
    # Change: Added class-level summary docstring to improve readability.
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

    def _load_price_frame(self, ticker: str) -> pd.DataFrame | None:
        """Load and sanitise a single ticker CSV for downstream concatenation."""
        # Change: Documented helper to clarify sanitisation expectations.
        csv_path = self.price_history_dir / f"{ticker}_hist.csv"
        if not csv_path.exists():
            logger.warning("Price history missing for %s", ticker)
            return None

        raw = pd.read_csv(csv_path)
        if {"Date", "Close"}.difference(raw.columns):
            logger.error("Unexpected columns in %s", csv_path.name)
            return None

        timestamps = pd.to_datetime(raw["Date"], utc=True, errors="coerce")
        valid_mask = ~timestamps.isna()
        if not valid_mask.any():
            logger.error("Unable to parse dates for %s", csv_path.name)
            return None

        frame = raw.loc[valid_mask, ["Close"]].copy()
        frame["Date"] = (
            timestamps.loc[valid_mask]
            .tz_convert(None)
            .strftime("%Y-%m-%d")
        )
        frame.dropna(subset=["Close"], inplace=True)
        if frame.empty:
            logger.warning("No valid price points retained for %s", ticker)
            return None

        frame.set_index("Date", inplace=True)
        return frame.rename(columns={"Close": ticker})

    def collate_portfolio(
        self,
        name: str,
        tickers: Sequence[str],
    ) -> pd.DataFrame:
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
        payload = {
            "name": name,
            "requested_tickers": list(requested),
            "included_tickers": list(included),
            "dropped_tickers": sorted(requested_set - included_set),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "artifact_version": self.settings.artifact_version,
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
