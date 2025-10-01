"""Price history fetching abstractions."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PriceHistoryError(RuntimeError):
    """Raised when a price history request fails."""


class PriceFetcher(ABC):
    """Abstract base class for price history providers."""

    @abstractmethod
    def fetch_history(
        self,
        ticker: str,
        *,
        period: str,
        interval: str,
        start: Optional[str],
        end: Optional[str],
    ) -> pd.DataFrame:
        raise NotImplementedError


class YFinancePriceFetcher(PriceFetcher):
    """yfinance-backed fetcher with thin logging and validation."""

    def __init__(self, history_kwargs: Optional[Dict[str, object]] = None) -> None:
        self._history_overrides = history_kwargs or {}

    def _lazy_module(self):
        try:
            import yfinance as yf  # type: ignore
        except ImportError as exc:
            raise PriceHistoryError(
                "yfinance must be installed to download market data."
            ) from exc
        return yf

    def fetch_history(
        self,
        ticker: str,
        *,
        period: str,
        interval: str,
        start: Optional[str],
        end: Optional[str],
    ) -> pd.DataFrame:
        yf = self._lazy_module()
        request_payload: Dict[str, object] = {"interval": interval}
        if start:
            request_payload["start"] = start
        if end:
            request_payload["end"] = end
        if not start and not end:
            request_payload["period"] = period

        request_payload.update(self._history_overrides)

        logger.info("Fetching history for %s", ticker)
        try:
            history = yf.Ticker(ticker).history(**request_payload)
        except Exception as exc:  # pragma: no cover - network issues
            raise PriceHistoryError(f"Failed to download history for {ticker}: {exc}") from exc

        if history.empty:
            raise PriceHistoryError(f"No data returned for {ticker}")

        return history
