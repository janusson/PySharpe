"""High level orchestration for portfolio downloads."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from pysharpe.config import PySharpeSettings, get_settings

from .collation import CollationService
from .fetcher import PriceFetcher, YFinancePriceFetcher
from .portfolio import PortfolioDefinition, PortfolioRepository

logger = logging.getLogger(__name__)


class PortfolioDownloadWorkflow:
    """Orchestrate fetching and collating portfolios.

    Example:
        >>> from pysharpe.data.workflows import PortfolioDownloadWorkflow
        >>> workflow = PortfolioDownloadWorkflow()
        >>> isinstance(workflow, PortfolioDownloadWorkflow)
        True
    """

    def __init__(
        self,
        *,
        settings: PySharpeSettings | None = None,
        repository: PortfolioRepository | None = None,
        fetcher: PriceFetcher | None = None,
        collation: CollationService | None = None,
        price_history_dir: Path | None = None,
        export_dir: Path | None = None,
        portfolio_dir: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = repository or PortfolioRepository(self.settings, directory=portfolio_dir)
        self.fetcher = fetcher or YFinancePriceFetcher()
        self.collation = collation or CollationService(
            self.fetcher,
            self.settings,
            price_history_dir=price_history_dir,
            export_dir=export_dir,
        )

    def process_portfolio(
        self,
        portfolio: PortfolioDefinition,
        *,
        period: str,
        interval: str,
        start: str | None,
        end: str | None,
    ) -> pd.DataFrame:
        """Download and collate a single portfolio definition.

        Args:
            portfolio: Portfolio metadata describing tickers and path.
            period: Rolling window requested when explicit dates are absent.
            interval: Sampling interval for historical data.
            start: Optional start date in ISO format.
            end: Optional end date in ISO format.

        Returns:
            Collated price history (empty DataFrame when no data is available).

        Example:
            >>> from pysharpe.data.portfolio import PortfolioDefinition
            >>> from pathlib import Path
            >>> workflow = PortfolioDownloadWorkflow(fetcher=None)  # doctest: +SKIP
            >>> workflow.process_portfolio(  # doctest: +SKIP
            ...     PortfolioDefinition('demo', ('AAPL',), Path('demo.csv')),
            ...     period='1y', interval='1d', start=None, end=None,
            ... )
        """

        logger.info("Processing portfolio %s", portfolio.name)
        return self.collation.process_portfolio(
            portfolio.name,
            portfolio.tickers,
            period=period,
            interval=interval,
            start=start,
            end=end,
        )

    def process_portfolios(
        self,
        names: Iterable[str] | None,
        *,
        period: str,
        interval: str,
        start: str | None,
        end: str | None,
    ) -> Dict[str, pd.DataFrame]:
        """Batch process one or more portfolio definitions.

        Args:
            names: Optional iterable of portfolio names. When ``None`` every
                tracked portfolio is processed.
            period: Rolling window requested when explicit dates are absent.
            interval: Sampling interval for historical data.
            start: Optional ISO start date.
            end: Optional ISO end date.

        Returns:
            Mapping of portfolio name to collated DataFrame for each successful run.

        Example:
            >>> workflow = PortfolioDownloadWorkflow()
            >>> workflow.process_portfolios([], period='1y', interval='1d', start=None, end=None)
            {}
        """

        results: dict[str, pd.DataFrame] = {}
        definitions = list(self.repository.iter_definitions(names))
        for definition in definitions:
            frame = self.process_portfolio(
                definition,
                period=period,
                interval=interval,
                start=start,
                end=end,
            )
            if not frame.empty:
                results[definition.name] = frame
        return results
