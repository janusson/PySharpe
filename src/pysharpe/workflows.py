"""High-level workflows combining data ingestion and optimisation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from pysharpe.config import get_settings
from pysharpe.data import PortfolioDownloadWorkflow
from pysharpe.optimization.models import OptimisationResult
from pysharpe.portfolio_optimization import optimise_portfolio, optimise_all_portfolios

logger = logging.getLogger(__name__)


def download_portfolios(
    *,
    portfolio_names: Optional[Iterable[str]] = None,
    portfolio_dir: Path | None = None,
    price_history_dir: Path | None = None,
    export_dir: Path | None = None,
    period: str = "max",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Download and collate portfolios leveraging the data workflow.

    Args:
        portfolio_names: Optional iterable of portfolio names to process.
        portfolio_dir: Override for the portfolio CSV directory.
        price_history_dir: Override for raw price data directory.
        export_dir: Override for collated export directory.
        period: Rolling window used when explicit dates are missing.
        interval: Sampling interval for downloads.
        start: Optional ISO start date.
        end: Optional ISO end date.

    Returns:
        Mapping of portfolio name to collated DataFrame.

    Example:
        >>> from pysharpe.workflows import download_portfolios
        >>> download_portfolios(period='1y', interval='1d')  # doctest: +SKIP
        {'demo': ...}
    """

    settings = get_settings()
    workflow = PortfolioDownloadWorkflow(
        settings=settings,
        portfolio_dir=portfolio_dir or settings.portfolio_dir,
        price_history_dir=price_history_dir or settings.price_history_dir,
        export_dir=export_dir or settings.export_dir,
    )

    results = workflow.process_portfolios(
        names=portfolio_names,
        period=period,
        interval=interval,
        start=start,
        end=end,
    )
    if not results:
        logger.warning("No portfolios processed")
    return results


def optimise_portfolios(
    *,
    portfolio_names: Optional[Iterable[str]] = None,
    collated_dir: Path | None = None,
    output_dir: Path | None = None,
    time_constraint: Optional[str] = None,
    make_plot: bool = True,
) -> Dict[str, OptimisationResult]:
    """Optimise one or more portfolios and persist artefacts.

    Args:
        portfolio_names: Optional iterable of portfolio names. When ``None`` all
            collated CSVs under ``collated_dir`` are processed.
        collated_dir: Directory containing collated price histories.
        output_dir: Directory for optimisation artefacts.
        time_constraint: Optional ISO date slice for the collated histories.
        make_plot: When ``True`` generate allocation pie charts.

    Returns:
        Mapping of portfolio name to :class:`OptimisationResult` objects.

    Example:
        >>> from pysharpe.workflows import optimise_portfolios
        >>> optimise_portfolios(make_plot=False)  # doctest: +SKIP
        {'demo': OptimisationResult(...)}
    """

    settings = get_settings()
    collated_root = Path(collated_dir or settings.export_dir)
    output_root = Path(output_dir or settings.export_dir)

    if portfolio_names:
        targets = list(portfolio_names)
    else:
        targets = [
            path.stem.replace("_collated", "")
            for path in collated_root.glob("*_collated.csv")
        ]
        if not targets:
            logger.warning("No collated portfolios discovered for optimisation")
            return {}

    results: Dict[str, OptimisationResult] = {}
    for name in targets:
        try:
            results[name] = optimise_portfolio(
                name,
                collated_dir=collated_root,
                output_dir=output_root,
                time_constraint=time_constraint,
                make_plot=make_plot,
            )
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", name, exc)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", name, exc)

    if not results:
        logger.warning("No portfolios optimised successfully")
    return results


__all__ = ["download_portfolios", "optimise_portfolios", "optimise_all_portfolios"]
