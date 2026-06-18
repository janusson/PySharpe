"""High-level workflows combining data ingestion and optimisation."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from pysharpe.config import ExecutionConfig, get_settings
from pysharpe.data import PortfolioDownloadWorkflow
from pysharpe.exceptions import PySharpeError
from pysharpe.optimization.models import OptimisationResult
from pysharpe.portfolio_optimization import optimise_all_portfolios, optimise_portfolio

logger = logging.getLogger(__name__)


def download_portfolios(
    *,
    portfolio_names: Iterable[str] | None = None,
    portfolio_dir: Path | None = None,
    price_history_dir: Path | None = None,
    export_dir: Path | None = None,
    period: str = "max",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
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
    portfolio_names: Iterable[str] | None = None,
    collated_dir: Path | None = None,
    output_dir: Path | None = None,
    time_constraint: str | None = None,
    mer_mapping: dict[str, float] | None = None,
    max_portfolio_mer: float | None = None,
    geo_mapping: dict[str, str] | None = None,
    geo_lower_bounds: dict[str, float] | None = None,
    geo_upper_bounds: dict[str, float] | None = None,
    make_plot: bool = True,
    category_map: dict[str, str] | None = None,
    include_unmapped_categories: bool = True,
    return_model: str = "shrinkage",
    base_currency: str = "CAD",
    max_weight: float = 0.20,
    shrinkage_floor: float = 0.3,
    execution_config: ExecutionConfig | None = None,
    proxy_map: dict[str, dict[str, object]] | None = None,
) -> dict[str, OptimisationResult]:
    """Optimise one or more portfolios and persist artefacts.

    Args:
        portfolio_names: Optional iterable of portfolio names. When ``None`` all
            collated CSVs under ``collated_dir`` are processed.
        collated_dir: Directory containing collated price histories.
        output_dir: Directory for optimisation artefacts.
        time_constraint: Optional ISO date slice for the collated histories.
        mer_mapping: Optional mapping of ticker to its MER (decimal).
        max_portfolio_mer: Maximum allowable weighted MER for the portfolio.
        geo_mapping: Optional mapping of ticker to geographic region.
        geo_lower_bounds: Minimum weight per region.
        geo_upper_bounds: Maximum weight per region.
        make_plot: When ``True`` generate allocation pie charts.
        category_map: Optional mapping of ticker -> category label used to group
            exposures before optimisation.
        include_unmapped_categories: When ``True`` keep tickers that are not
            present in ``category_map`` as standalone categories.
        return_model: Expected return calculation method. 'ema' or 'mean'. Defaults to 'ema'.
        base_currency: The target currency for all assets (default "CAD").
        max_weight: Maximum allowable weight for any single asset (default 0.20).
        shrinkage_floor: Minimum shrinkage intensity for 'shrinkage' return model
            (0.0-1.0, default 0.3). Higher values pull estimates more aggressively
            toward the grand mean, reducing recency bias.

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

    results: dict[str, OptimisationResult] = {}
    for name in targets:
        try:
            results[name] = optimise_portfolio(
                name,
                collated_dir=collated_root,
                output_dir=output_root,
                time_constraint=time_constraint,
                mer_mapping=mer_mapping,
                max_portfolio_mer=max_portfolio_mer,
                geo_mapping=geo_mapping,
                geo_lower_bounds=geo_lower_bounds,
                geo_upper_bounds=geo_upper_bounds,
                make_plot=make_plot,
                category_map=category_map,
                include_unmapped_categories=include_unmapped_categories,
                return_model=return_model,
                base_currency=base_currency,
                max_weight=max_weight,
                shrinkage_floor=shrinkage_floor,
                execution_config=execution_config,
                proxy_map=proxy_map,
            )
        except (FileNotFoundError, ValueError, PySharpeError) as exc:
            logger.warning("Skipping %s: %s", name, exc)

    if not results:
        logger.warning("No portfolios optimised successfully")
    return results


__all__ = ["download_portfolios", "optimise_portfolios", "optimise_all_portfolios"]
