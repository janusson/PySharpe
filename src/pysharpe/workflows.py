"""High-level workflows combining data ingestion and optimisation."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from pysharpe.config import ExecutionConfig, get_settings
from pysharpe.data import PortfolioDownloadWorkflow
from pysharpe.data.fetcher import apply_fx_conversion
from pysharpe.exceptions import PySharpeError
from pysharpe.optimization.models import OptimisationResult
from pysharpe.portfolio_optimization import optimise_all_portfolios, optimise_portfolio
from pysharpe.visualization.utils import require_matplotlib

if TYPE_CHECKING:  # pragma: no cover - type checking aide
    from matplotlib.axes import Axes

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


def plot_holdings_history(
    *,
    portfolio_names: Iterable[str] | None = None,
    collated_dir: Path | None = None,
    output_dir: Path | None = None,
    weights: dict[str, dict[str, float]] | None = None,
    time_constraint: str | None = None,
    base_currency: str = "CAD",
    apply_fx: bool = True,
    show: bool = False,
    title: str | None = None,
) -> Axes:
    """Plot cumulative returns for one or more portfolios on a single chart.

    Collated price histories are loaded for every portfolio in
    *portfolio_names*, optionally adjusted into *base_currency* via
    :func:`~pysharpe.data.fetcher.apply_fx_conversion`, and then cumulated
    — respecting the weights in *weights* when supplied, or assuming
    equal-weighting otherwise.

    This is the recommended entry-point for comparing hedged versus unhedged
    performance: pass ``apply_fx=True`` for the hedged variant and
    ``apply_fx=False`` (calling a second time) for the unhedged variant,
    then overlay the two Axes.

    Args:
        portfolio_names: Iterable of portfolio names to compare.  When
            ``None`` all collated CSVs under *collated_dir* are used.
        collated_dir: Directory containing ``*_collated.csv`` files.
            Defaults to the ``export_dir`` from settings.
        output_dir: Directory where the plot PNG is saved.  When ``None``
            the plot is not persisted to disk.
        weights: Optional mapping of portfolio name → ``{ticker: weight}``.
            Portfolios not present in the mapping are treated as equally
            weighted.
        time_constraint: Optional ISO date to filter the collated histories.
        base_currency: FX target currency applied when *apply_fx* is
            ``True`` (default ``"CAD"``).
        apply_fx: When ``True`` call :func:`apply_fx_conversion` on each
            price DataFrame before calculating returns.
        show: When ``True`` call ``plt.show()`` before returning.
        title: Optional plot title override.

    Returns:
        The matplotlib :class:`~matplotlib.axes.Axes` containing the plot.

    Example:
        >>> from pysharpe.workflows import plot_holdings_history
        >>> plot_holdings_history(  # doctest: +SKIP
        ...     portfolio_names=["demo"],
        ...     apply_fx=False,
        ... )
    """
    settings = get_settings()
    collated_root = Path(collated_dir or settings.export_dir)
    output_root = Path(output_dir) if output_dir else None

    if portfolio_names:
        targets = list(portfolio_names)
    else:
        targets = sorted(
            path.stem.replace("_collated", "")
            for path in collated_root.glob("*_collated.csv")
        )
        if not targets:
            raise FileNotFoundError(f"No collated portfolios found in {collated_root}")

    # --- load and optionally FX-convert each portfolio's price history ---
    portfolio_returns: dict[str, pd.Series] = {}

    for name in targets:
        csv_path = collated_root / f"{name}_collated.csv"
        if not csv_path.exists():
            logger.warning("Skipping %s: collated CSV not found at %s", name, csv_path)
            continue

        prices = pd.read_csv(csv_path, parse_dates=True, index_col="Date")
        if time_constraint:
            prices = prices.sort_index()
            prices = prices.loc[time_constraint:]

        if prices.empty:
            logger.warning("Skipping %s: empty after time constraint", name)
            continue

        if prices.isnull().values.any():
            prices = prices.ffill()

        if apply_fx:
            prices = apply_fx_conversion(prices, base_currency=base_currency)

        prices = prices.dropna()
        if prices.empty:
            logger.warning("Skipping %s: no data after FX conversion and dropna", name)
            continue

        alloc = weights.get(name) if weights else None

        if alloc is not None:
            # Weighted portfolio return.
            alloc_series = pd.Series(alloc)
            common = [t for t in prices.columns if t in alloc_series.index]
            if not common:
                logger.warning("Skipping %s: no tickers match supplied weights", name)
                continue
            sub = prices[common]
            daily_ret = sub.pct_change().dropna()
            w = alloc_series[common] / alloc_series[common].sum()
            weighted_ret = daily_ret.dot(w)
        else:
            # Equal-weighted basket.
            daily_ret = prices.pct_change().dropna()
            weighted_ret = daily_ret.mean(axis=1)

        cumulative = (1 + weighted_ret).cumprod()  # type: ignore[assignment]
        assert isinstance(cumulative, pd.Series)  # narrow for pyright
        cumulative = cumulative / cumulative.iloc[0]  # normalise to 1.0
        portfolio_returns[name] = cumulative

    if not portfolio_returns:
        raise ValueError("No valid portfolio return series to plot.")

    # --- plot ---
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (label, series) in enumerate(portfolio_returns.items()):
        colour = colours[idx % len(colours)]
        ax.plot(
            series.index,
            series.to_numpy(),
            label=label,
            linewidth=2,
            color=colour,
        )

    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Cumulative Return (normalised)", fontweight="bold")

    if title is None:
        start = max(s.index.min() for s in portfolio_returns.values())  # type: ignore[type-var]
        title = f"Cumulative Returns from {pd.Timestamp(start).strftime('%Y-%m-%d')}"
    ax.set_title(title, fontweight="bold")

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.7, linewidth=1)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)

    if output_root:
        output_root.mkdir(parents=True, exist_ok=True)
        stem = "_".join(sorted(portfolio_returns.keys()))
        save_path = output_root / f"{stem}_holdings_history.png"
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved holdings history plot to %s", save_path)

    if show:
        plt.show()

    return ax


__all__ = [
    "download_portfolios",
    "optimise_portfolios",
    "optimise_all_portfolios",
    "plot_holdings_history",
]
