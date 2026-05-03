"""Portfolio optimisation helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import ema_historical_return, mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

from pysharpe.analysis import apply_category_mapping
from pysharpe.config import get_settings
from pysharpe.optimization.models import (
    OptimisationPerformance,
    OptimisationResult,
    PortfolioWeights,
)
from pysharpe.visualization.utils import require_matplotlib

_SETTINGS = get_settings()
EXPORT_DIR = Path(_SETTINGS.export_dir)

logger = logging.getLogger(__name__)


class AllocationPlotStrategy(Protocol):
    """Callable responsible for handling allocation plot side-effects."""

    def __call__(self, result: OptimisationResult, output_dir: Path) -> None:
        """Execute the plotting side-effect."""


class _GenerateAllocationPlot:
    def __call__(self, result: OptimisationResult, output_dir: Path) -> None:
        try:
            _plot_allocation(result, output_dir)
        except RuntimeError as exc:
            logger.warning("Skipping allocation plot for %s: %s", result.name, exc)


class _SkipAllocationPlot:
    def __call__(self, result: OptimisationResult, output_dir: Path) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Allocation plot disabled for %s", result.name)


_PLOT_STRATEGIES: dict[bool, AllocationPlotStrategy] = {
    True: _GenerateAllocationPlot(),
    False: _SkipAllocationPlot(),
}


def _resolve_plot_strategy(make_plot: bool) -> AllocationPlotStrategy:
    """Return an allocation plotting strategy derived from *make_plot*."""

    return _PLOT_STRATEGIES[bool(make_plot)]


@lru_cache(maxsize=32)
def _cached_collated_prices(
    portfolio_name: str,
    collated_dir: str,
    time_constraint: str | None,
) -> pd.DataFrame:
    csv_path = Path(collated_dir) / f"{portfolio_name}_collated.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Collated prices not found for {portfolio_name}: {csv_path}"
        )

    frame = pd.read_csv(csv_path, parse_dates=True, index_col="Date")
    if time_constraint:
        frame = frame.sort_index()
        frame = frame.loc[time_constraint:]

    if frame.empty:
        raise ValueError(
            f"No data available for {portfolio_name} after applying constraint"
        )

    if frame.isnull().values.any():
        frame = frame.ffill()

    return frame


def _load_collated_prices(
    portfolio_name: str,
    collated_dir: Path,
    *,
    time_constraint: str | None = None,
) -> pd.DataFrame:
    cached = _cached_collated_prices(
        portfolio_name,
        str(Path(collated_dir).resolve()),
        time_constraint,
    )
    return cached.copy(deep=True)


def _prepare_output_dir(path: Path) -> Path:
    target = Path(path).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    return target


def _plot_allocation(result: OptimisationResult, output_dir: Path) -> Path:
    weights = result.weights.non_zero()
    if not weights:
        raise ValueError("No positive weights to plot")

    plt = require_matplotlib()
    target_dir = _prepare_output_dir(output_dir)
    output_path = target_dir / f"{result.name}_allocation.png"

    fig, ax = plt.subplots()
    ax.pie(weights.values(), labels=weights.keys(), autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    ax.set_title(f"{result.name} Allocation")
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def _save_weights(result: OptimisationResult, output_dir: Path) -> Path:
    target_dir = _prepare_output_dir(output_dir)
    output_path = target_dir / f"{result.name}_weights.txt"

    lines = ["ticker,weight"]
    for ticker, weight in result.weights.allocations.items():
        lines.append(f"{ticker},{weight:.8f}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _save_performance(result: OptimisationResult, output_dir: Path) -> Path:
    target_dir = _prepare_output_dir(output_dir)
    output_path = target_dir / f"{result.name}_performance.txt"

    perf = result.performance
    lines = [
        f"Expected annual return: {perf.expected_return * 100:.2f}%",
        f"Annual volatility: {perf.volatility * 100:.2f}%",
        f"Sharpe Ratio: {perf.sharpe_ratio:.2f}",
    ]
    if perf.portfolio_mer is not None:
        lines.append(f"Portfolio MER: {perf.portfolio_mer * 100:.3f}%")

    lines.append(f"Optimization Window: {perf.start_date} to {perf.end_date}")
    if perf.limiting_ticker:
        lines.append(f"Limiting Ticker: {perf.limiting_ticker}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def optimise_portfolio(
    portfolio_name: str,
    *,
    collated_dir: Path = EXPORT_DIR,
    output_dir: Path = EXPORT_DIR,
    time_constraint: str | None = None,
    asset_constraints: dict[str, float] | None = None,
    mer_mapping: dict[str, float] | None = None,
    max_portfolio_mer: float | None = None,
    geo_mapping: dict[str, str] | None = None,
    geo_lower_bounds: dict[str, float] | None = None,
    geo_upper_bounds: dict[str, float] | None = None,
    make_plot: bool = True,
    category_map: Mapping[str, str] | None = None,
    include_unmapped_categories: bool = True,
    return_model: str = "ema",
) -> OptimisationResult:
    """Optimise a portfolio using the PyPortfolioOpt max Sharpe workflow.

    Args:
        portfolio_name: Name of the portfolio (also the collated CSV stem).
        collated_dir: Directory containing ``*_collated.csv`` files.
        output_dir: Directory where optimisation artefacts are written.
        time_constraint: Optional ISO date to filter the collated history.
        asset_constraints: Optional dict with ``min_weight``/``max_weight`` keys
            applied as linear constraints.
        mer_mapping: Optional mapping of ticker to its MER (decimal).
        max_portfolio_mer: Maximum allowable weighted MER for the portfolio.
        geo_mapping: Optional mapping of ticker to geographic region.
        geo_lower_bounds: Minimum weight per region.
        geo_upper_bounds: Maximum weight per region.
        make_plot: When ``True`` generate a pie chart of positive weights.
        category_map: Optional mapping of ticker -> category label used to
            collapse highly correlated exposures before optimisation.
        include_unmapped_categories: When ``True`` retain tickers that do not
            appear in ``category_map`` as standalone categories.
        return_model: Expected return calculation method. 'ema' (Exponential
            Moving Average) or 'mean' (Arithmetic Mean). Defaults to 'ema'.

    Returns:
        :class:`OptimisationResult` containing weights and performance stats.

    Raises:
        FileNotFoundError: If the collated CSV is missing.
        ValueError: If the constraint removes all usable data.

    Example:
        >>> from pysharpe.portfolio_optimization import optimise_portfolio
        >>> optimise_portfolio('demo', make_plot=False)  # doctest: +SKIP
        OptimisationResult(...)
    """

    plot_strategy = _resolve_plot_strategy(make_plot)

    return _optimise_portfolio_impl(
        portfolio_name=portfolio_name,
        collated_dir=collated_dir,
        output_dir=output_dir,
        time_constraint=time_constraint,
        asset_constraints=asset_constraints,
        mer_mapping=mer_mapping,
        max_portfolio_mer=max_portfolio_mer,
        geo_mapping=geo_mapping,
        geo_lower_bounds=geo_lower_bounds,
        geo_upper_bounds=geo_upper_bounds,
        category_map=category_map,
        include_unmapped_categories=include_unmapped_categories,
        plot_strategy=plot_strategy,
        return_model=return_model,
    )


def _optimise_portfolio_impl(
    *,
    portfolio_name: str,
    collated_dir: Path,
    output_dir: Path,
    time_constraint: str | None,
    asset_constraints: dict[str, float] | None,
    mer_mapping: dict[str, float] | None,
    max_portfolio_mer: float | None,
    geo_mapping: dict[str, str] | None,
    geo_lower_bounds: dict[str, float] | None,
    geo_upper_bounds: dict[str, float] | None,
    category_map: Mapping[str, str] | None,
    include_unmapped_categories: bool,
    plot_strategy: AllocationPlotStrategy,
    return_model: str,
) -> OptimisationResult:
    """Implementation detail powering :func:`optimise_portfolio`."""

    prices = _load_collated_prices(
        portfolio_name, collated_dir, time_constraint=time_constraint
    )

    first_valid_dates = prices.apply(lambda col: col.first_valid_index())
    limiting_ticker = first_valid_dates.idxmax()
    prices = prices.dropna()
    start_date = prices.index.min().strftime("%Y-%m-%d")
    end_date = prices.index.max().strftime("%Y-%m-%d")

    if category_map:
        try:
            aggregation = apply_category_mapping(
                prices,
                category_map,
                include_unmapped=include_unmapped_categories,
            )
        except ValueError as exc:
            raise ValueError(
                f"No data available for {portfolio_name} after applying category mapping."
            ) from exc

        if aggregation.dropped:
            logger.warning(
                "Dropping tickers for %s without category assignment: %s",
                portfolio_name,
                ", ".join(sorted(aggregation.dropped)),
            )

        for category, members in aggregation.groups.items():
            if len(members) > 1 and logger.isEnabledFor(logging.INFO):
                logger.info(
                    "Grouped %s into category %s for optimisation",
                    ", ".join(members),
                    category,
                )

        prices = aggregation.prices

    if return_model.lower() == "ema":
        mu = ema_historical_return(prices)
    else:
        mu = mean_historical_return(prices)

    try:
        cov = CovarianceShrinkage(prices).ledoit_wolf()
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - sklearn optional
        logger.warning(
            "scikit-learn is missing. Falling back to sample covariance instead of Ledoit-Wolf shrinkage."
        )
        cov = prices.pct_change().dropna().cov()
    ef = EfficientFrontier(mu, cov)

    if asset_constraints:
        if "min_weight" in asset_constraints:
            ef.add_constraint(lambda w: w >= asset_constraints["min_weight"])
        if "max_weight" in asset_constraints:
            ef.add_constraint(lambda w: w <= asset_constraints["max_weight"])

    if geo_mapping:
        available_sectors = {geo_mapping.get(t) for t in ef.tickers if t in geo_mapping}

        safe_lower = {}
        for k, v in (geo_lower_bounds or {}).items():
            if k in available_sectors:
                safe_lower[k] = v
            else:
                logger.warning(
                    "Portfolio %s lacks assets in sector '%s'; ignoring lower bound constraint.",
                    portfolio_name,
                    k,
                )

        safe_upper = {
            k: v for k, v in (geo_upper_bounds or {}).items() if k in available_sectors
        }

        ef.add_sector_constraints(
            geo_mapping,
            sector_lower=safe_lower,
            sector_upper=safe_upper,
        )

    if mer_mapping and max_portfolio_mer is not None:
        aligned_mers = np.array([mer_mapping.get(t, 0.0) for t in ef.tickers])
        # For max_sharpe, w is scaled by k, so we must scale the RHS by sum(w)
        # Using subtraction form to satisfy DCP rules (affine <= constant)
        ef.add_constraint(
            lambda w: w @ aligned_mers - max_portfolio_mer * cp.sum(w) <= 0
        )

    ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    expected, volatility, sharpe = ef.portfolio_performance(verbose=False)

    portfolio_mer: float | None = None
    if mer_mapping:
        portfolio_mer = sum(
            weight * mer_mapping.get(ticker, 0.0)
            for ticker, weight in cleaned_weights.items()
        )

    result = OptimisationResult(
        name=portfolio_name,
        weights=PortfolioWeights(cleaned_weights),
        performance=OptimisationPerformance(
            expected,
            volatility,
            sharpe,
            start_date=start_date,
            end_date=end_date,
            limiting_ticker=str(limiting_ticker),
            portfolio_mer=portfolio_mer,
        ),
    )

    _save_weights(result, output_dir)
    _save_performance(result, output_dir)
    plot_strategy(result, output_dir)

    return result


def optimise_all_portfolios(
    collated_dir: Path = EXPORT_DIR,
    *,
    output_dir: Path = EXPORT_DIR,
    time_constraint: str | None = None,
    mer_mapping: dict[str, float] | None = None,
    max_portfolio_mer: float | None = None,
    geo_mapping: dict[str, str] | None = None,
    geo_lower_bounds: dict[str, float] | None = None,
    geo_upper_bounds: dict[str, float] | None = None,
    category_map: Mapping[str, str] | None = None,
    include_unmapped_categories: bool = True,
    return_model: str = "ema",
) -> dict[str, OptimisationResult]:
    """Optimise every collated portfolio located in ``collated_dir``.

    Args:
        collated_dir: Directory containing collated CSV files.
        output_dir: Directory for optimisation artefacts.
        time_constraint: Optional ISO date slice applied to every portfolio.
        mer_mapping: Optional mapping of ticker to its MER (decimal).
        max_portfolio_mer: Maximum allowable weighted MER for the portfolio.
        geo_mapping: Optional mapping of ticker to geographic region.
        geo_lower_bounds: Minimum weight per region.
        geo_upper_bounds: Maximum weight per region.
        category_map: Optional mapping of ticker -> category label applied to
            each portfolio prior to optimisation.
        include_unmapped_categories: Toggle for retaining unmapped tickers as
            standalone categories.
        return_model: Expected return calculation method. 'ema' or 'mean'.

    Returns:
        Mapping of portfolio name to :class:`OptimisationResult`.

    Example:
        >>> from pysharpe.portfolio_optimization import optimise_all_portfolios
        >>> optimise_all_portfolios(make_plot=False)  # doctest: +SKIP
        {'demo': OptimisationResult(...)}
    """

    results: Dict[str, OptimisationResult] = {}

    for path in Path(collated_dir).glob("*_collated.csv"):
        name = path.stem.replace("_collated", "")
        results[name] = optimise_portfolio(
            name,
            collated_dir=collated_dir,
            output_dir=output_dir,
            time_constraint=time_constraint,
            mer_mapping=mer_mapping,
            max_portfolio_mer=max_portfolio_mer,
            geo_mapping=geo_mapping,
            geo_lower_bounds=geo_lower_bounds,
            geo_upper_bounds=geo_upper_bounds,
            category_map=category_map,
            include_unmapped_categories=include_unmapped_categories,
            return_model=return_model,
        )

    return results


def main() -> None:  # pragma: no cover - interactive legacy flow
    collated_dir = EXPORT_DIR
    output_dir = EXPORT_DIR

    portfolio_files = [path for path in Path(_SETTINGS.portfolio_dir).glob("*.csv")]
    print(f"Found {len(portfolio_files)} portfolios:")
    for csv_path in portfolio_files:
        try:
            df = pd.read_csv(csv_path)
            print(f"- {csv_path.stem}: {len(df.index)} equities")
        except Exception as exc:
            print(f"  Unable to read {csv_path}: {exc}")

    proceed = (
        input("Do you want to optimize all portfolios found? (yes/no): ")
        .strip()
        .lower()
    )
    if proceed != "yes":
        print("Optimization aborted.")
        return

    for csv_path in portfolio_files:
        name = csv_path.stem
        try:
            optimise_portfolio(
                name,
                collated_dir=collated_dir,
                output_dir=output_dir,
                time_constraint="1980-01-01",
            )
        except Exception as exc:
            print(f"Error optimising {name}: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
