"""Portfolio optimisation helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Protocol, cast

import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import ema_historical_return, mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

from pysharpe.analysis import apply_category_mapping
from pysharpe.config import (
    ExecutionConfig,
    get_settings,
    get_ticker_metadata,
)
from pysharpe.data.fetcher import apply_fx_conversion
from pysharpe.optimization.expected_returns import (
    constant_expected_return,
    shrinkage_expected_return,
)
from pysharpe.optimization.models import (
    OptimisationPerformance,
    OptimisationResult,
    PortfolioWeights,
)
from pysharpe.optimization.sharpe_optimizer import (
    SharpeOptimizer,
    SharpeOptimizerConfig,
)
from pysharpe.visualization.utils import require_matplotlib

_SETTINGS = get_settings()
EXPORT_DIR = Path(_SETTINGS.export_dir)

logger = logging.getLogger(__name__)

MIN_ASSETS = 3


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
    mtime: float,  # cache-busting key only — not used in the body
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
    resolved_dir = str(Path(collated_dir).resolve())
    csv_path = Path(resolved_dir) / f"{portfolio_name}_collated.csv"
    try:
        mtime = csv_path.stat().st_mtime
    except FileNotFoundError:
        mtime = 0.0
    cached = _cached_collated_prices(
        portfolio_name,
        resolved_dir,
        time_constraint,
        mtime,
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
    ax.pie(list(weights.values()), labels=list(weights.keys()), autopct="%1.1f%%", startangle=90)
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
    return_model: str = "shrinkage",
    base_currency: str = "CAD",
    max_weight: float = 0.20,
    shrinkage_floor: float = 0.3,
    execution_config: ExecutionConfig | None = None,
    proxy_map: dict[str, dict[str, object]] | None = None,
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
        base_currency: The target currency for all assets (default "CAD").

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
        base_currency=base_currency,
        max_weight=max_weight,
        shrinkage_floor=shrinkage_floor,
        execution_config=execution_config,
        proxy_map=proxy_map,
    )


class ConfigurationError(ValueError):
    """Raised when portfolio configuration is incomplete or invalid."""

    pass


def optimise_from_prices(
    prices: pd.DataFrame,
    *,
    name: str = "",
    asset_constraints: dict[str, float] | None = None,
    mer_mapping: dict[str, float] | None = None,
    max_portfolio_mer: float | None = None,
    geo_mapping: dict[str, str] | None = None,
    geo_lower_bounds: dict[str, float] | None = None,
    geo_upper_bounds: dict[str, float] | None = None,
    category_map: Mapping[str, str] | None = None,
    include_unmapped_categories: bool = True,
    return_model: str = "shrinkage",
    base_currency: str = "CAD",
    max_weight: float = 0.20,
    shrinkage_floor: float = 0.3,
    execution_config: ExecutionConfig | None = None,
    proxy_map: dict[str, dict[str, object]] | None = None,
) -> OptimisationResult:
    """Canonical optimization kernel operating on an in-memory price DataFrame.

    No disk I/O — callers decide what to do with the result.  Both the CLI
    (via :func:`optimise_portfolio`) and the Streamlit dashboard call this
    function so the algorithm is always identical.

    Args:
        prices: Historical price DataFrame (rows = dates, columns = tickers).
        name: Portfolio label used in error messages and the result.
        asset_constraints: Optional dict with ``min_weight``/``max_weight`` keys
            applied as linear constraints.
        mer_mapping: Optional mapping of ticker to its MER (decimal).
        max_portfolio_mer: Maximum allowable weighted MER for the portfolio.
        geo_mapping: Optional mapping of ticker to geographic region.
        geo_lower_bounds: Minimum weight per region.
        geo_upper_bounds: Maximum weight per region.
        category_map: Optional mapping of ticker -> category label used to
            collapse highly correlated exposures before optimisation.
        include_unmapped_categories: When ``True`` retain tickers that do not
            appear in ``category_map`` as standalone categories.
        return_model: Expected return calculation method. 'ema' or 'mean'.
        base_currency: FX target currency applied before optimisation.
        max_weight: Maximum allowable weight for any single asset (default 0.20).
        execution_config: Optional execution settings controlling tax drag
            during expected-return estimation.
        proxy_map: Optional proxy metadata used to determine US domicile and
            CAD denomination per ticker.

    Returns:
        :class:`OptimisationResult` with weights and performance metrics.
    """
    if len(prices.columns) < MIN_ASSETS:
        raise ValueError(
            f"Portfolio optimization requires a minimum of {MIN_ASSETS} assets to ensure basic diversification. "
            "For fewer assets, use direct manual allocation."
        )

    if max_weight * len(prices.columns) < 1.0:
        raise ValueError(
            f"The max_weight constraint ({max_weight}) is too restrictive for "
            f"{len(prices.columns)} assets to sum to 1.0."
        )

    tickers = list(prices.columns)

    if mer_mapping or geo_mapping:
        missing_mer = [t for t in tickers if mer_mapping and t not in mer_mapping]
        missing_geo = [t for t in tickers if geo_mapping and t not in geo_mapping]

        if missing_mer or missing_geo:
            errors = []
            if missing_mer:
                errors.append(f"MER data missing for: {', '.join(missing_mer)}")
            if missing_geo:
                errors.append(f"Geographic data missing for: {', '.join(missing_geo)}")
            raise ConfigurationError(
                f"Configuration missing for portfolio '{name}'. " + "; ".join(errors)
            )

    first_valid_dates = prices.apply(lambda col: col.first_valid_index())
    limiting_ticker = first_valid_dates.idxmax()
    prices = prices.dropna()
    prices = apply_fx_conversion(prices, base_currency=base_currency)
    start_date = cast(pd.Timestamp, prices.index.min()).strftime("%Y-%m-%d")
    end_date = cast(pd.Timestamp, prices.index.max()).strftime("%Y-%m-%d")

    if category_map:
        try:
            aggregation = apply_category_mapping(
                prices,
                category_map,
                include_unmapped=include_unmapped_categories,
            )
        except ValueError as exc:
            raise ValueError(
                f"No data available for {name} after applying category mapping."
            ) from exc

        if aggregation.dropped:
            logger.warning(
                "Dropping tickers for %s without category assignment: %s",
                name,
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
    elif return_model.lower() == "shrinkage":
        mu = shrinkage_expected_return(prices, shrinkage_floor=shrinkage_floor)
    elif return_model.lower() == "constant":
        mu = constant_expected_return(prices)
    else:
        mu = mean_historical_return(prices)

    # --- Tax Drag: reduce expected returns for US-domiciled assets in TFSA/FHSA ---
    if execution_config is not None and execution_config.tax_drag_applies:
        drag = execution_config.annual_tax_drag
        tickers_affected: list[str] = []
        for ticker in mu.index:
            ticker_str = str(ticker)
            meta = get_ticker_metadata(ticker_str, proxy_map=proxy_map)
            if meta["is_us_domiciled"]:
                mu[ticker] -= drag
                tickers_affected.append(ticker_str)
        if tickers_affected:
            logger.info(
                "Applied %.2f%% tax drag to US-domiciled assets: %s",
                drag * 100,
                ", ".join(tickers_affected),
            )

    try:
        cov = CovarianceShrinkage(prices).ledoit_wolf()
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - sklearn optional
        logger.warning(
            "scikit-learn is missing. Falling back to sample covariance instead of Ledoit-Wolf shrinkage."
        )
        cov = prices.pct_change().dropna().cov()

    ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, max_weight))

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
                    name,
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
        aligned_mers = np.array([mer_mapping.get(str(t), 0.0) for t in ef.tickers])
        # For max_sharpe, w is scaled by k, so we must scale the RHS by sum(w)
        # Using subtraction form to satisfy DCP rules (affine <= constant)
        ef.add_constraint(
            lambda w: w @ aligned_mers - max_portfolio_mer * cp.sum(w) <= 0
        )

    ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in sqrt",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in power",
        )
        expected, volatility, sharpe = ef.portfolio_performance(verbose=False)

    expected = cast(float, expected)
    volatility = cast(float, volatility)
    sharpe = cast(float, sharpe)

    if np.isnan(volatility):
        volatility = 0.0
        if expected - 0.02 > 0:  # Assuming 2% risk-free rate for dummy calculation
            sharpe = np.inf
        else:
            sharpe = 0.0

    portfolio_mer: float | None = None
    if mer_mapping:
        portfolio_mer = sum(
            weight * mer_mapping.get(str(ticker), 0.0)
            for ticker, weight in cleaned_weights.items()
        )

    return OptimisationResult(
        name=name,
        weights=PortfolioWeights(cast(dict[str, float], cleaned_weights)),
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
    base_currency: str,
    max_weight: float = 0.20,
    shrinkage_floor: float = 0.3,
    execution_config: ExecutionConfig | None = None,
    proxy_map: dict[str, dict[str, object]] | None = None,
) -> OptimisationResult:
    """Implementation detail powering :func:`optimise_portfolio`."""

    prices = _load_collated_prices(
        portfolio_name, collated_dir, time_constraint=time_constraint
    )

    result = optimise_from_prices(
        prices,
        name=portfolio_name,
        asset_constraints=asset_constraints,
        mer_mapping=mer_mapping,
        max_portfolio_mer=max_portfolio_mer,
        geo_mapping=geo_mapping,
        geo_lower_bounds=geo_lower_bounds,
        geo_upper_bounds=geo_upper_bounds,
        category_map=category_map,
        include_unmapped_categories=include_unmapped_categories,
        return_model=return_model,
        base_currency=base_currency,
        max_weight=max_weight,
        shrinkage_floor=shrinkage_floor,
        execution_config=execution_config,
        proxy_map=proxy_map,
    )

    _save_weights(result, output_dir)
    _save_performance(result, output_dir)
    plot_strategy(result, output_dir)

    return result


def optimise_portfolio_for_sharpe(
    portfolio_name: str,
    *,
    collated_dir: Path = EXPORT_DIR,
    output_dir: Path = EXPORT_DIR,
    time_constraint: str | None = None,
    config: SharpeOptimizerConfig | None = None,
    mer_mapping: dict[str, float] | None = None,
    max_portfolio_mer: float | None = None,
    make_plot: bool = True,
    base_currency: str = "CAD",
    max_weight: float = 0.20,
) -> OptimisationResult:
    """Optimises a portfolio to maximize the Sharpe ratio using the custom SharpeOptimizer.

    Args:
        portfolio_name: Name of the portfolio (also the collated CSV stem).
        collated_dir: Directory containing ``*_collated.csv`` files.
        output_dir: Directory where optimisation artefacts are written.
        time_constraint: Optional ISO date to filter the collated history.
        config: Configuration for the SharpeOptimizer. If None, default settings are used.
        mer_mapping: Optional mapping of ticker to its MER (decimal).
        max_portfolio_mer: Maximum allowable weighted MER for the portfolio.
        make_plot: When ``True`` generate a pie chart of positive weights.
        base_currency: The target currency for all assets (default "CAD").
        max_weight: The maximum allowable weight for any single asset. Defaults to 0.20 (20%).

    Returns:
        :class:`OptimisationResult` containing weights and performance stats.

    Raises:
        FileNotFoundError: If the collated CSV is missing.
        ValueError: If the constraint removes all usable data or no assets are found.
    """
    plot_strategy = _resolve_plot_strategy(make_plot)

    prices = _load_collated_prices(
        portfolio_name, collated_dir, time_constraint=time_constraint
    )

    if len(prices.columns) < MIN_ASSETS:
        raise ValueError(
            f"Portfolio optimization requires a minimum of {MIN_ASSETS} assets to ensure basic diversification. "
            "For fewer assets, use direct manual allocation."
        )

    if max_weight * len(prices.columns) < 1.0:
        raise ValueError(
            f"The max_weight constraint ({max_weight}) is too restrictive for "
            f"{len(prices.columns)} assets to sum to 1.0."
        )

    first_valid_dates = prices.apply(lambda col: col.first_valid_index())
    limiting_ticker = first_valid_dates.idxmax()  # Used for metadata
    prices = prices.dropna()
    prices = apply_fx_conversion(prices, base_currency=base_currency)

    if prices.empty:
        raise ValueError(
            f"No data available for {portfolio_name} after applying time constraint "
            "and dropping NaNs for Sharpe optimization."
        )

    start_date = cast(pd.Timestamp, prices.index.min()).strftime("%Y-%m-%d")
    end_date = cast(pd.Timestamp, prices.index.max()).strftime("%Y-%m-%d")

    # Initialize and run the SharpeOptimizer
    if config is None:
        config = SharpeOptimizerConfig(
            max_weight=max_weight,
            mer_by_ticker=mer_mapping or {},
            max_portfolio_mer=max_portfolio_mer,
        )
    else:
        config.max_weight = max_weight
        if mer_mapping:
            config.mer_by_ticker = mer_mapping
        if max_portfolio_mer is not None:
            config.max_portfolio_mer = max_portfolio_mer

    sharpe_optimizer = SharpeOptimizer(prices=prices, config=config)
    optimization_result = sharpe_optimizer.optimize()

    # Calculate portfolio MER for performance reporting
    portfolio_mer: float | None = None
    if config and config.mer_by_ticker:
        portfolio_mer = sum(
            weight * config.mer_by_ticker.get(ticker, 0.0)
            for ticker, weight in optimization_result.weights.items()
        )

    result = OptimisationResult(
        name=portfolio_name,
        weights=PortfolioWeights(optimization_result.weights),
        performance=OptimisationPerformance(
            optimization_result.expected_return,
            optimization_result.volatility,
            optimization_result.sharpe_ratio,
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
    return_model: str = "shrinkage",
    sharpe_optimizer_config: SharpeOptimizerConfig | None = None,
    base_currency: str = "CAD",
    max_weight: float = 0.20,
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
        sharpe_optimizer_config: Optional configuration for Sharpe ratio optimization.
            If provided, `optimise_portfolio_for_sharpe` will be used instead of
            the default `optimise_portfolio`.
        base_currency: The target currency for all assets (default "CAD").
        max_weight: The maximum allowable weight for any single asset. Defaults to 0.20 (20%).

    Returns:
        Mapping of portfolio name to :class:`OptimisationResult`.

    Example:
        >>> from pysharpe.portfolio_optimization import optimise_all_portfolios
        >>> optimise_all_portfolios(make_plot=False)  # doctest: +SKIP
        {'demo': OptimisationResult(...)}
    """

    results: dict[str, OptimisationResult] = {}

    for path in Path(collated_dir).glob("*_collated.csv"):
        name = path.stem.replace("_collated", "")
        if sharpe_optimizer_config:
            logger.info(f"Optimising {name} for Sharpe ratio maximization.")
            results[name] = optimise_portfolio_for_sharpe(
                name,
                collated_dir=collated_dir,
                output_dir=output_dir,
                time_constraint=time_constraint,
                config=sharpe_optimizer_config,
                base_currency=base_currency,
                max_weight=max_weight,
            )
        else:
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
                base_currency=base_currency,
                max_weight=max_weight,
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
