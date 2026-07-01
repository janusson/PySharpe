"""Public interface for PySharpe with minimal import side effects.

This module exposes the most common entry points while deferring heavyweight
imports until the corresponding attribute is accessed. The approach prevents
partial initialisation errors that were previously triggered by circular
imports between top-level modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Final

from .config import (
    AssetTaxProfile,
    PySharpeSettings,
    TaxProfile,
    build_settings,
    calculate_withholding_tax_rate,
    get_settings,
)

_CONFIG_EXPORTS: tuple[str, ...] = (
    "AccountType",
    "AssetTaxProfile",
    "PySharpeSettings",
    "TaxProfile",
    "build_settings",
    "calculate_withholding_tax_rate",
    "get_settings",
)

_DIRECTORY_EXPORTS: tuple[str, ...] = (
    "DATA_DIR",
    "PORTFOLIO_DIR",
    "PRICE_HISTORY_DIR",
    "EXPORT_DIR",
    "INFO_DIR",
    "LOG_DIR",
)

_METRIC_EXPORTS: dict[str, tuple[str, str]] = {
    "compute_returns": ("pysharpe.metrics", "compute_returns"),
    "annualize_return": ("pysharpe.metrics", "annualize_return"),
    "annualize_volatility": ("pysharpe.metrics", "annualize_volatility"),
    "expected_return": ("pysharpe.metrics", "expected_return"),
    "sharpe_ratio": ("pysharpe.metrics", "sharpe_ratio"),
    "compute_realized_volatility": ("pysharpe.metrics", "compute_realized_volatility"),
}

_SETTINGS = get_settings()
DATA_DIR: Final[str] = _SETTINGS.data_dir
PORTFOLIO_DIR: Final[str] = _SETTINGS.portfolio_dir
PRICE_HISTORY_DIR: Final[str] = _SETTINGS.price_history_dir
EXPORT_DIR: Final[str] = _SETTINGS.export_dir
INFO_DIR: Final[str] = _SETTINGS.info_dir
LOG_DIR: Final[str] = _SETTINGS.log_dir

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    # Exceptions
    "PySharpeError": ("pysharpe.exceptions", "PySharpeError"),
    "DataIngestionError": ("pysharpe.exceptions", "DataIngestionError"),
    "DataValidationError": ("pysharpe.exceptions", "DataValidationError"),
    "ExecutionConfigError": ("pysharpe.exceptions", "ExecutionConfigError"),
    # Configuration
    "AccountType": ("pysharpe.config", "AccountType"),
    "AssetTaxProfile": ("pysharpe.config", "AssetTaxProfile"),
    "calculate_withholding_tax_rate": (
        "pysharpe.config",
        "calculate_withholding_tax_rate",
    ),
    "ExecutionConfig": ("pysharpe.config", "ExecutionConfig"),
    "load_execution_config": ("pysharpe.config", "load_execution_config"),
    "get_ticker_metadata": ("pysharpe.config", "get_ticker_metadata"),
    # Data collection helpers
    "PortfolioTickerReader": ("pysharpe.data_collector", "PortfolioTickerReader"),
    "SecurityDataCollector": ("pysharpe.data_collector", "SecurityDataCollector"),
    "download_portfolio_prices": (
        "pysharpe.data_collector",
        "download_portfolio_prices",
    ),
    "collate_prices": ("pysharpe.data_collector", "collate_prices"),
    "process_portfolio": ("pysharpe.data_collector", "process_portfolio"),
    "process_all_portfolios": (
        "pysharpe.data_collector",
        "process_all_portfolios",
    ),
    "get_csv_file_paths": ("pysharpe.data_collector", "get_csv_file_paths"),
    "read_tickers_from_file": (
        "pysharpe.data_collector",
        "read_tickers_from_file",
    ),
    "setup_logging": ("pysharpe.data_collector", "setup_logging"),
    # High-level workflows
    "download_portfolios": ("pysharpe.workflows", "download_portfolios"),
    "optimise_portfolios": ("pysharpe.workflows", "optimise_portfolios"),
    "plot_holdings_history": ("pysharpe.workflows", "plot_holdings_history"),
    # Optimisation helpers
    "optimise_portfolio": (
        "pysharpe.portfolio_optimization",
        "optimise_portfolio",
    ),
    "optimise_all_portfolios": (
        "pysharpe.portfolio_optimization",
        "optimise_all_portfolios",
    ),
    "PortfolioWeights": ("pysharpe.optimization", "PortfolioWeights"),
    "OptimisationPerformance": (
        "pysharpe.optimization",
        "OptimisationPerformance",
    ),
    "OptimisationResult": ("pysharpe.optimization", "OptimisationResult"),
    # Tax location engine
    "AssetLocationEngine": (
        "pysharpe.optimization.tax_location",
        "AssetLocationEngine",
    ),
    "AssetTaxCharacteristics": (
        "pysharpe.optimization.tax_location",
        "AssetTaxCharacteristics",
    ),
    "TaxProfile": ("pysharpe.optimization.tax_location", "TaxProfile"),
    "build_asset_characteristics": (
        "pysharpe.optimization.tax_location",
        "build_asset_characteristics",
    ),
    "build_asset_characteristics_batch": (
        "pysharpe.optimization.tax_location",
        "build_asset_characteristics_batch",
    ),
    # Execution / rebalancing
    "RebalancePlan": ("pysharpe.execution.rebalance", "RebalancePlan"),
    "build_rebalance_plan": ("pysharpe.execution.rebalance", "build_rebalance_plan"),
    "format_rebalance_plan": ("pysharpe.execution.rebalance", "format_rebalance_plan"),
    # Allocator
    "AllocationConfig": ("pysharpe.execution.allocator", "AllocationConfig"),
    "FxRoutingResult": ("pysharpe.execution.allocator", "FxRoutingResult"),
    "allocate_contribution": ("pysharpe.execution.allocator", "allocate_contribution"),
    "determine_fx_routing": ("pysharpe.execution.allocator", "determine_fx_routing"),
    "score_opportunities": ("pysharpe.execution.allocator", "score_opportunities"),
    # Tax tracker / TLH
    "ACBPosition": ("pysharpe.execution.tax_tracker", "ACBPosition"),
    "ACBTracker": ("pysharpe.execution.tax_tracker", "ACBTracker"),
    "TLHEngine": ("pysharpe.execution.tax_tracker", "TLHEngine"),
    "TLHRebalanceResult": ("pysharpe.execution.tax_tracker", "TLHRebalanceResult"),
    "TLHTrade": ("pysharpe.execution.tax_tracker", "TLHTrade"),
    "TradeRecord": ("pysharpe.execution.tax_tracker", "TradeRecord"),
    "analyze_tlh_opportunities": (
        "pysharpe.execution.tax_tracker",
        "analyze_tlh_opportunities",
    ),
    "format_tlh_rebalance_result": (
        "pysharpe.execution.tax_tracker",
        "format_tlh_rebalance_result",
    ),
    # Analysis helpers
    "apply_category_mapping": (
        "pysharpe.analysis",
        "apply_category_mapping",
    ),
    "load_category_map": ("pysharpe.analysis", "load_category_map"),
    # Visualisation helpers
    "DCAProjection": ("pysharpe.visualization", "DCAProjection"),
    "simulate_dca": ("pysharpe.visualization", "simulate_dca"),
    "plot_dca_projection": (
        "pysharpe.visualization",
        "plot_dca_projection",
    ),
    "plot_comparative_returns": (
        "pysharpe.visualization",
        "plot_comparative_returns",
    ),
    "plot_equity_curves": (
        "pysharpe.visualization",
        "plot_equity_curves",
    ),
}

_EXPORT_MAP.update(_METRIC_EXPORTS)

__all__ = (*_CONFIG_EXPORTS, *_DIRECTORY_EXPORTS, *_EXPORT_MAP)


def __getattr__(name: str) -> Any:  # pragma: no cover - thin dynamic dispatch
    """Resolve lazily exported attributes on first access and cache them."""

    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as error:
        available = ", ".join(sorted(__all__))
        message = (
            f"module 'pysharpe' has no attribute {name!r}. "
            f"Available exports: {available}"
        )
        raise AttributeError(message) from error

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value  # Cache to avoid repeated imports.
    return value


def __dir__() -> list[str]:  # pragma: no cover - proxy to improve discoverability
    """Surface lazily loaded attributes during interactive exploration tools."""

    return sorted({*globals(), *__all__})


if TYPE_CHECKING:  # pragma: no cover - import for static analysis only
    from pysharpe.analysis import (  # noqa: F401
        apply_category_mapping,
        load_category_map,
    )
    from pysharpe.config import (  # noqa: F401
        AccountType,
        AssetTaxProfile,
        ExecutionConfig,
        TaxProfile,
        calculate_withholding_tax_rate,
        get_ticker_metadata,
        load_execution_config,
    )
    from pysharpe.data_collector import (  # noqa: F401
        PortfolioTickerReader,
        SecurityDataCollector,
        collate_prices,
        download_portfolio_prices,
        get_csv_file_paths,
        process_all_portfolios,
        process_portfolio,
        read_tickers_from_file,
        setup_logging,
    )
    from pysharpe.exceptions import (  # noqa: F401
        DataIngestionError,
        DataValidationError,
        ExecutionConfigError,
        PySharpeError,
    )
    from pysharpe.execution.allocator import (  # noqa: F401
        AllocationConfig,
        FxRoutingResult,
        allocate_contribution,
        determine_fx_routing,
        score_opportunities,
    )
    from pysharpe.execution.rebalance import (  # noqa: F401
        RebalancePlan,
        build_rebalance_plan,
        format_rebalance_plan,
    )
    from pysharpe.execution.tax_tracker import (  # noqa: F401
        ACBPosition,
        ACBTracker,
        TLHEngine,
        TLHRebalanceResult,
        TLHTrade,
        TradeRecord,
        analyze_tlh_opportunities,
        format_tlh_rebalance_result,
    )
    from pysharpe.metrics import (  # noqa: F401
        annualize_return,
        annualize_volatility,
        compute_realized_volatility,
        compute_returns,
        expected_return,
        sharpe_ratio,
    )
    from pysharpe.optimization import (  # noqa: F401
        OptimisationPerformance,
        OptimisationResult,
        PortfolioWeights,
    )
    from pysharpe.optimization.tax_location import (  # noqa: F401
        AssetLocationEngine,
        AssetTaxCharacteristics,
        build_asset_characteristics,
        build_asset_characteristics_batch,
    )
    from pysharpe.portfolio_optimization import (  # noqa: F401
        optimise_all_portfolios,
        optimise_portfolio,
    )
    from pysharpe.visualization import (  # noqa: F401
        DCAProjection,
        plot_comparative_returns,
        plot_dca_projection,
        plot_equity_curves,
        simulate_dca,
    )
    from pysharpe.workflows import (  # noqa: F401
        download_portfolios,
        optimise_portfolios,
        plot_holdings_history,
    )
