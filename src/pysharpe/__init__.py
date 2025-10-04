"""Public interface for PySharpe with minimal import side effects.

This module exposes the most common entry points while deferring heavyweight
imports until the corresponding attribute is accessed. The approach prevents
partial initialisation errors that were previously triggered by circular
imports between top-level modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Final

from .config import PySharpeSettings, build_settings, get_settings

_CONFIG_EXPORTS: tuple[str, ...] = (
    "PySharpeSettings",
    "build_settings",
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
}

_SETTINGS = get_settings()
DATA_DIR: Final[str] = _SETTINGS.data_dir
PORTFOLIO_DIR: Final[str] = _SETTINGS.portfolio_dir
PRICE_HISTORY_DIR: Final[str] = _SETTINGS.price_history_dir
EXPORT_DIR: Final[str] = _SETTINGS.export_dir
INFO_DIR: Final[str] = _SETTINGS.info_dir
LOG_DIR: Final[str] = _SETTINGS.log_dir

_EXPORT_MAP: dict[str, tuple[str, str]] = {
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
    # Visualisation helpers
    "DCAProjection": ("pysharpe.visualization", "DCAProjection"),
    "simulate_dca": ("pysharpe.visualization", "simulate_dca"),
    "plot_dca_projection": (
        "pysharpe.visualization",
        "plot_dca_projection",
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
    from pysharpe.workflows import download_portfolios, optimise_portfolios  # noqa: F401
    from pysharpe.portfolio_optimization import (  # noqa: F401
        optimise_all_portfolios,
        optimise_portfolio,
    )
    from pysharpe.optimization import (  # noqa: F401
        OptimisationPerformance,
        OptimisationResult,
        PortfolioWeights,
    )
    from pysharpe.visualization import (  # noqa: F401
        DCAProjection,
        plot_dca_projection,
        simulate_dca,
    )
    from pysharpe.metrics import (  # noqa: F401
        annualize_return,
        annualize_volatility,
        compute_returns,
        expected_return,
        sharpe_ratio,
    )
