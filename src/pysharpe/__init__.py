"""PySharpe package exposing the legacy-style helpers."""

from .config import PySharpeSettings, build_settings, get_settings  # noqa: F401
from .data_collector import (  # noqa: F401
    DATA_DIR,
    EXPORT_DIR,
    INFO_DIR,
    LOG_DIR,
    PORTFOLIO_DIR,
    PRICE_HISTORY_DIR,
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
from .portfolio_optimization import (  # noqa: F401
    optimise_all_portfolios,
    optimise_portfolio,
)
from .visualization import (  # noqa: F401
    DCAProjection,
    plot_dca_projection,
    simulate_dca,
)
from .workflows import download_portfolios, optimise_portfolios  # noqa: F401
from .optimization import (  # noqa: F401
    OptimisationPerformance,
    OptimisationResult,
    PortfolioWeights,
)

__all__ = [
    "DATA_DIR",
    "PORTFOLIO_DIR",
    "PRICE_HISTORY_DIR",
    "EXPORT_DIR",
    "INFO_DIR",
    "LOG_DIR",
    "PySharpeSettings",
    "PortfolioTickerReader",
    "SecurityDataCollector",
    "get_csv_file_paths",
    "read_tickers_from_file",
    "download_portfolio_prices",
    "collate_prices",
    "process_portfolio",
    "process_all_portfolios",
    "setup_logging",
    "optimise_portfolio",
    "optimise_all_portfolios",
    "download_portfolios",
    "optimise_portfolios",
    "build_settings",
    "get_settings",
    "PortfolioWeights",
    "OptimisationPerformance",
    "OptimisationResult",
    "DCAProjection",
    "simulate_dca",
    "plot_dca_projection",
]
