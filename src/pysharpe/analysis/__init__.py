"""PySharpe analysis helpers."""

from .benchmarks import (
    BENCHMARK_MERS,
    CANADIAN_BENCHMARKS,
    build_benchmark_characteristics,
    fetch_benchmark_metrics,
)
from .categorization import (
    CategoryAggregation,
    apply_category_mapping,
    load_category_map,
)
from .comparison import compare_two_funds
from .portfolio_overlap import PortfolioOverlapResult, analyze_portfolio_overlap
from .tactical_momentum import (
    MarketDataProvenance,
    MomentumConfig,
    MomentumResearchResult,
    TacticalResearchResult,
    TrendConfig,
    WalkForwardConfig,
    WalkForwardResearchResult,
    run_cross_asset_momentum,
    run_tactical_strategy,
    run_trend_strategy,
    run_walk_forward_momentum,
    run_walk_forward_strategy,
)
from .time_series import GARCHVolatilityForecaster, VARModeler, check_stationarity

__all__ = [
    "BENCHMARK_MERS",
    "CANADIAN_BENCHMARKS",
    "build_benchmark_characteristics",
    "fetch_benchmark_metrics",
    "CategoryAggregation",
    "apply_category_mapping",
    "load_category_map",
    "compare_two_funds",
    "PortfolioOverlapResult",
    "analyze_portfolio_overlap",
    "MarketDataProvenance",
    "MomentumConfig",
    "MomentumResearchResult",
    "TacticalResearchResult",
    "TrendConfig",
    "WalkForwardConfig",
    "WalkForwardResearchResult",
    "run_cross_asset_momentum",
    "run_tactical_strategy",
    "run_trend_strategy",
    "run_walk_forward_momentum",
    "run_walk_forward_strategy",
    "GARCHVolatilityForecaster",
    "VARModeler",
    "check_stationarity",
]
