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
from .tactical_momentum import (
    MomentumConfig,
    MomentumResearchResult,
    run_cross_asset_momentum,
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
    "MomentumConfig",
    "MomentumResearchResult",
    "run_cross_asset_momentum",
    "GARCHVolatilityForecaster",
    "VARModeler",
    "check_stationarity",
]
