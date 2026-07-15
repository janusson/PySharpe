"""PySharpe analysis helpers."""

from .benchmarks import CANADIAN_BENCHMARKS, fetch_benchmark_metrics
from .categorization import (
    CategoryAggregation,
    apply_category_mapping,
    load_category_map,
)
from .comparison import compare_two_funds
from .time_series import GARCHVolatilityForecaster, VARModeler, check_stationarity

__all__ = [
    "CANADIAN_BENCHMARKS",
    "fetch_benchmark_metrics",
    "CategoryAggregation",
    "apply_category_mapping",
    "load_category_map",
    "compare_two_funds",
    "GARCHVolatilityForecaster",
    "VARModeler",
    "check_stationarity",
]
