"""PySharpe validation: trial ledger, overfitting diagnostics, resampling, and DSR."""

from .friction import (  # noqa: F401 - re-exported via __all__
    FrictionProfile,
    FrictionStep,
    stress_test_execution_friction,
)
from .ledger import (  # noqa: F401 - re-exported via __all__
    DuckDBLedger,
    ExecutionStatus,
    PBOResult,
    TrialRecord,
    compute_pbo,
    validate_economic_justification,
)
from .metrics import (  # noqa: F401 - re-exported via __all__
    ValidationMetrics,
    compute_dsr,
    compute_validation_metrics,
    estimate_effective_trials,
)
from .resampling import (  # noqa: F401 - re-exported via __all__
    BootstrapResult,
    PurgedFold,
    PurgedKFold,
    Regime,
    RegimeDependencyReport,
    RegimeDependencyWarning,
    RegimeLabeler,
    RegimeSegmentationResult,
    bootstrap_regime_paths,
    check_regime_dependency,
    compute_regime_survival_rates,
    optimal_block_length,
)
from .sample_size import (  # noqa: F401 - re-exported via __all__
    SampleReliability,
    calculate_min_btl,
    evaluate_trade_sample,
)

_FRICTION_EXPORTS: list[str] = [
    "FrictionProfile",
    "FrictionStep",
    "stress_test_execution_friction",
]

_LEDGER_EXPORTS: list[str] = [
    "DuckDBLedger",
    "ExecutionStatus",
    "PBOResult",
    "TrialRecord",
    "compute_pbo",
    "validate_economic_justification",
]

_RESAMPLING_EXPORTS: list[str] = [
    "BootstrapResult",
    "PurgedFold",
    "PurgedKFold",
    "Regime",
    "RegimeDependencyReport",
    "RegimeDependencyWarning",
    "RegimeLabeler",
    "RegimeSegmentationResult",
    "bootstrap_regime_paths",
    "check_regime_dependency",
    "compute_regime_survival_rates",
    "optimal_block_length",
]

_METRICS_EXPORTS: list[str] = [
    "ValidationMetrics",
    "compute_dsr",
    "compute_validation_metrics",
    "estimate_effective_trials",
]

_SAMPLE_SIZE_EXPORTS: list[str] = [
    "SampleReliability",
    "calculate_min_btl",
    "evaluate_trade_sample",
]

__all__: list[str] = [
    *_FRICTION_EXPORTS,
    *_LEDGER_EXPORTS,
    *_METRICS_EXPORTS,
    *_RESAMPLING_EXPORTS,
    *_SAMPLE_SIZE_EXPORTS,
]
