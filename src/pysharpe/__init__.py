"""Public interface for PySharpe with minimal import side effects.

This module exposes the most common entry points while deferring heavyweight
imports until the corresponding attribute is accessed. The approach prevents
partial initialisation errors that were previously triggered by circular
imports between top-level modules.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from .config import (  # noqa: F401 — used via _CONFIG_EXPORTS
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
    "sortino_ratio": ("pysharpe.metrics", "sortino_ratio"),
    "calmar_ratio": ("pysharpe.metrics", "calmar_ratio"),
    "tracking_error": ("pysharpe.metrics", "tracking_error"),
    "max_drawdown_duration": ("pysharpe.metrics", "max_drawdown_duration"),
    "compute_realized_volatility": ("pysharpe.metrics", "compute_realized_volatility"),
}

_SETTINGS = get_settings()
DATA_DIR: Final[Path] = _SETTINGS.data_dir
PORTFOLIO_DIR: Final[Path] = _SETTINGS.portfolio_dir
PRICE_HISTORY_DIR: Final[Path] = _SETTINGS.price_history_dir
EXPORT_DIR: Final[Path] = _SETTINGS.export_dir
INFO_DIR: Final[Path] = _SETTINGS.info_dir
LOG_DIR: Final[Path] = _SETTINGS.log_dir

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
    # Black-Litterman
    "blend_views": ("pysharpe.optimization.black_litterman", "blend_views"),
    "build_views_uncertainty": (
        "pysharpe.optimization.black_litterman",
        "build_views_uncertainty",
    ),
    "compute_implied_returns": (
        "pysharpe.optimization.black_litterman",
        "compute_implied_returns",
    ),
    # Covariance shrinkage estimators
    "compute_linear_shrinkage": (
        "pysharpe.optimization.estimators",
        "compute_linear_shrinkage",
    ),
    "compute_nonlinear_shrinkage": (
        "pysharpe.optimization.estimators",
        "compute_nonlinear_shrinkage",
    ),
    # HRP (non‑inversion fallback)
    "HierarchicalRiskParity": (
        "pysharpe.optimization.hrp",
        "HierarchicalRiskParity",
    ),
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
    # Execution / brokerage export
    "Brokerage": ("pysharpe.execution.brokerage", "Brokerage"),
    "BrokerageExportConfig": ("pysharpe.execution.brokerage", "BrokerageExportConfig"),
    "export_buy_orders": ("pysharpe.execution.brokerage", "export_buy_orders"),
    # Allocator
    "AllocationConfig": ("pysharpe.execution.allocator", "AllocationConfig"),
    "allocate_contribution": ("pysharpe.execution.allocator", "allocate_contribution"),
    "score_opportunities": ("pysharpe.execution.allocator", "score_opportunities"),
    # Execution / cash-flow rebalancing
    "CashFlowRebalanceResult": (
        "pysharpe.execution.cash_flow_rebalance",
        "CashFlowRebalanceResult",
    ),
    "RebalanceConfig": (
        "pysharpe.execution.cash_flow_rebalance",
        "RebalanceConfig",
    ),
    "allocate_contribution_cash_flow": (
        "pysharpe.execution.cash_flow_rebalance",
        "allocate_contribution_cash_flow",
    ),
    "evaluate_taxable_rebalance": (
        "pysharpe.execution.cash_flow_rebalance",
        "evaluate_taxable_rebalance",
    ),
    # Tax tracker / ACB
    "ACBPosition": ("pysharpe.execution.tax_tracker", "ACBPosition"),
    "ACBTracker": ("pysharpe.execution.tax_tracker", "ACBTracker"),
    "TradeRecord": ("pysharpe.execution.tax_tracker", "TradeRecord"),
    # Tax compliance guardrails
    "SuperficialLossGuardrail": (
        "pysharpe.guardrails.tax_compliance",
        "SuperficialLossGuardrail",
    ),
    "SuperficialLossViolation": (
        "pysharpe.guardrails.tax_compliance",
        "SuperficialLossViolation",
    ),
    "build_default_identical_map": (
        "pysharpe.guardrails.tax_compliance",
        "build_default_identical_map",
    ),
    # Analysis helpers
    "apply_category_mapping": (
        "pysharpe.analysis",
        "apply_category_mapping",
    ),
    "load_category_map": ("pysharpe.analysis", "load_category_map"),
    "compare_two_funds": (
        "pysharpe.analysis.comparison",
        "compare_two_funds",
    ),
    # Validation / trial ledger
    "DuckDBLedger": ("pysharpe.validation.ledger", "DuckDBLedger"),
    "ExecutionStatus": ("pysharpe.validation.ledger", "ExecutionStatus"),
    "PBOResult": ("pysharpe.validation.ledger", "PBOResult"),
    "TrialRecord": ("pysharpe.validation.ledger", "TrialRecord"),
    "compute_pbo": ("pysharpe.validation.ledger", "compute_pbo"),
    "validate_economic_justification": (
        "pysharpe.validation.ledger",
        "validate_economic_justification",
    ),
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
    # Validation / resampling
    "BootstrapResult": ("pysharpe.validation.resampling", "BootstrapResult"),
    "PurgedFold": ("pysharpe.validation.resampling", "PurgedFold"),
    "PurgedKFold": ("pysharpe.validation.resampling", "PurgedKFold"),
    "Regime": ("pysharpe.validation.resampling", "Regime"),
    "RegimeDependencyReport": (
        "pysharpe.validation.resampling",
        "RegimeDependencyReport",
    ),
    "RegimeDependencyWarning": (
        "pysharpe.validation.resampling",
        "RegimeDependencyWarning",
    ),
    "RegimeLabeler": ("pysharpe.validation.resampling", "RegimeLabeler"),
    "RegimeSegmentationResult": (
        "pysharpe.validation.resampling",
        "RegimeSegmentationResult",
    ),
    "bootstrap_regime_paths": (
        "pysharpe.validation.resampling",
        "bootstrap_regime_paths",
    ),
    "check_regime_dependency": (
        "pysharpe.validation.resampling",
        "check_regime_dependency",
    ),
    "compute_regime_survival_rates": (
        "pysharpe.validation.resampling",
        "compute_regime_survival_rates",
    ),
    "optimal_block_length": (
        "pysharpe.validation.resampling",
        "optimal_block_length",
    ),
    # Validation / friction stress-testing
    "FrictionProfile": ("pysharpe.validation.friction", "FrictionProfile"),
    "FrictionStep": ("pysharpe.validation.friction", "FrictionStep"),
    "stress_test_execution_friction": (
        "pysharpe.validation.friction",
        "stress_test_execution_friction",
    ),
    # Validation / DSR and aggregate metrics
    "ValidationMetrics": (
        "pysharpe.validation.metrics",
        "ValidationMetrics",
    ),
    "compute_dsr": (
        "pysharpe.validation.metrics",
        "compute_dsr",
    ),
    "compute_validation_metrics": (
        "pysharpe.validation.metrics",
        "compute_validation_metrics",
    ),
    "estimate_effective_trials": (
        "pysharpe.validation.metrics",
        "estimate_effective_trials",
    ),
    # Validation / sample-size
    "SampleReliability": (
        "pysharpe.validation.sample_size",
        "SampleReliability",
    ),
    "evaluate_trade_sample": (
        "pysharpe.validation.sample_size",
        "evaluate_trade_sample",
    ),
    "calculate_min_btl": (
        "pysharpe.validation.sample_size",
        "calculate_min_btl",
    ),
}

_EXPORT_MAP.update(_METRIC_EXPORTS)

__all__: tuple[str, ...] = (*_CONFIG_EXPORTS, *_DIRECTORY_EXPORTS, *_EXPORT_MAP)  # type: ignore[reportUnknownVariableType]


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
    from pysharpe.analysis.comparison import (  # noqa: F401
        compare_two_funds,
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
        allocate_contribution,
        score_opportunities,
    )
    from pysharpe.execution.brokerage import (  # noqa: F401
        Brokerage,
        BrokerageExportConfig,
        export_buy_orders,
    )
    from pysharpe.execution.cash_flow_rebalance import (  # noqa: F401
        CashFlowRebalanceResult,
        RebalanceConfig,
        allocate_contribution_cash_flow,
        evaluate_taxable_rebalance,
    )
    from pysharpe.execution.rebalance import (  # noqa: F401
        RebalancePlan,
        build_rebalance_plan,
        format_rebalance_plan,
    )
    from pysharpe.execution.tax_tracker import (  # noqa: F401
        ACBPosition,
        ACBTracker,
        TradeRecord,
    )
    from pysharpe.guardrails.tax_compliance import (  # noqa: F401
        SuperficialLossGuardrail,
        SuperficialLossViolation,
        build_default_identical_map,
    )
    from pysharpe.metrics import (  # noqa: F401
        annualize_return,
        annualize_volatility,
        calmar_ratio,
        compute_realized_volatility,
        compute_returns,
        expected_return,
        max_drawdown_duration,
        sharpe_ratio,
        sortino_ratio,
        tracking_error,
    )
    from pysharpe.optimization import (  # noqa: F401
        OptimisationPerformance,
        OptimisationResult,
        PortfolioWeights,
    )
    from pysharpe.optimization.black_litterman import (  # noqa: F401
        blend_views,
        build_views_uncertainty,
        compute_implied_returns,
    )
    from pysharpe.optimization.estimators import (  # noqa: F401
        compute_linear_shrinkage,
        compute_nonlinear_shrinkage,
    )
    from pysharpe.optimization.hrp import (  # noqa: F401
        HierarchicalRiskParity,
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
    from pysharpe.validation.friction import (  # noqa: F401
        FrictionProfile,
        FrictionStep,
        stress_test_execution_friction,
    )
    from pysharpe.validation.ledger import (  # noqa: F401
        DuckDBLedger,
        ExecutionStatus,
        PBOResult,
        TrialRecord,
        compute_pbo,
        validate_economic_justification,
    )
    from pysharpe.validation.metrics import (  # noqa: F401
        ValidationMetrics,
        compute_dsr,
        compute_validation_metrics,
        estimate_effective_trials,
    )
    from pysharpe.validation.resampling import (  # noqa: F401
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
    from pysharpe.validation.sample_size import (  # noqa: F401
        SampleReliability,
        calculate_min_btl,
        evaluate_trade_sample,
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
