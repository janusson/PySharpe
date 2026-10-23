"""Chronological out-of-sample evaluation for fixed tactical configurations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, cast

import pandas as pd

from ..exceptions import DataValidationError
from .tactical_momentum import (
    MarketDataProvenance,
    MomentumConfig,
    TrendConfig,
    _cost_assumptions,
    _json_safe,
    _risk_metrics,
    _strategy_name,
    run_tactical_strategy,
)


@dataclass(frozen=True)
class WalkForwardConfig:
    """Initial training period and fixed-size, non-overlapping test folds."""

    training_months: int
    test_months: int

    def __post_init__(self) -> None:
        values = (self.training_months, self.test_months)
        if any(
            not isinstance(value, int) or isinstance(value, bool) for value in values
        ):
            raise DataValidationError("walk-forward window lengths must be integers")
        if self.training_months < 1 or self.test_months < 1:
            raise DataValidationError(
                "walk-forward training_months and test_months must be positive"
            )


@dataclass(frozen=True)
class WalkForwardResearchResult:
    """Fold-level out-of-sample returns, risk, benchmarks, and provenance."""

    generated_at: pd.Timestamp
    strategy_config: MomentumConfig | TrendConfig
    walk_forward_config: WalkForwardConfig
    folds: tuple[dict[str, Any], ...]
    data_quality: dict[str, Any]
    data_provenance: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned, JSON-compatible walk-forward contract."""
        return _json_safe(
            {
                "schema_version": (
                    "1.2" if isinstance(self.strategy_config, TrendConfig) else "1.1"
                ),
                "evaluation": "fixed_configuration_walk_forward",
                "generated_at": self.generated_at,
                "strategy": {
                    "name": _strategy_name(self.strategy_config),
                    "config": asdict(self.strategy_config),
                },
                "walk_forward_config": asdict(self.walk_forward_config),
                "folds": self.folds,
                "data_quality": self.data_quality,
                "data_provenance": self.data_provenance,
                "cost_assumptions": _cost_assumptions(self.strategy_config),
            }
        )

    def to_json(self) -> str:
        """Serialize deterministically with standard JSON values only."""
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )


def run_walk_forward_strategy(
    total_return_prices: pd.DataFrame,
    execution_prices: pd.DataFrame,
    *,
    strategy_config: MomentumConfig | TrendConfig,
    walk_forward_config: WalkForwardConfig,
    currencies: dict[str, str] | None = None,
    fx_to_cad: pd.DataFrame | None = None,
    provenance: dict[str, MarketDataProvenance] | None = None,
) -> WalkForwardResearchResult:
    """Evaluate an unchanged strategy over chronological test windows.

    No parameter fitting occurs in this evaluation. Training periods establish
    the available history, and every signal continues to use only observations
    preceding its effective execution date. Test windows are contiguous and do
    not overlap.
    """
    training_months = walk_forward_config.training_months
    skip_months = (
        strategy_config.skip_months
        if isinstance(strategy_config, MomentumConfig)
        else 0
    )
    minimum_training_months = strategy_config.lookback_months + skip_months + 1
    if training_months < minimum_training_months:
        raise DataValidationError(
            "training history must exceed the configured strategy lookback and skip window"
        )
    full_result = run_tactical_strategy(
        total_return_prices,
        execution_prices,
        config=strategy_config,
        currencies=currencies,
        fx_to_cad=fx_to_cad,
        provenance=provenance,
    )
    index = pd.DatetimeIndex(full_result.backtest.index)
    month_keys = index.year * 100 + index.month
    month_ends = pd.Series(index=index, data=index).groupby(month_keys).last()
    if len(month_ends) <= training_months:
        raise DataValidationError(
            "insufficient history for an out-of-sample walk-forward fold"
        )

    fold_rows: list[dict[str, Any]] = []
    for fold_number, first_test_month in enumerate(
        range(training_months, len(month_ends), walk_forward_config.test_months),
        start=1,
    ):
        training_end = pd.Timestamp(month_ends.iloc[first_test_month - 1])
        test_start_candidates = index[index > training_end]
        if test_start_candidates.empty:
            break
        test_start = pd.Timestamp(test_start_candidates[0])
        last_test_month = min(
            first_test_month + walk_forward_config.test_months - 1,
            len(month_ends) - 1,
        )
        test_end = pd.Timestamp(month_ends.iloc[last_test_month])
        fold_index = index[(index >= test_start) & (index <= test_end)]
        if len(fold_index) < 1:
            continue

        strategy_returns = full_result.backtest.loc[fold_index, "return"]
        boundary_nav = cast(
            float, full_result.backtest.loc[training_end, "portfolio_value"]
        )
        strategy_equity = boundary_nav * (1.0 + strategy_returns).cumprod()
        baseline = pd.Series(
            [boundary_nav],
            index=pd.DatetimeIndex([training_end]),
            dtype=float,
        )
        strategy_equity = pd.concat([baseline, strategy_equity]).sort_index()
        benchmark_equity: dict[str, list[dict[str, Any]]] = {}
        benchmark_risk: dict[str, dict[str, float | None]] = {}
        for ticker in full_result.benchmarks.columns:
            benchmark_returns = (
                full_result.benchmarks[ticker]
                .pct_change(fill_method=None)
                .loc[fold_index]
            )
            equity = boundary_nav * (1.0 + benchmark_returns).cumprod()
            equity = pd.concat([baseline, equity]).sort_index()
            benchmark_equity[ticker] = [
                {"timestamp": date, "value": float(value)}
                for date, value in equity.items()
            ]
            benchmark_risk[ticker] = _risk_metrics(
                equity, strategy_config.periods_per_year
            )
        fold_rows.append(
            {
                "fold": fold_number,
                "training_start": pd.Timestamp(month_ends.iloc[0]),
                "training_end": training_end,
                "test_start": test_start,
                "test_end": test_end,
                "strategy_equity": [
                    {"timestamp": date, "value": float(value)}
                    for date, value in strategy_equity.items()
                ],
                "test_returns": [
                    {"timestamp": date, "return": float(value)}
                    for date, value in strategy_returns.items()
                ],
                "benchmark_equity": benchmark_equity,
                "risk": _risk_metrics(
                    strategy_equity, strategy_config.periods_per_year
                ),
                "benchmark_risk": benchmark_risk,
            }
        )
    if not fold_rows:
        raise DataValidationError(
            "insufficient history for an out-of-sample walk-forward fold"
        )
    return WalkForwardResearchResult(
        generated_at=full_result.generated_at,
        strategy_config=strategy_config,
        walk_forward_config=walk_forward_config,
        folds=tuple(fold_rows),
        data_quality=full_result.data_quality,
        data_provenance=full_result.data_provenance,
    )


def run_walk_forward_momentum(
    total_return_prices: pd.DataFrame,
    execution_prices: pd.DataFrame,
    *,
    strategy_config: MomentumConfig,
    walk_forward_config: WalkForwardConfig,
    currencies: dict[str, str] | None = None,
    fx_to_cad: pd.DataFrame | None = None,
    provenance: dict[str, MarketDataProvenance] | None = None,
) -> WalkForwardResearchResult:
    """Evaluate a fixed momentum configuration over chronological test windows."""
    if not isinstance(strategy_config, MomentumConfig):
        raise DataValidationError("strategy_config must be a MomentumConfig")
    return run_walk_forward_strategy(
        total_return_prices,
        execution_prices,
        strategy_config=strategy_config,
        walk_forward_config=walk_forward_config,
        currencies=currencies,
        fx_to_cad=fx_to_cad,
        provenance=provenance,
    )


__all__ = [
    "WalkForwardConfig",
    "WalkForwardResearchResult",
    "run_walk_forward_momentum",
    "run_walk_forward_strategy",
]
