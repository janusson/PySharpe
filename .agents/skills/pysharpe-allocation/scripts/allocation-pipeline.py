"""Standard Value Averaging allocation pipeline template.

This template demonstrates the canonical allocation flow used by PySharpe's
execution engine. Load this file via read_file when implementing or modifying
the allocator to ensure consistency with the established pipeline.

Flow:
1. Calculate path drift for each asset (current vs. target value).
2. Calculate valuation/mean-reversion signal.
3. Blend into opportunity score (60% drift + 40% valuation).
4. Normalize scores and allocate contribution budget.
5. Enforce min/max bounds per asset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AllocationConfig:
    """Configuration for the value averaging allocator."""

    path_drift_weight: float = 0.60
    valuation_weight: float = 0.40
    min_contribution_per_asset: float = 0.0
    max_contribution_per_asset: float | None = None
    total_budget: float = 1000.0


def calculate_path_drift(
    current_values: pd.Series,
    target_values: pd.Series,
) -> pd.Series:
    """Calculate how far current value deviates from target.

    Drift = (target - current) / target

    Positive drift means the asset is below target (buy more).
    Negative drift means the asset is above target (buy less or sell).

    Args:
        current_values: Current market value per asset.
        target_values: Target value per asset from the VA path.

    Returns:
        Drift fraction per asset (positive = below target).
    """
    drift = (target_values - current_values) / target_values.replace(0, np.nan)
    return drift.fillna(0.0)


def calculate_valuation_signal(
    current_prices: pd.Series,
    long_term_means: pd.Series,
) -> pd.Series:
    """Calculate mean-reversion valuation signal.

    Signal = (long_term_mean - current_price) / long_term_mean

    Positive signal means price is below long-term mean (undervalued).
    Negative signal means price is above long-term mean (overvalued).

    Args:
        current_prices: Current price per asset.
        long_term_means: Long-term average price per asset.

    Returns:
        Valuation signal per asset (positive = undervalued).
    """
    signal = (long_term_means - current_prices) / long_term_means.replace(0, np.nan)
    return signal.fillna(0.0)


def score_opportunities(
    path_drift: pd.Series,
    valuation_signal: pd.Series,
    config: AllocationConfig | None = None,
) -> pd.Series:
    """Compute blended opportunity score.

    Score = drift_weight * normalized_drift + valuation_weight * normalized_signal

    Args:
        path_drift: Drift fraction per asset.
        valuation_signal: Valuation signal per asset.
        config: Allocation configuration (uses defaults if None).

    Returns:
        Blended opportunity score per asset (higher = more attractive).
    """
    cfg = config or AllocationConfig()

    # Normalize each component to [0, 1] range across assets
    drift_norm = _minmax_normalize(path_drift)
    val_norm = _minmax_normalize(valuation_signal)

    score = cfg.path_drift_weight * drift_norm + cfg.valuation_weight * val_norm
    return score


def allocate_contribution(
    scores: pd.Series,
    config: AllocationConfig | None = None,
) -> pd.Series:
    """Convert opportunity scores to dollar allocation amounts.

    Args:
        scores: Opportunity score per asset.
        config: Allocation configuration.

    Returns:
        Dollar amount to allocate per asset.
    """
    cfg = config or AllocationConfig()

    # Normalize scores to sum-to-1 weights (only positive scores get allocation)
    positive_scores = scores.clip(lower=0.0)
    total_score = positive_scores.sum()

    if total_score == 0.0:
        return pd.Series(0.0, index=scores.index)

    weights = positive_scores / total_score
    allocations = weights * cfg.total_budget

    # Enforce per-asset bounds
    allocations = allocations.clip(lower=cfg.min_contribution_per_asset)
    if cfg.max_contribution_per_asset is not None:
        allocations = allocations.clip(upper=cfg.max_contribution_per_asset)

    # Rescale to fit within total budget
    total_allocated = allocations.sum()
    if total_allocated > 0.0 and total_allocated > cfg.total_budget:
        allocations = allocations * (cfg.total_budget / total_allocated)

    return allocations


def _minmax_normalize(series: pd.Series) -> pd.Series:
    """Normalize a series to [0, 1] range using min-max scaling.

    If all values are equal, returns 0.5 for all entries.
    """
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series(0.5, index=series.index)
    return (series - s_min) / (s_max - s_min)
