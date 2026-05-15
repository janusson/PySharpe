"""Base classes and protocols for portfolio optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, runtime_checkable


@dataclass(frozen=True)
class OptimizationResult:
    """Standardized result of a portfolio optimization.

    Attributes:
        weights: Mapping of ticker symbol to its optimized weight.
        expected_return: The annualized expected return of the optimized portfolio.
        volatility: The annualized volatility (standard deviation) of the optimized portfolio.
        sharpe_ratio: The calculated Sharpe ratio of the optimized portfolio.
    """

    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


@runtime_checkable
class PortfolioOptimizer(Protocol):
    """Protocol for portfolio optimization strategies."""

    def optimize(self) -> OptimizationResult:
        """Perform portfolio optimization and return the results.

        Returns:
            OptimizationResult: The result containing weights and performance metrics.
        """
        ...
