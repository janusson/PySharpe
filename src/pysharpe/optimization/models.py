"""Data structures for optimisation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PortfolioWeights:
    """Container for portfolio allocations keyed by ticker.

    Attributes:
        allocations: Mapping of ticker symbol to weight (fractions sum to ~1).

    Example:
        >>> from pysharpe.optimization.models import PortfolioWeights
        >>> weights = PortfolioWeights({"AAPL": 0.6, "MSFT": 0.4})
        >>> weights.non_zero()
        {'AAPL': 0.6, 'MSFT': 0.4}
    """

    allocations: Dict[str, float]

    def non_zero(self) -> Dict[str, float]:
        """Return allocations greater than zero."""

        return {ticker: weight for ticker, weight in self.allocations.items() if weight > 0}


@dataclass(frozen=True)
class OptimisationPerformance:
    """Performance metrics captured after optimisation.

    Example:
        >>> from pysharpe.optimization.models import OptimisationPerformance
        >>> perf = OptimisationPerformance(0.1, 0.15, 0.9)
        >>> perf.sharpe_ratio
        0.9
    """

    expected_return: float
    volatility: float
    sharpe_ratio: float


@dataclass(frozen=True)
class OptimisationResult:
    """Bundle of portfolio weights and performance metrics.

    Example:
        >>> from pysharpe.optimization.models import OptimisationPerformance, OptimisationResult, PortfolioWeights
        >>> result = OptimisationResult('demo', PortfolioWeights({'AAPL': 0.6}), OptimisationPerformance(0.1, 0.15, 0.9))
        >>> 'demo' in result.summary
        True
    """

    name: str
    weights: PortfolioWeights
    performance: OptimisationPerformance

    @property
    def summary(self) -> str:
        """Return a human-readable summary of key metrics."""

        return (
            f"{self.name}: expected {self.performance.expected_return:.2%}, "
            f"volatility {self.performance.volatility:.2%}, "
            f"sharpe {self.performance.sharpe_ratio:.2f}"
        )
