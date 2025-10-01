"""Data structures for optimisation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PortfolioWeights:
    allocations: Dict[str, float]

    def non_zero(self) -> Dict[str, float]:
        return {ticker: weight for ticker, weight in self.allocations.items() if weight > 0}


@dataclass(frozen=True)
class OptimisationPerformance:
    expected_return: float
    volatility: float
    sharpe_ratio: float


@dataclass(frozen=True)
class OptimisationResult:
    name: str
    weights: PortfolioWeights
    performance: OptimisationPerformance

    @property
    def summary(self) -> str:
        return (
            f"{self.name}: expected {self.performance.expected_return:.2%}, "
            f"volatility {self.performance.volatility:.2%}, "
            f"sharpe {self.performance.sharpe_ratio:.2f}"
        )
