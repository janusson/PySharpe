"""Data models for PySharpe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class PortfolioAllocation:
    """Represents a set of allocation weights for portfolio assets."""

    weights: Dict[str, float]

    def as_series(self) -> pd.Series:
        """Return the allocation weights as a pandas Series."""

        return pd.Series(self.weights, name="weight")


@dataclass
class PortfolioPerformance:
    """Captures the output metrics of an optimized portfolio."""

    expected_annual_return: float
    annual_volatility: float
    sharpe_ratio: float

    def as_dict(self) -> Dict[str, float]:
        """Return the performance metrics as a dictionary."""

        return {
            "expected_annual_return": self.expected_annual_return,
            "annual_volatility": self.annual_volatility,
            "sharpe_ratio": self.sharpe_ratio,
        }
