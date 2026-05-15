"""Module for Sharpe ratio portfolio optimization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class SharpeOptimizerConfig:
    """Configuration for the SharpeOptimizer.

    Attributes
    ----------
    risk_free_rate : float, optional
        The annual risk-free rate. Default is 0.00 (0%).
    target_return : float, optional
        Target portfolio return for optimization. Not used in simple Sharpe
        maximization, but can be useful for other optimization types.
        Default is None.
    mer_by_ticker : Dict[str, float], optional
        Dictionary where keys are ticker symbols and values are their annual
        Management Expense Ratios (MERs) as a percentage (e.g., 0.17 for 0.17%).
        These will be deducted from expected returns.
        Default is an empty dictionary, meaning no MER deductions.
    num_portfolios_monte_carlo : int, optional
        Number of random portfolios to generate for Monte Carlo simulation
        to find a good starting point for numerical optimization.
        Default is 10000.
    """

    risk_free_rate: float = 0.00
    target_return: Optional[float] = None
    mer_by_ticker: Dict[str, float] = field(default_factory=dict)
    num_portfolios_monte_carlo: int = 10000
    max_weight: float = 0.20


class SharpeOptimizer:
    """Optimizes portfolio weights to maximize the Sharpe ratio."""

    def __init__(
        self,
        prices: pd.DataFrame,
        config: Optional[SharpeOptimizerConfig] = None,
    ) -> None:
        """Initialize the optimizer with price data and configuration.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical asset prices where rows are dates (DatetimeIndex)
            and columns are asset tickers.
        config : SharpeOptimizerConfig, optional
            Configuration for the optimizer. If None, default settings are used.
        """
        if prices.empty:
            raise ValueError("Prices DataFrame cannot be empty.")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("Prices DataFrame must have a DatetimeIndex.")

        self.prices = prices.dropna()
        self.assets = self.prices.columns.tolist()
        self.config = config if config is not None else SharpeOptimizerConfig()

        if not self.assets:
            raise ValueError("No assets found in the prices DataFrame.")

        # Calculate daily returns
        self.returns = self.prices.pct_change().dropna()
        if self.returns.empty:
            raise ValueError(
                "Insufficient data to calculate returns after dropping NaNs."
            )

        self.num_periods_per_year = self._estimate_periods_per_year(self.prices.index)
        self.annualized_risk_free_rate = (
            self.config.risk_free_rate / self.num_periods_per_year
        )

        logger.info(
            f"SharpeOptimizer initialized with {len(self.assets)} assets. "
            f"Annual Risk-Free Rate: {self.config.risk_free_rate:.2%}. "
            f"Periods per year: {self.num_periods_per_year}."
        )

    def _estimate_periods_per_year(self, index: pd.DatetimeIndex) -> float:
        """Estimate the number of periods per year based on the frequency of the index."""
        if len(index) < 2:
            return 252.0  # Default to daily if not enough data

        # Calculate the median difference in days between consecutive dates
        diffs = index.to_series().diff().dropna().dt.days
        median_days = diffs.median()

        if median_days <= 1.05:  # Approximately daily
            return 252.0
        elif median_days <= 7.05:  # Approximately weekly
            return 52.0
        elif median_days <= 31.05:  # Approximately monthly
            return 12.0
        elif median_days <= 92.05:  # Approximately quarterly
            return 4.0
        elif median_days <= 366.05:  # Approximately annually
            return 1.0
        else:
            logger.warning(
                "Could not reliably determine data frequency. Assuming 252 periods per year."
            )
            return 252.0  # Fallback

    def calculate_portfolio_performance(
        self, weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculates portfolio return, volatility, and Sharpe ratio.

        Parameters
        ----------
        weights : np.ndarray
            A NumPy array of asset weights.

        Returns
        -------
        Tuple[float, float, float]
            (annualized_return, annualized_volatility, sharpe_ratio)
        """
        weights = np.array(weights)  # Ensure it's a numpy array
        if not np.isclose(weights.sum(), 1.0):
            logger.warning(
                f"Weights do not sum to 1.0 ({weights.sum()}). Normalizing weights."
            )
            weights = weights / weights.sum()

        portfolio_return = (
            np.sum(self.returns.mean() * weights) * self.num_periods_per_year
        )
        portfolio_volatility = np.sqrt(
            np.dot(
                weights.T,
                np.dot(self.returns.cov() * self.num_periods_per_year, weights),
            )
        )

        # Apply MER deductions
        for i, ticker in enumerate(self.assets):
            mer_annual = (
                self.config.mer_by_ticker.get(ticker, 0.0) / 100.0
            )  # Convert % to decimal
            portfolio_return -= weights[i] * mer_annual

        if portfolio_volatility == 0:
            sharpe_ratio = 0.0  # Avoid division by zero
        else:
            sharpe_ratio = (
                portfolio_return - self.config.risk_free_rate
            ) / portfolio_volatility

        return portfolio_return, portfolio_volatility, sharpe_ratio

    def _objective_function(self, weights: np.ndarray) -> float:
        """Objective function to minimize (negative Sharpe ratio)."""
        _, _, sharpe_ratio = self.calculate_portfolio_performance(weights)
        return -sharpe_ratio

    def _generate_random_portfolios(self, num_portfolios: int) -> pd.DataFrame:
        """Generates random portfolios for Monte Carlo simulation."""
        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in range(num_portfolios):
            weights = np.random.random(len(self.assets))
            weights /= np.sum(weights)
            weights_record.append(tuple(weights))

            p_return, p_volatility, p_sharpe = self.calculate_portfolio_performance(
                weights
            )
            results[0, i] = p_return
            results[1, i] = p_volatility
            results[2, i] = p_sharpe

        return pd.DataFrame(
            {"Return": results[0], "Volatility": results[1], "Sharpe": results[2]},
            index=pd.MultiIndex.from_arrays(
                [range(num_portfolios), weights_record], names=["Portfolio", "Weights"]
            ),
        )

    def optimize(self) -> OptimizationResult:
        """Optimizes portfolio weights to maximize the Sharpe ratio.

        Returns
        -------
        OptimizationResult
            Standardized result containing weights and performance metrics.
        """
        num_assets = len(self.assets)
        if num_assets == 0:
            return OptimizationResult({}, 0.0, 0.0, 0.0)

        if self.config.max_weight * num_assets < 1.0:
            raise ValueError(
                f"The max_weight constraint ({self.config.max_weight}) is too restrictive "
                f"for {num_assets} assets to sum to 1.0."
            )

        # Constraints and Bounds
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0.0, self.config.max_weight) for _ in range(num_assets))

        # Initial guess: equal weighting or best from Monte Carlo
        initial_guess = np.array([1.0 / num_assets] * num_assets)
        if self.config.num_portfolios_monte_carlo > 0:
            logger.debug(
                f"Generating {self.config.num_portfolios_monte_carlo} random portfolios "
                "to find a good initial guess."
            )
            random_portfolios = self._generate_random_portfolios(
                self.config.num_portfolios_monte_carlo
            )
            # Find the portfolio with the highest Sharpe ratio
            max_sharpe_idx = random_portfolios["Sharpe"].idxmax()
            initial_guess = np.array(
                max_sharpe_idx[1]
            )  # idx is (portfolio_num, weights_tuple)

        logger.info("Starting numerical optimization for Sharpe ratio maximization.")
        optimal_results = minimize(
            self._objective_function,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not optimal_results.success:
            logger.error(f"Optimization failed: {optimal_results.message}")
            # Fallback to equal weighting if optimization fails
            optimal_weights_array = np.array([1.0 / num_assets] * num_assets)
        else:
            optimal_weights_array = optimal_results.x
            # Normalize to ensure sum to 1.0 due to potential floating point inaccuracies
            optimal_weights_array /= np.sum(optimal_weights_array)

        optimal_weights = dict(zip(self.assets, optimal_weights_array))

        # Calculate final performance using the optimized weights
        p_return, p_volatility, p_sharpe = self.calculate_portfolio_performance(
            optimal_weights_array
        )

        logger.info(
            f"Optimization complete. Max Sharpe: {p_sharpe:.4f} "
            f"with weights: {optimal_weights}"
        )

        return OptimizationResult(
            weights=optimal_weights,
            expected_return=p_return,
            volatility=p_volatility,
            sharpe_ratio=p_sharpe,
        )
