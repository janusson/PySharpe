"""Backtesting engine for historical portfolio simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Container for backtest simulation results.

    Attributes
    ----------
    portfolio_value : pd.Series
        Total simulated dollar value of the portfolio at each time step.
    historical_weights : pd.DataFrame
        Asset allocation fractions recorded at each time step.
    rebalance_events : pd.DatetimeIndex
        Dates on which a rebalancing action was triggered and executed.
    """

    portfolio_value: pd.Series
    historical_weights: pd.DataFrame
    rebalance_events: pd.DatetimeIndex


class HistoricalBacktester:
    """Simulates chronological portfolio performance with drift and rebalancing.

    This engine calculates the evolving value of a portfolio given starting capital,
    target allocations, and an array of historical prices. It models the drift of
    asset weights over time as prices change and enforces rebalancing rules based
    on calendar frequency or strict deviation bands.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        target_weights: dict[str, float],
        initial_capital: float = 10000.0,
        rebalance_freq: str | None = None,
        abs_band: float | None = None,
        rel_band: float | None = None,
    ) -> None:
        """Initialize the backtester with price data and rule configurations.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical asset prices where rows are dates (DatetimeIndex) and columns
            are asset tickers.
        target_weights : dict of str to float
            The ideal allocation fraction for each asset (e.g., `{"AAPL": 0.6, "TLT": 0.4}`).
        initial_capital : float, default 10000.0
            The starting cash value injected into the simulation at t=0.
        rebalance_freq : str, optional
            A pandas frequency string dictating calendar rebalancing. Common values
            include 'M' (monthly end), 'Q' (quarterly), or 'A' (annual). If None,
            the portfolio is not rebalanced based on the calendar.
        abs_band : float, optional
            Absolute deviation threshold (e.g., 0.05). If an asset's weight drifts
            more than this fraction from its target, a rebalance is triggered.
        rel_band : float, optional
            Relative deviation threshold (e.g., 0.20). If an asset's weight drifts
            by a percentage of its target greater than this bound, a rebalance
            is triggered.

        Raises
        ------
        ValueError
            If no assets in `target_weights` match the columns in `prices`, or if
            the target weights sum to zero.
        """
        self.prices = prices.dropna().sort_index()
        self.target_weights = target_weights
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.abs_band = abs_band
        self.rel_band = rel_band

        # Identify valid assets
        available_assets = set(self.prices.columns)
        target_assets = set(self.target_weights.keys())
        self.assets = sorted(list(available_assets.intersection(target_assets)))

        if not self.assets:
            raise ValueError(
                "No overlapping assets found between prices and target weights."
            )

        # Construct target vector
        raw_targets = np.array([self.target_weights[a] for a in self.assets])
        total_target = raw_targets.sum()
        if total_target <= 0:
            raise ValueError("Total target weight must be positive.")

        self.targets = raw_targets / total_target

    def run(self) -> BacktestResult:
        """Execute the chronological backtest simulation.

        Steps through the provided price history day-by-day. At each step, it:
        1. Updates the total portfolio value based on previously held share counts.
        2. Computes the current drifted weights.
        3. Checks if any rebalance condition (calendar, absolute band, relative band)
           is met.
        4. If met, recalculates share counts to match `target_weights` exactly at
           current prices.
        5. Records the daily state.

        Returns
        -------
        BacktestResult
            An object containing the portfolio value series, the historical weight
            drift dataframe, and a DatetimeIndex of dates where rebalancing occurred.
        """
        price_matrix = self.prices[self.assets].values
        dates = self.prices.index
        n_days, n_assets = price_matrix.shape

        if n_days == 0:
            return BacktestResult(
                pd.Series(dtype=float),
                pd.DataFrame(columns=self.assets),
                pd.DatetimeIndex([]),
            )

        # Precompute calendar rebalance triggers
        calendar_triggers = np.zeros(n_days, dtype=bool)
        if self.rebalance_freq:
            # Group by period and mark the last day of each period
            period_idx = dates.to_period(self.rebalance_freq)
            is_period_end = np.zeros(n_days, dtype=bool)
            if n_days > 1:
                is_period_end[:-1] = period_idx[:-1] != period_idx[1:]
            # The last element is always a period end in this logic
            if n_days > 0:
                is_period_end[-1] = True
            calendar_triggers = is_period_end

        # Simulation State
        current_shares = np.zeros(n_assets)

        # Result Containers
        portfolio_values = np.zeros(n_days)
        daily_weights = np.zeros((n_days, n_assets))
        rebalance_dates = []

        # Initial Setup (Day 0)
        # Assume buy at Close of Day 0
        initial_prices = price_matrix[0]
        current_shares = (self.initial_capital * self.targets) / initial_prices
        portfolio_values[0] = self.initial_capital
        daily_weights[0] = self.targets

        for t in range(1, n_days):
            current_prices = price_matrix[t]

            # 1. Calculate current value based on holdings
            asset_values = current_shares * current_prices
            total_value = np.sum(asset_values)
            current_weights = asset_values / total_value

            # 2. Check Rebalance Triggers
            should_rebalance = False

            # A. Calendar
            if calendar_triggers[t]:
                should_rebalance = True

            # B. Absolute Drift
            if not should_rebalance and self.abs_band is not None:
                abs_drift = np.abs(current_weights - self.targets)
                if np.any(abs_drift > self.abs_band):
                    should_rebalance = True

            # C. Relative Drift
            if not should_rebalance and self.rel_band is not None:
                nonzero_mask = self.targets > 1e-6
                if np.any(nonzero_mask):
                    rel_drift = (
                        np.abs(
                            current_weights[nonzero_mask] - self.targets[nonzero_mask]
                        )
                        / self.targets[nonzero_mask]
                    )
                    if np.any(rel_drift > self.rel_band):
                        should_rebalance = True

            # 3. Execute Rebalance or Record Drift
            if should_rebalance:
                current_shares = (total_value * self.targets) / current_prices
                current_weights = self.targets
                rebalance_dates.append(dates[t])

            # 4. Update records
            portfolio_values[t] = total_value
            daily_weights[t] = current_weights

        return BacktestResult(
            portfolio_value=pd.Series(
                portfolio_values, index=dates, name="Portfolio Value"
            ),
            historical_weights=pd.DataFrame(
                daily_weights, index=dates, columns=self.assets
            ),
            rebalance_events=pd.DatetimeIndex(rebalance_dates).normalize(),
        )
