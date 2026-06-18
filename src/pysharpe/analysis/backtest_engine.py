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
        vol_threshold: float | None = None,
        fee_per_trade: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> None:
        """Initialize the backtester with price data and rule configurations.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical asset prices where rows are dates (DatetimeIndex) and columns
            are asset tickers.
        target_weights : dict of str to float
            The ideal allocation fraction for each asset (e.g., `{"AAPL": 0.6}`).
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
        vol_threshold : float, optional
            Annualized 20-day realized volatility threshold. If the portfolio's
            rolling volatility exceeds this limit, a rebalance is forced to reset
            allocations during market stress.
        fee_per_trade : float, default 0.0
            Fixed transaction cost in dollars per asset traded during a rebalance.
        slippage_pct : float, default 0.0
            Variable transaction cost representing slippage or spread, as a fraction
            of the total dollar amount traded.

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
        self.vol_threshold = vol_threshold
        self.fee_per_trade = fee_per_trade
        self.slippage_pct = slippage_pct

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
        3. Checks if any rebalance condition (calendar, absolute band, relative band,
           or volatility threshold) is met.
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
            # Translate newer pandas offset aliases (ME/QE/YE) to period aliases (M/Q/Y)
            # `to_period` requires the older aliases on pandas >= 2.2
            _PERIOD_ALIAS_MAP = {
                "ME": "M",
                "QE": "Q",
                "YE": "Y",
                "MS": "M",
                "QS": "Q",
                "YS": "Y",
            }
            period_freq = _PERIOD_ALIAS_MAP.get(
                self.rebalance_freq, self.rebalance_freq
            )
            # Group by period and mark the last day of each period
            period_idx = dates.to_period(period_freq)
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
        initial_target_dollars = self.initial_capital * self.targets

        # Calculate initial fees
        initial_num_trades = np.sum(initial_target_dollars > 1e-4)
        initial_slippage = np.sum(initial_target_dollars) * self.slippage_pct
        initial_fixed_fees = initial_num_trades * self.fee_per_trade
        total_initial_fees = initial_slippage + initial_fixed_fees

        actual_starting_capital = self.initial_capital - total_initial_fees

        current_shares = (actual_starting_capital * self.targets) / initial_prices
        portfolio_values[0] = actual_starting_capital
        daily_weights[0] = self.targets

        for t in range(1, n_days):
            current_prices = price_matrix[t]

            # 1. Calculate current value based on holdings
            asset_values = current_shares * current_prices
            total_value = np.sum(asset_values)
            portfolio_values[t] = total_value
            if total_value <= 0:
                break  # Portfolio wiped out; stop simulation
            current_weights = asset_values / total_value

            # 2. Check Rebalance Triggers
            should_rebalance = False

            # A. Calendar
            if calendar_triggers[t]:
                should_rebalance = True

            # B. Volatility Threshold
            if not should_rebalance and self.vol_threshold is not None and t >= 20:
                # Calculate 20-day realized volatility of portfolio log returns
                # We need the last 21 values to get 20 returns
                window_values = portfolio_values[t - 20 : t + 1]
                log_returns = np.log(window_values[1:] / window_values[:-1])
                # Annualize using 252 trading days
                rolling_vol = np.std(log_returns, ddof=1) * np.sqrt(252)
                if rolling_vol > self.vol_threshold:
                    should_rebalance = True

            # C. Absolute Drift
            if not should_rebalance and self.abs_band is not None:
                abs_drift = np.abs(current_weights - self.targets)
                if np.any(abs_drift > self.abs_band):
                    should_rebalance = True

            # D. Relative Drift
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
                # Calculate required trades before fees
                ideal_target_shares = (total_value * self.targets) / current_prices
                shares_traded = np.abs(ideal_target_shares - current_shares)
                dollars_traded = shares_traded * current_prices

                # Apply transaction costs
                num_trades = np.sum(dollars_traded > 1e-4)
                total_slippage = np.sum(dollars_traded) * self.slippage_pct
                total_fixed_fees = num_trades * self.fee_per_trade
                total_fees = total_slippage + total_fixed_fees

                # Deduct fees
                total_value -= total_fees

                # Recalculate shares with adjusted capital
                current_shares = (total_value * self.targets) / current_prices
                current_weights = self.targets
                rebalance_dates.append(dates[t])

            # 4. Update daily weights
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


from typing import Callable

from ..optimization.base import PortfolioOptimizer


class WalkForwardBacktester:
    """Performs a walk-forward optimization and backtest.

    Iteratively trains a PortfolioOptimizer on a rolling window of historical data
    and simulates the performance of the optimized weights on a subsequent out-of-sample
    test window.
    """

    def __init__(
        self,
        optimizer_factory: Callable[[pd.DataFrame], PortfolioOptimizer],
        train_window_days: int = 252,
        test_window_days: int = 21,
        initial_capital: float = 10000.0,
    ) -> None:
        """Initialize the walk-forward backtester.

        Parameters
        ----------
        optimizer_factory : Callable[[pd.DataFrame], PortfolioOptimizer]
            A factory function that takes a DataFrame of historical prices and
            returns an instance implementing the PortfolioOptimizer protocol.
        train_window_days : int, default 252
            Number of days to use for optimization in each step.
        test_window_days : int, default 21
            Number of days to simulate using the optimized weights before re-optimizing.
        initial_capital : float, default 10000.0
            Starting capital for the backtest.
        """
        self.optimizer_factory = optimizer_factory
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.initial_capital = initial_capital

    def run(self, prices: pd.DataFrame) -> BacktestResult:
        """Run the walk-forward backtest.

        Parameters
        ----------
        prices : pd.DataFrame
            Full historical price data.

        Returns
        -------
        BacktestResult
            The combined result of the walk-forward backtest.
        """
        prices = prices.dropna().sort_index()
        dates = prices.index
        n_days = len(dates)

        if n_days < self.train_window_days + self.test_window_days:
            raise ValueError(
                "Insufficient data for walk-forward backtest with the given window sizes."
            )

        portfolio_values = []
        historical_weights = []
        rebalance_events = []

        current_capital = self.initial_capital
        start_idx = 0

        # Run until we can't form a full training window
        while start_idx + self.train_window_days < n_days:
            train_end_idx = start_idx + self.train_window_days
            test_end_idx = min(train_end_idx + self.test_window_days, n_days)

            # 1. Train
            train_data = prices.iloc[start_idx:train_end_idx]
            optimizer = self.optimizer_factory(train_data)
            opt_result = optimizer.optimize()
            target_weights = opt_result.weights

            # 2. Test
            test_data = prices.iloc[train_end_idx:test_end_idx]

            # Use HistoricalBacktester to simulate the test window
            # Start the test window with the ending capital of the previous window
            sub_backtester = HistoricalBacktester(
                prices=test_data,
                target_weights=target_weights,
                initial_capital=current_capital,
                # Rebalance only at the start of the test window (implicitly by target_weights)
                rebalance_freq=None,
            )
            sub_result = sub_backtester.run()

            # 3. Accumulate results
            portfolio_values.append(sub_result.portfolio_value)
            historical_weights.append(sub_result.historical_weights)
            rebalance_events.append(dates[train_end_idx])

            # Update capital for the next iteration
            current_capital = sub_result.portfolio_value.iloc[-1]

            # Step forward
            start_idx += self.test_window_days

        # Combine results
        combined_values = pd.concat(portfolio_values)
        # Ensure unique index by keeping the last value of any duplicates (though edges shouldn't overlap if sliced properly)
        combined_values = combined_values[
            ~combined_values.index.duplicated(keep="last")
        ]

        combined_weights = pd.concat(historical_weights)
        combined_weights = combined_weights[
            ~combined_weights.index.duplicated(keep="last")
        ]

        return BacktestResult(
            portfolio_value=combined_values,
            historical_weights=combined_weights,
            rebalance_events=pd.DatetimeIndex(rebalance_events),
        )

    def compare_to_benchmark(
        self, prices: pd.DataFrame, benchmark_ticker: str = "SPY"
    ) -> pd.DataFrame:
        """Compare the walk-forward equity curve against a buy-and-hold benchmark.

        Parameters
        ----------
        prices : pd.DataFrame
            Full historical price data containing the benchmark ticker.
        benchmark_ticker : str, default "SPY"
            The ticker symbol of the benchmark asset.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing 'Strategy' and 'Benchmark' equity curves,
            normalized to start at the same initial capital.
        """
        if benchmark_ticker not in prices.columns:
            raise ValueError(
                f"Benchmark ticker '{benchmark_ticker}' not found in prices data."
            )

        # Run strategy
        result = self.run(prices)
        strategy_curve = result.portfolio_value

        # Calculate benchmark curve
        # Align benchmark with the strategy's dates
        bench_prices = prices[benchmark_ticker].loc[strategy_curve.index]
        bench_returns = bench_prices.pct_change().fillna(0)
        benchmark_curve = self.initial_capital * (1 + bench_returns).cumprod()

        comparison = pd.DataFrame(
            {"Strategy": strategy_curve, "Benchmark": benchmark_curve}
        )

        return comparison
