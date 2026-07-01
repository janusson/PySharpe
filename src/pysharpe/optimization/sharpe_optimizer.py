"""Module for Sharpe ratio portfolio optimization with asset-location awareness.

Optimises across an N-asset × M-account matrix, using tax-adjusted expected
returns from the :class:`AssetLocationEngine` to simultaneously solve the
asset allocation **and** asset location problems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pysharpe.exceptions import ExecutionConfigError
from pysharpe.optimization.tax_location import (
    AccountType,
    AssetLocationEngine,
    AssetTaxCharacteristics,
    TaxProfile,
)

from .base import OptimizationResult

logger = logging.getLogger(__name__)

# Sentinel for "no net-return difference across accounts" (backward compat).
_UNCONSTRAINED_ACCOUNT = "__unconstrained__"


@dataclass
class SharpeOptimizerConfig:
    """Configuration for the SharpeOptimizer with 2D asset-location support.

    Attributes
    ----------
    risk_free_rate : float
        Annual risk-free rate. Default is 0.025 (2.5 %).
    mer_by_ticker : dict[str, float]
        Ticker → annual MER decimal (e.g. ``0.0017``).  When
        ``asset_characteristics`` includes MER values those take precedence.
    max_portfolio_mer : float or None
        Maximum allowable weighted MER for the entire portfolio.
    tax_profile : TaxProfile
        Investor's marginal tax profile for the :class:`AssetLocationEngine`.
    asset_characteristics : dict[str, AssetTaxCharacteristics]
        Ticker → detailed tax characteristics (domicile, income fractions,
        MER, dividend yield).  Used by the engine to compute tax-adjusted
        returns per (asset, account) pair.
    account_capacities : dict[AccountType, float]
        Fractional capacity per account (e.g. ``{TFSA: 0.20, RRSP: 0.50,
        NON_REG: 0.30}``).  Must sum to ≤ 1.0.  When empty the optimiser
        falls back to classic 1-D (asset-only) behaviour.
    num_portfolios_monte_carlo : int
        Random portfolios to generate for the initial guess. Default 10 000.
    max_weight : float
        Per-asset maximum weight (applied across all accounts for that
        asset).  Default 0.20.
    """

    risk_free_rate: float = 0.025
    mer_by_ticker: dict[str, float] = field(default_factory=dict)
    max_portfolio_mer: float | None = None
    tax_profile: TaxProfile = field(
        default_factory=lambda: TaxProfile(marginal_tax_rate=0.40)
    )
    asset_characteristics: dict[str, AssetTaxCharacteristics] = field(
        default_factory=dict
    )
    account_capacities: dict[AccountType, float] = field(default_factory=dict)
    num_portfolios_monte_carlo: int = 10000
    max_weight: float = 0.20


class SharpeOptimizer:
    """Optimizes (asset × account) weights to maximise the Sharpe ratio.

    When ``account_capacities`` is non-empty the solver operates on an
    **N × M** variable space — one weight per (asset, account) pair — and
    enforces per-account capacity constraints.  Tax-adjusted net returns
    are computed by the :class:`AssetLocationEngine` for each pair.

    With an empty ``account_capacities`` the optimiser delegates to the
    classic 1-D path for backward compatibility.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        config: SharpeOptimizerConfig | None = None,
    ) -> None:
        if prices.empty:
            raise ValueError("Prices DataFrame cannot be empty.")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("Prices DataFrame must have a DatetimeIndex.")

        self.prices = prices.dropna()
        self.assets = self.prices.columns.tolist()
        self.config = config if config is not None else SharpeOptimizerConfig()

        if not self.assets:
            raise ValueError("No assets found in the prices DataFrame.")

        self.returns = self.prices.pct_change().dropna()
        if self.returns.empty:
            raise ValueError(
                "Insufficient data to calculate returns after dropping NaNs."
            )

        self.num_periods_per_year = self._estimate_periods_per_year(
            cast(pd.DatetimeIndex, self.prices.index)
        )
        self._rf_period = self.config.risk_free_rate / self.num_periods_per_year

        # ------------------------------------------------------------------
        # Build the 2-D variable layout
        # ------------------------------------------------------------------
        capacities = self.config.account_capacities
        if capacities:
            # Validate capacities
            total_cap = sum(capacities.values())
            if total_cap > 1.0 + 1e-9:
                raise ValueError(
                    f"Account capacities sum to {total_cap:.4f} (> 1.0). "
                    "They must be fractional and sum to ≤ 1.0."
                )
            self._accounts: list[AccountType] = list(capacities.keys())
            self._account_values: list[str] = [a.value for a in self._accounts]
            self._capacity_array = np.array(
                [capacities[a] for a in self._accounts], dtype=float
            )
        else:
            # Backward-compat: single synthetic account
            self._accounts = []
            self._account_values = [_UNCONSTRAINED_ACCOUNT]
            self._capacity_array = np.array([1.0])

        self._num_assets = len(self.assets)
        self._num_accounts = len(self._account_values)
        self._num_vars = self._num_assets * self._num_accounts

        # ------------------------------------------------------------------
        # Gross expected returns (asset-level, annualised)
        # ------------------------------------------------------------------
        self._gross_returns = np.array(
            self.returns.mean() * self.num_periods_per_year
        )  # shape (N,)

        # ------------------------------------------------------------------
        # 2-D net-return matrix  (N × M)
        # ------------------------------------------------------------------
        self._net_returns = self._build_net_return_matrix()

        # ------------------------------------------------------------------
        # Asset-level covariance (N × N)
        # ------------------------------------------------------------------
        self._cov = self.returns.cov().values * self.num_periods_per_year

        logger.info(
            "SharpeOptimizer initialised: %d assets × %d accounts = %d variables. "
            "Risk-free rate: %.2f%%. Periods/year: %.0f.",
            self._num_assets,
            self._num_accounts,
            self._num_vars,
            self.config.risk_free_rate * 100,
            self.num_periods_per_year,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_net_return_matrix(self) -> np.ndarray:
        """Build (N × M) matrix of tax-adjusted annualised expected returns.

        When no accounts are configured the single-column result is the
        gross return less any configured MER.
        """
        engine = AssetLocationEngine(self.config.tax_profile)

        if not self._accounts:
            # Classic 1-D path — deduct MER only
            net = np.zeros((self._num_assets, 1))
            for i, ticker in enumerate(self.assets):
                mer = self.config.mer_by_ticker.get(ticker, 0.0)
                char = self.config.asset_characteristics.get(ticker)
                if char is not None:
                    mer = char.mer
                net[i, 0] = self._gross_returns[i] - mer
            return net

        # 2-D path — use AssetLocationEngine
        net = np.zeros((self._num_assets, self._num_accounts))
        for i, ticker in enumerate(self.assets):
            char = self.config.asset_characteristics.get(ticker)
            gross = self._gross_returns[i]
            for j, acct in enumerate(self._accounts):
                if char is not None:
                    net[i, j] = engine.compute_tax_adjusted_return(
                        gross, char, acct.value
                    )
                else:
                    # No characteristics → use MER from config only
                    mer = self.config.mer_by_ticker.get(ticker, 0.0)
                    net[i, j] = gross - mer
        return net

    @staticmethod
    def _estimate_periods_per_year(index: pd.DatetimeIndex) -> float:
        if len(index) < 2:
            return 252.0
        diffs = index.to_series().diff().dropna().dt.days
        median_days = diffs.median()
        if median_days <= 1.05:
            return 252.0
        elif median_days <= 7.05:
            return 52.0
        elif median_days <= 31.05:
            return 12.0
        elif median_days <= 92.05:
            return 4.0
        elif median_days <= 366.05:
            return 1.0
        logger.warning(
            "Could not reliably determine data frequency. Assuming 252 periods/year."
        )
        return 252.0

    # ------------------------------------------------------------------
    # 2-D → asset-level aggregation
    # ------------------------------------------------------------------

    def _asset_weights_from_2d(self, weights_2d: np.ndarray) -> np.ndarray:
        """Sum 2-D weights across accounts to get per-asset weights.  (N,)"""
        return weights_2d.reshape(self._num_assets, self._num_accounts).sum(axis=1)

    # ------------------------------------------------------------------
    # Performance calculation
    # ------------------------------------------------------------------

    def calculate_portfolio_performance(
        self, weights: np.ndarray
    ) -> tuple[float, float, float]:
        """Compute (return, volatility, Sharpe) for a 2-D weight vector.

        Parameters
        ----------
        weights : np.ndarray
            Flat array of length *N* × *M* (or *N* for classic 1-D).

        Returns
        -------
        tuple[float, float, float]
            (annualized_return, annualized_volatility, sharpe_ratio)
        """
        weights = np.asarray(weights, dtype=float)

        # Normalise if needed
        total = weights.sum()
        if not np.isclose(total, 1.0) and total > 0:
            logger.debug("Weights sum to %.6f; normalising.", total)
            weights = weights / total

        # --- Expected return (sum-product of 2-D weights × net returns) ---
        if weights.size == self._num_vars:
            # 2-D path
            w2d = weights.reshape(self._num_assets, self._num_accounts)
            portfolio_return = float(np.sum(w2d * self._net_returns))
        else:
            # 1-D fallback
            portfolio_return = float(np.dot(weights, self._gross_returns))
            for i, ticker in enumerate(self.assets):
                mer = self.config.mer_by_ticker.get(ticker, 0.0)
                char = self.config.asset_characteristics.get(ticker)
                if char is not None:
                    mer = char.mer
                portfolio_return -= weights[i] * mer

        # --- Volatility (asset-level covariance) ---
        if weights.size == self._num_vars:
            asset_w = self._asset_weights_from_2d(weights)
        else:
            asset_w = weights
        var = float(asset_w @ self._cov @ asset_w)
        portfolio_volatility = float(np.sqrt(max(var, 0.0)))

        if portfolio_volatility == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (
                portfolio_return - self.config.risk_free_rate
            ) / portfolio_volatility

        return portfolio_return, portfolio_volatility, sharpe_ratio

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _objective_function(self, weights: np.ndarray) -> float:
        _, _, sr = self.calculate_portfolio_performance(weights)
        return -sr

    # ------------------------------------------------------------------
    # Monte Carlo initial guess
    # ------------------------------------------------------------------

    def _generate_random_portfolios(self, num_portfolios: int) -> pd.DataFrame:
        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in range(num_portfolios):
            w = np.random.random(self._num_vars)
            w /= w.sum()
            weights_record.append(tuple(w))
            r, v, s = self.calculate_portfolio_performance(w)
            results[0, i] = r
            results[1, i] = v
            results[2, i] = s

        return pd.DataFrame(
            {"Return": results[0], "Volatility": results[1], "Sharpe": results[2]},
            index=pd.MultiIndex.from_arrays(
                [range(num_portfolios), weights_record],
                names=["Portfolio", "Weights"],
            ),
        )

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------

    def optimize(self) -> OptimizationResult:
        """Run the 2-D (or classic 1-D) Sharpe-maximisation.

        Returns
        -------
        OptimizationResult
            Weights keyed by ``(ticker, account_value)`` tuples in 2-D mode,
            or by plain ticker in 1-D mode.
        """
        if self._num_assets == 0:
            return OptimizationResult({}, 0.0, 0.0, 0.0)

        if self.config.max_weight * self._num_assets < 1.0:
            raise ValueError(
                f"max_weight ({self.config.max_weight}) too restrictive "
                f"for {self._num_assets} assets."
            )

        # --- Bounds ---
        bounds = tuple((0.0, self.config.max_weight) for _ in range(self._num_vars))

        # --- Constraints ---
        constraints: list[dict] = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

        # Account capacity constraints (only in 2-D mode)
        if self._accounts:
            for j in range(self._num_accounts):
                # Sum of weights for account j across all assets ≤ capacity[j]
                def _account_constraint(x: np.ndarray, _j: int = j) -> float:
                    w2d = x.reshape(self._num_assets, self._num_accounts)
                    return self._capacity_array[_j] - w2d[:, _j].sum()

                constraints.append({"type": "ineq", "fun": _account_constraint})

        # --- Aggregate MER constraint (applied to asset-level weights) ---
        if self.config.max_portfolio_mer is not None and self.config.mer_by_ticker:
            mer_array = np.array(
                [self.config.mer_by_ticker.get(a, 0.0) for a in self.assets]
            )
            # Override with characteristics MER when available
            for i, ticker in enumerate(self.assets):
                char = self.config.asset_characteristics.get(ticker)
                if char is not None:
                    mer_array[i] = char.mer

            min_mer = float(mer_array.min())
            if min_mer > self.config.max_portfolio_mer:
                raise ExecutionConfigError(
                    f"max_portfolio_mer ({self.config.max_portfolio_mer:.4%}) "
                    f"infeasible: lowest individual MER is {min_mer:.4%}."
                )

            def _mer_constraint(x: np.ndarray) -> float:
                if x.size == self._num_vars and self._accounts:
                    asset_w = self._asset_weights_from_2d(x)
                else:
                    asset_w = x
                return self.config.max_portfolio_mer - np.dot(asset_w, mer_array)

            constraints.append({"type": "ineq", "fun": _mer_constraint})

        # --- Initial guess ---
        initial_guess = np.full(self._num_vars, 1.0 / self._num_vars)
        if self.config.num_portfolios_monte_carlo > 0:
            try:
                rp = self._generate_random_portfolios(
                    self.config.num_portfolios_monte_carlo
                )
                best_idx = rp["Sharpe"].idxmax()
                # best_idx is a tuple (Portfolio, Weights) from the MultiIndex;
                # index [1] extracts the weight vector.
                best_tuple = cast(tuple, best_idx)
                initial_guess = np.array(best_tuple[1])
            except Exception:
                logger.debug("Monte Carlo initial-guess failed; using equal weights.")

        # --- Solve ---
        try:
            result = minimize(
                self._objective_function,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
        except np.linalg.LinAlgError as exc:
            logger.error(
                "Linear-algebra error: %s. Falling back to equal weights.", exc
            )
            return self._fallback_result()

        if not result.success:
            logger.error("Optimisation failed: %s", result.message)
            return self._fallback_result()

        w_opt = result.x
        w_opt /= w_opt.sum()  # guard against float drift
        p_ret, p_vol, p_sr = self.calculate_portfolio_performance(w_opt)

        # --- Format weights (drop near-zero entries) ---
        if self._accounts:
            w2d = w_opt.reshape(self._num_assets, self._num_accounts)
            weights_map: dict = {}
            for i, ticker in enumerate(self.assets):
                for j, acct_val in enumerate(self._account_values):
                    w = float(w2d[i, j])
                    if w > 1e-8:
                        weights_map[(ticker, acct_val)] = w
        else:
            weights_map = {
                t: float(w) for t, w in zip(self.assets, w_opt.tolist()) if w > 1e-8
            }

        logger.info(
            "Optimisation complete. Sharpe=%.4f, return=%.4f%%, vol=%.4f%%.",
            p_sr,
            p_ret * 100,
            p_vol * 100,
        )
        return OptimizationResult(
            weights=weights_map,
            expected_return=float(p_ret),
            volatility=float(p_vol),
            sharpe_ratio=float(p_sr),
        )

    def _fallback_result(self) -> OptimizationResult:
        """Equal-weight fallback when optimisation fails."""
        w = np.full(self._num_vars, 1.0 / self._num_vars)
        p_ret, p_vol, p_sr = self.calculate_portfolio_performance(w)

        if self._accounts:
            w2d = w.reshape(self._num_assets, self._num_accounts)
            weights_map = {}
            for i, ticker in enumerate(self.assets):
                for j, acct_val in enumerate(self._account_values):
                    w_val = float(w2d[i, j])
                    if w_val > 1e-8:
                        weights_map[(ticker, acct_val)] = w_val
        else:
            weights_map = {
                t: float(w_val)
                for t, w_val in zip(self.assets, w.tolist())
                if w_val > 1e-8
            }

        return OptimizationResult(
            weights=weights_map,
            expected_return=float(p_ret),
            volatility=float(p_vol),
            sharpe_ratio=float(p_sr),
        )
