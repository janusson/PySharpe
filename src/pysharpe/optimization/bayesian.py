"""Bayesian Portfolio Optimization using PyMC.

This module provides tools for estimating the posterior distribution of asset returns
using Bayesian methods, serving as a robust foundation for portfolio allocation models
like Black-Litterman.
"""

import logging

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.optimize import minimize

from .base import OptimizationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compilation-error detection helpers
# ---------------------------------------------------------------------------
_compilation_error_types: tuple[type[Exception], ...] = ()
try:
    from pytensor.link.c.exceptions import (
        CompileError as _PytensorCompileError,  # type: ignore[import-untyped]
    )

    _compilation_error_types = (_PytensorCompileError,)
except ImportError:  # pragma: no cover - pytensor may not expose this module
    pass

_COMPILATION_KEYWORDS: tuple[str, ...] = (
    "compile",
    "compilation",
    "linker",
    "c compiler",
    "gcc",
    "clang",
    "g++",
    "ld returned",
    "cannot compile",
)


def _is_compilation_error(exc: BaseException) -> bool:
    """Return ``True`` when *exc* looks like a C-compiler / linker failure."""
    if isinstance(exc, _compilation_error_types):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in _COMPILATION_KEYWORDS)


class BayesianOptimizer:
    """Estimates the posterior distribution of asset returns using PyMC
    and optimizes weights."""

    def __init__(
        self,
        returns: pd.DataFrame | None = None,
        random_seed: int | None = 42,
        risk_free_rate: float = 0.0,
    ):
        """Initialize the Bayesian Optimizer.

        Args:
            returns (Optional[pd.DataFrame]): Historical asset returns.
            random_seed (Optional[int]): Seed for reproducibility of MCMC sampling.
            risk_free_rate (float): Annual risk-free rate for Sharpe calculation.
        """
        self.returns = returns
        self.random_seed = random_seed
        self.risk_free_rate = risk_free_rate
        self.trace_ = None
        self.model_ = None
        self.assets_ = None

    # ------------------------------------------------------------------
    # Compile-mode safety net
    # ------------------------------------------------------------------
    @staticmethod
    def _enable_fast_compile() -> None:
        """Configure PyTensor to use FAST_COMPILE (pure-Python) mode.

        Bypasses the C-compiler / linker path entirely so that PyMC can
        function in environments where a working C toolchain is unavailable
        (e.g. macOS with Python 3.13 and no command-line developer tools).
        """
        import pytensor

        pytensor.config.mode = "FAST_COMPILE"

    def fit_returns_model(
        self,
        returns: pd.DataFrame | None = None,
        draws: int = 1000,
        tune: int = 1000,
        target_accept: float = 0.9,
        **kwargs: object,
    ) -> az.InferenceData:
        """Fit a Bayesian multivariate normal model to historical returns.

        This method estimates the mean vector and covariance matrix of asset returns.

        If the initial MCMC sampling attempt fails because of a C-compiler or
        linker error (common on macOS with Python 3.13 when the Xcode
        command-line tools are not installed), the method automatically falls
        back to PyTensor's ``FAST_COMPILE`` pure-Python mode and retries
        sampling.

        Args:
            returns (Optional[pd.DataFrame]): Historical asset returns.
                If None, uses self.returns.
            draws (int): Number of MCMC samples to draw.
            tune (int): Number of tuning steps for the sampler.
            target_accept (float): Target acceptance rate for NUTS.
            **kwargs: Additional keyword arguments to pass to pm.sample().

        Returns:
            az.InferenceData: The trace containing posterior samples.
        """
        if returns is None:
            returns = self.returns

        if returns is None:
            raise ValueError(
                "Returns data must be provided either in __init__ or fit_returns_model."
            )

        if not isinstance(returns, pd.DataFrame):
            raise TypeError("Returns must be a pandas DataFrame.")
        if returns.empty:
            raise ValueError("Returns DataFrame cannot be empty.")

        self.assets_ = returns.columns.tolist()
        n_assets = len(self.assets_)
        data = returns.values

        def _sample(extra_kwargs: dict[str, object]) -> pm.Model:
            """Build the model and draw posterior samples."""
            with pm.Model() as model:
                # Priors for the mean returns
                mu = pm.Normal("mu", mu=0.0, sigma=0.1, shape=n_assets)

                # Priors for the covariance matrix using LKJ Cholesky
                sd_dist = pm.HalfCauchy.dist(beta=0.1, shape=n_assets)
                chol, corr, stds = pm.LKJCholeskyCov(
                    "chol", n=n_assets, eta=2.0, sd_dist=sd_dist, compute_corr=True
                )

                # Reconstruct the covariance matrix as a Deterministic node
                cov = pm.Deterministic(  # noqa: F841
                    "cov", pm.math.dot(chol, chol.T)
                )

                # Likelihood
                pm.MvNormal("obs", mu=mu, chol=chol, observed=data)

                # Sample from the posterior
                logger.info(
                    "Starting MCMC sampling with %d draws and %d tuning steps...",
                    draws,
                    tune,
                )
                self.trace_ = pm.sample(
                    draws=draws,
                    tune=tune,
                    target_accept=target_accept,
                    random_seed=self.random_seed,
                    return_inferencedata=True,
                    progressbar=False,
                    **extra_kwargs,
                    **kwargs,
                )
                return model

        try:
            self.model_ = _sample({})
        except Exception as exc:
            if _is_compilation_error(exc):
                logger.warning(
                    "Local C compiler/linker failed. "
                    + "Falling back to pure-Python FAST_COMPILE mode"
                    + " for PyMC/PyTensor."
                )
                BayesianOptimizer._enable_fast_compile()
                self.model_ = _sample({})
            else:
                raise

        logger.info("MCMC sampling completed.")
        return self.trace_

    def get_posterior_estimates(self) -> tuple[pd.Series, pd.DataFrame]:
        """Extract the posterior mean returns and covariance matrix.

        Returns:
            tuple[pd.Series, pd.DataFrame]: A tuple containing:
                - Expected returns (posterior mean of 'mu').
                - Expected covariance matrix (posterior mean of 'cov').
        """
        if self.trace_ is None or self.model_ is None:
            raise RuntimeError(
                "Model has not been fitted. Call `fit_returns_model` first."
            )

        # Extract posterior samples
        posterior = self.trace_.posterior

        # Calculate expected returns (mean of the posterior distribution for 'mu')
        expected_returns = posterior["mu"].mean(dim=["chain", "draw"]).values

        # Calculate expected covariance matrix
        # (mean of the posterior distribution for 'cov')
        expected_cov = posterior["cov"].mean(dim=["chain", "draw"]).values

        return (
            pd.Series(
                expected_returns, index=self.assets_, name="Posterior Mean Returns"
            ),
            pd.DataFrame(expected_cov, index=self.assets_, columns=self.assets_),
        )

    def optimize(self) -> OptimizationResult:
        """Perform Bayesian portfolio optimization.

        Fits the model and then maximizes the Sharpe ratio using posterior estimates.

        Returns:
            OptimizationResult: The result containing weights and performance metrics.
        """
        if self.trace_ is None:
            self.fit_returns_model()

        mu, cov = self.get_posterior_estimates()
        n_assets = len(self.assets_)

        # Standard Sharpe maximization using posterior means
        def objective(weights):
            p_return = np.sum(mu.values * weights) * 252  # Annualized
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov.values * 252, weights)))
            sharpe = (p_return - self.risk_free_rate) / p_vol if p_vol > 0 else 0
            return -sharpe

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        initial_guess = np.array([1.0 / n_assets] * n_assets)

        res = minimize(
            objective,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not res.success:
            logger.warning(f"Bayesian optimization failed: {res.message}")
            weights_array = initial_guess
        else:
            weights_array = res.x

        weights_dict = dict(zip(self.assets_, weights_array))

        # Calculate performance metrics
        p_return = np.sum(mu.values * weights_array) * 252
        p_vol = np.sqrt(
            np.dot(weights_array.T, np.dot(cov.values * 252, weights_array))
        )
        p_sharpe = (p_return - self.risk_free_rate) / p_vol if p_vol > 0 else 0

        return OptimizationResult(
            weights=weights_dict,
            expected_return=p_return,
            volatility=p_vol,
            sharpe_ratio=p_sharpe,
        )
