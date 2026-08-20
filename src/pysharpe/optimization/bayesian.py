"""Bayesian Portfolio Optimization using PyMC.

This module provides tools for estimating the posterior distribution of asset returns
using Bayesian methods, serving as a robust foundation for portfolio allocation models
like Black-Litterman.
"""

import logging
import os
import shutil
from typing import cast

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from pypfopt.efficient_frontier import EfficientFrontier
from scipy.optimize import minimize

from pysharpe.exceptions import DataValidationError

from .base import OptimizationResult
from .estimators import ensure_strictly_psd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# C-compiler toolchain detection
# ---------------------------------------------------------------------------


def _has_c_compiler() -> bool:
    """Return ``True`` if a C/C++ compiler appears available on the system.

    Checks (in order of authority):

    1. ``CXX`` / ``CC`` environment variables — if set, verifies the
       named compiler is resolvable on ``PATH``.
    2. Common compiler names — ``clang``, ``gcc``, ``g++``, ``cc``, ``c++``
       — probed via :func:`shutil.which`.

    This is a lightweight pre-flight check; it does **not** verify that the
    compiler can actually link or produce working binaries.
    """
    for var in ("CXX", "CC"):
        compiler = os.environ.get(var)
        if compiler and shutil.which(compiler):
            return True

    for compiler in ("clang", "gcc", "g++", "cc", "c++"):
        if shutil.which(compiler):
            return True

    return False


# ---------------------------------------------------------------------------
# PyTensor compilation-error type (imported once at module level)
# ---------------------------------------------------------------------------

try:
    from pytensor.link.c.exceptions import (  # type: ignore[import-untyped]
        CompileError as _PytensorCompileError,
    )
except ImportError:  # pragma: no cover - pytensor may not expose this module
    _PytensorCompileError = None  # type: ignore[assignment]


def _is_compilation_error(exc: BaseException) -> bool:
    """Return ``True`` when *exc* is a known PyTensor C-compilation error.

    Uses PyTensor's own ``CompileError`` exception type rather than brittle
    string-matching against error messages.
    """
    return _PytensorCompileError is not None and isinstance(exc, _PytensorCompileError)


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
        import pytensor  # type: ignore[import-untyped]

        _config = getattr(pytensor, "config", None)
        if _config is not None:
            _config.mode = "FAST_COMPILE"

    # ------------------------------------------------------------------
    # Compilation cache warm-up
    # ------------------------------------------------------------------
    @classmethod
    def warm_compilation_cache(cls) -> bool:
        """Pre-warm the PyTensor compilation cache and verify the toolchain.

        First performs a **preemptive** check via :func:`_has_c_compiler` to
        determine whether a C/C++ compiler toolchain is available.  If no
        compiler is found, ``FAST_COMPILE`` pure-Python mode is enabled
        immediately — without invoking PyTensor's compilation path at all.

        When a compiler **is** detected, a minimal PyTensor graph is compiled
        as a functional probe.  If that probe fails (e.g. the compiler exists
        but cannot link), the method falls back to ``FAST_COMPILE``.

        This is safe to call early during CLI startup — it is non-blocking
        and does not mutate any instance state.

        Returns:
            ``True`` when the C-compiler toolchain is functional (no
            fallback needed), ``False`` when PyTensor was switched to
            ``FAST_COMPILE`` mode.
        """
        try:
            import pytensor  # type: ignore[import-untyped]
        except ImportError:
            logger.info(
                "PyTensor is not installed; skipping compilation cache warm-up."
            )
            return True  # Nothing to warm — not a failure.

        # --- Preemptive compiler check ---------------------------------------
        if not _has_c_compiler():
            logger.warning(
                "No C compiler (clang/gcc) found on PATH and CXX/CC not set. "
                "Switching PyTensor to FAST_COMPILE pure-Python mode. "
                "Install Xcode CLI tools (macOS) or a C++ compiler to restore "
                "full performance for Bayesian estimation."
            )
            cls._enable_fast_compile()
            return False

        # --- Functional probe: compile a minimal graph -----------------------
        try:
            import pytensor.tensor as pt  # type: ignore[import-untyped]
        except ImportError:
            logger.info("pytensor.tensor not importable; skipping probe.")
            return True

        _config = getattr(pytensor, "config", None)
        original_mode: str = _config.mode if _config is not None else "FAST_COMPILE"

        try:
            x = pt.scalar("x")
            f = pytensor.function([x], x + 1)  # type: ignore[reportPrivateImportUsage]
            f(0)
            logger.info(
                "PyTensor C-compiler toolchain verified successfully (mode=%s).",
                original_mode,
            )
            return True
        except Exception as exc:
            logger.warning(
                "C compiler found but PyTensor compilation failed (%s). "
                "Switching to FAST_COMPILE pure-Python mode.",
                exc,
            )
            cls._enable_fast_compile()
            return False

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

        If the initial MCMC sampling attempt fails with a PyTensor compilation
        error (e.g. a C++ linker failure despite a compiler being present),
        the method automatically falls back to ``FAST_COMPILE`` pure-Python
        mode and retries sampling.  The preemptive compiler check in
        :meth:`warm_compilation_cache` handles the common case where no
        compiler is installed at all.

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
                    "cov",
                    pm.math.dot(chol, chol.T),  # type: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
                )

                # Likelihood
                pm.MvNormal("obs", mu=mu, chol=chol, observed=data)

                # Sample from the posterior
                logger.info(
                    "Starting MCMC sampling with %d draws and %d tuning steps...",
                    draws,
                    tune,
                )
                self.trace_ = pm.sample(  # type: ignore[reportCallIssue]
                    draws=draws,
                    tune=tune,
                    target_accept=target_accept,
                    random_seed=self.random_seed,
                    return_inferencedata=True,
                    progressbar=False,
                    **extra_kwargs,  # type: ignore[reportArgumentType]
                    **kwargs,  # type: ignore[reportArgumentType]
                )
                return model

        # Attempt sampling; fall back to FAST_COMPILE on PyTensor compile errors.
        try:
            self.model_ = _sample({})
        except Exception as exc:
            if _is_compilation_error(exc):
                logger.warning(
                    "MCMC sampling failed due to a compiler/linker error. "
                    "Falling back to FAST_COMPILE pure-Python mode."
                )
                BayesianOptimizer._enable_fast_compile()
                self.model_ = _sample({})
            else:
                raise

        logger.info("MCMC sampling completed.")
        return cast(az.InferenceData, self.trace_)

    def get_posterior_estimates(self) -> tuple[pd.Series, pd.DataFrame]:
        """Extract the posterior mean returns and covariance matrix.

        The posterior covariance is the mean over MCMC draws of the
        LKJ-prior covariance — a Bayesian-shrunk estimate, never the raw
        sample covariance.  It is symmetrised and eigen-clipped to strict
        positive definiteness so downstream solvers (including
        :class:`EfficientFrontier`) cannot fail on a near-singular matrix.

        Returns:
            tuple[pd.Series, pd.DataFrame]: A tuple containing:
                - Expected returns (posterior mean of 'mu').
                - Expected covariance matrix (posterior mean of 'cov'),
                  strictly positive definite.

        Raises:
            RuntimeError: If the model has not been fitted.
            DataValidationError: If the posterior estimates are non-finite.
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

        if not np.all(np.isfinite(expected_returns)):
            raise DataValidationError(
                "Posterior mean returns contain non-finite values; the MCMC "
                "chain did not converge to a valid posterior."
            )

        # Bayesian shrinkage + strict positive-definiteness guarantee.
        expected_cov = ensure_strictly_psd(expected_cov)

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
        assets = cast("list[str]", self.assets_)
        n_assets = len(assets)

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
            raise RuntimeError(f"Bayesian portfolio optimisation failed: {res.message}")
        weights_array = res.x

        weights_dict = dict(zip(assets, weights_array))

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

    def optimize_efficient_frontier(
        self,
        *,
        risk_free_rate: float | None = None,
        weight_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> OptimizationResult:
        """Maximise the Sharpe ratio on a PyPortfolioOpt ``EfficientFrontier``
        using Bayesian posterior estimates.

        The frontier is built from the posterior mean returns and the
        **Bayesian-shrunk posterior covariance** returned by
        :meth:`get_posterior_estimates` — the raw sample covariance is never
        used.  The posterior covariance is already eigen-clipped to strict
        positive definiteness, so the convex solver cannot fail on a
        near-singular matrix.

        Args:
            risk_free_rate: Annual risk-free rate.  Defaults to the instance
                value; posterior means are assumed daily-period returns, so
                the rate is de-annualised by 252 for the solver.
            weight_bounds: ``(min, max)`` per-asset weight bounds.  Default
                long-only ``(0.0, 1.0)``.

        Returns:
            OptimizationResult: The result containing weights and performance
            metrics.

        Raises:
            RuntimeError: If the convex solver fails (failures are surfaced,
                never silently replaced with equal weights).
        """
        if self.trace_ is None:
            self.fit_returns_model()

        mu, cov = self.get_posterior_estimates()
        rf = self.risk_free_rate if risk_free_rate is None else risk_free_rate
        assets = list(cov.columns)

        try:
            ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
            ef.max_sharpe(risk_free_rate=rf / 252.0)
            cleaned = cast(dict[str, float], ef.clean_weights())
        except Exception as exc:
            raise RuntimeError(
                f"Bayesian efficient-frontier optimisation failed: {exc}"
            ) from exc

        weights = {ticker: float(cleaned.get(ticker, 0.0)) for ticker in assets}
        w = np.array([weights[t] for t in assets], dtype=np.float64)

        mu_arr = np.asarray(mu, dtype=np.float64)
        cov_arr = np.asarray(cov, dtype=np.float64)

        p_return = float(np.sum(mu_arr * w)) * 252.0
        p_vol = float(np.sqrt(w @ (cov_arr * 252.0) @ w))
        p_sharpe = (p_return - rf) / p_vol if p_vol > 0 else 0.0

        return OptimizationResult(
            weights=weights,
            expected_return=p_return,
            volatility=p_vol,
            sharpe_ratio=p_sharpe,
        )
