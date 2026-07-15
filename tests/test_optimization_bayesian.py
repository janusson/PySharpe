"""Tests for the Bayesian optimizer.

.. note::

    **Canadian ETF Context** — Bayesian posterior estimation of asset
    returns uses synthetic CAD-denominated return series.  PyMC sampling
    provides uncertainty-aware expected returns and covariance for
    efficient frontier analysis, not for predictive trading.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from conftest import _SAMPLING_SKIP_REASON, pymc_sampling_works

from pysharpe.optimization.bayesian import (
    BayesianOptimizer,
    _is_compilation_error,
)


@pytest.fixture
def sample_returns():
    """Create synthetic returns for two assets."""
    np.random.seed(42)
    n_obs = 100
    returns = pd.DataFrame(
        {
            "Asset_A": np.random.normal(0.001, 0.02, n_obs),
            "Asset_B": np.random.normal(0.002, 0.03, n_obs),
        }
    )
    return returns


def test_bayesian_optimizer_init():
    """Test initialization of BayesianOptimizer."""
    optimizer = BayesianOptimizer(random_seed=123)
    assert optimizer.random_seed == 123
    assert optimizer.trace_ is None
    assert optimizer.model_ is None


@pytest.mark.skipif(
    not pymc_sampling_works(),
    reason=_SAMPLING_SKIP_REASON,
)
def test_fit_returns_model(sample_returns):
    """Test fitting the returns model (with small samples for speed)."""
    optimizer = BayesianOptimizer(random_seed=42)
    # Using small draws/tune and single core for fast test execution
    trace = optimizer.fit_returns_model(
        sample_returns, draws=100, tune=100, chains=1, cores=1
    )

    assert trace is not None
    assert "mu" in trace.posterior
    assert "cov" in trace.posterior
    assert optimizer.assets_ == ["Asset_A", "Asset_B"]


@pytest.mark.skipif(
    not pymc_sampling_works(),
    reason=_SAMPLING_SKIP_REASON,
)
def test_get_posterior_estimates(sample_returns):
    """Test extracting posterior estimates."""
    optimizer = BayesianOptimizer(random_seed=42)
    optimizer.fit_returns_model(sample_returns, draws=100, tune=100, chains=1, cores=1)

    mu_post, cov_post = optimizer.get_posterior_estimates()

    assert isinstance(mu_post, pd.Series)
    assert isinstance(cov_post, pd.DataFrame)
    assert mu_post.index.tolist() == ["Asset_A", "Asset_B"]
    assert cov_post.columns.tolist() == ["Asset_A", "Asset_B"]
    assert cov_post.index.tolist() == ["Asset_A", "Asset_B"]

    # Basic sanity check on values
    assert mu_post.mean() < 0.1  # Should be reasonably small
    assert np.all(np.diag(cov_post) > 0)  # Variances must be positive


def test_fit_returns_model_invalid_input():
    """Test error handling for invalid input."""
    optimizer = BayesianOptimizer()

    with pytest.raises(TypeError):
        optimizer.fit_returns_model([1, 2, 3])  # Not a DataFrame

    with pytest.raises(ValueError):
        optimizer.fit_returns_model(pd.DataFrame())  # Empty DataFrame


def test_compilation_error_detection():
    """Test that _is_compilation_error correctly identifies compilation failures."""
    # Errors whose messages contain compiler keywords should be detected
    for msg in (
        "gcc: error: linker command failed with exit code 1",
        "clang: error: unable to execute command: Segmentation fault",
        "Compilation failed: cannot compile C code",
        "ld returned 1 exit status",
    ):
        assert _is_compilation_error(RuntimeError(msg)), f"Should detect: {msg}"

    # Unrelated errors should pass through
    assert not _is_compilation_error(ValueError("Invalid shape"))
    assert not _is_compilation_error(RuntimeError("convergence failure"))


def test_fast_compile_fallback_on_compile_error(sample_returns):
    """Test that a C-compiler/linker error triggers the FAST_COMPILE fallback.

    When ``pm.sample`` raises an error whose message matches compilation-
    failure keywords, the optimizer should log a warning, enable PyTensor's
    FAST_COMPILE mode, and retry sampling successfully.
    """
    optimizer = BayesianOptimizer(random_seed=42)

    # Simulate a trace-like return for the second (successful) attempt.
    fake_trace = MagicMock(name="trace")

    # The first call to pm.sample raises a linker error; the second succeeds.
    fake_error = RuntimeError("gcc: error: linker command failed")

    with patch.object(BayesianOptimizer, "_enable_fast_compile") as mock_fast_compile:
        with patch("pysharpe.optimization.bayesian.pm.sample") as mock_sample:
            mock_sample.side_effect = [fake_error, fake_trace]

            trace = optimizer.fit_returns_model(
                sample_returns, draws=100, tune=100, chains=1, cores=1
            )

            mock_fast_compile.assert_called_once()
            assert mock_sample.call_count == 2

    assert trace is fake_trace


def test_no_fallback_for_non_compilation_errors(sample_returns):
    """Test that non-compilation errors are re-raised without fallback."""
    optimizer = BayesianOptimizer(random_seed=42)

    # An ordinary ValueError should NOT trigger the FAST_COMPILE fallback.
    real_error = ValueError("Something else broke")

    with patch("pysharpe.optimization.bayesian.pm.sample") as mock_sample:
        mock_sample.side_effect = real_error

        with pytest.raises(ValueError, match="Something else broke"):
            optimizer.fit_returns_model(
                sample_returns, draws=100, tune=100, chains=1, cores=1
            )

    assert mock_sample.call_count == 1


def test_fast_compile_static_method():
    """Test that _enable_fast_compile correctly sets PyTensor's config mode."""
    try:
        import pytensor
    except ImportError:  # pragma: no cover
        pytest.skip("PyTensor is not installed.")

    original = pytensor.config.mode
    try:
        BayesianOptimizer._enable_fast_compile()
        assert pytensor.config.mode == "FAST_COMPILE"
    finally:
        pytensor.config.mode = original


def test_get_estimates_before_fit():
    """Test calling get_posterior_estimates before fitting."""
    optimizer = BayesianOptimizer()
    with pytest.raises(RuntimeError):
        optimizer.get_posterior_estimates()


# ---------------------------------------------------------------------------
# warm_compilation_cache tests
# ---------------------------------------------------------------------------


def test_warm_compilation_cache_success():
    """Test that warm_compilation_cache returns True when the C toolchain works."""
    try:
        import pytensor
    except ImportError:  # pragma: no cover
        pytest.skip("PyTensor is not installed.")

    original_mode = pytensor.config.mode
    try:
        # Ensure we start from the default (C-compiler) mode.
        pytensor.config.mode = "FAST_RUN"
        result = BayesianOptimizer.warm_compilation_cache()
        # In CI/dev environments with a working C compiler this should be True.
        # On macOS without Xcode CLI tools it may return False (fallback).
        assert isinstance(result, bool)
        if result:
            # Mode should be unchanged when the probe succeeded.
            assert pytensor.config.mode == "FAST_RUN"
        else:
            # Fallback was activated — mode should be FAST_COMPILE.
            assert pytensor.config.mode == "FAST_COMPILE"
    finally:
        pytensor.config.mode = original_mode


def test_warm_compilation_cache_fallback_on_compile_error():
    """Test that a compilation error triggers the FAST_COMPILE fallback."""
    import importlib.util

    if importlib.util.find_spec("pytensor") is None:  # pragma: no cover
        pytest.skip("PyTensor is not installed.")

    compile_error = RuntimeError("gcc: error: linker command failed")

    with patch.object(BayesianOptimizer, "_enable_fast_compile") as mock_fast_compile:
        with patch("pytensor.function") as mock_fn:
            mock_fn.side_effect = compile_error

            result = BayesianOptimizer.warm_compilation_cache()

            mock_fast_compile.assert_called_once()
            assert result is False


def test_warm_compilation_cache_non_compilation_error_is_silent():
    """Test that non-compilation errors do not trigger fallback and return True."""
    import importlib.util

    if importlib.util.find_spec("pytensor") is None:  # pragma: no cover
        pytest.skip("PyTensor is not installed.")

    other_error = ValueError("Something else broke")

    with patch.object(BayesianOptimizer, "_enable_fast_compile") as mock_fast_compile:
        with patch("pytensor.function") as mock_fn:
            mock_fn.side_effect = other_error

            result = BayesianOptimizer.warm_compilation_cache()

            mock_fast_compile.assert_not_called()
            assert result is True
