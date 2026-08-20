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
from pypfopt.efficient_frontier import EfficientFrontier

from pysharpe.exceptions import DataValidationError
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
    """Test that _is_compilation_error detects PyTensor CompileError, not
    arbitrary exceptions with compiler-like messages."""
    # Try to import PyTensor's real CompileError type.
    try:
        from pytensor.link.c.exceptions import CompileError
    except ImportError:
        CompileError = None

    if CompileError is not None:
        # A genuine PyTensor CompileError should be detected.
        assert _is_compilation_error(CompileError("linker failure")), (
            "Should detect genuine CompileError"
        )

    # Arbitrary exceptions — even with compiler-related messages — should
    # NOT be detected (the old string-matching heuristic is removed).
    for msg in (
        "gcc: error: linker command failed with exit code 1",
        "clang: error: unable to execute command: Segmentation fault",
        "Compilation failed: cannot compile C code",
        "ld returned 1 exit status",
    ):
        assert not _is_compilation_error(RuntimeError(msg)), (
            f"Should NOT detect string-matched error: {msg}"
        )

    # Unrelated errors should not be detected.
    assert not _is_compilation_error(ValueError("Invalid shape"))
    assert not _is_compilation_error(RuntimeError("convergence failure"))


def test_fast_compile_fallback_on_compile_error(sample_returns):
    """Test that a PyTensor compilation error triggers the FAST_COMPILE fallback.

    When ``pm.sample`` raises a ``pytensor.link.c.exceptions.CompileError``,
    the optimizer should enable FAST_COMPILE mode and retry sampling.
    """
    optimizer = BayesianOptimizer(random_seed=42)

    # Try to use PyTensor's real CompileError; fall back to RuntimeError if
    # the import fails (which means the type-based check won't fire, so the
    # RuntimeError will propagate as a non-compilation error — skip the test).
    try:
        from pytensor.link.c.exceptions import CompileError as _CE
    except ImportError:
        pytest.skip("PyTensor CompileError not importable; cannot simulate.")

    fake_trace = MagicMock(name="trace")
    fake_error = _CE("gcc: linker command failed")

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


def test_warm_compilation_cache_any_probe_error_triggers_fallback():
    """Test that any exception during the functional probe triggers fallback.

    Unlike the old behaviour (which only fell back on string-matched
    compiler errors), the new ``warm_compilation_cache`` treats **any**
    failure of the functional probe as grounds to enable FAST_COMPILE.
    This is safer: if the probe fails for an unexpected reason, the
    compiler toolchain is likely not functional.
    """
    import importlib.util

    if importlib.util.find_spec("pytensor") is None:  # pragma: no cover
        pytest.skip("PyTensor is not installed.")

    other_error = ValueError("Something else broke")

    with patch.object(BayesianOptimizer, "_enable_fast_compile") as mock_fast_compile:
        with patch("pytensor.function") as mock_fn:
            mock_fn.side_effect = other_error

            result = BayesianOptimizer.warm_compilation_cache()

            # Any probe failure should trigger FAST_COMPILE fallback.
            mock_fast_compile.assert_called_once()
            assert result is False


# ---------------------------------------------------------------------------
# EfficientFrontier integration: posterior (shrunk) covariance, never sample
# ---------------------------------------------------------------------------


def _make_posterior_mock(mu_values: np.ndarray, cov_values: np.ndarray) -> MagicMock:
    """Build a fake InferenceData whose posterior contains the given estimates."""
    mu_mock = MagicMock()
    mu_mock.mean.return_value.values = mu_values
    cov_mock = MagicMock()
    cov_mock.mean.return_value.values = cov_values
    posterior = MagicMock()
    posterior.__getitem__.side_effect = lambda key: {
        "mu": mu_mock,
        "cov": cov_mock,
    }[key]
    trace = MagicMock()
    trace.posterior = posterior
    return trace


class TestPosteriorCovarianceHardening:
    """Posterior estimates must be strictly PSD without MCMC sampling."""

    def test_posterior_covariance_is_eigen_clipped(self) -> None:
        optimizer = BayesianOptimizer(random_seed=42)
        optimizer.assets_ = ["A", "B"]
        optimizer.trace_ = _make_posterior_mock(
            mu_values=np.array([0.001, 0.002]),
            cov_values=np.array([[1.0, 1.0], [1.0, 1.0]]),  # singular
        )
        optimizer.model_ = MagicMock()

        mu, cov = optimizer.get_posterior_estimates()
        assert np.all(np.isfinite(cov.values))
        assert np.all(np.linalg.eigvalsh(cov.values) > 0.0)
        np.testing.assert_allclose(cov.values, cov.values.T)
        assert list(mu.index) == ["A", "B"]

    def test_non_finite_posterior_returns_raise(self) -> None:
        optimizer = BayesianOptimizer(random_seed=42)
        optimizer.assets_ = ["A", "B"]
        optimizer.trace_ = _make_posterior_mock(
            mu_values=np.array([np.nan, 0.002]),
            cov_values=np.array([[1.0, 0.2], [0.2, 1.0]]),
        )
        optimizer.model_ = MagicMock()

        with pytest.raises(DataValidationError, match="non-finite"):
            optimizer.get_posterior_estimates()


class TestEfficientFrontierIntegration:
    """The EfficientFrontier must receive the posterior covariance, never
    the raw sample covariance."""

    def test_frontier_receives_posterior_covariance_not_sample(
        self, sample_returns: pd.DataFrame
    ) -> None:
        optimizer = BayesianOptimizer(returns=sample_returns, random_seed=42)
        optimizer.trace_ = MagicMock()  # skip the auto-fit path

        posterior_mu = pd.Series(
            [0.001, 0.002], index=["Asset_A", "Asset_B"], name="Posterior Mean Returns"
        )
        posterior_cov = pd.DataFrame(
            [[1e-4, 4e-6], [4e-6, 2e-4]],
            index=["Asset_A", "Asset_B"],
            columns=["Asset_A", "Asset_B"],
        )
        # Deliberately different from the sample covariance.
        sample_cov = sample_returns.cov().to_numpy(dtype=float)
        assert not np.allclose(posterior_cov.values, sample_cov, atol=1e-6)

        captured: dict[str, object] = {}
        original = EfficientFrontier

        def spy_constructor(
            expected_returns: object,
            cov_matrix: object,
            **kwargs: object,
        ) -> object:
            captured["cov"] = cov_matrix
            captured["mu"] = expected_returns
            return original(expected_returns, cov_matrix, **kwargs)  # type: ignore[arg-type]

        with patch.object(BayesianOptimizer, "get_posterior_estimates") as mock_get:
            mock_get.return_value = (posterior_mu, posterior_cov)
            with patch(
                "pysharpe.optimization.bayesian.EfficientFrontier",
                side_effect=spy_constructor,
            ):
                result = optimizer.optimize_efficient_frontier()

        # The covariance handed to EfficientFrontier is the posterior estimate…
        cov_seen = np.asarray(captured["cov"], dtype=np.float64)
        np.testing.assert_allclose(cov_seen, posterior_cov.values, rtol=1e-12)
        # …and NOT the raw sample covariance.
        assert not np.allclose(cov_seen, sample_cov, atol=1e-6)

        assert abs(sum(result.weights.values()) - 1.0) < 1e-9
        assert result.volatility > 0

    def test_frontier_survives_near_singular_posterior(
        self, sample_returns: pd.DataFrame
    ) -> None:
        """A near-singular posterior covariance is eigen-clipped so the
        convex solver completes instead of crashing."""
        optimizer = BayesianOptimizer(returns=sample_returns, random_seed=42)
        optimizer.trace_ = MagicMock()  # skip the auto-fit path

        posterior_mu = pd.Series(
            [0.001, 0.002], index=["Asset_A", "Asset_B"], name="Posterior Mean Returns"
        )
        # Duplicated rows → rank 1, singular in exact arithmetic.
        posterior_cov = pd.DataFrame(
            [[1e-4, 1e-4], [1e-4, 1e-4]],
            index=["Asset_A", "Asset_B"],
            columns=["Asset_A", "Asset_B"],
        )

        with patch.object(BayesianOptimizer, "get_posterior_estimates") as mock_get:
            mock_get.return_value = (posterior_mu, posterior_cov)
            result = optimizer.optimize_efficient_frontier()

        assert abs(sum(result.weights.values()) - 1.0) < 1e-9
        assert all(0.0 <= w <= 1.0 for w in result.weights.values())

    def test_frontier_failure_is_surfaced_not_silent(
        self, sample_returns: pd.DataFrame
    ) -> None:
        """Solver failures must raise, never silently return equal weights."""
        optimizer = BayesianOptimizer(returns=sample_returns, random_seed=42)
        optimizer.trace_ = MagicMock()  # skip the auto-fit path

        posterior_mu = pd.Series(
            [0.001, 0.002], index=["Asset_A", "Asset_B"], name="Posterior Mean Returns"
        )
        posterior_cov = pd.DataFrame(
            [[1e-4, 4e-6], [4e-6, 2e-4]],
            index=["Asset_A", "Asset_B"],
            columns=["Asset_A", "Asset_B"],
        )

        with patch.object(BayesianOptimizer, "get_posterior_estimates") as mock_get:
            mock_get.return_value = (posterior_mu, posterior_cov)
            with patch(
                "pysharpe.optimization.bayesian.EfficientFrontier",
                side_effect=ValueError("infeasible"),
            ):
                with pytest.raises(RuntimeError, match="failed"):
                    optimizer.optimize_efficient_frontier()


# ---------------------------------------------------------------------------
# pytest.MonkeyPatch isolation: the PyMC sampler is mocked out entirely so
# these tests pass on any CI runner, even when FAST_COMPILE mode is broken.
# ---------------------------------------------------------------------------


class TestPymcSamplerIsolation:
    """fit_returns_model with ``pm.sample`` monkeypatched away.

    None of these tests invoke PyTensor's compilation path; they verify the
    orchestrating logic around the sampler.
    """

    def test_fit_with_mocked_sampler(
        self, sample_returns: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_trace = MagicMock(name="fake_trace")

        calls: list[dict[str, object]] = []

        def fake_sample(**kwargs: object) -> MagicMock:
            calls.append(kwargs)
            return fake_trace

        monkeypatch.setattr("pysharpe.optimization.bayesian.pm.sample", fake_sample)

        optimizer = BayesianOptimizer(random_seed=123)
        trace = optimizer.fit_returns_model(
            sample_returns, draws=50, tune=50, chains=1, cores=1
        )

        assert trace is fake_trace
        assert optimizer.trace_ is fake_trace
        assert optimizer.model_ is not None
        assert len(calls) == 1
        assert calls[0]["draws"] == 50
        assert calls[0]["tune"] == 50

    def test_fast_compile_fallback_uses_monkeypatch(
        self, sample_returns: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A compile error on the first sampling attempt triggers the
        FAST_COMPILE fallback exactly once, then retries."""
        try:
            from pytensor.link.c.exceptions import CompileError as _CE
        except ImportError:  # pragma: no cover
            pytest.skip("PyTensor CompileError not importable.")

        fake_trace = MagicMock(name="fake_trace")
        fallback_calls: list[object] = []

        monkeypatch.setattr(
            "pysharpe.optimization.bayesian.pm.sample",
            MagicMock(side_effect=[_CE("linker failed"), fake_trace]),
        )
        monkeypatch.setattr(
            BayesianOptimizer,
            "_enable_fast_compile",
            staticmethod(lambda: fallback_calls.append("called")),
        )

        optimizer = BayesianOptimizer(random_seed=42)
        trace = optimizer.fit_returns_model(
            sample_returns, draws=50, tune=50, chains=1, cores=1
        )

        assert trace is fake_trace
        assert fallback_calls == ["called"]

    def test_non_compile_error_propagates_under_monkeypatch(
        self, sample_returns: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "pysharpe.optimization.bayesian.pm.sample",
            MagicMock(side_effect=ValueError("divergence catastrophe")),
        )
        monkeypatch.setattr(
            BayesianOptimizer,
            "_enable_fast_compile",
            staticmethod(lambda: pytest.fail("fallback must not fire")),
        )

        optimizer = BayesianOptimizer(random_seed=42)
        with pytest.raises(ValueError, match="divergence catastrophe"):
            optimizer.fit_returns_model(sample_returns, draws=50, tune=50)

    def test_warm_compilation_cache_without_compiler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No C toolchain → FAST_COMPILE is enabled preemptively; the
        functional probe is never attempted."""
        fallback_calls: list[object] = []

        monkeypatch.setattr(
            "pysharpe.optimization.bayesian._has_c_compiler", lambda: False
        )
        monkeypatch.setattr(
            BayesianOptimizer,
            "_enable_fast_compile",
            staticmethod(lambda: fallback_calls.append("called")),
        )

        assert BayesianOptimizer.warm_compilation_cache() is False
        assert fallback_calls == ["called"]

    def test_warm_compilation_cache_probe_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fallback_calls: list[object] = []

        monkeypatch.setattr(
            "pysharpe.optimization.bayesian._has_c_compiler", lambda: True
        )
        monkeypatch.setattr(
            BayesianOptimizer,
            "_enable_fast_compile",
            staticmethod(lambda: fallback_calls.append("called")),
        )

        try:
            import pytensor
        except ImportError:  # pragma: no cover
            pytest.skip("PyTensor is not installed.")

        monkeypatch.setattr(
            pytensor, "function", MagicMock(side_effect=ValueError("probe boom"))
        )
        assert BayesianOptimizer.warm_compilation_cache() is False
        assert fallback_calls == ["called"]

    def test_warm_compilation_cache_probe_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fallback_calls: list[object] = []

        monkeypatch.setattr(
            "pysharpe.optimization.bayesian._has_c_compiler", lambda: True
        )
        monkeypatch.setattr(
            BayesianOptimizer,
            "_enable_fast_compile",
            staticmethod(lambda: fallback_calls.append("called")),
        )

        try:
            import pytensor
        except ImportError:  # pragma: no cover
            pytest.skip("PyTensor is not installed.")

        monkeypatch.setattr(pytensor, "function", MagicMock(return_value=lambda x: x))
        assert BayesianOptimizer.warm_compilation_cache() is True
        assert fallback_calls == []
