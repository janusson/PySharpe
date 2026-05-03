import numpy as np
import pandas as pd
import pytest

from pysharpe.optimization.bayesian import BayesianOptimizer


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


def test_get_estimates_before_fit():
    """Test calling get_posterior_estimates before fitting."""
    optimizer = BayesianOptimizer()
    with pytest.raises(RuntimeError):
        optimizer.get_posterior_estimates()
