import numpy as np
import pandas as pd
import pytest

from pysharpe.analysis.time_series import (
    GARCHVolatilityForecaster,
    VARModeler,
    check_stationarity,
)


@pytest.fixture
def synthetic_stationary_series():
    """Generate a simple stationary series (white noise)."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0, 1, 100), name="WhiteNoise")


@pytest.fixture
def synthetic_non_stationary_series():
    """Generate a non-stationary series (random walk)."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0, 1, 100).cumsum(), name="RandomWalk")


@pytest.fixture
def synthetic_multivariate_data():
    """Generate simple multivariate data for VAR."""
    np.random.seed(42)
    n = 100
    # Create two correlated series
    x = np.random.normal(0, 1, n)
    y = 0.5 * x + np.random.normal(0, 0.5, n)
    return pd.DataFrame({"Asset1": x, "Asset2": y})


def test_stationarity_stationary(synthetic_stationary_series):
    """Test ADF on a known stationary series."""
    result = check_stationarity(synthetic_stationary_series)
    assert result["is_stationary"] is True
    assert result["p_value"] < 0.05


def test_stationarity_non_stationary(synthetic_non_stationary_series):
    """Test ADF on a known non-stationary series."""
    result = check_stationarity(synthetic_non_stationary_series)
    assert result["is_stationary"] is False
    assert result["p_value"] > 0.05


def test_stationarity_invalid_input():
    with pytest.raises(TypeError):
        check_stationarity([1, 2, 3])
    with pytest.raises(ValueError):
        check_stationarity(pd.Series(dtype=float))


def test_garch_forecaster(synthetic_stationary_series):
    """Test GARCH fitting and forecasting."""
    # Scale up variance to avoid optimizer warnings in arch
    returns = synthetic_stationary_series * 100

    forecaster = GARCHVolatilityForecaster(p=1, q=1)
    forecaster.fit(returns)

    assert forecaster.model_result_ is not None

    forecast = forecaster.forecast(horizon=3)
    assert isinstance(forecast, pd.Series)
    assert len(forecast) == 3
    assert forecast.name == "forecasted_variance"


def test_garch_unfitted_error():
    forecaster = GARCHVolatilityForecaster()
    with pytest.raises(RuntimeError):
        forecaster.forecast()


def test_var_modeler(synthetic_multivariate_data):
    """Test VAR fitting and forecasting."""
    modeler = VARModeler(maxlags=2)
    modeler.fit(synthetic_multivariate_data)

    assert modeler.model_result_ is not None

    forecast = modeler.forecast(steps=4)
    assert isinstance(forecast, pd.DataFrame)
    assert forecast.shape == (4, 2)
    assert forecast.columns.tolist() == ["Asset1", "Asset2"]


def test_var_invalid_input(synthetic_stationary_series):
    modeler = VARModeler()
    with pytest.raises(TypeError):
        modeler.fit(synthetic_stationary_series.values)
    with pytest.raises(ValueError):
        modeler.fit(pd.DataFrame(synthetic_stationary_series))  # Only 1 column


def test_var_unfitted_error():
    modeler = VARModeler()
    with pytest.raises(RuntimeError):
        modeler.forecast()
# Append these tests to the file
def test_garch_invalid_input():
    forecaster = GARCHVolatilityForecaster()
    with pytest.raises(TypeError):
        forecaster.fit([1, 2, 3])


def test_var_zero_lag_fallback():
    """Test that VAR forces 1 lag if 0 is selected to allow forecasting."""
    np.random.seed(42)
    data = pd.DataFrame(
        {"A": np.random.normal(0, 1, 100), "B": np.random.normal(0, 1, 100)}
    )
    modeler = VARModeler(maxlags=1)
    modeler.fit(data)
    assert modeler.model_result_.k_ar >= 1
    forecast = modeler.forecast(steps=1)
    assert not forecast.empty
