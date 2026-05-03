"""Time-Series Analysis and Volatility Modeling.

This module provides tools for advanced econometric analysis of financial time series,
including stationarity testing, GARCH volatility forecasting, and Vector Autoregression (VAR).
"""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


def check_stationarity(
    series: pd.Series, significance_level: float = 0.05
) -> Dict[str, Union[float, bool, dict]]:
    """Perform the Augmented Dickey-Fuller test for stationarity.

    Args:
        series (pd.Series): The time series to test.
        significance_level (float): The p-value threshold for rejecting the null hypothesis.

    Returns:
        Dict: A dictionary containing the test statistic, p-value, critical values,
              and a boolean indicating if the series is stationary.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    if series.empty:
        raise ValueError("Input series cannot be empty.")

    # Drop NaNs before testing
    clean_series = series.dropna()

    # Perform ADF test
    adf_result = adfuller(clean_series, autolag="AIC")

    result = {
        "test_statistic": adf_result[0],
        "p_value": adf_result[1],
        "used_lag": adf_result[2],
        "nobs": adf_result[3],
        "critical_values": adf_result[4],
        "is_stationary": bool(adf_result[1] < significance_level),
    }

    logger.debug(
        f"Stationarity test for {series.name}: p-value = {result['p_value']:.4f}"
    )
    return result


class GARCHVolatilityForecaster:
    """Forecasts asset volatility using Generalized Autoregressive Conditional Heteroskedasticity (GARCH)."""

    def __init__(
        self, p: int = 1, q: int = 1, mean: str = "Constant", vol: str = "GARCH"
    ):
        """Initialize the GARCH model parameters.

        Args:
            p (int): Lag order of the symmetric innovation.
            q (int): Lag order of lagged volatility.
            mean (str): Name of the mean model (e.g., 'Constant', 'Zero', 'AR').
            vol (str): Name of the volatility model (e.g., 'GARCH', 'EGARCH').
        """
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.model_result_ = None

    def fit(self, returns: pd.Series, **kwargs) -> "GARCHVolatilityForecaster":
        """Fit the GARCH model to historical returns.

        Args:
            returns (pd.Series): The return series (should be scaled, e.g., multiplied by 100).
            **kwargs: Additional arguments passed to the fit method of arch_model.

        Returns:
            GARCHVolatilityForecaster: The fitted instance.
        """
        if not isinstance(returns, pd.Series):
            raise TypeError("Returns must be a pandas Series.")

        # The arch package often works better with scaled returns (percentages)
        # We assume the user has scaled them, or we let the arch optimizer handle it.
        # Warnings might appear if the scale is too small.
        self.am_ = arch_model(returns, vol=self.vol, p=self.p, q=self.q, mean=self.mean)

        logger.info(f"Fitting {self.vol}({self.p},{self.q}) model...")
        # Use disp='off' to keep console output clean during fitting
        self.model_result_ = self.am_.fit(disp="off", **kwargs)
        return self

    def forecast(self, horizon: int = 5) -> pd.Series:
        """Forecast future volatility.

        Args:
            horizon (int): Number of periods to forecast.

        Returns:
            pd.Series: The forecasted variance (not standard deviation).
        """
        if self.model_result_ is None:
            raise RuntimeError("Model has not been fitted. Call `fit` first.")

        forecasts = self.model_result_.forecast(horizon=horizon, reindex=False)
        # Return the expected variance for the requested horizon
        # The variance forecast is usually in the 'variance' attribute of the result
        var_forecast = forecasts.variance.iloc[-1]
        var_forecast.name = "forecasted_variance"
        return var_forecast


class VARModeler:
    """Models interdependencies between multiple time series using Vector Autoregression."""

    def __init__(self, maxlags: Optional[int] = None, ic: str = "aic"):
        """Initialize the VAR modeler.

        Args:
            maxlags (Optional[int]): Maximum number of lags to check for order selection.
            ic (str): Information criterion to use for lag length selection ('aic', 'bic', 'fpe', 'hqic').
        """
        self.maxlags = maxlags
        self.ic = ic
        self.model_result_ = None
        self.data_columns_ = None

    def fit(self, data: pd.DataFrame) -> "VARModeler":
        """Fit the VAR model.

        Args:
            data (pd.DataFrame): Multivariate time series data.

        Returns:
            VARModeler: The fitted instance.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame.")
        if data.shape[1] < 2:
            raise ValueError("VAR requires at least two time series.")

        self.data_columns_ = data.columns.tolist()

        model = VAR(data)
        logger.info(f"Fitting VAR model (criterion={self.ic})...")
        self.model_result_ = model.fit(maxlags=self.maxlags, ic=self.ic)

        # statsmodels VAR forecast fails if selected lags == 0
        if self.model_result_.k_ar == 0:
            logger.warning(
                "VAR order selection chose 0 lags. Forcing 1 lag to allow forecasting."
            )
            self.model_result_ = model.fit(maxlags=1)

        logger.info(f"Selected VAR order (lags): {self.model_result_.k_ar}")

        return self

    def forecast(self, steps: int = 5) -> pd.DataFrame:
        """Forecast future values.

        Args:
            steps (int): Number of steps to forecast.

        Returns:
            pd.DataFrame: The forecasted values.
        """
        if self.model_result_ is None:
            raise RuntimeError("Model has not been fitted. Call `fit` first.")

        # Get the last k_ar observations to seed the forecast
        lag_order = self.model_result_.k_ar
        y_seed = self.model_result_.endog[-lag_order:]

        pred = self.model_result_.forecast(y_seed, steps=steps)

        # Create a DataFrame with integer index for future steps
        idx = pd.RangeIndex(start=1, stop=steps + 1, name="step")
        return pd.DataFrame(pred, index=idx, columns=self.data_columns_)
