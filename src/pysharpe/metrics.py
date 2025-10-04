"""Financial performance metrics utilities.

The helpers in this module focus on numeric correctness and ergonomics for
common portfolio analytics tasks. Each function accepts ``pandas`` Series or
DataFrames, validates the input, and returns results with matching dimensional
ity so they slot naturally into notebook or production workflows.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

PandasLike = pd.Series | pd.DataFrame


def _coerce_to_dataframe(data: PandasLike) -> Tuple[pd.DataFrame, bool]:
    """Normalise inputs to a DataFrame while tracking the original shape."""

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("Input must be a pandas Series or DataFrame.")

    if isinstance(data, pd.Series):
        frame = data.to_frame(name=data.name or "value")
        return frame, True
    return data.copy(), False


def _prep_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.apply(pd.to_numeric, errors="coerce")
    cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    cleaned = cleaned.dropna(how="all")
    if cleaned.empty:
        raise ValueError("Input must contain at least one finite observation per column.")
    return cleaned


def compute_returns(
    prices: PandasLike,
    *,
    method: str = "simple",
    dropna: bool = True,
) -> PandasLike:
    """Compute periodic returns from a price series.

    Args:
        prices: Ordered price levels for one or more assets.
        method: ``"simple"`` for percentage change, ``"log"`` for log returns.
        dropna: When ``True`` the initial NaN row is removed. Set to ``False``
            when callers prefer to retain alignment with the original index.

    Returns:
        Returns with the same dimensionality as ``prices``.

    Raises:
        TypeError: If *prices* is not a Series or DataFrame.
        ValueError: If *method* is not recognised.

    Example:
        >>> import pandas as pd
        >>> from pysharpe import metrics
        >>> prices = pd.Series([100, 102, 101], dtype=float)
        >>> metrics.compute_returns(prices).round(4).tolist()
        [0.02, -0.0098]
    """

    frame, was_series = _coerce_to_dataframe(prices)
    numeric = _prep_numeric(frame)

    if method not in {"simple", "log"}:
        raise ValueError("method must be 'simple' or 'log'")

    shifted = numeric.shift(1)
    if method == "log":
        returns = np.log(numeric / shifted)
    else:
        returns = numeric.pct_change()

    returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    if dropna:
        returns = returns.dropna(how="all")

    if was_series:
        return returns.iloc[:, 0]
    return returns


def annualize_return(
    returns: PandasLike,
    *,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """Compute the geometric annualised return.

    Args:
        returns: Periodic returns expressed as decimal fractions.
        periods_per_year: Observation frequency (252 for daily, 12 for monthly).

    Returns:
        The annualised return as a float for Series input or a Series for
        DataFrame input.

    Raises:
        TypeError: If ``returns`` is neither Series nor DataFrame.
        ValueError: If there are fewer than one finite observations per column.

    Example:
        >>> import pandas as pd
        >>> from pysharpe import metrics
        >>> rets = pd.Series([0.01, 0.02, -0.005], dtype=float)
        >>> metrics.annualize_return(rets, periods_per_year=252)  # doctest: +ELLIPSIS
        0.3...
    """

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    frame, was_series = _coerce_to_dataframe(returns)
    numeric = _prep_numeric(frame)

    counts = numeric.count()
    if (counts == 0).any():
        raise ValueError("Each column must contain at least one observation.")

    compounded = (1 + numeric).prod(skipna=True)
    exponent = periods_per_year / counts
    annualised = compounded.pow(exponent) - 1

    if was_series:
        return float(annualised.iloc[0])
    return annualised


def annualize_volatility(
    returns: PandasLike,
    *,
    periods_per_year: int = 252,
    ddof: int = 1,
) -> float | pd.Series:
    """Annualise the standard deviation of periodic returns.

    Args:
        returns: Periodic returns expressed as decimal fractions.
        periods_per_year: Observation frequency (252 for daily, 12 for monthly).
        ddof: Delta degrees of freedom passed to ``pandas.Series.std``.

    Returns:
        Annualised volatility matching the dimensionality of ``returns``.

    Raises:
        TypeError: If ``returns`` is not a pandas object.
        ValueError: If insufficient observations exist for the chosen *ddof*.

    Example:
        >>> import pandas as pd
        >>> from pysharpe import metrics
        >>> rets = pd.Series([0.01, -0.02, 0.015], dtype=float)
        >>> metrics.annualize_volatility(rets, periods_per_year=252)  # doctest: +ELLIPSIS
        0.2...
    """

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if ddof < 0:
        raise ValueError("ddof must be non-negative")

    frame, was_series = _coerce_to_dataframe(returns)
    numeric = _prep_numeric(frame)

    counts = numeric.count()
    if (counts <= ddof).any():
        raise ValueError("Not enough observations to compute volatility with the given ddof.")

    stdev = numeric.std(ddof=ddof)
    annualised = stdev * np.sqrt(periods_per_year)

    if was_series:
        return float(annualised.iloc[0])
    return annualised


def expected_return(
    returns: PandasLike,
    *,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """Compute the arithmetic annualised expected return.

    Args:
        returns: Periodic returns as decimal fractions.
        periods_per_year: Observation frequency used to annualise the mean.

    Returns:
        Annualised expected return of the series. For DataFrame input a Series
        is returned with one expected value per column.

    Example:
        >>> import pandas as pd
        >>> from pysharpe import metrics
        >>> rets = pd.Series([0.01, 0.02, -0.005], dtype=float)
        >>> metrics.expected_return(rets, periods_per_year=252)  # doctest: +ELLIPSIS
        0.6...
    """

    frame, was_series = _coerce_to_dataframe(returns)
    numeric = _prep_numeric(frame)
    mean_returns = numeric.mean() * periods_per_year

    if was_series:
        return float(mean_returns.iloc[0])
    return mean_returns


def sharpe_ratio(
    returns: PandasLike,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """Calculate the annualised Sharpe ratio.

    Args:
        returns: Periodic returns as decimal fractions.
        risk_free_rate: Annual risk-free rate expressed as a decimal.
        periods_per_year: Observation frequency of the input ``returns``.

    Returns:
        Sharpe ratio of the supplied returns.

    Raises:
        ValueError: If volatility is zero for any column.

    Example:
        >>> import pandas as pd
        >>> from pysharpe import metrics
        >>> rets = pd.Series([0.01, 0.015, -0.005], dtype=float)
        >>> metrics.sharpe_ratio(rets, risk_free_rate=0.02, periods_per_year=252)  # doctest: +ELLIPSIS
        -0.1...
    """

    annualised_return = annualize_return(returns, periods_per_year=periods_per_year)
    annualised_volatility = annualize_volatility(
        returns,
        periods_per_year=periods_per_year,
    )

    if isinstance(annualised_volatility, pd.Series):
        zero_mask = annualised_volatility.apply(lambda value: np.isclose(value, 0.0))
        if zero_mask.any():
            problematic = ", ".join(annualised_volatility[zero_mask].index)
            raise ValueError(f"Volatility is zero; Sharpe ratio undefined for: {problematic}")
        excess = annualised_return - risk_free_rate
        return excess / annualised_volatility

    if np.isclose(annualised_volatility, 0.0):
        raise ValueError("Volatility is zero; Sharpe ratio undefined.")

    excess_return = annualised_return - risk_free_rate
    return float(excess_return / annualised_volatility)


__all__ = [
    "compute_returns",
    "annualize_return",
    "annualize_volatility",
    "expected_return",
    "sharpe_ratio",
]
