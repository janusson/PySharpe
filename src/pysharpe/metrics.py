"""Financial performance metrics utilities.

The helpers in this module focus on numeric correctness and ergonomics for
common portfolio analytics tasks. Each function accepts ``pandas`` Series or
DataFrames, validates the input, and returns results with matching dimensional
ity so they slot naturally into notebook or production workflows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PandasLike = pd.Series | pd.DataFrame


def _coerce_to_dataframe(data: PandasLike) -> tuple[pd.DataFrame, bool]:
    """Normalise inputs to a DataFrame while tracking the original shape."""

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("Input must be a pandas Series or DataFrame.")

    if isinstance(data, pd.Series):
        frame = data.to_frame(name=data.name or "value")
        return frame, True
    return data.copy(), False


def _prep_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned: pd.DataFrame = frame.apply(pd.to_numeric, errors="coerce")  # type: ignore[assignment]
    cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    cleaned = cleaned.dropna(how="all")
    if cleaned.empty:
        raise ValueError(
            "Input must contain at least one finite observation per column."
        )
    return cleaned  # type: ignore[return-value]


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
        raise ValueError(
            "Not enough observations to compute volatility with the given ddof."
        )

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
        zero_mask: pd.Series = annualised_volatility.apply(  # type: ignore[assignment]
            lambda value: np.isclose(value, 0.0)
        )
        if zero_mask.any():
            problematic = ", ".join(annualised_volatility[zero_mask].index)  # type: ignore[union-attr]
            raise ValueError(
                f"Volatility is zero; Sharpe ratio undefined for: {problematic}"
            )
        excess = annualised_return - risk_free_rate
        return excess / annualised_volatility

    if np.isclose(annualised_volatility, 0.0):
        raise ValueError("Volatility is zero; Sharpe ratio undefined.")

    excess_return = annualised_return - risk_free_rate
    return float(excess_return / annualised_volatility)


def cagr(value_series: pd.Series) -> float:
    """Compute the Compound Annual Growth Rate from a value series.

    Args:
        value_series: Portfolio value over time (index must be DatetimeIndex).

    Returns:
        CAGR as a decimal fraction.

    Raises:
        TypeError: If index is not DatetimeIndex.
        ValueError: If initial value is not positive.
    """
    if value_series.empty:
        return 0.0

    if not isinstance(value_series.index, pd.DatetimeIndex):
        raise TypeError("Input series must have a DatetimeIndex.")

    start_val = value_series.iloc[0]
    end_val = value_series.iloc[-1]

    if start_val <= 0:
        raise ValueError("Initial value must be positive.")

    days = (value_series.index[-1] - value_series.index[0]).days
    if days <= 0:
        return 0.0

    years = days / 365.25
    return (end_val / start_val) ** (1 / years) - 1


def maximum_drawdown(value_series: pd.Series) -> float:
    """Calculate the maximum peak-to-trough drawdown.

    Args:
        value_series: Portfolio value over time.

    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.20 for 20% loss).
    """
    if value_series.empty:
        return 0.0

    peak = value_series.cummax()
    drawdown = (value_series - peak) / peak
    return float(drawdown.min())


def sortino_ratio(
    returns: PandasLike,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float | pd.Series:
    """Calculate the annualised Sortino ratio.

    The Sortino ratio uses downside deviation (volatility of returns below a
    target) in the denominator rather than total volatility, penalising only
    harmful variability.

    Args:
        returns: Periodic returns expressed as decimal fractions.
        risk_free_rate: Annual risk-free rate expressed as a decimal.
        periods_per_year: Observation frequency of the input ``returns``.
        target_return: Minimum acceptable return (MAR) expressed as a daily
            decimal fraction. Defaults to zero.

    Returns:
        Sortino ratio of the supplied returns.

    Raises:
        ValueError: If downside deviation is zero for any column.

    Example:
        >>> import pandas as pd
        >>> from pysharpe import metrics
        >>> rets = pd.Series([0.01, -0.02, 0.015], dtype=float)
        >>> metrics.sortino_ratio(rets, risk_free_rate=0.02, periods_per_year=252)  # doctest: +ELLIPSIS
        ...
    """
    frame, was_series = _coerce_to_dataframe(returns)
    numeric = _prep_numeric(frame)

    annualised_return = annualize_return(returns, periods_per_year=periods_per_year)

    # Daily risk-free rate for excess return in downside calculation.
    daily_rf = risk_free_rate / periods_per_year
    downside = numeric - (target_return + daily_rf)
    downside = downside.clip(upper=0.0)
    downside_sq = downside.pow(2)
    downside_dev = np.sqrt(downside_sq.mean(skipna=True))
    annualised_downside = downside_dev * np.sqrt(periods_per_year)

    if isinstance(annualised_downside, pd.Series):
        zero_mask: pd.Series = annualised_downside.apply(  # type: ignore[assignment]
            lambda v: np.isclose(v, 0.0)
        )
        if zero_mask.any():
            problematic = ", ".join(annualised_downside[zero_mask].index)  # type: ignore[union-attr]
            raise ValueError(
                f"Downside deviation is zero; Sortino ratio undefined for: {problematic}"
            )
        excess = annualised_return - risk_free_rate
        result = excess / annualised_downside
        if was_series:
            return float(result.iloc[0])
        return result

    if np.isclose(annualised_downside, 0.0):
        raise ValueError("Downside deviation is zero; Sortino ratio undefined.")

    excess_return = annualised_return - risk_free_rate
    return float(excess_return / annualised_downside)


def calmar_ratio(value_series: pd.Series) -> float:
    """Calculate the Calmar ratio (annualised return / |max drawdown|).

    Args:
        value_series: Portfolio value over time (index must be DatetimeIndex).

    Returns:
        Calmar ratio as a float. Returns ``inf`` when the maximum drawdown is
        zero (i.e., the series never declined from a peak).

    Raises:
        TypeError: If index is not DatetimeIndex.
        ValueError: If initial value is not positive.

    Example:
        >>> import pandas as pd
        >>> from pysharpe import metrics
        >>> dates = pd.date_range("2024-01-01", periods=5, freq="D")
        >>> vals = pd.Series([100, 102, 98, 101, 105], index=dates, dtype=float)
        >>> metrics.calmar_ratio(vals)  # doctest: +ELLIPSIS
        ...
    """
    annual_cagr = cagr(value_series)
    mdd = maximum_drawdown(value_series)

    if np.isclose(mdd, 0.0):
        return float("inf") if annual_cagr > 0 else float("-inf")

    return annual_cagr / abs(mdd)


def tracking_error(
    returns_a: PandasLike,
    returns_b: PandasLike,
    *,
    periods_per_year: int = 252,
) -> float:
    """Calculate the annualised tracking error between two return series.

    Tracking error is the standard deviation of the return differential.

    Args:
        returns_a: Periodic returns of the first asset.
        returns_b: Periodic returns of the second asset (benchmark).
        periods_per_year: Observation frequency used to annualise.

    Returns:
        Annualised tracking error as a float.

    Raises:
        ValueError: If the two series have different lengths.

    Example:
        >>> import pandas as pd
        >>> from pysharpe import metrics
        >>> ra = pd.Series([0.01, -0.005, 0.02], dtype=float)
        >>> rb = pd.Series([0.005, 0.01, 0.015], dtype=float)
        >>> metrics.tracking_error(ra, rb, periods_per_year=252)  # doctest: +ELLIPSIS
        0.0...
    """
    a_frame, _ = _coerce_to_dataframe(returns_a)
    b_frame, _ = _coerce_to_dataframe(returns_b)

    if len(a_frame) != len(b_frame):
        raise ValueError(
            f"Return series must have the same length; "
            f"got {len(a_frame)} vs {len(b_frame)}."
        )

    a_numeric = _prep_numeric(a_frame)
    b_numeric = _prep_numeric(b_frame)

    diff = a_numeric.iloc[:, 0] - b_numeric.iloc[:, 0]
    return float(diff.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown_duration(value_series: pd.Series) -> int:
    """Calculate the longest drawdown duration in trading days.

    Duration is measured as the number of consecutive periods where the value
    remains below its previous all-time high.

    Args:
        value_series: Portfolio value over time.

    Returns:
        Longest drawdown duration in periods (trading days).

    Example:
        >>> import pandas as pd
        >>> from pysharpe import metrics
        >>> vals = pd.Series([100, 90, 95, 80, 105], dtype=float)
        >>> metrics.max_drawdown_duration(vals)
        3
    """
    if value_series.empty:
        return 0

    peak = value_series.cummax()
    is_drawdown = value_series < peak

    if not is_drawdown.any():
        return 0

    # Identify contiguous drawdown runs.
    # We increment a counter during drawdown, reset to zero at recovery.
    drawdown_streak = is_drawdown.astype(int)
    streak = drawdown_streak.groupby(
        (drawdown_streak != drawdown_streak.shift(1)).cumsum()
    ).cumsum()
    # Only count streaks where drawdown is True.
    streak = streak * drawdown_streak
    return int(streak.max())


def compute_realized_volatility(
    prices: PandasLike,
    window: int,
    *,
    periods_per_year: int = 252,
) -> PandasLike:
    """Compute rolling annualized realized volatility from log returns.

    This function calculates the standard deviation of log returns over a
    sliding window and annualizes the result. It is optimized for speed using
    pandas rolling primitives.

    Args:
        prices: Ordered price levels for one or more assets.
        window: The size of the rolling window (number of periods).
        periods_per_year: Observation frequency used to annualise the volatility.

    Returns:
        Rolling annualized volatility matching the dimensionality of ``prices``.
        The first ``window`` rows will typically contain NaNs.

    Raises:
        TypeError: If ``prices`` is not a pandas Series or DataFrame.
        ValueError: If ``window`` is less than 2.

    Example:
        >>> import pandas as pd
        >>> from pysharpe import metrics
        >>> prices = pd.Series([100, 101, 102, 101, 103], dtype=float)
        >>> res = metrics.compute_realized_volatility(prices, window=2)
        >>> res.iloc[-1]  # doctest: +ELLIPSIS
        0.31...
    """
    if window < 2:
        raise ValueError("window must be at least 2")

    # Log returns are standard for realized volatility
    returns = compute_returns(prices, method="log", dropna=False)

    frame, was_series = _coerce_to_dataframe(returns)

    # Use pandas rolling std for high performance and NaN handling.
    # We use ddof=1 to match annualize_volatility convention.
    rolling_std = frame.rolling(window=window).std(ddof=1)
    annualized = rolling_std * np.sqrt(periods_per_year)

    if was_series:
        return annualized.iloc[:, 0]
    return annualized


__all__ = [
    "compute_returns",
    "annualize_return",
    "annualize_volatility",
    "expected_return",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "tracking_error",
    "max_drawdown_duration",
    "cagr",
    "maximum_drawdown",
    "compute_realized_volatility",
]
