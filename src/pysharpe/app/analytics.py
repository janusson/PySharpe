"""Analytics helpers for the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from pypfopt import EfficientFrontier

from pysharpe import metrics
from pysharpe.optimization import PortfolioWeights

WarningHandler = Callable[[str], None]


@dataclass
class MetricResults:
    """Container holding the output of portfolio metric calculations."""

    returns: pd.DataFrame
    expected: pd.Series
    volatility: pd.Series
    sharpe: pd.Series


def compute_metrics(price_frame: pd.DataFrame) -> MetricResults:
    """Compute metrics using PySharpe helpers and ensure aligned indices."""

    if price_frame.empty:
        raise ValueError(
            "Price data is empty; please adjust tickers, dates, or upload a richer "
            "dataset."
        )

    returns = price_frame.pct_change().dropna(how="all")
    if returns.empty:
        raise ValueError("Insufficient price history to compute portfolio metrics.")

    column_index = returns.columns
    expected = metrics.expected_return(returns)
    volatility = metrics.annualize_volatility(returns)
    sharpe = metrics.sharpe_ratio(returns)

    expected = expected.reindex(column_index)
    volatility = volatility.reindex(column_index)
    sharpe = sharpe.reindex(column_index)

    return MetricResults(
        returns=returns,
        expected=expected,
        volatility=volatility,
        sharpe=sharpe,
    )


def optimise_weights(
    metrics_result: MetricResults, on_warning: WarningHandler | None = None
) -> PortfolioWeights | None:
    """Optimise portfolio weights using EfficientFrontier."""

    if metrics_result.returns.empty:
        return None

    mu = metrics_result.expected
    cov = metrics_result.returns.cov() * 252

    try:
        frontier = EfficientFrontier(mu, cov)
        frontier.max_sharpe()
        cleaned = frontier.clean_weights()
    except Exception as exc:  # pragma: no cover - shown in UI
        message = f"Optimisation failed: {exc}"
        if on_warning:
            on_warning(message)
        return None

    return PortfolioWeights(cleaned)


__all__ = [
    "MetricResults",
    "compute_metrics",
    "optimise_weights",
]
