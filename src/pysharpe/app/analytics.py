"""Analytics helpers for the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pypfopt.risk_models import CovarianceShrinkage

from pysharpe import metrics
from pysharpe.optimization.expected_returns import shrinkage_expected_return
from pysharpe.optimization.tax_location import (
    AccountType,
    AssetLocationEngine,
    AssetTaxCharacteristics,
    TaxProfile,
)

WarningHandler = None  # retained for any external callers that import the name

#: Risk-free rate used by the harmonized comparison pipeline.  Matches
#: PyPortfolioOpt's ``portfolio_performance`` default so the optimized,
#: custom-mix, and benchmark rows are directly comparable.
DEFAULT_RISK_FREE_RATE: float = 0.02

#: Bayes-Stein shrinkage floor used by ``optimise_from_prices``.  The
#: comparison pipeline must use the same estimator so the rows are
#: mathematically equivalent.
DEFAULT_SHRINKAGE_FLOOR: float = 0.3


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


@dataclass
class AdjustedMetricResults:
    """Tax- and MER-adjusted metrics computed from Bayes-Stein shrunk returns.

    Attributes:
        returns: Period returns of the aligned evaluation frame.
        expected: Annualized expected returns after shrinkage, MER, and
            account-specific tax drag (foreign withholding tax + income tax).
        expected_pre_drag: Annualized shrunk expected returns before any
            MER/tax drag is deducted (transparency aid).
        volatility: Annualized volatility of the aligned evaluation frame.
        sharpe: ``(expected - risk_free_rate) / volatility``.
    """

    returns: pd.DataFrame
    expected: pd.Series
    expected_pre_drag: pd.Series
    volatility: pd.Series
    sharpe: pd.Series


def _apply_account_drag(
    mu: pd.Series,
    *,
    tax_profile: TaxProfile | None,
    asset_characteristics: dict[str, AssetTaxCharacteristics] | None,
    account: str,
) -> pd.Series:
    """Subtract MER and account-specific tax drag from shrunk expected returns.

    Mirrors the optimizer's net-return path: each asset's pre-drag return is
    routed through
    :meth:`AssetLocationEngine.compute_tax_adjusted_return
    <pysharpe.optimization.tax_location.AssetLocationEngine.compute_tax_adjusted_return>`
    which deducts MER, foreign withholding tax, and income tax drag for the
    target account.  Assets without characteristics pass through unadjusted
    (no MER/tax information available), consistent with the optimizer's 1-D
    fallback.
    """
    if tax_profile is None:
        return mu.copy()
    engine = AssetLocationEngine(tax_profile)
    chars = asset_characteristics or {}
    adjusted = mu.copy()
    for ticker, ret in adjusted.items():
        char = chars.get(ticker)
        if char is not None:
            adjusted[ticker] = engine.compute_tax_adjusted_return(
                float(ret), char, account
            )
    return adjusted


def compute_adjusted_metrics(
    price_frame: pd.DataFrame,
    *,
    reference_prices: pd.DataFrame | None = None,
    tax_profile: TaxProfile | None = None,
    asset_characteristics: dict[str, AssetTaxCharacteristics] | None = None,
    account: str = AccountType.NON_REG.value,
    shrinkage_floor: float = DEFAULT_SHRINKAGE_FLOOR,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> AdjustedMetricResults:
    """Compute harmonized metrics using shrunk, tax- and MER-adjusted returns.

    The raw historical means used by :func:`compute_metrics` are replaced by
    Bayes-Stein shrunk expected returns (:func:`shrinkage_expected_return`)
    so benchmark and custom-mix rows use the same estimator as the optimizer.

    When *reference_prices* is provided (e.g. the benchmark price series),
    its columns are appended to the evaluation frame before shrinkage so the
    benchmarks are pulled toward the same cross-sectional grand mean as the
    asset universe they are compared against.  Portfolio columns always keep
    the portfolio-only shrunk values, matching the optimizer's estimator
    exactly.  Without a reference frame, shrinkage degenerates to the raw
    mean for single assets (its documented fallback) — there is no
    cross-section to shrink toward.

    When *tax_profile* is provided, each shrunk return is reduced by MER and
    account-specific tax drag via the :class:`AssetLocationEngine`.  The
    default account is Non-Registered; callers with an explicit asset-location
    split can pass a different *account*.

    Args:
        price_frame: Historical prices (rows = dates, columns = tickers).
        reference_prices: Optional additional price series (e.g. benchmarks)
            evaluated jointly for shrinkage.
        tax_profile: Investor's marginal tax profile.  When ``None`` no
            MER/tax drag is applied (raw shrunk metrics).
        asset_characteristics: Ticker → tax characteristics used by the
            engine.  Assets missing from the mapping are left unadjusted.
        account: Account label for the drag calculation (default NON_REG).
        shrinkage_floor: Minimum Bayes-Stein shrinkage intensity.
        risk_free_rate: Annual risk-free rate used for Sharpe ratios.

    Returns:
        AdjustedMetricResults with shrunk, drag-adjusted expected returns.
    """

    if price_frame.empty:
        raise ValueError(
            "Price data is empty; please adjust tickers, dates, or upload a richer "
            "dataset."
        )

    frame = price_frame.dropna()
    if frame.empty:
        raise ValueError("Insufficient price history to compute portfolio metrics.")

    # Append reference columns (e.g. benchmarks) not already held by the
    # portfolio so shrinkage operates on the joint cross-section.
    extra_cols: list[str] = []
    if reference_prices is not None and not reference_prices.empty:
        extra_cols = [
            col for col in reference_prices.columns if col not in frame.columns
        ]
    if extra_cols:
        joint = pd.concat([frame, reference_prices[extra_cols]], axis=1)
    else:
        joint = frame
    joint = joint.dropna()
    if joint.empty:
        raise ValueError(
            "No overlapping price history between the portfolio and the "
            "reference series."
        )

    mu_joint = shrinkage_expected_return(joint, shrinkage_floor=shrinkage_floor)

    if extra_cols:
        # Portfolio assets keep the portfolio-only shrunk values (identical to
        # optimise_from_prices); only the reference columns come from the
        # joint run.
        mu_portfolio = shrinkage_expected_return(frame, shrinkage_floor=shrinkage_floor)
        mu = mu_joint.copy()
        for ticker in frame.columns:
            mu[ticker] = mu_portfolio[ticker]
    else:
        mu = mu_joint

    returns = joint.pct_change().dropna()
    if returns.empty:
        raise ValueError("Insufficient price history to compute portfolio metrics.")

    column_index = joint.columns
    expected_pre_drag = mu.reindex(column_index)
    expected = _apply_account_drag(
        expected_pre_drag,
        tax_profile=tax_profile,
        asset_characteristics=asset_characteristics,
        account=account,
    )
    volatility = metrics.annualize_volatility(returns).reindex(column_index)

    sharpe = pd.Series(0.0, index=column_index, dtype=float)
    nonzero = volatility > 0
    sharpe[nonzero] = (expected[nonzero] - risk_free_rate) / volatility[nonzero]

    return AdjustedMetricResults(
        returns=returns,
        expected=expected,
        expected_pre_drag=expected_pre_drag,
        volatility=volatility,
        sharpe=sharpe,
    )


def evaluate_adjusted_performance(
    weights: dict[str, float],
    price_frame: pd.DataFrame,
    *,
    expected_returns: pd.Series,
    tax_profile: TaxProfile | None = None,
    asset_characteristics: dict[str, AssetTaxCharacteristics] | None = None,
    account: str = AccountType.NON_REG.value,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> tuple[float, float, float]:
    """Evaluate a weight vector with shrunk, tax- and MER-adjusted returns.

    Replaces the raw-mean evaluation previously used for the equal-weight and
    custom-mix rows so every row of the dashboard comparison table shares the
    optimizer's Bayes-Stein estimator and the :class:`AssetLocationEngine`
    drag model.  Volatility uses the same Ledoit-Wolf shrunk covariance as
    :func:`pysharpe.portfolio_optimization.optimise_from_prices`.

    Args:
        weights: Ticker → weight mapping (does not need to sum to 1).
        price_frame: Historical prices used for the covariance estimate.
        expected_returns: Shrunk (pre-drag) annualized expected returns.
        tax_profile: Investor's marginal tax profile.  When ``None`` only the
            shrinkage change applies (no MER/tax drag).
        asset_characteristics: Ticker → tax characteristics for the engine.
        account: Account label for the drag calculation (default NON_REG).
        risk_free_rate: Annual risk-free rate used for the Sharpe ratio.

    Returns:
        tuple[float, float, float]: (annualized_return, annualized_volatility,
        sharpe_ratio).
    """

    if price_frame.empty:
        raise ValueError("Price data is empty; cannot evaluate portfolio.")
    assets = list(price_frame.columns)
    if not assets:
        raise ValueError("Price data contains no assets.")

    weights_array = np.array([float(weights.get(asset, 0.0)) for asset in assets])
    total = float(weights_array.sum())
    if total <= 0:
        raise ValueError("Portfolio weights must sum to a positive value.")
    if not np.isclose(total, 1.0):
        weights_array = weights_array / total

    mu_adj = _apply_account_drag(
        expected_returns.reindex(assets),
        tax_profile=tax_profile,
        asset_characteristics=asset_characteristics,
        account=account,
    )

    try:
        cov = CovarianceShrinkage(price_frame.dropna()).ledoit_wolf()
    except Exception as exc:
        raise RuntimeError(
            "Failed to compute Ledoit-Wolf shrinkage covariance for the "
            f"comparison table: {exc}"
        ) from exc

    portfolio_return = float(weights_array @ mu_adj.to_numpy(dtype=float))
    variance = float(weights_array @ cov @ weights_array)
    portfolio_volatility = float(np.sqrt(max(variance, 0.0)))

    if portfolio_volatility == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    return portfolio_return, portfolio_volatility, sharpe_ratio


__all__ = [
    "AdjustedMetricResults",
    "DEFAULT_RISK_FREE_RATE",
    "DEFAULT_SHRINKAGE_FLOOR",
    "MetricResults",
    "compute_adjusted_metrics",
    "compute_metrics",
    "evaluate_adjusted_performance",
]
