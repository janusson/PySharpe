"""Smart contribution allocation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AllocationConfig:
    """Configuration for the contribution allocator.

    Attributes
    ----------
    weight_underweight : float, optional
        How much weight to give to portfolio drift (underweight vs target) in
        the final opportunity score. Default is 0.6.
    weight_valuation : float, optional
        How much weight to give to fundamental valuation in the final
        opportunity score. Default is 0.4.
    weight_pe : float, optional
        Weight of the P/E ratio within the valuation score. Default is 0.4.
    weight_pb : float, optional
        Weight of the P/B ratio within the valuation score. Default is 0.3.
    weight_div_yield : float, optional
        Weight of the dividend yield within the valuation score. Default is 0.2.
    weight_momentum : float, optional
        Weight of 6-month momentum within the valuation score. Default is 0.1.
    min_allocation_dollars : float, optional
        Minimum dollar amount to allocate to a single asset to avoid tiny
        fractional buys. Default is 25.0.
    valuation_epsilon : float, optional
        A small constant added to valuation scores so that assets with terrible
        valuations still receive *some* allocation if they are severely
        underweight. Default is 0.1.
    """

    weight_underweight: float = 0.6
    weight_valuation: float = 0.4

    weight_pe: float = 0.4
    weight_pb: float = 0.3
    weight_div_yield: float = 0.2
    weight_momentum: float = 0.1

    min_allocation_dollars: float = 25.0
    valuation_epsilon: float = 0.1


def _zscore(series: pd.Series) -> pd.Series:
    """Calculate standard z-score, safely handling NaNs and zero std.

    Parameters
    ----------
    series : pd.Series
        The input data series.

    Returns
    -------
    pd.Series
        The z-scored series. Returns 0.0 for elements where standard deviation
        is zero or NaN.
    """
    s = series.astype(float)
    mean = s.mean(skipna=True)
    std = s.std(skipna=True)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


def _minmax_01(series: pd.Series) -> pd.Series:
    """Scale values to the [0, 1] range, safely handling edge cases.

    Parameters
    ----------
    series : pd.Series
        The input data series to scale.

    Returns
    -------
    pd.Series
        The scaled series. Returns 0.5 (neutral) if the series is constant or
        contains only NaNs.
    """
    s = series.astype(float)
    s_min = s.min(skipna=True)
    s_max = s.max(skipna=True)
    if np.isnan(s_min) or np.isnan(s_max) or s_min == s_max:
        return pd.Series(0.5, index=s.index)  # neutral
    return (s - s_min) / (s_max - s_min)


def score_opportunities(
    df: pd.DataFrame,
    config: Optional[AllocationConfig] = None,
) -> pd.DataFrame:
    """Evaluate and rank investment opportunities based on drift and valuation.

    This function calculates a composite "opportunity score" for each asset in
    a portfolio. It combines how severely an asset is underweight relative to
    its target allocation with a multi-factor fundamental valuation score.

    Parameters
    ----------
    df : pd.DataFrame
        Current portfolio state and optional fundamental data.
        Required columns: 'ticker', 'current_value', 'target_weight'.
        Optional columns: 'pe_ratio', 'pb_ratio', 'div_yield', 'momentum_6m'.
    config : AllocationConfig, optional
        Configuration dictating the weights of different scoring components.
        If None, default values are used.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame sorted by descending 'opportunity_score',
        with the following new columns appended:
        - 'current_weight': Actual weight in the current portfolio.
        - 'underweight': Percentage points below target (floored at 0).
        - 'underweight_score': Normalized drift score [0, 1].
        - 'valuation_score': Composite fundamental score [0, 1].
        - 'opportunity_score': Final blended ranking score.
    """
    if config is None:
        config = AllocationConfig()

    df = df.copy()

    # 1) Portfolio weights / underweights
    portfolio_value = df["current_value"].sum()
    if portfolio_value <= 0:
        # If portfolio is empty, treat all current weights as 0.
        df["current_weight"] = 0.0
    else:
        df["current_weight"] = df["current_value"] / portfolio_value

    df["target_weight"] = df["target_weight"].fillna(0.0).clip(lower=0.0)

    df["underweight"] = (df["target_weight"] - df["current_weight"]).clip(lower=0.0)

    max_under = df["underweight"].max()
    if max_under <= 0:
        df["underweight_score"] = 0.0
    else:
        df["underweight_score"] = df["underweight"] / max_under

    # 2) Valuation factors
    # Lower PE and PB are "better": invert the z-score
    pe_z = _zscore(df.get("pe_ratio", pd.Series(np.nan, index=df.index)))
    pb_z = _zscore(df.get("pb_ratio", pd.Series(np.nan, index=df.index)))

    pe_score = -pe_z  # low PE => higher score
    pb_score = -pb_z  # low PB => higher score

    # Higher dividend yield is good
    div_z = _zscore(df.get("div_yield", pd.Series(np.nan, index=df.index)))
    div_score = div_z

    # Momentum: higher is good, but with low weight by default
    if "momentum_6m" in df.columns:
        mom_z = _zscore(df["momentum_6m"])
        mom_score = mom_z
    else:
        mom_score = pd.Series(0.0, index=df.index)

    # Normalize each to [0, 1] for stability
    pe_n = _minmax_01(pe_score)
    pb_n = _minmax_01(pb_score)
    div_n = _minmax_01(div_score)
    mom_n = _minmax_01(mom_score)

    # Weighted valuation score in [0, 1]
    w_pe = config.weight_pe
    w_pb = config.weight_pb
    w_div = config.weight_div_yield
    w_mom = config.weight_momentum

    total_w = w_pe + w_pb + w_div + w_mom
    if total_w == 0:
        total_w = 1.0

    valuation_score = (
        w_pe * pe_n + w_pb * pb_n + w_div * div_n + w_mom * mom_n
    ) / total_w

    df["valuation_score"] = valuation_score

    # 3) Final opportunity score
    wu = config.weight_underweight
    wv = config.weight_valuation
    if wu + wv == 0:
        wu, wv = 0.5, 0.5

    df["opportunity_score"] = wu * df["underweight_score"] + wv * df["valuation_score"]

    # Sort for convenience
    df = df.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    return df


def allocate_contribution(
    scored_df: pd.DataFrame,
    contribution_dollars: float,
    config: Optional[AllocationConfig] = None,
) -> pd.DataFrame:
    """Distribute a cash contribution based on opportunity scores.

    This function takes the ranked output from `score_opportunities` and assigns
    dollar amounts to the most attractive assets. It uses a "need" metric
    calculated as `underweight * (epsilon + valuation_score)`. The allocator
    also enforces a minimum buy threshold to avoid generating tiny orders.

    Parameters
    ----------
    scored_df : pd.DataFrame
        The output from `score_opportunities`, containing at minimum the
        'current_value', 'underweight', and 'valuation_score' columns.
    contribution_dollars : float
        The total amount of cash to deploy. Must be strictly positive.
    config : AllocationConfig, optional
        Configuration object. If None, default settings are applied.

    Returns
    -------
    pd.DataFrame
        The input DataFrame sorted descending by recommended dollar amount,
        with new columns:
        - 'raw_allocation': Unconstrained theoretical dollar assignment.
        - 'recommended_allocation': Final dollar amount to buy.
        - 'recommended_weight_increase': Estimated portfolio weight impact.
        - 'allocation_rank': Rank of the recommended allocation size.

    Raises
    ------
    ValueError
        If `contribution_dollars` is not strictly positive.
    """
    if config is None:
        config = AllocationConfig()

    if contribution_dollars <= 0:
        raise ValueError("contribution_dollars must be positive.")

    df = scored_df.copy()

    # Build a "need" weight: underweight * (epsilon + valuation_score)
    val = df["valuation_score"].fillna(0.0)
    under = df["underweight"].fillna(0.0)

    need_raw = under * (config.valuation_epsilon + val)

    # If everything is fully at or above target, fall back to pure valuation
    if need_raw.sum() <= 0:
        need_raw = config.valuation_epsilon + val

    total_need = need_raw.sum()
    if total_need <= 0:
        # degenerate; split equally
        alloc_weights = pd.Series(1.0 / len(df), index=df.index)
    else:
        alloc_weights = need_raw / total_need

    df["raw_allocation"] = alloc_weights * contribution_dollars

    # Enforce minimum allocation per name; re-normalize the rest
    min_alloc = config.min_allocation_dollars
    df["recommended_allocation"] = 0.0

    # First, assign 0 to very small allocations and mark the "eligible" set
    eligible = df["raw_allocation"] >= min_alloc
    if eligible.any():
        # Work only on eligible names; renormalize
        sub = df.loc[eligible, "raw_allocation"]
        sub_weights = sub / sub.sum()
        df.loc[eligible, "recommended_allocation"] = sub_weights * contribution_dollars
    else:
        # If no-one clears the min threshold, just give the largest one the whole amount
        idx_max = df["raw_allocation"].idxmax()
        df.loc[idx_max, "recommended_allocation"] = contribution_dollars

    # Approximate weight increase (assuming portfolio_value + contribution)
    portfolio_value = df["current_value"].sum()
    total_after = portfolio_value + contribution_dollars

    if total_after > 0:
        df["recommended_weight_increase"] = df["recommended_allocation"] / total_after
    else:
        df["recommended_weight_increase"] = 0.0

    # Rank for display
    df["allocation_rank"] = df["recommended_allocation"].rank(
        ascending=False, method="min"
    )

    df = df.sort_values("recommended_allocation", ascending=False).reset_index(
        drop=True
    )
    return df
