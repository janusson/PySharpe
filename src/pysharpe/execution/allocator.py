"""Smart contribution allocation logic."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from pysharpe.config import (
    AccountType,
    AssetTaxProfile,
    calculate_withholding_tax_rate,
)

if TYPE_CHECKING:
    from pysharpe.config import ExecutionConfig

logger = logging.getLogger(__name__)


@dataclass
class AllocationConfig:
    """Configuration for the contribution allocator.

    Attributes
    ----------
    weight_underweight : float
        How much weight to give to portfolio drift (underweight vs target) in
        the final opportunity score. Default is 0.5 (50 %).
    weight_valuation : float
        How much weight to give to fundamental valuation in the final
        opportunity score. Default is 0.3 (30 %).
    weight_capital_efficiency : float
        How much weight to give to capital efficiency (tax-advantaged account
        prioritisation) in the final opportunity score. Default is 0.2 (20 %).
    account_room : Dict[AccountType, float]
        Available contribution room per Canadian registered account type.
        Used to prioritise tax-advantaged accounts during allocation and to
        enforce contribution caps. Default is an empty dictionary (no caps).
    asset_tax_profiles : dict[str, AssetTaxProfile]
        Mapping of ticker symbols to their tax profiles. Used to compute
        withholding-tax drag on spillover allocations to non-registered
        accounts. Default is an empty dictionary.
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

    weight_underweight: float = 0.5
    weight_valuation: float = 0.3
    weight_capital_efficiency: float = 0.2
    account_room: dict[AccountType, float] = field(default_factory=dict)
    asset_tax_profiles: dict[str, AssetTaxProfile] = field(default_factory=dict)

    time_series_factors: dict[str, tuple[float, Literal["positive", "negative"]]] = (
        field(
            default_factory=lambda: {
                "pe_ratio": (0.4, "negative"),
                "pb_ratio": (0.3, "negative"),
                "div_yield": (0.2, "positive"),
                "momentum_6m": (0.1, "positive"),
            }
        )
    )

    min_allocation_dollars: float = 25.0
    valuation_epsilon: float = 0.1
    trend_factors: dict[str, float] = field(
        default_factory=lambda: {
            "ma_crossover_signal": 0.2,
        }
    )


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
    config: AllocationConfig | None = None,
) -> pd.DataFrame:
    """Evaluate and rank investment opportunities based on drift, valuation,
    and capital efficiency.

    This function calculates a composite "opportunity score" for each asset in
    a portfolio. It combines (1) how severely an asset is underweight relative
    to its target allocation, (2) a multi-factor fundamental valuation score,
    and (3) a capital-efficiency score that prioritises tax-advantaged accounts
    (TFSA/RRSP/FHSA) with remaining contribution room.

    Parameters
    ----------
    df : pd.DataFrame
        Current portfolio state and optional fundamental data.
        Required columns: 'ticker', 'current_value', 'target_weight'.
        Optional columns: 'target_account' (AccountType or str),
        'pe_ratio', 'pb_ratio', 'div_yield', 'momentum_6m'.
    config : AllocationConfig, optional
        Configuration dictating the weights of different scoring components.
        If None, default values are used (Drift 0.5, Valuation 0.3,
        Capital Efficiency 0.2).

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame sorted by descending 'opportunity_score',
        with the following new columns appended:
        - 'current_weight': Actual weight in the current portfolio.
        - 'underweight': Percentage points below target (floored at 0).
        - 'underweight_score': Normalized drift score [0, 1].
        - 'valuation_score': Composite fundamental score [0, 1].
        - 'capital_efficiency_score': Tax-advantaged account priority [0, 1].
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

    # 2) Valuation factors (dynamic based on config)
    weighted_normalized_scores = []
    total_w = 0.0

    # Process time-series factors (e.g., PE, PB, Div Yield, Momentum)
    for factor_name, (weight, direction) in config.time_series_factors.items():
        if factor_name not in df.columns:
            logger.warning(
                f"Time-series factor '{factor_name}' not found in DataFrame; skipping."
            )
            continue

        series = df[factor_name]
        z_score = _zscore(series)

        # Apply direction: negative factors (e.g., PE) are inverted
        factor_score = -z_score if direction == "negative" else z_score

        # Normalize to [0, 1] for stability
        normalized_score = _minmax_01(factor_score)

        weighted_normalized_scores.append(weight * normalized_score)
        total_w += weight

    # Process trend factors (e.g., MA Crossover Signals)
    for factor_name, weight in config.trend_factors.items():
        if factor_name not in df.columns:
            logger.warning(
                f"Trend factor '{factor_name}' not found in DataFrame; skipping."
            )
            continue

        # Trend factors are expected to be pre-scored (e.g., -1, 0, 1). Normalize to [0, 1]
        series = df[factor_name]
        normalized_score = (series + 1) / 2  # Maps -1->0, 0->0.5, 1->1

        weighted_normalized_scores.append(weight * normalized_score)
        total_w += weight

    if total_w == 0 or not weighted_normalized_scores:
        valuation_score = pd.Series(
            0.5, index=df.index
        )  # Neutral if no factors or weights
    else:
        valuation_score = sum(weighted_normalized_scores) / total_w

    df["valuation_score"] = valuation_score

    # 3) Capital efficiency score — prioritise tax-advantaged accounts with room
    account_room = config.account_room
    if "target_account" in df.columns and account_room:
        capital_scores = []
        for _, row in df.iterrows():
            raw_acct = row.get("target_account")
            if raw_acct is None or (isinstance(raw_acct, float) and np.isnan(raw_acct)):
                capital_scores.append(0.5)  # Neutral when no account is specified
                continue

            try:
                acct = AccountType(str(raw_acct).upper().strip())
            except ValueError:
                capital_scores.append(0.5)
                continue

            remaining = account_room.get(acct, 0.0)
            if acct == AccountType.NON_REGISTERED:
                # Non-registered always has unlimited room; neutral score
                capital_scores.append(0.5)
            elif remaining > 0:
                # Tax-advantaged account with room → high priority
                capital_scores.append(1.0)
            else:
                # Tax-advantaged account with no room left → low priority
                capital_scores.append(0.0)
        df["capital_efficiency_score"] = capital_scores
    else:
        # No account info or no room data → neutral score for all
        df["capital_efficiency_score"] = 0.5

    # 4) Final opportunity score — blended from three pillars
    wu = config.weight_underweight
    wv = config.weight_valuation
    wce = config.weight_capital_efficiency

    df["opportunity_score"] = (
        wu * df["underweight_score"]
        + wv * df["valuation_score"]
        + wce * df["capital_efficiency_score"]
    )

    # Sort for convenience
    df = df.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    return df


@dataclass
class FxRoutingResult:
    """Result of an FX routing decision between standard conversion and
    Norbert's Gambit.

    Attributes
    ----------
    use_norberts_gambit : bool
        ``True`` when Norbert's Gambit is cheaper than the standard
        percentage-based currency conversion.
    standard_cost : float
        Estimated cost of the standard conversion in CAD.
    norberts_gambit_cost : float
        Estimated cost of Norbert's Gambit in CAD.
    savings : float
        Dollar amount saved by using the recommended method (always ≥ 0).
    execution_steps : list[str]
        Ordered checklist for the investor.
    """

    use_norberts_gambit: bool
    standard_cost: float
    norberts_gambit_cost: float
    savings: float
    execution_steps: list[str]


def determine_fx_routing(
    transaction_value: float,
    config: ExecutionConfig,
) -> FxRoutingResult:
    """Decide whether to use Norbert's Gambit for a US-asset purchase.

    Compares the standard percentage-based FX conversion fee against the
    fixed-commission cost of Norbert's Gambit (buy DLR.TO → journal to
    DLR-U.TO → sell for USD).

    Parameters
    ----------
    transaction_value : float
        Size of the transaction in CAD that requires currency conversion.
    config : ExecutionConfig
        Execution settings containing ``fx_fee_bps``,
        ``norberts_commission``, ``norberts_spread_bps``, and
        ``norberts_drift_risk_bps``.

    Returns
    -------
    FxRoutingResult
    """
    if transaction_value <= 0:
        return FxRoutingResult(
            use_norberts_gambit=False,
            standard_cost=0.0,
            norberts_gambit_cost=0.0,
            savings=0.0,
            execution_steps=[],
        )

    fee = config.fx_fee_decimal
    commission = config.norberts_commission
    spread_pct = config.norberts_spread_decimal
    drift_pct = config.norberts_drift_decimal
    dlr_price = config.norberts_dlr_price_cad

    # Standard FX cost: T * f
    standard_cost = transaction_value * fee

    # Norbert's Gambit cost: 2*C + T*spread + T*drift
    ng_cost = 2 * commission + transaction_value * (spread_pct + drift_pct)

    use_ng = ng_cost < standard_cost
    savings = abs(standard_cost - ng_cost)

    # Build execution checklist when Norbert's Gambit is recommended
    steps: list[str] = []
    if use_ng:
        shares = math.floor(transaction_value / dlr_price) if dlr_price > 0 else 0
        if shares > 0:
            dlr_cost = shares * dlr_price
            steps.append(
                f"Step 1: Buy {shares} shares of DLR.TO at ~${dlr_price:.2f}/share "
                f"(≈${dlr_cost:,.2f} CAD + ${commission:.2f} commission)"
            )
        else:
            steps.append(
                f"Step 1: Buy DLR.TO with ~${transaction_value:,.2f} CAD "
                f"(+ ${commission:.2f} commission)"
            )
        steps.append(
            "Step 2: Request journaling of DLR.TO → DLR-U.TO (takes 2–3 business days)"
        )
        if shares > 0:
            steps.append(
                f"Step 3: Sell {shares} shares of DLR-U.TO to receive USD "
                f"(− ${commission:.2f} commission)"
            )
        else:
            steps.append(
                f"Step 3: Sell DLR-U.TO shares to receive USD "
                f"(− ${commission:.2f} commission)"
            )
        steps.append("Step 4: Use USD proceeds to purchase the target US security")

    return FxRoutingResult(
        use_norberts_gambit=use_ng,
        standard_cost=round(standard_cost, 2),
        norberts_gambit_cost=round(ng_cost, 2),
        savings=round(savings, 2),
        execution_steps=steps,
    )


def allocate_contribution(
    scored_df: pd.DataFrame,
    contribution_dollars: float,
    config: AllocationConfig | None = None,
) -> pd.DataFrame:
    """Distribute a cash contribution based on opportunity scores with
    account-room awareness.

    The allocator prioritises tax-advantaged accounts (TFSA, RRSP, FHSA) up to
    their contribution limits.  Any cash that exceeds a registered account's
    remaining room *spills over* into the non-registered allocation, where
    US withholding-tax drag is reapplied to the affected shares.

    When no ``target_account`` column or ``account_room`` mapping is present,
    the function falls back to the classic unconstrained allocation behaviour.

    Parameters
    ----------
    scored_df : pd.DataFrame
        The output from :func:`score_opportunities`, containing at minimum the
        ``'current_value'``, ``'underweight'``, and ``'valuation_score'``
        columns.  When ``'target_account'`` is present and
        ``config.account_room`` is non-empty, account caps are enforced.
    contribution_dollars : float
        The total amount of cash to deploy. Must be strictly positive.
    config : AllocationConfig, optional
        Configuration object. If None, default settings are applied.

    Returns
    -------
    pd.DataFrame
        The input DataFrame sorted descending by recommended dollar amount,
        with new columns:
        - ``'raw_allocation'``: Unconstrained theoretical dollar assignment.
        - ``'recommended_allocation'``: Final dollar amount to buy.
        - ``'spillover_allocation'``: Portion redirected to non-registered.
        - ``'recommended_weight_increase'``: Estimated portfolio weight impact.
        - ``'allocation_rank'``: Rank of the recommended allocation size.

    Raises
    ------
    ValueError
        If ``contribution_dollars`` is not strictly positive.
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

    # --- Account-room-aware allocation ---
    has_accounts = "target_account" in df.columns and bool(config.account_room)

    df["recommended_allocation"] = 0.0
    df["spillover_allocation"] = 0.0

    if not has_accounts:
        # Classic unconstrained path (backward compatible)
        _apply_min_allocation(df, contribution_dollars, config)
    else:
        # Account-aware path with spillover
        _allocate_with_account_room(df, contribution_dollars, config)

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


def _apply_min_allocation(
    df: pd.DataFrame,
    contribution_dollars: float,
    config: AllocationConfig,
    *,
    source_col: str = "raw_allocation",
) -> None:
    """Enforce minimum allocation threshold; re-normalize the rest.

    Operates on the DataFrame in-place, setting ``recommended_allocation``.
    When ``source_col`` is ``"recommended_allocation"`` (used after
    account-aware allocation), the thresholds and weights are derived from
    the already-constrained recommended amounts rather than raw.
    """
    min_alloc = config.min_allocation_dollars
    eligible = df[source_col] >= min_alloc
    if eligible.any():
        sub = df.loc[eligible, source_col]
        sub_weights = sub / sub.sum()
        df.loc[eligible, "recommended_allocation"] = sub_weights * contribution_dollars
    else:
        idx_max = df[source_col].idxmax()
        df.loc[idx_max, "recommended_allocation"] = contribution_dollars


def _apply_min_allocation_by_account(
    df: pd.DataFrame,
    contribution_dollars: float,
    config: AllocationConfig,
    room: dict[AccountType, float],
) -> None:
    """Enforce minimum allocation within each account group, capping at room.

    Unlike the classic ``_apply_min_allocation``, this operates per-account
    to avoid breaking contribution limits during re-normalization.
    """
    min_alloc = config.min_allocation_dollars

    for idx in df.index:
        rec = df.at[idx, "recommended_allocation"]
        if rec > 0 and rec < min_alloc:
            # Tiny allocation — zero it out and return the cash to spillover
            acct = _resolve_account(df.loc[idx])
            if acct is not None and acct != AccountType.NON_REGISTERED:
                room[acct] = room.get(acct, 0.0) + rec
            df.at[idx, "recommended_allocation"] = 0.0


def _resolve_account(row: pd.Series) -> AccountType | None:
    """Parse the ``target_account`` column into an AccountType, or None."""
    raw = row.get("target_account")
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    try:
        return AccountType(str(raw).upper().strip())
    except ValueError:
        return None


def _allocate_with_account_room(
    df: pd.DataFrame,
    contribution_dollars: float,
    config: AllocationConfig,
) -> None:
    """Account-aware allocation that caps registered contributions and
    redirects spillover to non-registered accounts.

    Operates on the DataFrame in-place, setting ``recommended_allocation``
    and ``spillover_allocation``.
    """
    # Snapshot remaining room per account
    room: dict[AccountType, float] = dict(config.account_room)
    spillover = 0.0

    # Sort by opportunity score descending so highest-priority assets get
    # first access to limited tax-advantaged room.
    ranked = df.sort_values("opportunity_score", ascending=False)

    for idx in ranked.index:
        raw = df.at[idx, "raw_allocation"]
        if raw <= 0:
            continue

        acct = _resolve_account(df.loc[idx])
        if acct is None or acct == AccountType.NON_REGISTERED:
            # Non-reg or unknown → unlimited room; allocate directly
            df.at[idx, "recommended_allocation"] = raw
            continue

        # Tax-advantaged account — check remaining room
        remaining = room.get(acct, 0.0)
        if remaining >= raw:
            # Full allocation fits within room
            df.at[idx, "recommended_allocation"] = raw
            room[acct] = remaining - raw
        elif remaining > 0:
            # Partial allocation: use up remaining room, spillover the rest
            df.at[idx, "recommended_allocation"] = remaining
            df.at[idx, "spillover_allocation"] = raw - remaining
            spillover += raw - remaining
            room[acct] = 0.0

            ticker_label = df.at[idx, "ticker"] if "ticker" in df.columns else str(idx)
            logger.info(
                "%s: %s room exhausted (allocated $%.2f of $%.2f raw). "
                "$%.2f spills to non-registered.",
                ticker_label,
                acct.value,
                remaining,
                raw,
                raw - remaining,
            )
        else:
            # No room left — entire amount spills over
            df.at[idx, "spillover_allocation"] = raw
            spillover += raw

            ticker_label = df.at[idx, "ticker"] if "ticker" in df.columns else str(idx)
            logger.debug(
                "%s: %s has no remaining room — $%.2f spills to non-registered.",
                ticker_label,
                acct.value,
                raw,
            )

    # --- Re-allocate spillover to non-registered assets ---
    if spillover > 0:
        # Find assets targeted for non-registered (or unknown) accounts
        nonreg_mask = pd.Series(False, index=df.index)
        for idx in df.index:
            acct = _resolve_account(df.loc[idx])
            nonreg_mask.at[idx] = acct is None or acct == AccountType.NON_REGISTERED

        nonreg_df = df.loc[nonreg_mask]

        if nonreg_df.empty:
            logger.warning(
                "$%.2f in spillover but no non-registered assets to absorb it. "
                "Spillover is left unallocated.",
                spillover,
            )
        else:
            # Distribute spillover proportionally by opportunity score
            scores = nonreg_df["opportunity_score"].clip(lower=0.0)
            if scores.sum() <= 0:
                weights = pd.Series(1.0 / len(nonreg_df), index=nonreg_df.index)
            else:
                weights = scores / scores.sum()

            for idx in nonreg_df.index:
                extra = weights.at[idx] * spillover
                df.at[idx, "recommended_allocation"] += extra

                # Apply withholding-tax drag logging for spillover shares
                ticker = df.at[idx, "ticker"] if "ticker" in df.columns else str(idx)
                profile = config.asset_tax_profiles.get(ticker)
                if profile is not None and extra > 1e-4:
                    wht_rate = calculate_withholding_tax_rate(
                        AccountType.NON_REGISTERED, profile
                    )
                    tax_drag = profile.dividend_yield * wht_rate
                    if tax_drag > 1e-4:
                        logger.info(
                            "Spillover to non-reg: %s receives $%.2f "
                            "(tax drag=%.2f bps on yield=%.2f%%).",
                            ticker,
                            extra,
                            tax_drag * 10000,
                            profile.dividend_yield * 100,
                        )

    # --- Enforce minimum allocation on the final recommended amounts ---
    # Re-normalize within each account group to preserve room caps.
    _apply_min_allocation_by_account(df, contribution_dollars, config, room)
