"""Smart contribution allocation logic with tax-shelter awareness.

Routes new cash to specific account wrappers, respecting contribution
limits and dynamically re-routing overflow to the non-registered account
with on-the-fly opportunity-score recalculation.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from pysharpe.config import AccountType, TaxProfile
from pysharpe.optimization.tax_location import (
    AssetLocationEngine,
    AssetTaxCharacteristics,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class AllocationConfig:
    """Configuration for the contribution allocator.

    Attributes
    ----------
    weight_underweight : float
        Weight for portfolio drift (underweight vs target). Default 0.5.
    weight_valuation : float
        Weight for fundamental valuation. Default 0.3.
    weight_tax_efficiency : float
        Weight for tax-efficiency (via :class:`AssetLocationEngine`).
        Default 0.2.
    available_contribution_room : dict[AccountType, float]
        Remaining contribution room per registered account type.  The
        allocator deducts from this as it assigns dollars.
    tax_profile : TaxProfile
        Investor's marginal tax profile forwarded to the engine.
    asset_characteristics : dict[str, AssetTaxCharacteristics]
        Ticker → detailed tax characteristics for the engine.
    min_allocation_dollars : float
        Minimum dollar amount per allocation (default $25).
    valuation_epsilon : float
        Small constant added to valuation scores so severely underweight
        assets still receive some allocation. Default 0.1.
    """

    weight_underweight: float = 0.5
    weight_valuation: float = 0.3
    weight_tax_efficiency: float = 0.2
    # Backward-compat alias — also accepted from JSON / kwargs
    weight_capital_efficiency: float = field(default=0.2, repr=False)
    available_contribution_room: dict[AccountType, float] = field(default_factory=dict)
    # Backward-compat alias
    account_room: dict[AccountType, float] = field(default_factory=dict, repr=False)
    tax_profile: TaxProfile = field(
        default_factory=lambda: TaxProfile(marginal_tax_rate=0.40)
    )
    asset_characteristics: dict[str, AssetTaxCharacteristics] = field(
        default_factory=dict
    )

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
        default_factory=lambda: {"ma_crossover_signal": 0.2}
    )

    def __post_init__(self) -> None:
        # Backward-compat: sync old field names to canonical ones
        if self.weight_capital_efficiency != 0.2 and self.weight_tax_efficiency == 0.2:
            object.__setattr__(
                self, "weight_tax_efficiency", self.weight_capital_efficiency
            )
        if self.account_room and not self.available_contribution_room:
            object.__setattr__(
                self, "available_contribution_room", dict(self.account_room)
            )


# ---------------------------------------------------------------------------
# Scoring utilities (unchanged)
# ---------------------------------------------------------------------------


def _zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mean = s.mean(skipna=True)
    std = s.std(skipna=True)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


def _minmax_01(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    s_min = s.min(skipna=True)
    s_max = s.max(skipna=True)
    if np.isnan(s_min) or np.isnan(s_max) or s_min == s_max:
        return pd.Series(0.5, index=s.index)
    return (s - s_min) / (s_max - s_min)


# ---------------------------------------------------------------------------
# Adaptive pillar scaling
# ---------------------------------------------------------------------------

# Authoritative baseline weights when tax location is non-differentiable.
# These match the core 60/40 investment-heuristic split mandated by the
# project investment philosophy.
_CORE_UNDERWEIGHT_WEIGHT: float = 0.6
_CORE_VALUATION_WEIGHT: float = 0.4


def _is_tax_location_differentiable(
    df: pd.DataFrame,
    chars: dict[str, AssetTaxCharacteristics],
) -> bool:
    """Return ``True`` when tax-efficiency scoring provides
    a meaningful differentiation signal across the evaluated assets.

    Tax location is *non-differentiable* (returns ``False``) when:

    * No asset characteristics are configured, OR
    * There is no ``target_account`` column, OR
    * Every row targets the same account wrapper (eliminating
      relative tax-optimisation differentials).
    """
    if not chars:
        return False
    if "target_account" not in df.columns:
        return False

    accounts = df["target_account"].dropna()
    if accounts.empty:
        return False

    # Normalise to uppercase, stripped strings for robust comparison
    unique = {str(a).upper().strip() for a in accounts}
    return len(unique) > 1


# ---------------------------------------------------------------------------
# Opportunity scoring
# ---------------------------------------------------------------------------


def score_opportunities(
    df: pd.DataFrame,
    config: AllocationConfig | None = None,
) -> pd.DataFrame:
    """Evaluate and rank investment opportunities with tax-efficiency scoring.

    Each row represents one (asset, account) pair.  The composite score
    blends **drift** (underweight vs target), **valuation** (multi-factor
    fundamentals), and **tax efficiency** (via
    :class:`AssetLocationEngine.compute_tax_efficiency_score`).

    Required columns: ``'ticker'``, ``'current_value'``, ``'target_weight'``.
    Optional: ``'target_account'`` (for tax scoring), ``'pe_ratio'``,
    ``'pb_ratio'``, ``'div_yield'``, ``'momentum_6m'``.

    Parameters
    ----------
    df : pd.DataFrame
    config : AllocationConfig or None

    Returns
    -------
    pd.DataFrame
        Sorted by ``'opportunity_score'`` descending.
    """
    if config is None:
        config = AllocationConfig()

    df = df.copy()
    engine = AssetLocationEngine(config.tax_profile)

    # 1) Portfolio weights / underweights
    portfolio_value = df["current_value"].sum()
    df["current_weight"] = (
        0.0 if portfolio_value <= 0 else df["current_value"] / portfolio_value
    )
    df["target_weight"] = df["target_weight"].fillna(0.0).clip(lower=0.0)
    df["underweight"] = (df["target_weight"] - df["current_weight"]).clip(lower=0.0)

    max_under = df["underweight"].max()
    df["underweight_score"] = 0.0 if max_under <= 0 else df["underweight"] / max_under

    # 2) Valuation factors
    weighted_normalized_scores = []
    total_w = 0.0

    for factor_name, (weight, direction) in config.time_series_factors.items():
        if factor_name not in df.columns:
            continue
        # df[factor_name] is a single column — narrow to Series for the type-checker
        series: pd.Series = df[factor_name]  # type: ignore[assignment]
        factor_score = -_zscore(series) if direction == "negative" else _zscore(series)
        weighted_normalized_scores.append(weight * _minmax_01(factor_score))
        total_w += weight

    for factor_name, weight in config.trend_factors.items():
        if factor_name not in df.columns:
            continue
        series: pd.Series = df[factor_name]  # type: ignore[assignment]
        weighted_normalized_scores.append(weight * (series + 1) / 2)
        total_w += weight

    if total_w == 0 or not weighted_normalized_scores:
        df["valuation_score"] = 0.5
    else:
        df["valuation_score"] = sum(weighted_normalized_scores) / total_w

    # 3) Tax-efficiency score via AssetLocationEngine
    has_account_col = "target_account" in df.columns
    chars = config.asset_characteristics

    if has_account_col and chars:
        tax_scores = []
        for _, row in df.iterrows():
            ticker = row.get("ticker")
            raw_acct = row.get("target_account")
            char = chars.get(ticker) if ticker else None

            if (
                char is None
                or raw_acct is None
                or (isinstance(raw_acct, float) and np.isnan(raw_acct))
            ):
                tax_scores.append(0.5)
                continue

            try:
                acct_str = str(raw_acct).upper().strip()
                # Normalise through the engine's internal lookup
                tax_scores.append(engine.compute_tax_efficiency_score(char, acct_str))
            except (ValueError, KeyError):
                tax_scores.append(0.5)
        df["tax_efficiency_score"] = tax_scores
    else:
        df["tax_efficiency_score"] = 0.5

    # 4) Final blended score with adaptive pillar scaling
    #
    # When tax location is non-differentiable (unconfigured tax
    # characteristics or a single uniform account), the three-pillar
    # 50/30/20 split collapses into the authoritative 60/40 investment-
    # heuristic baseline, bypassing the tax-efficiency pillar entirely.
    if _is_tax_location_differentiable(df, chars):
        df["opportunity_score"] = (
            config.weight_underweight * df["underweight_score"]
            + config.weight_valuation * df["valuation_score"]
            + config.weight_tax_efficiency * df["tax_efficiency_score"]
        )
    else:
        df["opportunity_score"] = (
            _CORE_UNDERWEIGHT_WEIGHT * df["underweight_score"]
            + _CORE_VALUATION_WEIGHT * df["valuation_score"]
        )

    return df.sort_values("opportunity_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Contribution allocation
# ---------------------------------------------------------------------------


def allocate_contribution(
    scored_df: pd.DataFrame,
    contribution_dollars: float,
    config: AllocationConfig | None = None,
) -> pd.DataFrame:
    """Distribute cash to the highest-ranked (asset, account) opportunities,
    deducting from each account's contribution room.  When a tax-advantaged
    account hits its limit, overflow is dynamically re-routed to
    ``NON_REG`` with an on-the-fly opportunity-score recalculation that
    reflects the lost tax efficiency.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Output from :func:`score_opportunities`.
    contribution_dollars : float
        Cash to deploy.  Must be > 0.
    config : AllocationConfig or None

    Returns
    -------
    pd.DataFrame
        With columns ``raw_allocation``, ``recommended_allocation``,
        ``spillover_allocation``, ``spillover_account``, and
        ``recalculated_score`` (when applicable).
    """
    if config is None:
        config = AllocationConfig()
    if contribution_dollars <= 0:
        raise ValueError("contribution_dollars must be positive.")

    df = scored_df.copy()
    engine = AssetLocationEngine(config.tax_profile)

    # --- Need-based raw allocation ---
    val = df["valuation_score"].fillna(0.0)
    under = df["underweight"].fillna(0.0)
    need_raw = under * (config.valuation_epsilon + val)
    if need_raw.sum() <= 0:
        need_raw = config.valuation_epsilon + val

    total_need = need_raw.sum()
    if total_need <= 0:
        alloc_weights = pd.Series(1.0 / len(df), index=df.index)
    else:
        alloc_weights = need_raw / total_need

    df["raw_allocation"] = alloc_weights * contribution_dollars

    # --- Account-aware allocation with dynamic re-routing ---
    has_accounts = "target_account" in df.columns and bool(
        config.available_contribution_room
    )

    df["recommended_allocation"] = 0.0
    df["spillover_allocation"] = 0.0
    df["spillover_account"] = None
    df["recalculated_score"] = np.nan

    if not has_accounts:
        _apply_min_allocation(df, contribution_dollars, config)
    else:
        _allocate_with_dynamic_reroute(df, contribution_dollars, config, engine)

    # --- Weight increase ---
    portfolio_value = df["current_value"].sum()
    total_after = portfolio_value + contribution_dollars
    df["recommended_weight_increase"] = (
        df["recommended_allocation"] / total_after if total_after > 0 else 0.0
    )

    df["allocation_rank"] = df["recommended_allocation"].rank(
        ascending=False, method="min"
    )
    return df.sort_values("recommended_allocation", ascending=False).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_account(row: pd.Series) -> AccountType | None:
    raw = row.get("target_account")
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    try:
        return AccountType(str(raw).upper().strip())
    except ValueError:
        return None


def _apply_min_allocation(
    df: pd.DataFrame,
    contribution_dollars: float,
    config: AllocationConfig,
    *,
    source_col: str = "raw_allocation",
) -> None:
    min_alloc = config.min_allocation_dollars
    eligible = df[source_col] >= min_alloc
    if eligible.any():
        sub = df.loc[eligible, source_col]
        df.loc[eligible, "recommended_allocation"] = (
            sub / sub.sum() * contribution_dollars
        )
    else:
        idx_max = df[source_col].idxmax()
        df.loc[idx_max, "recommended_allocation"] = contribution_dollars


def _allocate_with_dynamic_reroute(
    df: pd.DataFrame,
    contribution_dollars: float,
    config: AllocationConfig,
    engine: AssetLocationEngine,
) -> None:
    """Account-aware loop that creates new NON_REG rows on overflow and re-sorts.

    When a tax-advantaged account's contribution room is exhausted, the
    overflow capital is placed into a **new** dataframe row targeting
    ``NON_REG``.  Its tax-efficiency and composite scores are recalculated
    on the fly, and the row participates in subsequent sort-allocate
    iterations so that the highest-value opportunities always receive
    capital first.
    """
    room = deepcopy(dict(config.available_contribution_room))
    chars = config.asset_characteristics
    processed: set = set()

    # Allocate fresh indices for new rows we may append
    idx_max = df.index.max()
    next_idx = int(idx_max) + 1 if not df.empty else 0  # type: ignore[arg-type]

    # Safety cap: each original row can produce at most one NON_REG spillover
    # row, so total rows ≤ 2 × initial_rows.  Add a generous 10× buffer and
    # an absolute floor of 10 000 to guard against logic regressions.
    max_iterations = max(10 * len(df) + 1, 10_000)
    iteration = 0

    while True:
        iteration += 1
        if iteration > max_iterations:
            logger.error(
                "Allocation loop exceeded max_iterations (%d). "
                "Breaking to prevent infinite loop.  %d rows processed, "
                "%d rows remain.",
                max_iterations,
                len(processed),
                len(df) - len(processed),
            )
            break

        # ---- pick the highest-scored row that hasn't been finalised ----
        unprocessed_mask = ~df.index.isin(processed)
        unprocessed: pd.DataFrame = df.loc[unprocessed_mask]  # type: ignore[assignment]
        if unprocessed.empty:
            break

        unprocessed = unprocessed.sort_values("opportunity_score", ascending=False)
        top_idx = unprocessed.index[0]
        row = df.loc[top_idx]
        raw = float(row.get("raw_allocation", 0.0))

        # NaN raw_allocation (e.g. from divide-by-zero upstream) is unrecoverable
        if raw <= 0 or np.isnan(raw):
            processed.add(top_idx)
            continue

        acct = _resolve_account(row)
        ticker = str(row.get("ticker", "")) if "ticker" in df.columns else str(top_idx)

        # ---- NON_REG / unknown → unlimited room → allocate directly ----
        if acct is None or acct == AccountType.NON_REG:
            df.at[top_idx, "recommended_allocation"] = raw
            processed.add(top_idx)
            continue

        # ---- Tax-advantaged account — enforce room cap ----
        remaining = room.get(acct, 0.0)

        if remaining >= raw:
            # Full allocation within remaining room
            df.at[top_idx, "recommended_allocation"] = raw
            room[acct] = remaining - raw
            processed.add(top_idx)

        elif remaining > 0:
            # Partial fill — the portion beyond room spills to a new NON_REG row
            df.at[top_idx, "recommended_allocation"] = remaining
            overflow = raw - remaining
            df.at[top_idx, "spillover_allocation"] = overflow
            df.at[top_idx, "spillover_account"] = AccountType.NON_REG.value
            room[acct] = 0.0
            processed.add(top_idx)

            _append_overflow_row(
                df, next_idx, row, overflow, ticker, config, engine, chars
            )
            next_idx += 1

            logger.info(
                "%s: %s room exhausted (allocated $%.2f of $%.2f). $%.2f spills to %s.",
                ticker,
                acct.value,
                remaining,
                raw,
                overflow,
                AccountType.NON_REG.value,
            )

        else:
            # No room at all — entire amount spills to a new NON_REG row
            df.at[top_idx, "recommended_allocation"] = 0.0
            df.at[top_idx, "spillover_allocation"] = raw
            df.at[top_idx, "spillover_account"] = AccountType.NON_REG.value
            room[acct] = 0.0
            processed.add(top_idx)

            _append_overflow_row(df, next_idx, row, raw, ticker, config, engine, chars)
            next_idx += 1

            logger.debug(
                "%s: %s has no remaining room — $%.2f spills to %s.",
                ticker,
                acct.value,
                raw,
                AccountType.NON_REG.value,
            )

    # --- Apply minimum allocation ---
    _apply_min_allocation(df, contribution_dollars, config)


def _append_overflow_row(
    df: pd.DataFrame,
    new_idx: int,
    source_row: pd.Series,
    overflow_amount: float,
    ticker: str,
    config: AllocationConfig,
    engine: AssetLocationEngine,
    chars: dict[str, AssetTaxCharacteristics],
) -> None:
    """Append a new NON_REG row to *df* for the overflow capital.

    The new row inherits the source row's drift and valuation pillars but
    recalculates ``tax_efficiency_score`` and ``opportunity_score`` for the
    non-registered account context.
    """
    new_row = source_row.copy()
    new_row["target_account"] = AccountType.NON_REG.value
    new_row["raw_allocation"] = overflow_amount
    new_row["recommended_allocation"] = 0.0
    new_row["spillover_allocation"] = 0.0
    new_row["spillover_account"] = None
    new_row["recalculated_score"] = np.nan

    # Recalculate tax-efficiency for NON_REG
    char = chars.get(ticker)
    if char is not None:
        try:
            tax_score = engine.compute_tax_efficiency_score(
                char, AccountType.NON_REG.value
            )
            new_row["tax_efficiency_score"] = tax_score
            # pd.Series.get returns object; the defaults guarantee float at runtime
            uw_score = float(new_row.get("underweight_score", 0.0))  # type: ignore[arg-type]
            val_score = float(new_row.get("valuation_score", 0.0))  # type: ignore[arg-type]
            new_row["opportunity_score"] = (
                config.weight_underweight * uw_score
                + config.weight_valuation * val_score
                + config.weight_tax_efficiency * tax_score
            )
        except (ValueError, KeyError) as exc:
            logger.debug(
                "Could not recalculate tax-efficiency for %s in NON_REG: %s",
                ticker,
                exc,
            )

    df.loc[new_idx] = new_row
