"""Tax-efficient cash-flow rebalancing for Canadian multi-account portfolios.

Implements zero-sell contribution allocation with a hybrid 200/175 bps
threshold guardrail that suppresses taxable sales in Non-Registered accounts
unless drift exceeds +2.0% above target, and then only sells down to +1.75%.

References
----------
Perold, A. F., & Sharpe, W. F. (1988). "Dynamic Strategies for Asset Allocation."
Financial Analysts Journal, 44(1), 16-27.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from pysharpe.optimization.tax_location import (
    AccountType,
    AssetLocationEngine,
    AssetTaxCharacteristics,
    TaxProfile,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Basis‑point guardrail thresholds for Non‑Registered (taxable) accounts.
# Sells are suppressed unless drift exceeds the upper limit; when triggered,
# the sell only brings the asset down to the soft ceiling, preserving
# geometric compounding by minimising capital‑gains realisation.
_UPPER_THRESHOLD_BPS: float = 200.0  # +2.00 % above target
_SOFT_CEILING_BPS: float = 175.0  # +1.75 % above target (sell-down target)
_BPS_TO_DECIMAL: float = 0.0001  # 1 bp = 0.01 %

# Default low-end floor to prevent tiny allocations.
_MIN_ALLOCATION_DOLLARS: float = 1.0

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CashFlowRebalanceResult:
    """Result of a tax-efficient cash-flow rebalancing pass.

    Attributes
    ----------
    buy_orders
        DataFrame with columns ``ticker``, ``account``, ``buy_amount``,
        ``current_weight``, ``target_weight``, ``new_weight``.
    sell_orders
        DataFrame with columns ``ticker``, ``account``, ``sell_amount``,
        ``drift_bps``, ``reason``.  Empty when no taxable sales are triggered.
    cash_remaining
        Unallocated cash after rounding or minimum‑allocation filtering.
    """

    buy_orders: pd.DataFrame
    sell_orders: pd.DataFrame
    cash_remaining: float = 0.0

    @property
    def taxable_sell_triggered(self) -> bool:
        """``True`` when at least one taxable sell was generated."""
        return not self.sell_orders.empty


@dataclass
class RebalanceConfig:
    """Configuration for the cash-flow rebalancing allocator.

    Attributes
    ----------
    upper_threshold_bps
        Hard upper threshold in basis points.  Sells are suppressed in
        Non‑Registered accounts unless drift exceeds this value.  Default
        200 bps (+2.00 %).
    soft_ceiling_bps
        When a taxable sell is triggered, the sell amount is sized to
        bring the asset's drift down to this level (not to zero).  Default
        175 bps (+1.75 %).
    tax_profile
        Investor's marginal tax profile, forwarded to the
        :class:`AssetLocationEngine`.
    asset_characteristics
        Mapping of ticker → :class:`AssetTaxCharacteristics` for the
        2‑D asset location engine.
    min_allocation_dollars
        Minimum dollar amount per individual buy or sell order (default $1).
    """

    upper_threshold_bps: float = _UPPER_THRESHOLD_BPS
    soft_ceiling_bps: float = _SOFT_CEILING_BPS
    tax_profile: TaxProfile = field(
        default_factory=lambda: TaxProfile(marginal_tax_rate=0.40)
    )
    asset_characteristics: dict[str, AssetTaxCharacteristics] = field(
        default_factory=dict
    )
    min_allocation_dollars: float = _MIN_ALLOCATION_DOLLARS

    def __post_init__(self) -> None:
        if self.soft_ceiling_bps >= self.upper_threshold_bps:
            raise ValueError(
                f"soft_ceiling_bps ({self.soft_ceiling_bps}) must be strictly "
                f"less than upper_threshold_bps ({self.upper_threshold_bps})"
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_drift_bps(
    current_weight: float,
    target_weight: float,
) -> float:
    """Return the weight drift in basis points, rounded to avoid float
    artifacts near guardrail thresholds."""
    return round((current_weight - target_weight) * 10_000.0, 6)


def _is_taxable(account: str | AccountType) -> bool:
    """Return ``True`` when the account is Non‑Registered (taxable)."""
    if isinstance(account, AccountType):
        return account == AccountType.NON_REG
    return account.upper().strip() in {
        "NON_REG",
        "NON-REG",
        "TAXABLE",
        "MARGIN",
        "CASH",
        "UNREGISTERED",
    }


def _determine_optimal_account(
    ticker: str,
    characteristics: dict[str, AssetTaxCharacteristics],
    engine: AssetLocationEngine,
) -> AccountType:
    """Return the tax‑optimal account wrapper for *ticker*.

    Uses the 2‑D asset location engine to compute the account with the
    lowest total tax drag for this asset.
    """
    char = characteristics.get(ticker)
    if char is None:
        return AccountType.NON_REG

    candidates: list[AccountType] = [
        AccountType.TFSA,
        AccountType.RRSP,
        AccountType.NON_REG,
    ]

    best_account: AccountType = AccountType.NON_REG
    best_drag: float = float("inf")

    for acct in candidates:
        try:
            drag = engine.compute_total_drag(char, acct.value)
        except (ValueError, KeyError):
            drag = 1.0  # Maximum penalty for unknown account.
        if drag < best_drag:
            best_drag = drag
            best_account = acct

    return best_account


# ---------------------------------------------------------------------------
# evaluate_taxable_rebalance
# ---------------------------------------------------------------------------


def evaluate_taxable_rebalance(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    account_types: Mapping[str, str | list[str]],
    current_values: Mapping[str, Mapping[str, float]] | None = None,
    *,
    config: RebalanceConfig | None = None,
) -> pd.DataFrame:
    """Evaluate whether any Non‑Reg assets have breached the taxable‑sell guardrail.

    For each asset held (or partially held) in a Non‑Registered account,
    compares the current weight to the target weight.  If the drift exceeds
    ``upper_threshold_bps`` (+200 bps by default), a **partial sell order**
    is generated to bring the drift down to ``soft_ceiling_bps`` (+175 bps).

    Assets in registered accounts (TFSA, RRSP, FHSA, etc.) are never
    flagged for selling — capital gains within those wrappers are tax‑exempt
    and no restriction applies.

    Parameters
    ----------
    current_weights
        Mapping of ``ticker → current_weight`` (decimal fraction, e.g. 0.25).
    target_weights
        Mapping of ``ticker → target_weight`` (decimal fraction).
    account_types
        Mapping of ``ticker → account_type``.  If a ticker is held across
        multiple accounts, pass a list of account type strings.
    current_values
        Optional ``{ticker: {account: dollar_value}}`` for sizing sells
        in absolute dollars.  When omitted, sell amounts are expressed
        as weight deltas only.
    config
        :class:`RebalanceConfig` with threshold overrides.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``account``, ``current_weight``,
        ``target_weight``, ``drift_bps``, ``sell_weight``,
        ``sell_amount`` (NaN if current_values not provided),
        ``reason``.  Empty DataFrame when no taxable sells are triggered.
    """
    if config is None:
        config = RebalanceConfig()

    rows: list[dict[str, object]] = []

    for ticker, target_w in target_weights.items():
        current_w = float(current_weights.get(ticker, 0.0))
        drift_bps = _compute_drift_bps(current_w, target_w)

        # Only over-weight assets are candidates for selling.
        if drift_bps <= 0.0:
            continue

        # Determine which accounts hold this ticker.
        accts_raw = account_types.get(ticker)
        if accts_raw is None:
            continue

        accts: list[str] = (
            [accts_raw] if isinstance(accts_raw, str) else list(accts_raw)
        )

        for acct_str in accts:
            if not _is_taxable(acct_str):
                continue  # Registered accounts: no sell restriction.

            # Guardrail check.
            if drift_bps <= config.upper_threshold_bps:
                # Below or at threshold: suppress sell.
                continue

            # Drift exceeds upper threshold → generate partial sell.
            # Sell down to soft_ceiling_bps, not to zero.
            sell_weight_bps = drift_bps - config.soft_ceiling_bps
            sell_weight = sell_weight_bps * _BPS_TO_DECIMAL

            # Absolute dollar amount (if values are available).
            sell_amount: float | None = None
            if current_values is not None:
                acct_vals = current_values.get(ticker, {})
                acct_val = acct_vals.get(acct_str, 0.0)
                if acct_val > 0.0 and current_w > 0.0:
                    # The fraction of this asset's total value held in this
                    # specific account.
                    frac_in_acct = acct_val / (
                        sum(acct_vals.values()) if sum(acct_vals.values()) > 0 else 1.0
                    )
                    # Scale the sell weight by the account's share.
                    sell_amount = acct_val * (sell_weight / current_w) * frac_in_acct

            rows.append(
                {
                    "ticker": ticker,
                    "account": acct_str,
                    "current_weight": current_w,
                    "target_weight": target_w,
                    "drift_bps": drift_bps,
                    "sell_weight": max(sell_weight, 0.0),
                    "sell_amount": sell_amount,
                    "reason": (
                        f"Drift {drift_bps:.1f} bps exceeds "
                        f"threshold {config.upper_threshold_bps:.0f} bps; "
                        f"selling to {config.soft_ceiling_bps:.0f} bps ceiling."
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "account",
                "current_weight",
                "target_weight",
                "drift_bps",
                "sell_weight",
                "sell_amount",
                "reason",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values("drift_bps", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# allocate_contribution_cash_flow
# ---------------------------------------------------------------------------


def allocate_contribution_cash_flow(
    cash_amount: float,
    current_balances: pd.DataFrame,
    target_weights: Mapping[str, float],
    account_types: Mapping[str, str],
    opportunity_scores: Mapping[str, float],
    *,
    config: RebalanceConfig | None = None,
) -> CashFlowRebalanceResult:
    """Allocate new capital without triggering taxable sales.

    This is a **buy‑only** rebalancer: it deploys fresh cash toward
    underweight assets using a tax‑optimal account routing strategy.
    No assets are sold in the standard pass — taxable sells are only
    permitted through the separate :func:`evaluate_taxable_rebalance`
    guardrail.

    Algorithm
    ---------
    1. Compute current portfolio weights from *current_balances*.
    2. Identify underweight assets (current weight < target weight).
    3. If cash covers all underweight drift → allocate to exact targets,
       with any remainder distributed proportionally to target weights.
    4. If cash is insufficient → route 100 % to underweight assets,
       weighted by their opportunity‑score deficits.  Assets with the
       largest relative shortfall receive proportionally more capital.

    Parameters
    ----------
    cash_amount
        Dollar amount of fresh capital to deploy.  Must be > 0.
    current_balances
        DataFrame with columns ``ticker``, ``account``, ``current_value``.
        Each row is a (ticker, account) holding.
    target_weights
        Mapping of ``ticker → target_weight`` (decimal fractions summing
        to 1.0).
    account_types
        Mapping of ``ticker → optimal_account``.  This is the preferred
        account wrapper for each ticker, typically determined by the
        :class:`AssetLocationEngine`.  See
        :func:`_determine_optimal_account` for automated selection.
    opportunity_scores
        Mapping of ``ticker → score`` (higher = more attractive buy).
        Typically sourced from :func:`~pysharpe.execution.allocator.score_opportunities`.
    config
        :class:`RebalanceConfig` with threshold and tax‑profile overrides.

    Returns
    -------
    CashFlowRebalanceResult
        Buy orders, any guardrail‑triggered sell orders, and residual cash.
    """
    if cash_amount <= 0.0:
        raise ValueError("cash_amount must be positive.")

    if config is None:
        config = RebalanceConfig()

    # ------------------------------------------------------------------
    # 1. Build the portfolio state
    # ------------------------------------------------------------------
    required_cols = {"ticker", "account", "current_value"}
    missing = required_cols - set(current_balances.columns)
    if missing:
        raise ValueError(
            f"current_balances must have columns {sorted(required_cols)}, "
            f"missing: {sorted(missing)}"
        )

    # Aggregate total value per ticker.
    ticker_values = current_balances.groupby("ticker")["current_value"].sum().to_dict()
    total_portfolio_value = sum(ticker_values.values())

    if total_portfolio_value <= 0.0:
        # Edge case: empty portfolio — distribute equally to all targets.
        return _empty_portfolio_allocate(
            cash_amount, target_weights, account_types, config
        )

    # Current weights.
    current_weights: dict[str, float] = {
        ticker: val / total_portfolio_value for ticker, val in ticker_values.items()
    }
    # Add missing tickers with zero weight.
    for ticker in target_weights:
        if ticker not in current_weights:
            current_weights[ticker] = 0.0

    # ------------------------------------------------------------------
    # 2. Identify underweight assets and compute target dollar amounts
    # ------------------------------------------------------------------
    underweight: dict[str, float] = {}
    target_dollars: dict[str, float] = {}
    total_underweight_dollars: float = 0.0

    new_portfolio_value = total_portfolio_value + cash_amount

    for ticker, target_w in target_weights.items():
        current_w = current_weights.get(ticker, 0.0)
        target_val = target_w * new_portfolio_value
        current_val = ticker_values.get(ticker, 0.0)
        target_dollars[ticker] = target_val

        shortfall = max(target_val - current_val, 0.0)
        if shortfall > 0.0:
            underweight[ticker] = shortfall
            total_underweight_dollars += shortfall

    # ------------------------------------------------------------------
    # 3. Allocate cash
    # ------------------------------------------------------------------
    buy_orders: dict[str, dict[str, float]] = {}  # {ticker: {account: amount}}

    if total_underweight_dollars <= 0.0:
        # All assets are at or above target — distribute proportionally.
        _allocate_proportional(
            buy_orders,
            cash_amount,
            target_weights,
            account_types,
            config,
        )
    elif cash_amount >= total_underweight_dollars:
        # Cash is sufficient to cover all underweights.
        # First, allocate exactly to close the underweight gaps.
        for ticker, shortfall in underweight.items():
            acct = account_types.get(ticker, "NON_REG")
            buy_orders.setdefault(ticker, {})[acct] = (
                buy_orders.get(ticker, {}).get(acct, 0.0) + shortfall
            )

        # Remainder distributed proportionally to target weights.
        remainder = cash_amount - total_underweight_dollars
        if remainder > config.min_allocation_dollars:
            _allocate_proportional(
                buy_orders,
                remainder,
                target_weights,
                account_types,
                config,
            )
    else:
        # Cash shortfall — route 100 % to underweight assets weighted by
        # opportunity-score deficits.
        _allocate_by_opportunity_deficit(
            buy_orders,
            cash_amount,
            underweight,
            opportunity_scores,
            account_types,
            config,
        )

    # ------------------------------------------------------------------
    # 4. Build the buy‑orders DataFrame
    # ------------------------------------------------------------------
    buy_rows: list[dict[str, object]] = []
    for ticker, acct_map in buy_orders.items():
        for acct, amount in acct_map.items():
            if amount < config.min_allocation_dollars:
                continue
            current_w = current_weights.get(ticker, 0.0)
            target_w = target_weights.get(ticker, 0.0)
            current_val = ticker_values.get(ticker, 0.0)
            new_val = current_val + amount
            new_w = new_val / new_portfolio_value if new_portfolio_value > 0 else 0.0

            buy_rows.append(
                {
                    "ticker": ticker,
                    "account": acct,
                    "buy_amount": amount,
                    "current_weight": current_w,
                    "target_weight": target_w,
                    "new_weight": new_w,
                }
            )

    buy_df = pd.DataFrame(buy_rows)
    if not buy_df.empty:
        buy_df = buy_df.sort_values("buy_amount", ascending=False).reset_index(
            drop=True
        )

    # ------------------------------------------------------------------
    # 5. Evaluate taxable‑sell guardrail (post‑buy state)
    # ------------------------------------------------------------------
    # Recompute weights after buys for the guardrail evaluation.
    post_buy_weights: dict[str, float] = dict(current_weights)
    for ticker, acct_map in buy_orders.items():
        total_buy = sum(acct_map.values())
        post_buy_weights[ticker] = (
            ticker_values.get(ticker, 0.0) + total_buy
        ) / new_portfolio_value

    # Build the current_values structure for sell‑amount sizing.
    current_values_nested: dict[str, dict[str, float]] = {}
    for _, row in current_balances.iterrows():
        t = str(row["ticker"])
        a = str(row["account"])
        v = float(row["current_value"])
        current_values_nested.setdefault(t, {})[a] = v

    # Build account_types list for multi-account holdings.
    acct_types_list: dict[str, list[str]] = {}
    for _, row in current_balances.iterrows():
        t = str(row["ticker"])
        a = str(row["account"])
        acct_types_list.setdefault(t, []).append(a)

    sell_df = evaluate_taxable_rebalance(
        current_weights=post_buy_weights,
        target_weights=dict(target_weights),
        account_types=acct_types_list,
        current_values=current_values_nested,
        config=config,
    )

    # Compute residual cash.
    total_allocated = buy_df["buy_amount"].sum() if not buy_df.empty else 0.0
    cash_remaining = cash_amount - total_allocated

    return CashFlowRebalanceResult(
        buy_orders=buy_df,
        sell_orders=sell_df,
        cash_remaining=max(cash_remaining, 0.0),
    )


# ---------------------------------------------------------------------------
# Allocation strategies (internal)
# ---------------------------------------------------------------------------


def _allocate_proportional(
    buy_orders: dict[str, dict[str, float]],
    cash: float,
    target_weights: Mapping[str, float],
    account_types: Mapping[str, str],
    config: RebalanceConfig,
) -> None:
    """Distribute *cash* proportionally to target weights."""
    total_weight = sum(target_weights.values())
    if total_weight <= 0.0:
        return

    for ticker, target_w in target_weights.items():
        alloc = cash * (target_w / total_weight)
        if alloc < config.min_allocation_dollars:
            continue
        acct = account_types.get(ticker, "NON_REG")
        buy_orders.setdefault(ticker, {})[acct] = (
            buy_orders.get(ticker, {}).get(acct, 0.0) + alloc
        )


def _allocate_by_opportunity_deficit(
    buy_orders: dict[str, dict[str, float]],
    cash: float,
    underweight: dict[str, float],
    opportunity_scores: Mapping[str, float],
    account_types: Mapping[str, str],
    config: RebalanceConfig,
) -> None:
    """Route all cash to underweight assets, weighted by opportunity deficit.

    The allocation weight for each underweight asset is proportional to:

        weight_i = underweight_i × (1.0 − score_i)

    where *score_i* is normalised to [0, 1].  This ensures assets with
    the **largest shortfall and lowest opportunity score** receive the
    most capital — targeting the most severe imbalances first.
    """
    # Normalise opportunity scores to [0, 1].
    scores = {t: float(opportunity_scores.get(t, 0.5)) for t in underweight}
    if scores:
        min_s = min(scores.values())
        max_s = max(scores.values())
        score_range = max_s - min_s
        if score_range > 0.0:
            scores = {t: (s - min_s) / score_range for t, s in scores.items()}
        else:
            scores = {t: 0.5 for t in scores}

    # Compute deficit weight = underweight × (1 − normalised_score).
    # Higher normalised score → lower deficit weight (less urgent).
    deficit_weights: dict[str, float] = {}
    total_deficit_weight: float = 0.0

    for ticker, shortfall in underweight.items():
        norm_score = scores.get(ticker, 0.5)
        # Invert: low score → high urgency → larger allocation weight.
        dw = shortfall * (1.0 - norm_score)
        deficit_weights[ticker] = dw
        total_deficit_weight += dw

    if total_deficit_weight <= 0.0:
        # All scores are 1.0 or shortfalls are zero → equal allocation.
        n = len(underweight)
        for ticker in underweight:
            alloc = cash / n
            if alloc < config.min_allocation_dollars:
                continue
            acct = account_types.get(ticker, "NON_REG")
            buy_orders.setdefault(ticker, {})[acct] = (
                buy_orders.get(ticker, {}).get(acct, 0.0) + alloc
            )
        return

    for ticker in underweight:
        alloc = cash * (deficit_weights[ticker] / total_deficit_weight)
        if alloc < config.min_allocation_dollars:
            continue
        acct = account_types.get(ticker, "NON_REG")
        buy_orders.setdefault(ticker, {})[acct] = (
            buy_orders.get(ticker, {}).get(acct, 0.0) + alloc
        )


def _empty_portfolio_allocate(
    cash_amount: float,
    target_weights: Mapping[str, float],
    account_types: Mapping[str, str],
    config: RebalanceConfig,
) -> CashFlowRebalanceResult:
    """Handle the edge case where the portfolio has zero current value."""
    total_weight = sum(target_weights.values())
    if total_weight <= 0.0:
        return CashFlowRebalanceResult(
            buy_orders=pd.DataFrame(
                columns=[
                    "ticker",
                    "account",
                    "buy_amount",
                    "current_weight",
                    "target_weight",
                    "new_weight",
                ]
            ),
            sell_orders=pd.DataFrame(
                columns=[
                    "ticker",
                    "account",
                    "current_weight",
                    "target_weight",
                    "drift_bps",
                    "sell_weight",
                    "sell_amount",
                    "reason",
                ]
            ),
            cash_remaining=cash_amount,
        )

    buy_rows: list[dict[str, object]] = []
    for ticker, target_w in target_weights.items():
        alloc = cash_amount * (target_w / total_weight)
        if alloc < config.min_allocation_dollars:
            continue
        acct = account_types.get(ticker, "NON_REG")
        buy_rows.append(
            {
                "ticker": ticker,
                "account": acct,
                "buy_amount": alloc,
                "current_weight": 0.0,
                "target_weight": target_w,
                "new_weight": target_w,
            }
        )

    buy_df = pd.DataFrame(buy_rows)
    total_allocated = buy_df["buy_amount"].sum() if not buy_df.empty else 0.0

    return CashFlowRebalanceResult(
        buy_orders=buy_df,
        sell_orders=pd.DataFrame(
            columns=[
                "ticker",
                "account",
                "current_weight",
                "target_weight",
                "drift_bps",
                "sell_weight",
                "sell_amount",
                "reason",
            ]
        ),
        cash_remaining=max(cash_amount - total_allocated, 0.0),
    )


__all__ = [
    "CashFlowRebalanceResult",
    "RebalanceConfig",
    "allocate_contribution_cash_flow",
    "evaluate_taxable_rebalance",
]
