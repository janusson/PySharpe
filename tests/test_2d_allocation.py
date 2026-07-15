"""Integration tests for the 2-D Asset Location Matrix model.

.. note::

    **Canadian Investment Heuristics** — The allocation engine implements a
    deterministic Value Averaging (VA) strategy for broad-market, CAD-
    denominated index ETFs (e.g., VFV, VDY, QQC).

    The opportunity score is a strictly defined weighted sum:

    * 60% Path Drift (underweight_score): how far current value deviates
      from the compounding target value path.
    * 40% Valuation/Mean-Reversion (valuation_score): fundamental signal.

    When tax characteristics are absent or all assets target the same
    account, the blend collapses to the authoritative 60/40 baseline,
    bypassing the tax-efficiency pillar entirely (see
    :func:`pysharpe.execution.allocator._is_tax_location_differentiable`).

    These weights are stable.  Do not alter them without explicit
    authorization.

Validates:
* Sharpe-optimiser weight convergence in the N×M variable space.
* Per-account capacity constraints are respected.
* Contribution-allocator overflow logic: when a tax-advantaged wrapper
  hits its contribution cap, capital cascades into new NON_REG rows with
  on-the-fly opportunity-score recalculation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysharpe.config import AccountType
from pysharpe.execution.allocator import (
    AllocationConfig,
    _append_overflow_row,
    _resolve_account,
    allocate_contribution,
    score_opportunities,
)
from pysharpe.optimization.sharpe_optimizer import (
    SharpeOptimizer,
    SharpeOptimizerConfig,
)
from pysharpe.optimization.tax_location import (
    AssetLocationEngine,
    AssetTaxCharacteristics,
    TaxProfile,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TICKERS = ["VFV", "QQC", "VDY"]


def _make_prices(
    tickers: list[str] | None = None,
    n_days: int = 504,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic price history with distinct per-asset drift/vol."""
    rng = np.random.default_rng(seed)
    tickers = tickers or TICKERS
    dates = pd.bdate_range("2022-01-03", periods=n_days, freq="B")
    drifts = {"VFV": 0.0004, "QQC": 0.0005, "VDY": 0.0003}
    vols = {"VFV": 0.012, "QQC": 0.016, "VDY": 0.009}
    prices = pd.DataFrame(index=dates, dtype=float)
    for t in tickers:
        mu = drifts.get(t, 0.0004)
        sigma = vols.get(t, 0.012)
        rets = rng.normal(loc=mu, scale=sigma, size=n_days)
        prices[t] = 100.0 * np.cumprod(1 + rets)
    return prices


def _make_tax_chars() -> dict[str, AssetTaxCharacteristics]:
    """Representative ETF tax characteristics.

    Income fractions must sum to 1.0 per the ``AssetTaxCharacteristics``
    post-init validator.
    """
    return {
        "VFV": AssetTaxCharacteristics(
            ticker="VFV",
            dividend_yield=0.012,
            is_cad_wrapped_us_equity=True,
            income_frac_eligible_dividends=0.0,
            income_frac_foreign_income=1.0,
            mer=0.0009,
        ),
        "QQC": AssetTaxCharacteristics(
            ticker="QQC",
            dividend_yield=0.005,
            is_cad_wrapped_us_equity=True,
            income_frac_eligible_dividends=0.0,
            income_frac_foreign_income=1.0,
            mer=0.0020,
        ),
        "VDY": AssetTaxCharacteristics(
            ticker="VDY",
            dividend_yield=0.035,
            is_cad_wrapped_us_equity=False,
            income_frac_eligible_dividends=0.8,
            income_frac_foreign_income=0.2,
            mer=0.0022,
        ),
    }


@pytest.fixture()
def prices_2d() -> pd.DataFrame:
    """Price DataFrame with DatetimeIndex for the optimiser."""
    return _make_prices()


@pytest.fixture()
def tax_profile() -> TaxProfile:
    """Mid-to-high marginal bracket."""
    return TaxProfile(marginal_tax_rate=0.45)


@pytest.fixture()
def asset_chars() -> dict[str, AssetTaxCharacteristics]:
    return _make_tax_chars()


@pytest.fixture()
def account_capacities() -> dict[AccountType, float]:
    """Fractional capacities summing to 1.0."""
    return {
        AccountType.TFSA: 0.25,
        AccountType.RRSP: 0.35,
        AccountType.NON_REG: 0.40,
    }


@pytest.fixture()
def optimizer_config(
    tax_profile: TaxProfile,
    asset_chars: dict[str, AssetTaxCharacteristics],
    account_capacities: dict[AccountType, float],
) -> SharpeOptimizerConfig:
    return SharpeOptimizerConfig(
        risk_free_rate=0.03,
        tax_profile=tax_profile,
        asset_characteristics=asset_chars,
        account_capacities=account_capacities,
        max_weight=0.50,
        num_portfolios_monte_carlo=3000,
    )


# ---------------------------------------------------------------------------
# 2-D Optimiser tests
# ---------------------------------------------------------------------------


class Test2DOptimizerWeights:
    """Verify the SLSQP solver converges to feasible 2-D weight matrices."""

    def test_total_weights_sum_to_one(
        self,
        prices_2d: pd.DataFrame,
        optimizer_config: SharpeOptimizerConfig,
    ) -> None:
        """Grand total of all (ticker × account) weights must equal 1.0."""
        opt = SharpeOptimizer(prices_2d, optimizer_config)
        result = opt.optimize()

        total = sum(result.weights.values())
        assert abs(total - 1.0) < 1e-5, f"Total weight = {total:.8f}, expected 1.0"

    def test_weights_non_negative(
        self,
        prices_2d: pd.DataFrame,
        optimizer_config: SharpeOptimizerConfig,
    ) -> None:
        """Every individual weight must be ≥ 0."""
        opt = SharpeOptimizer(prices_2d, optimizer_config)
        result = opt.optimize()

        for key, w in result.weights.items():
            assert w >= -1e-9, f"Weight for {key} is negative: {w}"

    def test_per_account_capacity_respected(
        self,
        prices_2d: pd.DataFrame,
        optimizer_config: SharpeOptimizerConfig,
        account_capacities: dict[AccountType, float],
    ) -> None:
        """Sum of weights assigned to each account must not exceed its cap."""
        opt = SharpeOptimizer(prices_2d, optimizer_config)
        result = opt.optimize()

        per_account: dict[str, float] = {}
        for (_, acct_str), w in result.weights.items():
            per_account[acct_str] = per_account.get(acct_str, 0.0) + w

        for acct, cap in account_capacities.items():
            actual = per_account.get(acct.value, 0.0)
            assert actual <= cap + 1e-6, (
                f"{acct.value} allocation {actual:.6f} exceeds cap {cap:.4f}"
            )

    def test_2d_keys_are_tuples(
        self,
        prices_2d: pd.DataFrame,
        optimizer_config: SharpeOptimizerConfig,
    ) -> None:
        """In 2-D mode every weight key must be a (ticker, account) tuple."""
        opt = SharpeOptimizer(prices_2d, optimizer_config)
        result = opt.optimize()

        for key in result.weights:
            assert isinstance(key, tuple), f"Expected tuple key, got {type(key)}"
            assert len(key) == 2, f"Expected 2-element key, got {key}"
            assert isinstance(key[0], str), f"First element not str: {key[0]}"
            assert isinstance(key[1], str), f"Second element not str: {key[1]}"

    def test_1d_fallback_when_capacities_empty(
        self,
        prices_2d: pd.DataFrame,
        tax_profile: TaxProfile,
        asset_chars: dict[str, AssetTaxCharacteristics],
    ) -> None:
        """Classic 1-D path when account_capacities is empty."""
        config = SharpeOptimizerConfig(
            risk_free_rate=0.03,
            tax_profile=tax_profile,
            asset_characteristics=asset_chars,
            max_weight=0.50,
            # No account_capacities → 1-D backward-compat
        )
        opt = SharpeOptimizer(prices_2d, config)
        result = opt.optimize()

        total = sum(result.weights.values())
        assert abs(total - 1.0) < 1e-5

        # Keys must be plain ticker strings in 1-D mode
        for key in result.weights:
            assert isinstance(key, str), (
                f"Expected str key in 1-D mode, got {type(key)}"
            )

    def test_config_validates_capacity_sum(
        self,
        prices_2d: pd.DataFrame,
        tax_profile: TaxProfile,
        asset_chars: dict[str, AssetTaxCharacteristics],
    ) -> None:
        """Capacities summing > 1.0 must raise ValueError."""
        bad_caps = {
            AccountType.TFSA: 0.60,
            AccountType.RRSP: 0.60,
        }
        config = SharpeOptimizerConfig(
            risk_free_rate=0.03,
            tax_profile=tax_profile,
            asset_characteristics=asset_chars,
            account_capacities=bad_caps,
        )
        with pytest.raises(ValueError, match="sum to"):
            SharpeOptimizer(prices_2d, config)


# ---------------------------------------------------------------------------
# Allocator / spillover tests
# ---------------------------------------------------------------------------


def _make_scored_df(
    tickers: list[str] | None = None,
    tax_chars: dict[str, AssetTaxCharacteristics] | None = None,
    tax_profile: TaxProfile | None = None,
) -> pd.DataFrame:
    """Build a minimal DataFrame suitable for ``score_opportunities``.

    Each row is one (asset, account) pair.  Target weights are deliberately
    skewed so that VFV (US equity) gets the largest allocation — this
    ensures tax-efficiency scoring has material impact.
    """
    tickers = tickers or TICKERS
    tax_chars = tax_chars or _make_tax_chars()
    tax_profile = tax_profile or TaxProfile(marginal_tax_rate=0.45)

    rows = []
    for ticker in tickers:
        for acct in (AccountType.TFSA, AccountType.RRSP, AccountType.NON_REG):
            rows.append(
                {
                    "ticker": ticker,
                    "target_account": acct.value,
                    "current_value": 1000.0,
                    "target_weight": (
                        0.50 if ticker == "VFV" else 0.30 if ticker == "QQC" else 0.20
                    ),
                    "pe_ratio": (
                        18.0 if ticker == "VFV" else 25.0 if ticker == "QQC" else 14.0
                    ),
                    "pb_ratio": (
                        3.0 if ticker == "VFV" else 5.0 if ticker == "QQC" else 1.8
                    ),
                    "div_yield": (
                        0.012
                        if ticker == "VFV"
                        else 0.005
                        if ticker == "QQC"
                        else 0.035
                    ),
                    "momentum_6m": (
                        0.08 if ticker == "VFV" else 0.12 if ticker == "QQC" else 0.04
                    ),
                }
            )

    df = pd.DataFrame(rows)
    config = AllocationConfig(
        tax_profile=tax_profile,
        asset_characteristics=tax_chars,
    )
    return score_opportunities(df, config)


class TestAllocatorOverflow:
    """Verify spillover logic when tax-advantaged room is exhausted."""

    def test_scored_df_has_required_columns(self) -> None:
        """score_opportunities must produce tax_efficiency_score column."""
        df = _make_scored_df()
        assert "opportunity_score" in df.columns
        assert "tax_efficiency_score" in df.columns
        assert "underweight_score" in df.columns
        assert "valuation_score" in df.columns
        assert bool(df["opportunity_score"].notna().all()), (
            "opportunity_score contains NaN"
        )

    def test_tax_efficiency_varies_by_account(self) -> None:
        """VFV should have materially different tax scores across accounts."""
        df = _make_scored_df()
        vfv_rows = df[df["ticker"] == "VFV"]
        scores = vfv_rows["tax_efficiency_score"].tolist()
        unique = {round(s, 4) for s in scores}
        assert len(unique) > 1, f"VFV tax scores identical across accounts: {scores}"

    def test_overflow_creates_new_row(
        self,
        asset_chars: dict[str, AssetTaxCharacteristics],
        tax_profile: TaxProfile,
    ) -> None:
        """When TFSA room is exhausted, a new NON_REG row must appear."""
        df = _make_scored_df(tax_chars=asset_chars, tax_profile=tax_profile)
        original_len = len(df)

        # Tight room: only $100 in TFSA, forcing overflow for top-ranked rows
        alloc_config = AllocationConfig(
            available_contribution_room={AccountType.TFSA: 100.0},
            tax_profile=tax_profile,
            asset_characteristics=asset_chars,
            min_allocation_dollars=1.0,
        )

        result = allocate_contribution(
            df, contribution_dollars=10000.0, config=alloc_config
        )

        # The result should have more rows than the original (overflow rows)
        assert len(result) > original_len, (
            f"Expected overflow rows; got {len(result)} rows (original: {original_len})"
        )

        # At least one overflow row should target NON_REG
        new_rows = result[result.index >= original_len]
        non_reg_new = new_rows[new_rows["target_account"] == AccountType.NON_REG.value]
        assert len(non_reg_new) > 0, "No new NON_REG overflow rows were created"

    def test_overflow_row_has_recalculated_score(
        self,
        asset_chars: dict[str, AssetTaxCharacteristics],
        tax_profile: TaxProfile,
    ) -> None:
        """Overflow rows must have a valid opportunity_score."""
        df = _make_scored_df(tax_chars=asset_chars, tax_profile=tax_profile)
        original_len = len(df)

        alloc_config = AllocationConfig(
            available_contribution_room={AccountType.TFSA: 50.0},
            tax_profile=tax_profile,
            asset_characteristics=asset_chars,
            min_allocation_dollars=1.0,
        )

        result = allocate_contribution(
            df, contribution_dollars=5000.0, config=alloc_config
        )
        new_rows = result[result.index >= original_len]

        for _, row in new_rows.iterrows():
            if row.get("target_account") == AccountType.NON_REG.value:
                score = row.get("opportunity_score")
                assert score is not None and not (
                    isinstance(score, float) and np.isnan(score)
                ), "Overflow row missing opportunity_score"

    def test_non_reg_room_is_unlimited(
        self,
        asset_chars: dict[str, AssetTaxCharacteristics],
        tax_profile: TaxProfile,
    ) -> None:
        """NON_REG rows must always receive their full raw allocation."""
        df = _make_scored_df(tax_chars=asset_chars, tax_profile=tax_profile)

        # Only NON_REG room — should fully allocate without overflow
        alloc_config = AllocationConfig(
            available_contribution_room={AccountType.NON_REG: 1e9},
            tax_profile=tax_profile,
            asset_characteristics=asset_chars,
        )

        result = allocate_contribution(
            df, contribution_dollars=5000.0, config=alloc_config
        )
        # NON_REG-targeted rows must receive their full allocation
        non_reg_mask = result["target_account"] == AccountType.NON_REG.value
        assert bool((result.loc[non_reg_mask, "recommended_allocation"] > 0).all()), (
            "NON_REG rows did not receive allocation"
        )
        # TFSA/RRSP rows have zero room, so overflow rows are expected
        assert len(result) >= len(df)

    def test_spillover_rows_have_correct_account(
        self,
        asset_chars: dict[str, AssetTaxCharacteristics],
        tax_profile: TaxProfile,
    ) -> None:
        """Every spillover_row's ``spillover_account`` must be NON_REG."""
        df = _make_scored_df(tax_chars=asset_chars, tax_profile=tax_profile)
        original_len = len(df)

        alloc_config = AllocationConfig(
            available_contribution_room={
                AccountType.TFSA: 80.0,
                AccountType.RRSP: 80.0,
            },
            tax_profile=tax_profile,
            asset_characteristics=asset_chars,
            min_allocation_dollars=1.0,
        )

        result = allocate_contribution(
            df, contribution_dollars=5000.0, config=alloc_config
        )

        # Original rows that have spillover_account set
        spill_rows = result[
            result["spillover_account"].notna() & (result.index < original_len)
        ]
        for _, row in spill_rows.iterrows():
            assert row["spillover_account"] == AccountType.NON_REG.value, (
                f"Expected NON_REG spillover, got {row['spillover_account']}"
            )

    def test_no_infinite_loop_with_zero_room(
        self,
        asset_chars: dict[str, AssetTaxCharacteristics],
        tax_profile: TaxProfile,
    ) -> None:
        """Allocation must terminate even when all registered room is zero."""
        df = _make_scored_df(tax_chars=asset_chars, tax_profile=tax_profile)

        alloc_config = AllocationConfig(
            available_contribution_room={
                AccountType.TFSA: 0.0,
                AccountType.RRSP: 0.0,
            },
            tax_profile=tax_profile,
            asset_characteristics=asset_chars,
        )

        # Should not hang — must return within normal time
        result = allocate_contribution(
            df, contribution_dollars=5000.0, config=alloc_config
        )
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Allocator scoring tests
# ---------------------------------------------------------------------------


class TestAllocatorScoring:
    """Verify composite scoring blends drift, valuation, and tax efficiency."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_base_df(
        target_account: str | None = None,
    ) -> pd.DataFrame:
        """Build a minimal 3-ticker DataFrame parametrised by account."""
        rows = []
        for ticker in TICKERS:
            row = {
                "ticker": ticker,
                "current_value": 1000.0,
                "target_weight": (
                    0.50 if ticker == "VFV" else 0.30 if ticker == "QQC" else 0.20
                ),
                "pe_ratio": (
                    18.0 if ticker == "VFV" else 25.0 if ticker == "QQC" else 14.0
                ),
                "pb_ratio": (
                    3.0 if ticker == "VFV" else 5.0 if ticker == "QQC" else 1.8
                ),
                "div_yield": (
                    0.012 if ticker == "VFV" else 0.005 if ticker == "QQC" else 0.035
                ),
                "momentum_6m": (
                    0.08 if ticker == "VFV" else 0.12 if ticker == "QQC" else 0.04
                ),
            }
            if target_account is not None:
                row["target_account"] = target_account
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_composite_score_is_weighted_blend(self) -> None:
        """opportunity_score must equal the weighted sum of its components."""
        df = _make_scored_df()
        config = AllocationConfig()

        for _, row in df.iterrows():
            expected = (
                config.weight_underweight * row["underweight_score"]
                + config.weight_valuation * row["valuation_score"]
                + config.weight_tax_efficiency * row["tax_efficiency_score"]
            )
            ticker = row.get("ticker", "?")
            acct = row.get("target_account", "?")
            assert abs(row["opportunity_score"] - expected) < 1e-9, (
                f"Composite score mismatch for {ticker}/{acct}"
            )

    def test_default_weights_sum_to_one(self) -> None:
        """Default config weights should sum to 1.0."""
        config = AllocationConfig()
        total = (
            config.weight_underweight
            + config.weight_valuation
            + config.weight_tax_efficiency
        )
        assert abs(total - 1.0) < 1e-9

    def test_tax_neutral_scales_to_60_40(self) -> None:
        """When asset_characteristics are omitted, scoring collapses to
        the authoritative 60/40 investment-heuristic baseline."""
        df = self._build_base_df(target_account="TFSA")
        config = AllocationConfig()  # empty asset_characteristics
        result = score_opportunities(df, config)

        for _, row in result.iterrows():
            expected = 0.6 * row["underweight_score"] + 0.4 * row["valuation_score"]
            ticker = row.get("ticker", "?")
            assert abs(row["opportunity_score"] - expected) < 1e-9, (
                f"60/40 scaling failed for {ticker}: "
                f"got {row['opportunity_score']:.6f}, expected {expected:.6f}"
            )

    def test_uniform_account_scales_to_60_40(self) -> None:
        """When all rows target the same account wrapper, relative
        tax-location differentials vanish and scoring collapses to 60/40."""
        chars = _make_tax_chars()
        df = self._build_base_df(target_account="RRSP")
        config = AllocationConfig(
            asset_characteristics=chars,
            tax_profile=TaxProfile(marginal_tax_rate=0.45),
        )
        result = score_opportunities(df, config)

        for _, row in result.iterrows():
            expected = 0.6 * row["underweight_score"] + 0.4 * row["valuation_score"]
            ticker = row.get("ticker", "?")
            assert abs(row["opportunity_score"] - expected) < 1e-9, (
                f"Uniform-account 60/40 scaling failed for {ticker}: "
                f"got {row['opportunity_score']:.6f}, expected {expected:.6f}"
            )

    def test_mixed_accounts_uses_three_pillar_blend(self) -> None:
        """When multiple accounts are present alongside configured tax
        characteristics, the full three-pillar blend (50/30/20 default)
        should be used."""
        df = _make_scored_df()  # TFSA, RRSP, NON_REG
        config = AllocationConfig(
            asset_characteristics=_make_tax_chars(),
            tax_profile=TaxProfile(marginal_tax_rate=0.45),
        )
        result = score_opportunities(df, config)

        for _, row in result.iterrows():
            expected = (
                config.weight_underweight * row["underweight_score"]
                + config.weight_valuation * row["valuation_score"]
                + config.weight_tax_efficiency * row["tax_efficiency_score"]
            )
            ticker = row.get("ticker", "?")
            assert abs(row["opportunity_score"] - expected) < 1e-9, (
                f"Three-pillar blend mismatch for {ticker}: "
                f"got {row['opportunity_score']:.6f}, expected {expected:.6f}"
            )


# ---------------------------------------------------------------------------
# _resolve_account helper tests
# ---------------------------------------------------------------------------


class TestResolveAccount:
    def test_valid_account_types(self) -> None:
        row = pd.Series({"target_account": "TFSA"})
        assert _resolve_account(row) == AccountType.TFSA

    def test_case_insensitive(self) -> None:
        row = pd.Series({"target_account": "tfsa"})
        assert _resolve_account(row) == AccountType.TFSA

    def test_whitespace_tolerant(self) -> None:
        row = pd.Series({"target_account": "  RRSP  "})
        assert _resolve_account(row) == AccountType.RRSP

    def test_invalid_returns_none(self) -> None:
        row = pd.Series({"target_account": "INVALID"})
        assert _resolve_account(row) is None

    def test_nan_returns_none(self) -> None:
        row = pd.Series({"target_account": np.nan})
        assert _resolve_account(row) is None

    def test_none_returns_none(self) -> None:
        row = pd.Series({"target_account": None})
        assert _resolve_account(row) is None


# ---------------------------------------------------------------------------
# _append_overflow_row unit test
# ---------------------------------------------------------------------------


class TestAppendOverflowRow:
    def test_appended_row_has_non_reg_target(self) -> None:
        """The appended row must target NON_REG and have recalculated score."""
        chars = _make_tax_chars()
        tax_profile = TaxProfile(marginal_tax_rate=0.45)
        engine = AssetLocationEngine(tax_profile)

        df = _make_scored_df(tax_chars=chars, tax_profile=tax_profile)
        source_row = df.iloc[0].copy()
        original_len = len(df)

        _append_overflow_row(
            df,
            new_idx=original_len,
            source_row=source_row,
            overflow_amount=250.0,
            ticker="VFV",
            config=AllocationConfig(
                tax_profile=tax_profile,
                asset_characteristics=chars,
            ),
            engine=engine,
            chars=chars,
        )

        assert len(df) == original_len + 1
        new_row = df.iloc[-1]
        assert new_row["target_account"] == AccountType.NON_REG.value
        score = new_row.get("opportunity_score")
        assert score is not None and not (isinstance(score, float) and np.isnan(score))
