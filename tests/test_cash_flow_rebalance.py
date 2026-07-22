"""Tests for ``pysharpe.execution.cash_flow_rebalance``.

.. note::

    All tests use synthetic data.  No network calls.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pysharpe.execution.cash_flow_rebalance import (
    CashFlowRebalanceResult,
    RebalanceConfig,
    _compute_drift_bps,
    _is_taxable,
    allocate_contribution_cash_flow,
    evaluate_taxable_rebalance,
)
from pysharpe.optimization.tax_location import AccountType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_balances() -> pd.DataFrame:
    """Two‑asset portfolio, 50/50, $100k total."""
    return pd.DataFrame(
        {
            "ticker": ["VFV", "VDY"],
            "account": ["NON_REG", "TFSA"],
            "current_value": [50_000.0, 50_000.0],
        }
    )


@pytest.fixture
def multi_account_balances() -> pd.DataFrame:
    """Three‑asset portfolio across multiple accounts, $100k total."""
    return pd.DataFrame(
        {
            "ticker": ["VFV", "VFV", "VDY", "QQC"],
            "account": ["NON_REG", "TFSA", "TFSA", "RRSP"],
            "current_value": [30_000.0, 20_000.0, 40_000.0, 10_000.0],
        }
    )


@pytest.fixture
def target_weights() -> dict[str, float]:
    return {"VFV": 0.50, "VDY": 0.40, "QQC": 0.10}


@pytest.fixture
def account_types() -> dict[str, str]:
    return {"VFV": "NON_REG", "VDY": "TFSA", "QQC": "RRSP"}


@pytest.fixture
def opportunity_scores() -> dict[str, float]:
    return {"VFV": 0.3, "VDY": 0.8, "QQC": 0.5}


# ---------------------------------------------------------------------------
# _compute_drift_bps
# ---------------------------------------------------------------------------


class TestComputeDriftBps:
    def test_overweight(self) -> None:
        assert _compute_drift_bps(0.52, 0.50) == pytest.approx(200.0)

    def test_underweight(self) -> None:
        assert _compute_drift_bps(0.48, 0.50) == pytest.approx(-200.0)

    def test_at_target(self) -> None:
        assert _compute_drift_bps(0.50, 0.50) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _is_taxable
# ---------------------------------------------------------------------------


class TestIsTaxable:
    def test_non_reg_enum(self) -> None:
        assert _is_taxable(AccountType.NON_REG)

    def test_non_reg_string(self) -> None:
        assert _is_taxable("NON_REG")

    def test_non_reg_alias(self) -> None:
        assert _is_taxable("taxable")
        assert _is_taxable("MARGIN")

    def test_tfsa_not_taxable(self) -> None:
        assert not _is_taxable(AccountType.TFSA)
        assert not _is_taxable("TFSA")

    def test_rrsp_not_taxable(self) -> None:
        assert not _is_taxable(AccountType.RRSP)
        assert not _is_taxable("RRSP")


# ---------------------------------------------------------------------------
# evaluate_taxable_rebalance
# ---------------------------------------------------------------------------


class TestEvaluateTaxableRebalance:
    """Guardrail threshold tests."""

    def test_no_sell_when_at_target(self) -> None:
        orders = evaluate_taxable_rebalance(
            current_weights={"A": 0.50, "B": 0.50},
            target_weights={"A": 0.50, "B": 0.50},
            account_types={"A": "NON_REG", "B": "TFSA"},
        )
        assert len(orders) == 0

    def test_no_sell_when_underweight_non_reg(self) -> None:
        """Underweight in NON_REG should never trigger a sell."""
        orders = evaluate_taxable_rebalance(
            current_weights={"A": 0.48, "B": 0.52},
            target_weights={"A": 0.50, "B": 0.50},
            account_types={"A": "NON_REG", "B": "TFSA"},
        )
        assert len(orders) == 0

    def test_no_sell_tfsa_always(self) -> None:
        """TFSA overweights never trigger taxable sells."""
        orders = evaluate_taxable_rebalance(
            current_weights={"A": 0.45, "B": 0.55},
            target_weights={"A": 0.50, "B": 0.50},
            account_types={"A": "NON_REG", "B": "TFSA"},
        )
        # B is overweight in TFSA but TFSA is not taxable → no sell.
        assert len(orders) == 0

    def test_suppress_at_190_bps(self) -> None:
        """Drift at +190 bps is below the +200 bps threshold → no sell."""
        orders = evaluate_taxable_rebalance(
            current_weights={"A": 0.519, "B": 0.481},
            target_weights={"A": 0.50, "B": 0.50},
            account_types={"A": "NON_REG", "B": "TFSA"},
        )
        assert len(orders) == 0

    def test_suppress_at_exactly_200_bps(self) -> None:
        """Drift at exactly +200 bps should NOT trigger (strictly greater)."""
        # 0.5200 − 0.5000 = 0.0200 = 200 bps → not > 200 bps.
        orders = evaluate_taxable_rebalance(
            current_weights={"A": 0.520, "B": 0.480},
            target_weights={"A": 0.50, "B": 0.50},
            account_types={"A": "NON_REG", "B": "TFSA"},
        )
        assert len(orders) == 0

    def test_trigger_at_210_bps(self) -> None:
        """Drift at +210 bps exceeds threshold → sell down to +175 bps."""
        orders = evaluate_taxable_rebalance(
            current_weights={"A": 0.521, "B": 0.479},
            target_weights={"A": 0.50, "B": 0.50},
            account_types={"A": "NON_REG", "B": "TFSA"},
        )
        assert len(orders) == 1
        row = orders.iloc[0]
        assert row["ticker"] == "A"
        assert row["drift_bps"] == pytest.approx(210.0)
        # Sell weight: 210 − 175 = 35 bps = 0.0035
        assert row["sell_weight"] == pytest.approx(0.0035)

    def test_trigger_at_300_bps(self) -> None:
        """Larger drift → larger sell (down to +175 bps)."""
        orders = evaluate_taxable_rebalance(
            current_weights={"A": 0.530, "B": 0.470},
            target_weights={"A": 0.50, "B": 0.50},
            account_types={"A": "NON_REG", "B": "TFSA"},
        )
        assert len(orders) == 1
        row = orders.iloc[0]
        # 300 − 175 = 125 bps = 0.0125
        assert row["sell_weight"] == pytest.approx(0.0125)

    def test_sell_down_to_soft_ceiling_not_zero(self) -> None:
        """Verify the sell brings drift to +175 bps, not to 0."""
        target_w = 0.50
        current_w = 0.522  # +220 bps
        orders = evaluate_taxable_rebalance(
            current_weights={"A": current_w, "B": 1.0 - current_w},
            target_weights={"A": target_w, "B": 1.0 - target_w},
            account_types={"A": "NON_REG", "B": "TFSA"},
        )
        row = orders.iloc[0]
        # Post‑sell weight: 0.522 − 0.0045 = 0.5175 → drift = 0.0175 = 175 bps
        post_sell_w = current_w - row["sell_weight"]
        post_drift = (post_sell_w - target_w) * 10_000
        assert post_drift == pytest.approx(175.0, abs=0.1)

    def test_custom_thresholds(self) -> None:
        """Custom config overrides thresholds."""
        config = RebalanceConfig(upper_threshold_bps=100.0, soft_ceiling_bps=75.0)
        # Drift at +120 bps → above custom 100 bps threshold.
        orders = evaluate_taxable_rebalance(
            current_weights={"A": 0.512, "B": 0.488},
            target_weights={"A": 0.50, "B": 0.50},
            account_types={"A": "NON_REG", "B": "TFSA"},
            config=config,
        )
        assert len(orders) == 1
        # 120 − 75 = 45 bps = 0.0045
        assert orders.iloc[0]["sell_weight"] == pytest.approx(0.0045)

    def test_rejects_invalid_config_thresholds(self) -> None:
        with pytest.raises(ValueError, match="strictly less"):
            RebalanceConfig(upper_threshold_bps=100.0, soft_ceiling_bps=100.0)

    def test_with_absolute_dollar_values(self) -> None:
        """When current_values is provided, sell_amount is populated."""
        orders = evaluate_taxable_rebalance(
            current_weights={"VFV": 0.522, "VDY": 0.478},
            target_weights={"VFV": 0.50, "VDY": 0.50},
            account_types={"VFV": "NON_REG", "VDY": "TFSA"},
            current_values={"VFV": {"NON_REG": 52_200.0}, "VDY": {"TFSA": 47_800.0}},
        )
        assert len(orders) == 1
        assert not pd.isna(orders.iloc[0]["sell_amount"])
        assert orders.iloc[0]["sell_amount"] > 0

    def test_multi_account_holding(self) -> None:
        """Asset held in both NON_REG and TFSA — only NON_REG triggers sell."""
        orders = evaluate_taxable_rebalance(
            current_weights={"VFV": 0.55, "VDY": 0.45},
            target_weights={"VFV": 0.50, "VDY": 0.50},
            account_types={"VFV": ["NON_REG", "TFSA"], "VDY": "TFSA"},
        )
        assert len(orders) == 1
        assert orders.iloc[0]["account"] == "NON_REG"


# ---------------------------------------------------------------------------
# allocate_contribution_cash_flow — basic properties
# ---------------------------------------------------------------------------


class TestAllocateContributionBasic:
    """Basic correctness of ``allocate_contribution_cash_flow``."""

    def test_equal_targets_no_drift(self, simple_balances: pd.DataFrame) -> None:
        """Equal current → equal allocation."""
        result = allocate_contribution_cash_flow(
            cash_amount=10_000.0,
            current_balances=simple_balances,
            target_weights={"VFV": 0.50, "VDY": 0.50},
            account_types={"VFV": "NON_REG", "VDY": "TFSA"},
            opportunity_scores={"VFV": 0.5, "VDY": 0.5},
        )
        buys = result.buy_orders
        assert buys["buy_amount"].sum() == pytest.approx(10_000.0, abs=0.01)

    def test_buy_amounts_non_negative(self, simple_balances: pd.DataFrame) -> None:
        result = allocate_contribution_cash_flow(
            cash_amount=5_000.0,
            current_balances=simple_balances,
            target_weights={"VFV": 0.50, "VDY": 0.50},
            account_types={"VFV": "NON_REG", "VDY": "TFSA"},
            opportunity_scores={"VFV": 0.5, "VDY": 0.5},
        )
        assert (result.buy_orders["buy_amount"] >= 0).all()

    def test_correct_accounts_assigned(self, simple_balances: pd.DataFrame) -> None:
        result = allocate_contribution_cash_flow(
            cash_amount=10_000.0,
            current_balances=simple_balances,
            target_weights={"VFV": 0.50, "VDY": 0.50},
            account_types={"VFV": "NON_REG", "VDY": "TFSA"},
            opportunity_scores={"VFV": 0.5, "VDY": 0.5},
        )
        accts = set(result.buy_orders["account"])
        assert "NON_REG" in accts or "TFSA" in accts

    def test_cash_remaining_non_negative(self, simple_balances: pd.DataFrame) -> None:
        result = allocate_contribution_cash_flow(
            cash_amount=10_000.0,
            current_balances=simple_balances,
            target_weights={"VFV": 0.50, "VDY": 0.50},
            account_types={"VFV": "NON_REG", "VDY": "TFSA"},
            opportunity_scores={"VFV": 0.5, "VDY": 0.5},
        )
        assert result.cash_remaining >= 0.0

    def test_new_weights_are_sensible(self, simple_balances: pd.DataFrame) -> None:
        result = allocate_contribution_cash_flow(
            cash_amount=10_000.0,
            current_balances=simple_balances,
            target_weights={"VFV": 0.50, "VDY": 0.50},
            account_types={"VFV": "NON_REG", "VDY": "TFSA"},
            opportunity_scores={"VFV": 0.5, "VDY": 0.5},
        )
        new_weights = result.buy_orders.set_index("ticker")["new_weight"]
        assert sum(new_weights) == pytest.approx(1.0, abs=0.01)

    def test_rejects_non_positive_cash(self, simple_balances: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="positive"):
            allocate_contribution_cash_flow(
                cash_amount=0.0,
                current_balances=simple_balances,
                target_weights={"VFV": 0.50},
                account_types={"VFV": "NON_REG"},
                opportunity_scores={"VFV": 0.5},
            )

    def test_rejects_missing_columns(self) -> None:
        bad = pd.DataFrame({"ticker": ["A"], "current_value": [100.0]})
        with pytest.raises(ValueError, match="must have columns"):
            allocate_contribution_cash_flow(
                cash_amount=100.0,
                current_balances=bad,
                target_weights={"A": 1.0},
                account_types={"A": "NON_REG"},
                opportunity_scores={"A": 0.5},
            )

    def test_empty_portfolio(self) -> None:
        """Empty portfolio → distribute according to target weights."""
        empty = pd.DataFrame(columns=["ticker", "account", "current_value"])
        result = allocate_contribution_cash_flow(
            cash_amount=10_000.0,
            current_balances=empty,
            target_weights={"VFV": 0.50, "VDY": 0.50},
            account_types={"VFV": "NON_REG", "VDY": "TFSA"},
            opportunity_scores={"VFV": 0.5, "VDY": 0.5},
        )
        assert len(result.buy_orders) == 2

    def test_deterministic(self, multi_account_balances: pd.DataFrame) -> None:
        """Same inputs → same outputs."""
        kwargs = dict(
            cash_amount=8_000.0,
            current_balances=multi_account_balances,
            target_weights={"VFV": 0.50, "VDY": 0.40, "QQC": 0.10},
            account_types={"VFV": "NON_REG", "VDY": "TFSA", "QQC": "RRSP"},
            opportunity_scores={"VFV": 0.3, "VDY": 0.8, "QQC": 0.5},
        )
        r1 = allocate_contribution_cash_flow(**kwargs)
        r2 = allocate_contribution_cash_flow(**kwargs)
        pd.testing.assert_frame_equal(r1.buy_orders, r2.buy_orders)


# ---------------------------------------------------------------------------
# allocate_contribution_cash_flow — cash shortfall routing
# ---------------------------------------------------------------------------


class TestCashShortfallRouting:
    """Verify that when cash is insufficient, it routes to underweight assets."""

    def test_only_underweight_assets_receive_cash(
        self,
    ) -> None:
        """VDY is at or above target → should get $0."""
        # VFV: 50k, VDY: 42k, QQC: 8k → total = 100k.
        # After 5k cash: total = 105k.
        # VFV target: 52.5k, current: 50k → shortfall 2.5k.
        # VDY target: 42.0k, current: 42k → at target.
        # QQC target: 10.5k, current: 8k → shortfall 2.5k.
        balances = pd.DataFrame(
            {
                "ticker": ["VFV", "VDY", "QQC"],
                "account": ["NON_REG", "TFSA", "RRSP"],
                "current_value": [50_000.0, 42_000.0, 8_000.0],
            }
        )
        result = allocate_contribution_cash_flow(
            cash_amount=5_000.0,
            current_balances=balances,
            target_weights={"VFV": 0.50, "VDY": 0.40, "QQC": 0.10},
            account_types={"VFV": "NON_REG", "VDY": "TFSA", "QQC": "RRSP"},
            opportunity_scores={"VFV": 0.3, "VDY": 0.8, "QQC": 0.5},
        )
        vdy_buys = result.buy_orders[result.buy_orders["ticker"] == "VDY"]
        assert vdy_buys["buy_amount"].sum() == pytest.approx(0.0, abs=0.01), (
            "VDY is not underweight and should not receive cash"
        )

    def test_cash_shortfall_goes_to_underweight_only(
        self,
    ) -> None:
        """With limited cash, only underweight tickers get buys."""
        # VFV: 40k, VDY: 45k, QQC: 15k → total = 100k.
        # After 2k cash: total = 102k.
        # VFV target: 51k, current: 40k → shortfall 11k (underweight).
        # VDY target: 40.8k, current: 45k → surplus (overweight).
        # QQC target: 10.2k, current: 15k → surplus (overweight).
        balances = pd.DataFrame(
            {
                "ticker": ["VFV", "VDY", "QQC"],
                "account": ["NON_REG", "TFSA", "RRSP"],
                "current_value": [40_000.0, 45_000.0, 15_000.0],
            }
        )
        result = allocate_contribution_cash_flow(
            cash_amount=2_000.0,
            current_balances=balances,
            target_weights={"VFV": 0.50, "VDY": 0.40, "QQC": 0.10},
            account_types={"VFV": "NON_REG", "VDY": "TFSA", "QQC": "RRSP"},
            opportunity_scores={"VFV": 0.3, "VDY": 0.8, "QQC": 0.5},
        )
        tickers_with_buys = set(
            result.buy_orders[result.buy_orders["buy_amount"] > 0]["ticker"]
        )
        # VDY and QQC are overweight → should not receive buys.
        assert "VDY" not in tickers_with_buys, (
            f"VDY should not receive cash in shortfall: {tickers_with_buys}"
        )
        assert "QQC" not in tickers_with_buys, (
            f"QQC should not receive cash in shortfall: {tickers_with_buys}"
        )

    def test_cash_fully_allocated_in_shortfall(
        self,
        multi_account_balances: pd.DataFrame,
    ) -> None:
        """All cash should be deployed even in shortfall."""
        result = allocate_contribution_cash_flow(
            cash_amount=3_000.0,
            current_balances=multi_account_balances,
            target_weights={"VFV": 0.50, "VDY": 0.40, "QQC": 0.10},
            account_types={"VFV": "NON_REG", "VDY": "TFSA", "QQC": "RRSP"},
            opportunity_scores={"VFV": 0.3, "VDY": 0.8, "QQC": 0.5},
        )
        total_allocated = result.buy_orders["buy_amount"].sum()
        assert total_allocated + result.cash_remaining == pytest.approx(
            3_000.0, abs=0.01
        )

    def test_sufficient_cash_covers_all_underweights(
        self,
        multi_account_balances: pd.DataFrame,
    ) -> None:
        """With enough cash, all underweights are filled to target."""
        result = allocate_contribution_cash_flow(
            cash_amount=100_000.0,  # More than enough
            current_balances=multi_account_balances,
            target_weights={"VFV": 0.50, "VDY": 0.40, "QQC": 0.10},
            account_types={"VFV": "NON_REG", "VDY": "TFSA", "QQC": "RRSP"},
            opportunity_scores={"VFV": 0.3, "VDY": 0.8, "QQC": 0.5},
        )
        buys = result.buy_orders
        # All three tickers should appear.
        tickers = set(buys["ticker"])
        assert tickers == {"VFV", "VDY", "QQC"}


# ---------------------------------------------------------------------------
# allocate_contribution_cash_flow — integration with guardrail
# ---------------------------------------------------------------------------


class TestIntegrationWithGuardrail:
    """End‑to‑end tests combining allocation and guardrail evaluation."""

    def test_taxable_sell_triggered_post_buy(self) -> None:
        """After buying, an existing overweight Non‑Reg asset may breach guardrail."""
        balances = pd.DataFrame(
            {
                "ticker": ["VFV", "VDY"],
                "account": ["NON_REG", "TFSA"],
                "current_value": [55_000.0, 45_000.0],  # VFV at 55% (overweight)
            }
        )
        result = allocate_contribution_cash_flow(
            cash_amount=10_000.0,
            current_balances=balances,
            target_weights={"VFV": 0.50, "VDY": 0.50},
            account_types={"VFV": "NON_REG", "VDY": "TFSA"},
            opportunity_scores={"VFV": 0.5, "VDY": 0.5},
        )
        # VFV at 55% = +500 bps → should trigger sell.
        assert result.taxable_sell_triggered
        assert len(result.sell_orders) > 0

    def test_no_sell_when_below_threshold(self) -> None:
        """At exactly 200 bps, no taxable sell is triggered."""
        balances = pd.DataFrame(
            {
                "ticker": ["VFV", "VDY"],
                "account": ["NON_REG", "TFSA"],
                "current_value": [52_000.0, 48_000.0],  # VFV at 52% = +200 bps
            }
        )
        result = allocate_contribution_cash_flow(
            cash_amount=1.0,  # Minimal cash to satisfy validation
            current_balances=balances,
            target_weights={"VFV": 0.50, "VDY": 0.50},
            account_types={"VFV": "NON_REG", "VDY": "TFSA"},
            opportunity_scores={"VFV": 0.5, "VDY": 0.5},
        )
        # VFV is at 52% = 200 bps above 50% → not strictly > 200.
        assert not result.taxable_sell_triggered


# ---------------------------------------------------------------------------
# RebalanceConfig
# ---------------------------------------------------------------------------


class TestRebalanceConfig:
    def test_defaults(self) -> None:
        cfg = RebalanceConfig()
        assert cfg.upper_threshold_bps == 200.0
        assert cfg.soft_ceiling_bps == 175.0
        assert cfg.min_allocation_dollars == 1.0

    def test_custom_thresholds(self) -> None:
        cfg = RebalanceConfig(upper_threshold_bps=150.0, soft_ceiling_bps=125.0)
        assert cfg.upper_threshold_bps == 150.0
        assert cfg.soft_ceiling_bps == 125.0

    def test_ceiling_must_be_below_threshold(self) -> None:
        with pytest.raises(ValueError):
            RebalanceConfig(upper_threshold_bps=100.0, soft_ceiling_bps=110.0)


# ---------------------------------------------------------------------------
# CashFlowRebalanceResult
# ---------------------------------------------------------------------------


class TestResultDataclass:
    def test_empty_result(self) -> None:
        r = CashFlowRebalanceResult(
            buy_orders=pd.DataFrame(),
            sell_orders=pd.DataFrame(),
        )
        assert len(r.buy_orders) == 0
        assert len(r.sell_orders) == 0
        assert not r.taxable_sell_triggered

    def test_sell_triggered_flag(self) -> None:
        r = CashFlowRebalanceResult(
            buy_orders=pd.DataFrame(),
            sell_orders=pd.DataFrame({"a": [1]}),
        )
        assert r.taxable_sell_triggered
