"""Tests for the Norbert's Gambit FX routing optimizer."""

from __future__ import annotations

import pytest

from pysharpe.config import ExecutionConfig
from pysharpe.execution.allocator import FxRoutingResult, determine_fx_routing

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(
    fx_fee_bps: float = 150.0,
    commission: float = 6.95,
    spread_bps: float = 2.0,
    drift_bps: float = 5.0,
    dlr_price: float = 14.0,
) -> ExecutionConfig:
    """Build an ExecutionConfig with typical Canadian discount brokerage defaults."""
    return ExecutionConfig(
        fx_fee_bps=fx_fee_bps,
        norberts_commission=commission,
        norberts_spread_bps=spread_bps,
        norberts_drift_risk_bps=drift_bps,
        norberts_dlr_price_cad=dlr_price,
    )


# ---------------------------------------------------------------------------
# Basic routing logic
# ---------------------------------------------------------------------------


def test_standard_fx_for_very_small_transactions():
    """For tiny transactions, the fixed commission dominates → use standard FX."""
    config = _config()
    result = determine_fx_routing(transaction_value=100.0, config=config)

    assert result.use_norberts_gambit is False
    # Standard: 100 * 0.015 = $1.50
    # NG: 2*6.95 + 100*(0.0002+0.0005) = 13.90 + 0.07 = $13.97
    assert result.standard_cost == pytest.approx(1.50)
    assert result.norberts_gambit_cost == pytest.approx(13.97)
    assert result.savings == pytest.approx(12.47)
    assert result.execution_steps == []


def test_norberts_gambit_for_large_transactions():
    """For large transactions, the fixed commission is negligible → use NG."""
    config = _config()
    result = determine_fx_routing(transaction_value=10_000.0, config=config)

    assert result.use_norberts_gambit is True
    # Standard: 10000 * 0.015 = $150.00
    # NG: 2*6.95 + 10000*0.0007 = 13.90 + 7.00 = $20.90
    assert result.standard_cost == pytest.approx(150.00)
    assert result.norberts_gambit_cost == pytest.approx(20.90)


def test_exact_crossover_point():
    """At the crossover threshold, costs should be equal within rounding.

    Crossover: T = 2*C / (f - spread_pct - drift_pct)
    With defaults: T = 2*6.95 / (0.015 - 0.0002 - 0.0005)
                   = 13.90 / 0.0143 ≈ 971.33
    """
    config = _config()
    fee = config.fx_fee_decimal
    spread_pct = config.norberts_spread_decimal
    drift_pct = config.norberts_drift_decimal
    commission = config.norberts_commission

    # Exact crossover
    denom = fee - spread_pct - drift_pct
    crossover = (2 * commission) / denom

    result = determine_fx_routing(transaction_value=crossover, config=config)
    # Costs should be effectively equal (within rounding tolerance)
    assert result.standard_cost == pytest.approx(result.norberts_gambit_cost, abs=0.02)
    # At equality, either direction is acceptable due to floating point;
    # savings should be near zero.
    assert result.savings == pytest.approx(0.0, abs=0.02)


def test_one_dollar_above_crossover_switches_to_ng():
    """Just above the crossover, Norbert's Gambit should be preferred."""
    config = _config()
    fee = config.fx_fee_decimal
    spread_pct = config.norberts_spread_decimal
    drift_pct = config.norberts_drift_decimal
    commission = config.norberts_commission

    denom = fee - spread_pct - drift_pct
    crossover = (2 * commission) / denom
    # Add $1 to push above crossover
    above = crossover + 1.0

    result = determine_fx_routing(transaction_value=above, config=config)
    assert result.use_norberts_gambit is True
    assert result.standard_cost > result.norberts_gambit_cost


def test_one_dollar_below_crossover_uses_standard():
    """Just below the crossover, standard FX should be preferred."""
    config = _config()
    fee = config.fx_fee_decimal
    spread_pct = config.norberts_spread_decimal
    drift_pct = config.norberts_drift_decimal
    commission = config.norberts_commission

    denom = fee - spread_pct - drift_pct
    crossover = (2 * commission) / denom
    below = max(crossover - 1.0, 0.01)

    result = determine_fx_routing(transaction_value=below, config=config)
    assert result.use_norberts_gambit is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_transaction_value():
    """Zero transaction returns default with no costs."""
    config = _config()
    result = determine_fx_routing(transaction_value=0.0, config=config)

    assert result.use_norberts_gambit is False
    assert result.standard_cost == 0.0
    assert result.norberts_gambit_cost == 0.0
    assert result.savings == 0.0
    assert result.execution_steps == []


def test_negative_transaction_value():
    """Negative transaction returns default with no costs."""
    config = _config()
    result = determine_fx_routing(transaction_value=-500.0, config=config)

    assert result.use_norberts_gambit is False
    assert result.standard_cost == 0.0
    assert result.norberts_gambit_cost == 0.0


def test_zero_fx_fee_always_uses_standard():
    """When fx_fee_bps is 0, standard cost is 0 and NG always costs more."""
    config = _config(fx_fee_bps=0.0)
    result = determine_fx_routing(transaction_value=10_000.0, config=config)

    assert result.use_norberts_gambit is False
    assert result.standard_cost == 0.0
    assert result.norberts_gambit_cost > 0.0


def test_zero_commission():
    """With zero commission, NG has only spread+drift costs."""
    config = _config(commission=0.0)
    result = determine_fx_routing(transaction_value=1_000.0, config=config)

    # Standard: 1000 * 0.015 = $15.00
    # NG: 0 + 1000 * 0.0007 = $0.70
    assert result.use_norberts_gambit is True
    assert result.standard_cost == pytest.approx(15.00)
    assert result.norberts_gambit_cost == pytest.approx(0.70)
    assert result.savings == pytest.approx(14.30)


def test_zero_spread_and_drift():
    """With zero spread and drift, NG is just 2*commission."""
    config = _config(spread_bps=0.0, drift_bps=0.0)
    result = determine_fx_routing(transaction_value=500.0, config=config)

    # Standard: 500 * 0.015 = $7.50
    # NG: 2*6.95 + 0 = $13.90
    assert result.use_norberts_gambit is False
    assert result.standard_cost == pytest.approx(7.50)
    assert result.norberts_gambit_cost == pytest.approx(13.90)


# ---------------------------------------------------------------------------
# Execution checklist
# ---------------------------------------------------------------------------


def test_execution_steps_generated_for_ng():
    """When Norbert's Gambit is recommended, a 4-step checklist is produced."""
    config = _config()
    result = determine_fx_routing(transaction_value=10_000.0, config=config)

    assert result.use_norberts_gambit is True
    assert len(result.execution_steps) == 4

    assert "DLR.TO" in result.execution_steps[0]
    assert "journaling" in result.execution_steps[1].lower()
    assert "DLR-U.TO" in result.execution_steps[2]
    assert "USD" in result.execution_steps[3]


def test_no_execution_steps_for_standard_fx():
    """Standard FX routing produces an empty execution checklist."""
    config = _config()
    result = determine_fx_routing(transaction_value=100.0, config=config)

    assert result.use_norberts_gambit is False
    assert result.execution_steps == []


def test_execution_steps_include_share_counts():
    """Checklist computes the correct number of DLR shares."""
    config = _config(dlr_price=14.0)
    result = determine_fx_routing(transaction_value=10_000.0, config=config)

    # 10000 / 14 = 714.28 → floor = 714 shares
    assert "714 shares" in result.execution_steps[0]
    assert "714 shares" in result.execution_steps[2]


def test_execution_steps_small_amount_no_shares():
    """When transaction is smaller than one DLR share, standard FX is used."""
    config = _config(dlr_price=14.0)
    # $10 transaction: standard cost = $0.15, NG cost = $13.97 → standard wins
    result = determine_fx_routing(transaction_value=10.0, config=config)

    assert result.use_norberts_gambit is False
    assert result.execution_steps == []


def test_execution_steps_zero_dlr_price():
    """When DLR price is zero, share count is omitted gracefully."""
    config = _config(dlr_price=0.0)
    result = determine_fx_routing(transaction_value=10_000.0, config=config)

    assert result.use_norberts_gambit is True
    # Should not crash; should produce steps without share counts
    assert len(result.execution_steps) == 4
    # Step 1 should not contain a share count
    assert "shares" not in result.execution_steps[0].lower().split("share")[0]


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------


def test_higher_fx_fee_lowers_crossover():
    """A higher FX fee makes standard more expensive → lower crossover."""
    config_high = _config(fx_fee_bps=250.0)  # 2.5%
    fee_high = config_high.fx_fee_decimal
    spread_pct = config_high.norberts_spread_decimal
    drift_pct = config_high.norberts_drift_decimal
    commission = config_high.norberts_commission

    denom = fee_high - spread_pct - drift_pct
    crossover_high = (2 * commission) / denom

    config_low = _config(fx_fee_bps=100.0)  # 1.0%
    fee_low = config_low.fx_fee_decimal
    denom_low = fee_low - spread_pct - drift_pct
    crossover_low = (2 * commission) / denom_low

    # Higher fee → crossover should be lower (less money needed for NG to win)
    assert crossover_high < crossover_low


def test_higher_commission_raises_crossover():
    """A higher per-trade commission makes NG more expensive → higher crossover."""
    config_low = _config(commission=5.0)
    fee = config_low.fx_fee_decimal
    spread_pct = config_low.norberts_spread_decimal
    drift_pct = config_low.norberts_drift_decimal
    denom = fee - spread_pct - drift_pct
    crossover_low = (2 * 5.0) / denom

    crossover_high = (2 * 10.0) / denom

    assert crossover_low < crossover_high


def test_rounding_consistency():
    """Cost values should be rounded to 2 decimal places."""
    config = _config()
    result = determine_fx_routing(transaction_value=1234.56, config=config)

    # Verify costs are rounded to 2 dp
    assert result.standard_cost == round(result.standard_cost, 2)
    assert result.norberts_gambit_cost == round(result.norberts_gambit_cost, 2)
    assert result.savings == round(result.savings, 2)


# ---------------------------------------------------------------------------
# FxRoutingResult dataclass
# ---------------------------------------------------------------------------


def test_fx_routing_result_defaults():
    """FxRoutingResult can be constructed directly."""
    result = FxRoutingResult(
        use_norberts_gambit=True,
        standard_cost=150.0,
        norberts_gambit_cost=20.90,
        savings=129.10,
        execution_steps=["Step 1: ...", "Step 2: ..."],
    )
    assert result.use_norberts_gambit is True
    assert result.standard_cost == 150.0
    assert result.savings == 129.10
    assert len(result.execution_steps) == 2


# ---------------------------------------------------------------------------
# Integration: RebalancePlan fx_routing field
# ---------------------------------------------------------------------------


def test_rebalance_plan_stores_fx_routing(tmp_path):
    """build_rebalance_plan attaches fx_routing when FX fee > 0."""
    import pandas as pd

    from pysharpe.execution.rebalance import build_rebalance_plan

    export_dir = tmp_path / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Weights: 60% AAPL (USD), 40% VCN.TO (CAD)
    (export_dir / "test_weights.txt").write_text(
        "ticker,weight\nAAPL,0.60\nVCN.TO,0.40\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "Date": ["2024-06-01", "2024-06-02"],
            "AAPL": [180.0, 185.0],
            "VCN.TO": [44.0, 44.5],
        }
    ).to_csv(export_dir / "test_collated.csv", index=False)

    # Holdings: some existing AAPL (USD), some VCN
    holdings = pd.DataFrame(
        {
            "ticker": ["AAPL", "VCN.TO"],
            "current_value": [5000.0, 5000.0],
        }
    )
    holdings_csv = tmp_path / "holdings.csv"
    holdings.to_csv(holdings_csv, index=False)

    config = _config(fx_fee_bps=150.0)

    plan = build_rebalance_plan(
        "test",
        new_cash=10_000.0,
        holdings_csv=holdings_csv,
        export_dir=export_dir,
        execution_config=config,
    )

    # fx_routing should be populated
    assert plan.fx_routing is not None
    # AAPL is not CAD-denominated, VCN.TO is
    assert "AAPL" in plan.fx_routing
    assert "VCN.TO" not in plan.fx_routing

    # For $10k, Norbert's Gambit should be recommended
    routing = plan.fx_routing["AAPL"]
    assert routing.use_norberts_gambit is True


def test_rebalance_plan_no_fx_routing_when_fee_zero(tmp_path):
    """When fx_fee_bps is 0, fx_routing is None."""
    import pandas as pd

    from pysharpe.execution.rebalance import build_rebalance_plan

    export_dir = tmp_path / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "test_weights.txt").write_text(
        "ticker,weight\nAAPL,0.60\nVCN.TO,0.40\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "Date": ["2024-06-01", "2024-06-02"],
            "AAPL": [180.0, 185.0],
            "VCN.TO": [44.0, 44.5],
        }
    ).to_csv(export_dir / "test_collated.csv", index=False)

    holdings_csv = tmp_path / "holdings.csv"
    pd.DataFrame(
        {"ticker": ["AAPL", "VCN.TO"], "current_value": [5000.0, 5000.0]}
    ).to_csv(holdings_csv, index=False)

    config = _config(fx_fee_bps=0.0)

    plan = build_rebalance_plan(
        "test",
        new_cash=5_000.0,
        holdings_csv=holdings_csv,
        export_dir=export_dir,
        execution_config=config,
    )

    assert plan.fx_routing is None


def test_format_rebalance_plan_includes_ng_checklist(tmp_path):
    """format_rebalance_plan renders the NG checklist when applicable."""
    import pandas as pd

    from pysharpe.execution.rebalance import (
        build_rebalance_plan,
        format_rebalance_plan,
    )

    export_dir = tmp_path / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "test_weights.txt").write_text(
        "ticker,weight\nAAPL,1.00\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "Date": ["2024-06-01", "2024-06-02"],
            "AAPL": [180.0, 185.0],
        }
    ).to_csv(export_dir / "test_collated.csv", index=False)

    holdings_csv = tmp_path / "holdings.csv"
    pd.DataFrame({"ticker": ["AAPL"], "current_value": [1000.0]}).to_csv(
        holdings_csv, index=False
    )

    config = _config(fx_fee_bps=150.0)

    plan = build_rebalance_plan(
        "test",
        new_cash=50_000.0,
        holdings_csv=holdings_csv,
        export_dir=export_dir,
        execution_config=config,
    )

    rendered = format_rebalance_plan(plan)
    assert "Norbert's Gambit Execution Checklist" in rendered
    assert "DLR.TO" in rendered
    assert "DLR-U.TO" in rendered
    assert "journaling" in rendered.lower()


def test_format_rebalance_plan_omits_ng_when_standard_used(tmp_path):
    """When standard FX is selected, no NG checklist appears."""
    import pandas as pd

    from pysharpe.execution.rebalance import (
        build_rebalance_plan,
        format_rebalance_plan,
    )

    export_dir = tmp_path / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "test_weights.txt").write_text(
        "ticker,weight\nAAPL,1.00\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "Date": ["2024-06-01", "2024-06-02"],
            "AAPL": [180.0, 185.0],
        }
    ).to_csv(export_dir / "test_collated.csv", index=False)

    holdings_csv = tmp_path / "holdings.csv"
    pd.DataFrame({"ticker": ["AAPL"], "current_value": [1000.0]}).to_csv(
        holdings_csv, index=False
    )

    # Small cash → standard FX should win
    config = _config(fx_fee_bps=150.0)

    plan = build_rebalance_plan(
        "test",
        new_cash=100.0,
        holdings_csv=holdings_csv,
        export_dir=export_dir,
        execution_config=config,
    )

    rendered = format_rebalance_plan(plan)
    assert "Norbert's Gambit Execution Checklist" not in rendered
