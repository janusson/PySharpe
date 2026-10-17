"""Synthetic, deterministic checks for the standalone momentum research slice."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pysharpe import MomentumConfig as PublicMomentumConfig
from pysharpe import run_cross_asset_momentum as public_run_cross_asset_momentum
from pysharpe.analysis.tactical_momentum import (
    MomentumConfig,
    run_cross_asset_momentum,
)
from pysharpe.exceptions import DataValidationError


def _monthly_prices(months: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-31", periods=months, freq="ME")
    t = np.arange(months, dtype=float)
    total_return = pd.DataFrame(
        {
            "A": 100.0 * 1.02**t,
            "B": 100.0 * 1.005**t,
            "VFV": 100.0 * 1.01**t,
        },
        index=dates,
    )
    execution = total_return.copy()
    return total_return, execution


def _config(**overrides: object) -> MomentumConfig:
    values: dict[str, object] = {
        "universe": ("A", "B"),
        "lookback_months": 3,
        "skip_months": 1,
        "top_n": 1,
        "benchmark_ticker": "VFV",
        "periods_per_year": 12,
    }
    values.update(overrides)
    return MomentumConfig(**values)


def test_long_only_momentum_ranks_and_exports_versioned_result() -> None:
    total_return, execution = _monthly_prices()

    result = run_cross_asset_momentum(
        total_return,
        execution,
        config=_config(),
    )

    assert not result.signals.empty
    assert set(result.signals["ticker"]) == {"A"}
    assert (result.signals["target_weight"] == 1.0).all()
    assert (result.weights.to_numpy() >= 0.0).all()
    assert result.weights.sum(axis=1).max() <= 1.0
    assert {"VFV", "CASH"}.issubset(result.benchmarks.columns)

    payload = json.loads(result.to_json())
    assert payload["schema_version"] == "1.0"
    assert payload["strategy"]["name"] == "cross_asset_momentum"
    assert payload["signals"]
    assert payload["data_quality"]["return_basis"] == "net_of_expenses"
    assert "sharpe_ratio" in payload["risk"]
    assert result.to_json() == result.to_json()
    assert PublicMomentumConfig is MomentumConfig
    assert public_run_cross_asset_momentum is run_cross_asset_momentum


def test_signal_history_does_not_change_when_future_observations_change() -> None:
    total_return, execution = _monthly_prices()
    cutoff = total_return.index[20]
    original = run_cross_asset_momentum(
        total_return,
        execution,
        config=_config(),
    )

    changed_total_return = total_return.copy()
    changed_total_return.loc[changed_total_return.index > cutoff, "A"] *= 4.0
    changed = run_cross_asset_momentum(
        changed_total_return,
        execution,
        config=_config(),
    )

    original_past = original.signals.loc[original.signals["signal_date"] <= cutoff]
    changed_past = changed.signals.loc[changed.signals["signal_date"] <= cutoff]
    pd.testing.assert_frame_equal(
        original_past.reset_index(drop=True), changed_past.reset_index(drop=True)
    )


def test_incomplete_listing_history_is_excluded_until_full_window_exists() -> None:
    total_return, execution = _monthly_prices()
    total_return.loc[total_return.index[:18], "B"] = np.nan
    execution.loc[execution.index[:18], "B"] = np.nan
    total_return.loc[total_return.index[-1], "B"] = (
        total_return.loc[total_return.index[-1], "A"] * 20.0
    )

    result = run_cross_asset_momentum(
        total_return,
        execution,
        config=_config(top_n=2),
    )

    b_signals = result.signals.loc[result.signals["ticker"] == "B"]
    assert b_signals.empty or b_signals["signal_date"].min() >= total_return.index[22]
    assert "B" in result.data_quality["assets_with_incomplete_history"]


def test_interior_price_gap_in_held_asset_is_rejected_as_stale_data() -> None:
    total_return, execution = _monthly_prices()
    total_return.iloc[20, total_return.columns.get_loc("A")] = np.nan

    with pytest.raises(DataValidationError, match="interior missing"):
        run_cross_asset_momentum(total_return, execution, config=_config())


def test_cad_fx_alignment_uses_only_prior_rates_and_rejects_uncovered_lead() -> None:
    total_return, execution = _monthly_prices()
    total_return["USD_ASSET"] = total_return["A"]
    execution["USD_ASSET"] = execution["A"]
    total_return["A"] = total_return["B"]
    execution["A"] = execution["B"]
    fx = pd.DataFrame(
        {"USD": [1.25] * 9 + [1.50] * (len(total_return) - 17)},
        index=total_return.index[8:],
    )
    config = _config(universe=("A", "USD_ASSET"), top_n=1, max_fx_staleness_days=40)

    result = run_cross_asset_momentum(
        total_return,
        execution,
        config=config,
        currencies={"USD_ASSET": "USD"},
        fx_to_cad=fx,
    )

    assert result.data_quality["fx_uncovered_observations"] > 0
    assert result.data_quality["fx_alignment"] == "backward_only"
    assert (
        not result.signals.loc[
            result.signals["signal_date"] < total_return.index[8], "ticker"
        ]
        .eq("USD_ASSET")
        .any()
    )


def test_trading_costs_reduce_nav_and_turnover_is_reported() -> None:
    total_return, execution = _monthly_prices()
    config = _config(
        fee_per_order=1.0,
        slippage_bps=10.0,
        spread_bps=20.0,
    )

    result = run_cross_asset_momentum(total_return, execution, config=config)
    first_trade = result.trades.iloc[0]

    assert first_trade["order_count"] == 1
    assert first_trade["gross_traded_notional"] == pytest.approx(10_000.0)
    assert first_trade["cost"] == pytest.approx(21.0)
    assert first_trade["turnover"] == pytest.approx(1.0)
    assert (result.backtest["portfolio_value"] < 10_000.0).any()


def test_rebalance_turnover_uses_weights_drifted_through_current_return() -> None:
    total_return, execution = _monthly_prices()
    result = run_cross_asset_momentum(
        total_return,
        execution,
        config=_config(top_n=2),
    )

    second_trade = result.trades.iloc[1]
    trade_position = total_return.index.get_loc(pd.Timestamp(second_trade["date"]))
    a_return = (
        total_return["A"].iloc[trade_position]
        / total_return["A"].iloc[trade_position - 1]
        - 1
    )
    b_return = (
        total_return["B"].iloc[trade_position]
        / total_return["B"].iloc[trade_position - 1]
        - 1
    )
    drifted_a_weight = (
        0.5 * (1.0 + a_return) / (0.5 * (1.0 + a_return) + 0.5 * (1.0 + b_return))
    )
    expected_turnover = 2.0 * abs(drifted_a_weight - 0.5)

    assert second_trade["turnover"] == pytest.approx(expected_turnover)
    assert second_trade["turnover"] > 0.0


def test_benchmark_must_cover_full_reported_strategy_period() -> None:
    total_return, execution = _monthly_prices()
    total_return.loc[total_return.index[:6], "VFV"] = np.nan
    execution.loc[execution.index[:6], "VFV"] = np.nan

    with pytest.raises(
        DataValidationError, match="benchmark must cover simulation start"
    ):
        run_cross_asset_momentum(total_return, execution, config=_config())


def test_fx_conversion_fee_applies_only_to_foreign_trade_notional() -> None:
    total_return, execution = _monthly_prices()
    total_return["USD_ASSET"] = total_return["A"]
    execution["USD_ASSET"] = execution["A"]
    fx = pd.DataFrame({"USD": 1.35}, index=total_return.index)
    config = _config(
        universe=("USD_ASSET",),
        fx_fee_bps=10.0,
    )

    result = run_cross_asset_momentum(
        total_return,
        execution,
        config=config,
        currencies={"USD_ASSET": "USD"},
        fx_to_cad=fx,
    )

    assert result.trades.iloc[0]["fx_converted_notional"] == pytest.approx(10_000.0)
    assert result.trades.iloc[0]["cost"] == pytest.approx(10.0)


def test_split_adjusted_execution_prices_do_not_replace_total_return_valuation() -> (
    None
):
    total_return, execution = _monthly_prices()
    baseline = run_cross_asset_momentum(total_return, execution, config=_config())
    split_date = execution.index[20]
    execution.loc[split_date:, "A"] /= 2.0

    result = run_cross_asset_momentum(total_return, execution, config=_config())

    # The executable close validates order dates but does not mark the portfolio.
    pd.testing.assert_series_equal(
        baseline.backtest["portfolio_value"], result.backtest["portfolio_value"]
    )
    assert result.data_quality["execution_price_basis"] == "split_adjusted_close"


def test_insufficient_signal_history_fails_with_actionable_error() -> None:
    total_return, execution = _monthly_prices(months=4)

    with pytest.raises(DataValidationError, match="insufficient history"):
        run_cross_asset_momentum(
            total_return,
            execution,
            config=_config(lookback_months=12),
        )
