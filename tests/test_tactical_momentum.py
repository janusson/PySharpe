"""Synthetic, deterministic checks for the standalone momentum research slice."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pysharpe import MarketDataProvenance as PublicMarketDataProvenance
from pysharpe import MomentumConfig as PublicMomentumConfig
from pysharpe import WalkForwardConfig as PublicWalkForwardConfig
from pysharpe import WalkForwardResearchResult as PublicWalkForwardResearchResult
from pysharpe import run_cross_asset_momentum as public_run_cross_asset_momentum
from pysharpe import run_walk_forward_momentum as public_run_walk_forward_momentum
from pysharpe.analysis import MarketDataProvenance as AnalysisMarketDataProvenance
from pysharpe.analysis import WalkForwardConfig as AnalysisWalkForwardConfig
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
    assert payload["schema_version"] == "1.1"
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


def test_momentum_ranking_uses_cad_converted_foreign_total_returns() -> None:
    total_return, execution = _monthly_prices()
    total_return["USD_ASSET"] = 100.0
    execution["USD_ASSET"] = 100.0
    fx = pd.DataFrame(
        {"USD": 1.0 * 1.015 ** np.arange(len(total_return))},
        index=total_return.index,
    )
    config = _config(
        universe=("USD_ASSET", "B"),
        top_n=1,
        max_fx_staleness_days=40,
    )

    result = run_cross_asset_momentum(
        total_return,
        execution,
        config=config,
        currencies={"USD_ASSET": "USD"},
        fx_to_cad=fx,
    )

    recent_signals = result.signals.loc[
        result.signals["signal_date"] >= total_return.index[-6], "ticker"
    ]
    assert not recent_signals.empty
    assert recent_signals.eq("USD_ASSET").all()


def test_future_fx_changes_do_not_rewrite_past_momentum_signals() -> None:
    total_return, execution = _monthly_prices()
    total_return["USD_ASSET"] = 100.0
    execution["USD_ASSET"] = 100.0
    fx = pd.DataFrame(
        {"USD": 1.0 * 1.015 ** np.arange(len(total_return))},
        index=total_return.index,
    )
    config = _config(
        universe=("USD_ASSET", "B"),
        top_n=1,
        max_fx_staleness_days=40,
    )
    cutoff = total_return.index[20]
    original = run_cross_asset_momentum(
        total_return,
        execution,
        config=config,
        currencies={"USD_ASSET": "USD"},
        fx_to_cad=fx,
    )
    changed_fx = fx.copy()
    changed_fx.loc[changed_fx.index > cutoff, "USD"] *= 2.0
    changed = run_cross_asset_momentum(
        total_return,
        execution,
        config=config,
        currencies={"USD_ASSET": "USD"},
        fx_to_cad=changed_fx,
    )

    original_past = original.signals.loc[original.signals["signal_date"] <= cutoff]
    changed_past = changed.signals.loc[changed.signals["signal_date"] <= cutoff]
    pd.testing.assert_frame_equal(
        original_past.reset_index(drop=True), changed_past.reset_index(drop=True)
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


def test_market_data_provenance_is_validated_and_exported() -> None:
    from pysharpe.analysis import tactical_momentum as tactical

    total_return, execution = _monthly_prices()
    provenance = {
        f"{panel}:{ticker}": tactical.MarketDataProvenance(
            source="synthetic_fixture",
            retrieved_at=total_return.index[-1],
            currency="CAD",
            price_basis=basis,
            max_staleness_days=3,
        )
        for panel, frame, basis in (
            (
                "total_return_prices",
                total_return,
                "adjusted_total_return_net_of_expenses",
            ),
            ("execution_prices", execution, "split_adjusted_close"),
        )
        for ticker in frame.columns
    }

    result = run_cross_asset_momentum(
        total_return,
        execution,
        config=_config(),
        provenance=provenance,
    )
    payload = json.loads(result.to_json())

    assert payload["schema_version"] == "1.1"
    assert payload["data_quality"]["provenance_complete"] is True
    assert payload["data_provenance"]["total_return_prices:A"]["source"] == (
        "synthetic_fixture"
    )
    assert payload["data_provenance"]["execution_prices:A"]["stale"] is False
    assert payload["cost_assumptions"]["fee_per_order"] == 0.0


def test_stale_or_mislabeled_market_data_is_rejected() -> None:
    from pysharpe.analysis import tactical_momentum as tactical

    total_return, execution = _monthly_prices()
    provenance = {
        f"{panel}:{ticker}": tactical.MarketDataProvenance(
            source="fixture",
            retrieved_at=total_return.index[-1] + pd.Timedelta(days=10),
            currency="CAD",
            price_basis=basis,
            max_staleness_days=3,
        )
        for panel, frame, basis in (
            (
                "total_return_prices",
                total_return,
                "adjusted_total_return_net_of_expenses",
            ),
            ("execution_prices", execution, "split_adjusted_close"),
        )
        for ticker in frame.columns
    }

    with pytest.raises(DataValidationError, match="stale market data"):
        run_cross_asset_momentum(
            total_return,
            execution,
            config=_config(),
            provenance=provenance,
        )

    provenance["total_return_prices:A"] = tactical.MarketDataProvenance(
        source="fixture",
        retrieved_at=total_return.index[-1],
        currency="CAD",
        price_basis="unknown_basis",
        max_staleness_days=3,
    )
    with pytest.raises(DataValidationError, match="price basis"):
        run_cross_asset_momentum(
            total_return,
            execution,
            config=_config(),
            provenance=provenance,
        )


def test_fx_provenance_declares_conversion_direction_and_quote_basis() -> None:
    from pysharpe.analysis import tactical_momentum as tactical

    total_return, execution = _monthly_prices()
    total_return["USD_ASSET"] = 100.0
    execution["USD_ASSET"] = 100.0
    fx = pd.DataFrame({"USD": 1.35}, index=total_return.index)
    currencies = {"USD_ASSET": "USD"}
    provenance: dict[str, tactical.MarketDataProvenance] = {}
    for panel, frame, basis in (
        ("total_return_prices", total_return, "adjusted_total_return_net_of_expenses"),
        ("execution_prices", execution, "split_adjusted_close"),
    ):
        for ticker in frame.columns:
            if ticker == "B":
                continue
            provenance[f"{panel}:{ticker}"] = tactical.MarketDataProvenance(
                source="synthetic_fixture",
                retrieved_at=total_return.index[-1],
                currency=currencies.get(ticker, "CAD"),
                price_basis=basis,
            )
    provenance["fx_to_cad:USD"] = tactical.MarketDataProvenance(
        source="synthetic_fx_fixture",
        retrieved_at=fx.index[-1],
        currency="CAD/USD",
        price_basis="CAD_per_currency_unit",
        max_staleness_days=40,
    )

    result = run_cross_asset_momentum(
        total_return,
        execution,
        config=_config(
            universe=("A", "USD_ASSET"),
            top_n=1,
            max_fx_staleness_days=40,
        ),
        currencies=currencies,
        fx_to_cad=fx,
        provenance=provenance,
    )

    fx_provenance = result.data_provenance["fx_to_cad:USD"]
    assert fx_provenance["currency"] == "CAD/USD"
    assert fx_provenance["price_basis"] == "CAD_per_currency_unit"
    assert result.data_quality["provenance_complete"] is True


def test_missing_provenance_is_reported_without_inventing_a_source() -> None:
    total_return, execution = _monthly_prices()

    result = run_cross_asset_momentum(total_return, execution, config=_config())

    assert result.data_quality["provenance_complete"] is False
    assert result.data_quality["signal_freshness_verified"] is False
    assert "total_return_prices:A" in result.data_quality["unattributed_series"]
    assert result.to_dict()["data_provenance"] == {}


def test_provenance_with_no_series_observations_is_not_verified() -> None:
    from pysharpe.analysis import tactical_momentum as tactical

    total_return, execution = _monthly_prices()
    total_return = total_return.drop(columns="B")
    execution = execution.drop(columns="B")
    total_return["EMPTY"] = np.nan
    execution["EMPTY"] = np.nan
    provenance: dict[str, tactical.MarketDataProvenance] = {}
    for panel, frame, basis in (
        (
            "total_return_prices",
            total_return,
            "adjusted_total_return_net_of_expenses",
        ),
        ("execution_prices", execution, "split_adjusted_close"),
    ):
        for ticker in frame.columns:
            provenance[f"{panel}:{ticker}"] = tactical.MarketDataProvenance(
                source="synthetic_fixture",
                retrieved_at=total_return.index[-1],
                currency="CAD",
                price_basis=basis,
            )

    result = run_cross_asset_momentum(
        total_return,
        execution,
        config=_config(universe=("A", "EMPTY")),
        provenance=provenance,
    )

    assert result.data_quality["provenance_complete"] is False
    assert result.data_quality["signal_freshness_verified"] is False
    assert result.data_quality["unverified_series"] == [
        "execution_prices:EMPTY",
        "total_return_prices:EMPTY",
    ]


def test_walk_forward_uses_chronological_fixed_configuration_test_windows() -> None:
    from pysharpe.analysis import tactical_momentum as tactical

    total_return, execution = _monthly_prices(months=30)
    result = tactical.run_walk_forward_momentum(
        total_return,
        execution,
        strategy_config=_config(),
        walk_forward_config=tactical.WalkForwardConfig(
            training_months=8,
            test_months=6,
        ),
    )

    folds = result.to_dict()["folds"]
    assert len(folds) == 4
    assert folds[0]["test_start"] > folds[0]["training_end"]
    assert folds[0]["test_end"] < folds[1]["test_start"]
    assert all(fold["test_returns"] for fold in folds)
    assert all("VFV" in fold["benchmark_risk"] for fold in folds)
    assert result.to_dict()["strategy"]["config"]["top_n"] == 1
    assert result.to_dict()["cost_assumptions"]["fee_per_order"] == 0.0
    assert result.to_json() == result.to_json()


def test_walk_forward_fold_results_do_not_depend_on_later_prices() -> None:
    from pysharpe.analysis import tactical_momentum as tactical

    total_return, execution = _monthly_prices(months=30)
    walk_forward_config = tactical.WalkForwardConfig(
        training_months=8,
        test_months=6,
    )
    original = tactical.run_walk_forward_momentum(
        total_return,
        execution,
        strategy_config=_config(),
        walk_forward_config=walk_forward_config,
    )
    cutoff = pd.Timestamp(original.folds[0]["test_end"])
    changed = total_return.copy()
    changed.loc[changed.index > cutoff, "A"] *= 2.0
    modified = tactical.run_walk_forward_momentum(
        changed,
        execution,
        strategy_config=_config(),
        walk_forward_config=walk_forward_config,
    )

    assert original.folds[0] == modified.folds[0]


def test_walk_forward_fold_preserves_fixed_fees_at_actual_boundary_nav() -> None:
    from pysharpe.analysis import tactical_momentum as tactical

    total_return, execution = _monthly_prices(months=30)
    config = _config(initial_capital=1_000.0, fee_per_order=25.0)
    walk_forward_config = tactical.WalkForwardConfig(
        training_months=8,
        test_months=6,
    )
    full_result = run_cross_asset_momentum(
        total_return,
        execution,
        config=config,
    )
    result = tactical.run_walk_forward_momentum(
        total_return,
        execution,
        strategy_config=config,
        walk_forward_config=walk_forward_config,
    )
    fold = result.folds[0]
    training_end = pd.Timestamp(fold["training_end"])

    assert fold["strategy_equity"][0]["value"] == pytest.approx(
        full_result.backtest.loc[training_end, "portfolio_value"]
    )
    for row in fold["strategy_equity"][1:]:
        timestamp = pd.Timestamp(row["timestamp"])
        assert row["value"] == pytest.approx(
            full_result.backtest.loc[timestamp, "portfolio_value"]
        )


def test_walk_forward_rejects_insufficient_training_history() -> None:
    from pysharpe.analysis import tactical_momentum as tactical

    total_return, execution = _monthly_prices(months=10)

    with pytest.raises(DataValidationError, match="training history must exceed"):
        tactical.run_walk_forward_momentum(
            total_return,
            execution,
            strategy_config=_config(),
            walk_forward_config=tactical.WalkForwardConfig(
                training_months=4,
                test_months=2,
            ),
        )


def test_provenance_and_walk_forward_symbols_are_public() -> None:
    from pysharpe import analysis

    assert PublicMarketDataProvenance is AnalysisMarketDataProvenance
    assert PublicWalkForwardConfig is AnalysisWalkForwardConfig
    assert PublicWalkForwardResearchResult.__name__ == "WalkForwardResearchResult"
    assert public_run_walk_forward_momentum is analysis.run_walk_forward_momentum
