"""Synthetic tests for trend, breakout, walk-forward, and overlap research."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pysharpe import (
    PortfolioOverlapResult as PublicPortfolioOverlapResult,
)
from pysharpe import (
    TacticalResearchResult as PublicTacticalResearchResult,
)
from pysharpe import (
    TrendConfig as PublicTrendConfig,
)
from pysharpe import (
    analyze_portfolio_overlap as public_analyze_portfolio_overlap,
)
from pysharpe import (
    run_tactical_strategy as public_run_tactical_strategy,
)
from pysharpe import (
    run_trend_strategy as public_run_trend_strategy,
)
from pysharpe import (
    run_walk_forward_strategy as public_run_walk_forward_strategy,
)
from pysharpe.analysis.portfolio_overlap import analyze_portfolio_overlap
from pysharpe.analysis.tactical_momentum import (
    TrendConfig,
    run_tactical_strategy,
    run_trend_strategy,
)
from pysharpe.analysis.tactical_walk_forward import (
    WalkForwardConfig,
    run_walk_forward_strategy,
)
from pysharpe.exceptions import DataValidationError


def _histories(months: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2020-01-31", periods=months, freq="ME")
    steps = np.arange(months, dtype=float)
    total = pd.DataFrame(
        {
            "UP": 100.0 * 1.02**steps,
            "FLAT": np.full(months, 100.0),
            "VFV": 100.0 * 1.01**steps,
        },
        index=index,
    )
    return total, total.copy()


def _trend_config(**overrides: object) -> TrendConfig:
    values: dict[str, object] = {
        "universe": ("UP", "FLAT"),
        "lookback_months": 6,
        "top_n": 1,
        "benchmark_ticker": "VFV",
        "periods_per_year": 12,
    }
    values.update(overrides)
    return TrendConfig(**values)


@pytest.mark.parametrize("rule", ["moving_average", "breakout"])
def test_trend_rules_generate_next_session_long_only_signals(rule: str) -> None:
    total, execution = _histories()
    result = run_trend_strategy(
        total,
        execution,
        config=_trend_config(rule=rule),
    )

    assert not result.signals.empty
    assert set(result.signals["ticker"]) == {"UP"}
    assert (result.signals["score"] > 0).all()
    executable = result.signals.dropna(subset=["effective_date"])
    assert (executable["effective_date"] > executable["signal_date"]).all()
    assert (result.weights.to_numpy() >= 0.0).all()
    assert result.weights.sum(axis=1).max() <= 1.0
    assert {"VFV", "CASH"}.issubset(result.benchmarks.columns)
    payload = json.loads(result.to_json())
    assert payload["schema_version"] == "1.2"
    assert payload["strategy"]["name"] == rule
    assert isinstance(result, PublicTacticalResearchResult)
    assert PublicTrendConfig is TrendConfig
    assert public_run_trend_strategy is run_trend_strategy
    assert public_run_tactical_strategy is run_tactical_strategy


def test_trend_signals_are_unchanged_by_future_price_changes() -> None:
    total, execution = _histories()
    config = _trend_config(rule="breakout")
    cutoff = total.index[24]
    original = run_trend_strategy(total, execution, config=config)
    changed_total = total.copy()
    changed_total.loc[changed_total.index > cutoff, "UP"] *= 3.0
    changed = run_trend_strategy(changed_total, execution, config=config)

    left = original.signals.loc[original.signals["signal_date"] <= cutoff]
    right = changed.signals.loc[changed.signals["signal_date"] <= cutoff]
    pd.testing.assert_frame_equal(
        left.reset_index(drop=True), right.reset_index(drop=True)
    )


@pytest.mark.parametrize("rule", ["moving_average", "breakout"])
def test_trend_rules_exit_to_cash_when_their_exit_condition_fails(rule: str) -> None:
    total, execution = _histories()
    decline = np.arange(len(total) - 20, dtype=float)
    total.loc[total.index[20] :, "UP"] = 150.0 * 0.94**decline
    execution.loc[execution.index[20] :, "UP"] = total.loc[total.index[20] :, "UP"]
    result = run_trend_strategy(total, execution, config=_trend_config(rule=rule))

    assert result.weights["UP"].max() > 0.0
    assert result.weights.iloc[-1].sum() == pytest.approx(0.0)


def test_breakout_holds_below_prior_high_until_prior_window_floor_breaks() -> None:
    total, execution = _histories(18)
    levels = [100.0, 100.0, 100.0, 101.0, 102.0, 100.5, 100.0]
    total.loc[total.index[: len(levels)], "UP"] = levels
    execution.loc[execution.index[: len(levels)], "UP"] = levels
    result = run_trend_strategy(
        total,
        execution,
        config=_trend_config(rule="breakout", lookback_months=3),
    )

    signals = result.signals.loc[result.signals["ticker"] == "UP"]
    assert total.index[5] in signals["signal_date"].tolist()
    assert total.index[6] not in signals["signal_date"].tolist()
    assert (
        total.index[6]
        in result.signals.loc[
            result.signals["ticker"] == "CASH", "signal_date"
        ].tolist()
    )


def test_trend_history_requires_full_window_and_stays_out_until_listed() -> None:
    total, execution = _histories()
    total.loc[total.index[:30], "UP"] = np.nan
    execution.loc[execution.index[:30], "UP"] = np.nan
    result = run_trend_strategy(
        total,
        execution,
        config=_trend_config(rule="breakout", top_n=2),
    )

    up_signals = result.signals.loc[result.signals["ticker"] == "UP"]
    assert up_signals.empty or up_signals["signal_date"].min() >= total.index[36]


def test_trend_strategy_uses_existing_transaction_cost_model() -> None:
    total, execution = _histories()
    config = _trend_config(rule="moving_average", slippage_bps=10.0, spread_bps=20.0)
    result = run_trend_strategy(total, execution, config=config)

    assert (result.trades["cost"] > 0.0).any()
    assert result.trades["cost"].iloc[0] == pytest.approx(
        result.trades["gross_traded_notional"].iloc[0] * 0.002
    )


def test_total_return_levels_drive_signals_and_simulation_not_execution_quotes() -> (
    None
):
    total, execution = _histories()
    baseline = run_trend_strategy(total, execution, config=_trend_config())
    adjusted_execution = execution.copy()
    adjusted_execution["UP"] = 100.0
    changed_execution = run_trend_strategy(
        total, adjusted_execution, config=_trend_config()
    )

    pd.testing.assert_frame_equal(baseline.signals, changed_execution.signals)
    pd.testing.assert_frame_equal(baseline.backtest, changed_execution.backtest)


def test_trend_rule_uses_prior_fx_only_for_cad_rankings() -> None:
    total, execution = _histories()
    total["USD_UP"] = 100.0
    execution["USD_UP"] = 100.0
    fx = pd.DataFrame(
        {"USD": 1.0 * 1.02 ** np.arange(len(total))},
        index=total.index,
    )
    config = _trend_config(universe=("USD_UP", "FLAT"), top_n=1)
    result = run_trend_strategy(
        total,
        execution,
        config=config,
        currencies={"USD_UP": "USD"},
        fx_to_cad=fx,
    )

    assert set(result.signals.loc[result.signals["ticker"] != "CASH", "ticker"]) == {
        "USD_UP"
    }
    assert result.data_quality["fx_alignment"] == "backward_only"


def test_trend_strategy_rejects_insufficient_lookback_history() -> None:
    total, execution = _histories(6)
    with pytest.raises(DataValidationError, match="insufficient history"):
        run_trend_strategy(total, execution, config=_trend_config(lookback_months=12))


def test_trend_config_rejects_invalid_rules_and_fractional_months() -> None:
    with pytest.raises(DataValidationError, match="rule"):
        _trend_config(rule="unsupported")
    with pytest.raises(DataValidationError, match="integers"):
        _trend_config(lookback_months=6.5)


def test_fixed_walk_forward_accepts_trend_configuration() -> None:
    total, execution = _histories(54)
    result = run_walk_forward_strategy(
        total,
        execution,
        strategy_config=_trend_config(rule="breakout"),
        walk_forward_config=WalkForwardConfig(training_months=12, test_months=6),
    )

    assert len(result.folds) >= 2
    assert result.to_dict()["strategy"]["name"] == "breakout"
    assert all(fold["test_start"] > fold["training_end"] for fold in result.folds)


def test_walk_forward_earlier_folds_ignore_future_observations() -> None:
    total, execution = _histories(54)
    config = _trend_config(rule="breakout")
    walk_forward = WalkForwardConfig(training_months=12, test_months=6)
    original = run_walk_forward_strategy(
        total,
        execution,
        strategy_config=config,
        walk_forward_config=walk_forward,
    )
    first_fold_end = pd.Timestamp(original.folds[0]["test_end"])
    changed_total = total.copy()
    changed_execution = execution.copy()
    changed_total.loc[changed_total.index > first_fold_end, "UP"] *= 4.0
    changed_execution.loc[changed_execution.index > first_fold_end, "UP"] *= 4.0
    changed = run_walk_forward_strategy(
        changed_total,
        changed_execution,
        strategy_config=config,
        walk_forward_config=walk_forward,
    )

    assert changed.folds[0] == original.folds[0]


def test_overlap_uses_explicit_constituents_and_reports_unknown_coverage() -> None:
    result = analyze_portfolio_overlap(
        {
            "growth": {"FUND_A": 0.6, "FUND_UNKNOWN": 0.2},
            "balanced": {"FUND_B": 0.5, "DIRECT_X": 0.3},
        },
        fund_constituents={
            "FUND_A": {"DIRECT_X": 0.5, "DIRECT_Y": 0.3},
            "FUND_B": {"DIRECT_X": 0.4, "DIRECT_Z": 0.4},
        },
        fund_tickers={"FUND_A", "FUND_B", "FUND_UNKNOWN"},
    )

    assert result.portfolios["growth"]["known_exposure"]["DIRECT_X"] == pytest.approx(
        0.3
    )
    assert result.portfolios["growth"]["unknown_weight"] == pytest.approx(0.32)
    pair = result.pairwise[0]
    assert pair["shared_weight"] == pytest.approx(0.3)
    assert pair["overlap_of_smaller_known_portfolio"] == pytest.approx(0.3 / 0.48)
    payload = json.loads(result.to_json())
    assert payload["schema_version"] == "1.0"
    assert result.to_json() == result.to_json()
    assert isinstance(result, PublicPortfolioOverlapResult)
    assert public_analyze_portfolio_overlap is analyze_portfolio_overlap
    assert public_run_walk_forward_strategy is run_walk_forward_strategy


def test_overlap_rejects_invalid_weights_and_never_renormalizes_constituents() -> None:
    with pytest.raises(DataValidationError, match="weights"):
        analyze_portfolio_overlap({"bad": {"A": 1.1}}, fund_constituents={})

    result = analyze_portfolio_overlap(
        {"one": {"FUND": 1.0}},
        fund_constituents={"FUND": {"A": 0.4}},
        fund_tickers={"FUND"},
    )
    assert result.portfolios["one"]["known_exposure"]["A"] == pytest.approx(0.4)
    assert result.portfolios["one"]["unknown_weight"] == pytest.approx(0.6)
