"""Reproducible, UI-independent tactical portfolio research helpers.

The caller supplies separate, synchronized price panels:

* ``total_return_prices`` contains split- and distribution-adjusted index
  levels, already net of embedded fund expenses.
* ``execution_prices`` contains split-adjusted closes without distribution
  adjustments and is used to verify that trades had an executable close.

Foreign-currency assets require a CAD-per-unit FX series. FX is aligned only
backward in time and only within the configured calendar-day staleness limit.
No network access, brokerage connection, or order generation is performed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from ..exceptions import DataValidationError
from ..metrics import (
    annualize_volatility,
    cagr,
    maximum_drawdown,
    sharpe_ratio,
)


@dataclass(frozen=True)
class MomentumConfig:
    """Settings for a long-only, equal-weight cross-asset momentum strategy."""

    universe: tuple[str, ...]
    lookback_months: int = 12
    skip_months: int = 1
    top_n: int = 3
    benchmark_ticker: str = "VFV.TO"
    rebalance_frequency: str = "monthly"
    initial_capital: float = 10_000.0
    fee_per_order: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    fx_fee_bps: float = 0.0
    max_fx_staleness_days: int = 3
    max_price_staleness_days: int = 5
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if not self.universe or len(set(self.universe)) != len(self.universe):
            raise DataValidationError("universe must contain unique ticker symbols")
        if any(
            not isinstance(ticker, str) or not ticker.strip()
            for ticker in self.universe
        ):
            raise DataValidationError("universe symbols must be non-empty strings")
        if self.lookback_months < 1 or self.skip_months < 0 or self.top_n < 1:
            raise DataValidationError(
                "lookback_months and top_n must be positive; skip_months cannot be negative"
            )
        if self.rebalance_frequency != "monthly":
            raise DataValidationError("rebalance_frequency must be 'monthly'")
        if (
            not np.isfinite(self.initial_capital)
            or self.initial_capital <= 0
            or self.periods_per_year <= 0
        ):
            raise DataValidationError(
                "initial_capital and periods_per_year must be positive"
            )
        if not self.benchmark_ticker.strip():
            raise DataValidationError("benchmark_ticker must be a non-empty symbol")
        if self.max_fx_staleness_days < 0 or self.max_price_staleness_days < 0:
            raise DataValidationError("staleness limits cannot be negative")
        costs = (
            self.fee_per_order,
            self.slippage_bps,
            self.spread_bps,
            self.fx_fee_bps,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in costs):
            raise DataValidationError(
                "transaction costs must be finite and non-negative"
            )


@dataclass(frozen=True)
class MarketDataProvenance:
    """Source and freshness declaration for one supplied data series."""

    source: str
    retrieved_at: pd.Timestamp
    currency: str
    price_basis: str
    max_staleness_days: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise DataValidationError("provenance source must be non-empty")
        if (
            not isinstance(self.currency, str)
            or not self.currency.strip()
            or not isinstance(self.price_basis, str)
            or not self.price_basis.strip()
        ):
            raise DataValidationError(
                "provenance currency and price basis are required"
            )
        if (
            not isinstance(self.max_staleness_days, int)
            or isinstance(self.max_staleness_days, bool)
            or self.max_staleness_days < 0
        ):
            raise DataValidationError("provenance staleness limit cannot be negative")
        retrieved_at = pd.Timestamp(self.retrieved_at)
        if pd.isna(retrieved_at):
            raise DataValidationError("provenance retrieval timestamp is invalid")
        object.__setattr__(self, "retrieved_at", retrieved_at)


@dataclass
class MomentumResearchResult:
    """Signals, simulated portfolio path, benchmarks, and JSON export."""

    config: MomentumConfig
    generated_at: pd.Timestamp
    signals: pd.DataFrame
    weights: pd.DataFrame
    backtest: pd.DataFrame
    benchmarks: pd.DataFrame
    trades: pd.DataFrame
    risk: dict[str, float | None]
    benchmark_risk: dict[str, dict[str, float | None]]
    data_quality: dict[str, Any] = field(default_factory=dict)
    data_provenance: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned, JSON-compatible Northstar read contract."""
        return _json_safe(
            {
                "schema_version": "1.1",
                "generated_at": self.generated_at.isoformat(),
                "strategy": {
                    "name": "cross_asset_momentum",
                    "config": asdict(self.config),
                },
                "signals": _frame_records(self.signals),
                "weights": _frame_records(self.weights, include_index=True),
                "backtest": _frame_records(self.backtest, include_index=True),
                "benchmarks": _frame_records(self.benchmarks, include_index=True),
                "trades": _frame_records(self.trades),
                "risk": self.risk,
                "benchmark_risk": self.benchmark_risk,
                "data_quality": self.data_quality,
                "data_provenance": self.data_provenance,
                "cost_assumptions": _cost_assumptions(self.config),
            }
        )

    def to_json(self) -> str:
        """Serialize results deterministically with no non-standard NaN tokens."""
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )


def run_cross_asset_momentum(
    total_return_prices: pd.DataFrame,
    execution_prices: pd.DataFrame,
    *,
    config: MomentumConfig,
    currencies: dict[str, str] | None = None,
    fx_to_cad: pd.DataFrame | None = None,
    provenance: dict[str, MarketDataProvenance] | None = None,
) -> MomentumResearchResult:
    """Generate historical signals and simulate the specified strategy.

    ``total_return_prices`` must be an adjusted total-return index in each
    asset's local currency and already net of embedded expenses. ``execution_prices``
    must contain split-adjusted, distribution-unadjusted closes. The two panels
    must use the same timestamp index. FX rates are CAD per one unit of local
    currency (for example, 1.35 CAD/USD).
    """
    total = _validate_price_panel(total_return_prices, "total_return_prices")
    execution = _validate_price_panel(execution_prices, "execution_prices")
    if not total.index.equals(execution.index):
        raise DataValidationError(
            "total_return_prices and execution_prices must have identical timestamps"
        )
    required = set(config.universe) | {config.benchmark_ticker}
    missing = required - set(total.columns)
    if missing:
        raise DataValidationError(
            f"Missing total-return history for: {', '.join(sorted(missing))}"
        )
    missing_execution = required - set(execution.columns)
    if missing_execution:
        raise DataValidationError(
            f"Missing execution price columns for: {', '.join(sorted(missing_execution))}"
        )

    assets = sorted(required)
    total = total.loc[:, assets]
    execution = execution.loc[:, assets]
    for ticker in assets:
        _validate_leading_only_gaps(total[ticker], ticker, "total-return")
        _validate_leading_only_gaps(execution[ticker], ticker, "execution-price")

    currency_by_ticker = {ticker: "CAD" for ticker in assets}
    if currencies:
        currency_by_ticker.update(
            {
                ticker: str(currency).upper()
                for ticker, currency in currencies.items()
                if ticker in assets
            }
        )
    unknown = set(currency_by_ticker.values()) - {"CAD"}
    if unknown and fx_to_cad is None:
        raise DataValidationError(
            f"FX rates are required for currencies: {', '.join(sorted(unknown))}"
        )

    fx_aligned = _align_fx(
        index=pd.DatetimeIndex(total.index),
        currencies=sorted(unknown),
        fx_to_cad=fx_to_cad,
        max_staleness_days=config.max_fx_staleness_days,
    )
    provenance_details, provenance_missing, provenance_unverified = _data_provenance(
        provenance=provenance or {},
        assets=assets,
        currencies=currency_by_ticker,
        total_return_prices=total,
        execution_prices=execution,
        fx_to_cad=fx_to_cad,
        foreign_currencies=sorted(unknown),
        config=config,
    )
    stale_series = [
        key for key, details in provenance_details.items() if details["stale"]
    ]
    if stale_series:
        raise DataValidationError(
            f"stale market data prevents signal generation: {', '.join(stale_series)}"
        )
    cad_total = total.copy()
    cad_execution_fx = fx_aligned
    fx_uncovered = 0
    for ticker, currency in currency_by_ticker.items():
        if currency == "CAD":
            continue
        rate = fx_aligned[currency]
        fx_uncovered += int(rate.isna().sum())
        cad_total[ticker] = total[ticker] * rate
        _validate_leading_only_gaps(cad_total[ticker], ticker, "CAD-converted")

    monthly = cad_total.resample("ME").last()
    monthly_index = pd.DatetimeIndex(monthly.index)
    price_index = pd.DatetimeIndex(total.index)
    monthly_observation_dates = pd.DataFrame(index=monthly.index, columns=assets)
    for month_end in monthly_index:
        in_month = (price_index.year == month_end.year) & (
            price_index.month == month_end.month
        )
        month_dates = price_index[in_month]
        for ticker in assets:
            observed = total.loc[month_dates, ticker].dropna().index
            if len(observed):
                monthly_observation_dates.at[month_end, ticker] = observed[-1]
    signal_rows: list[dict[str, Any]] = []
    signal_targets: dict[pd.Timestamp, dict[str, float]] = {}
    first_signalable_month = config.lookback_months + config.skip_months
    if len(monthly) <= first_signalable_month:
        raise DataValidationError(
            "insufficient history for the configured momentum lookback and skip window"
        )

    date_index = price_index
    for month_pos in range(first_signalable_month, len(monthly)):
        signal_month = monthly_index[month_pos]
        signal_month_dates = date_index[
            (date_index.year == signal_month.year)
            & (date_index.month == signal_month.month)
        ]
        if not len(signal_month_dates):
            continue
        start_pos = month_pos - config.skip_months - config.lookback_months
        end_pos = month_pos - config.skip_months
        window = monthly.iloc[start_pos : end_pos + 1]
        scores = monthly.iloc[end_pos] / monthly.iloc[start_pos] - 1.0
        eligible = scores.loc[list(config.universe)].replace([np.inf, -np.inf], np.nan)
        observation_dates = monthly_observation_dates.iloc[start_pos : end_pos + 1]
        fresh = pd.DataFrame(index=observation_dates.index, columns=config.universe)
        for window_month in observation_dates.index:
            for ticker in config.universe:
                observed = observation_dates.at[window_month, ticker]
                fresh.at[window_month, ticker] = (
                    pd.notna(observed)
                    and (window_month - cast(pd.Timestamp, observed)).days
                    <= config.max_price_staleness_days
                )
        complete = window.loc[:, list(config.universe)].notna().all(axis=0) & fresh.loc[
            :, list(config.universe)
        ].astype(bool).all(axis=0)
        eligible = eligible.loc[complete & eligible.notna()]
        eligible = eligible.loc[eligible > -1.0]
        if eligible.empty:
            continue

        # Monthly aggregation drops missing listing history only if the entire
        # lookback window is present; ties resolve lexicographically.
        ranked = sorted(
            ((str(ticker), float(score)) for ticker, score in eligible.items()),
            key=lambda item: (-item[1], item[0]),
        )
        selected = ranked[: config.top_n]
        actual_signal_date = pd.Timestamp(signal_month_dates[-1])
        effective_candidates = date_index[date_index > actual_signal_date]
        effective_date = effective_candidates[0] if len(effective_candidates) else None
        if effective_date is not None:
            tradable = [
                (ticker, score)
                for ticker, score in selected
                if pd.notna(execution.at[effective_date, ticker])
                and (
                    currency_by_ticker[ticker] == "CAD"
                    or pd.notna(
                        cad_execution_fx.at[effective_date, currency_by_ticker[ticker]]
                    )
                )
            ]
        else:
            tradable = selected
        if tradable:
            target_weight = 1.0 / len(tradable)
            if effective_date is not None:
                signal_targets[pd.Timestamp(effective_date)] = {
                    ticker: target_weight for ticker, _score in tradable
                }
            for ticker, score in tradable:
                signal_rows.append(
                    {
                        "signal_date": actual_signal_date,
                        "effective_date": effective_date,
                        "ticker": ticker,
                        "score": float(score),
                        "target_weight": target_weight,
                    }
                )

    if not signal_rows:
        raise DataValidationError(
            "insufficient history or data coverage to generate an executable signal"
        )

    signals = pd.DataFrame(signal_rows)
    signals.sort_values(["signal_date", "ticker"], inplace=True, ignore_index=True)
    simulation_index = date_index
    cadence_returns = cad_total.pct_change(fill_method=None)
    cadence_returns = cadence_returns.replace([np.inf, -np.inf], np.nan)
    nav = config.initial_capital
    held_weights = {ticker: 0.0 for ticker in config.universe}
    portfolio_values: list[float] = []
    net_returns: list[float] = []
    weight_rows: list[dict[str, float]] = []
    trade_rows: list[dict[str, Any]] = []
    previous_value = nav

    for position, date in enumerate(simulation_index):
        if position == 0:
            period_return = 0.0
        else:
            row = cast(pd.Series, cadence_returns.loc[date])
            held_returns = []
            for ticker, weight in held_weights.items():
                if weight == 0.0:
                    continue
                value = cast(float, row[ticker])
                if not np.isfinite(value):
                    raise DataValidationError(
                        f"interior missing total-return observation while holding {ticker}"
                    )
                held_returns.append(weight * value)
            period_return = float(sum(held_returns))
        nav_before_cost = previous_value * (1.0 + period_return)
        if not np.isfinite(nav_before_cost) or nav_before_cost <= 0.0:
            raise DataValidationError("simulated portfolio value became non-positive")

        drifted_weights = held_weights.copy()
        if position > 0:
            growth = 1.0 + period_return
            drifted_weights = {
                ticker: (
                    weight
                    * (1.0 + cast(float, cadence_returns.at[date, ticker]))
                    / growth
                    if weight
                    else 0.0
                )
                for ticker, weight in held_weights.items()
            }
            if sum(drifted_weights.values()) > 1.0 + 1e-9:
                raise DataValidationError("portfolio weights exceeded 100 percent")

        if date in signal_targets:
            target = signal_targets[date]
            deltas = {
                ticker: target.get(ticker, 0.0) - drifted_weights.get(ticker, 0.0)
                for ticker in config.universe
            }
            traded = {
                ticker: delta
                for ticker, delta in deltas.items()
                if not np.isclose(delta, 0.0, atol=1e-12)
            }
            gross_notional = nav_before_cost * sum(
                abs(delta) for delta in traded.values()
            )
            order_count = len(traded)
            fx_notional = sum(
                nav_before_cost * abs(delta)
                for ticker, delta in traded.items()
                if currency_by_ticker[ticker] != "CAD"
            )
            variable_bps = config.slippage_bps + config.spread_bps / 2.0
            cost = (
                gross_notional * variable_bps / 10_000.0
                + fx_notional * config.fx_fee_bps / 10_000.0
                + order_count * config.fee_per_order
            )
            if cost >= nav_before_cost:
                raise DataValidationError("transaction costs exceed portfolio value")
            nav = nav_before_cost - cost
            trade_rows.append(
                {
                    "date": pd.Timestamp(date),
                    "gross_traded_notional": float(gross_notional),
                    "turnover": float(gross_notional / nav_before_cost),
                    "order_count": int(order_count),
                    "cost": float(cost),
                    "fx_converted_notional": float(fx_notional),
                }
            )
            held_weights = {
                ticker: target.get(ticker, 0.0) for ticker in config.universe
            }
        else:
            nav = nav_before_cost
            held_weights = drifted_weights

        portfolio_values.append(float(nav))
        net_returns.append(float(nav / previous_value - 1.0) if position else 0.0)
        weight_rows.append(held_weights.copy())
        previous_value = nav

    backtest = pd.DataFrame(
        {"portfolio_value": portfolio_values, "return": net_returns},
        index=simulation_index,
    )
    weights = pd.DataFrame(weight_rows, index=simulation_index).fillna(0.0)
    weights = weights.reindex(columns=config.universe, fill_value=0.0)
    benchmark_series: dict[str, pd.Series] = {}
    benchmark_values = cad_total[config.benchmark_ticker].dropna()
    if benchmark_values.empty:
        raise DataValidationError("benchmark has no CAD-convertible history")
    if benchmark_values.index[0] != simulation_index[0]:
        raise DataValidationError("benchmark must cover simulation start")
    benchmark_series[config.benchmark_ticker] = (
        config.initial_capital
        * cad_total[config.benchmark_ticker]
        / float(benchmark_values.iloc[0])
    )
    benchmark_series["CASH"] = pd.Series(
        config.initial_capital, index=simulation_index, dtype=float
    )
    benchmarks = pd.DataFrame(benchmark_series, index=simulation_index)
    strategy_risk = _risk_metrics(backtest["portfolio_value"], config.periods_per_year)
    benchmark_risk = {
        ticker: _risk_metrics(benchmarks[ticker], config.periods_per_year)
        for ticker in (config.benchmark_ticker, "CASH")
    }
    incomplete = [
        ticker
        for ticker in config.universe
        if total[ticker].isna().any() and total[ticker].first_valid_index() is not None
    ]
    data_quality = {
        "return_basis": "net_of_expenses",
        "execution_price_basis": "split_adjusted_close",
        "fx_alignment": "backward_only",
        "fx_uncovered_observations": fx_uncovered,
        "assets_with_incomplete_history": incomplete,
        "signal_count": int(len(signals)),
        "signal_warmup_months": config.lookback_months + config.skip_months,
        "data_end": pd.Timestamp(total.index[-1]).isoformat(),
        "stale_signals_executed": 0,
        "provenance_complete": not provenance_missing and not provenance_unverified,
        "signal_freshness_verified": not provenance_missing
        and not provenance_unverified,
        "unattributed_series": provenance_missing,
        "unverified_series": sorted(set(provenance_missing + provenance_unverified)),
        "stale_data_series": stale_series,
    }
    return MomentumResearchResult(
        config=config,
        generated_at=pd.Timestamp(total.index[-1]),
        signals=signals,
        weights=weights,
        backtest=backtest,
        benchmarks=benchmarks,
        trades=pd.DataFrame(trade_rows),
        risk=strategy_risk,
        benchmark_risk=benchmark_risk,
        data_quality=data_quality,
        data_provenance=provenance_details,
    )


def _validate_price_panel(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError(f"{label} must be a non-empty DataFrame")
    if frame.columns.has_duplicates:
        raise DataValidationError(f"{label} columns must be unique")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataValidationError(f"{label} must use a DatetimeIndex")
    if (
        frame.index.hasnans
        or frame.index.has_duplicates
        or not frame.index.is_monotonic_increasing
    ):
        raise DataValidationError(f"{label} timestamps must be unique and increasing")
    out = frame.apply(pd.to_numeric, errors="coerce").astype(float)
    invalid_numeric = frame.notna() & out.isna()
    if invalid_numeric.any().any():
        raise DataValidationError(f"{label} contains non-numeric observations")
    out.index = pd.DatetimeIndex(out.index)
    if (out <= 0.0).where(out.notna(), False).any().any():
        raise DataValidationError(f"{label} prices must be positive where present")
    if np.isinf(out.to_numpy()).any():
        raise DataValidationError(f"{label} contains infinite values")
    return out


def _validate_leading_only_gaps(series: pd.Series, ticker: str, label: str) -> None:
    valid = series.notna().to_numpy()
    if valid.any():
        first = int(np.argmax(valid))
        if series.iloc[first:].isna().any():
            raise DataValidationError(
                f"{label} has interior missing observations for {ticker}; "
                "repair the source history before simulation"
            )


def _align_fx(
    *,
    index: pd.DatetimeIndex,
    currencies: list[str],
    fx_to_cad: pd.DataFrame | None,
    max_staleness_days: int,
) -> pd.DataFrame:
    aligned = pd.DataFrame(index=index)
    if not currencies:
        return aligned
    if fx_to_cad is None or not set(currencies).issubset(fx_to_cad.columns):
        missing = set(currencies) - (
            set(fx_to_cad.columns) if fx_to_cad is not None else set()
        )
        raise DataValidationError(
            f"FX rates missing for currencies: {', '.join(sorted(missing))}"
        )
    if not isinstance(fx_to_cad.index, pd.DatetimeIndex):
        raise DataValidationError("fx_to_cad must use a DatetimeIndex")
    if fx_to_cad.columns.has_duplicates:
        raise DataValidationError("FX currency columns must be unique")
    if (
        fx_to_cad.index.hasnans
        or fx_to_cad.index.has_duplicates
        or not fx_to_cad.index.is_monotonic_increasing
    ):
        raise DataValidationError("FX timestamps must be unique and increasing")
    tolerance = pd.Timedelta(days=max_staleness_days)
    targets = pd.DataFrame({"timestamp": index})
    for currency in currencies:
        raw_source = fx_to_cad[currency]
        numeric_source = pd.to_numeric(raw_source, errors="coerce")
        if (raw_source.notna() & numeric_source.isna()).any():
            raise DataValidationError(
                f"FX rates for {currency} contain non-numeric values"
            )
        source = numeric_source.dropna()
        if (source <= 0.0).any() or not np.isfinite(source.to_numpy()).all():
            raise DataValidationError(
                f"FX rates for {currency} must be finite and positive"
            )
        values = pd.DataFrame(
            {"timestamp": source.index, "rate": source.to_numpy(dtype=float)}
        )
        try:
            joined = pd.merge_asof(
                targets,
                values.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                tolerance=tolerance,
            )
        except (ValueError, TypeError) as exc:
            raise DataValidationError(
                "FX and price timestamps must use compatible timezones"
            ) from exc
        aligned[currency] = joined["rate"].to_numpy()
    return aligned


def _data_provenance(
    *,
    provenance: dict[str, MarketDataProvenance],
    assets: list[str],
    currencies: dict[str, str],
    total_return_prices: pd.DataFrame,
    execution_prices: pd.DataFrame,
    fx_to_cad: pd.DataFrame | None,
    foreign_currencies: list[str],
    config: MomentumConfig,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    expected: dict[str, tuple[pd.Series, str, str, int]] = {}
    for ticker in assets:
        expected[f"total_return_prices:{ticker}"] = (
            total_return_prices[ticker],
            currencies[ticker],
            "adjusted_total_return_net_of_expenses",
            config.max_price_staleness_days,
        )
        expected[f"execution_prices:{ticker}"] = (
            execution_prices[ticker],
            currencies[ticker],
            "split_adjusted_close",
            config.max_price_staleness_days,
        )
    if fx_to_cad is not None:
        for currency in foreign_currencies:
            expected[f"fx_to_cad:{currency}"] = (
                fx_to_cad[currency],
                f"CAD/{currency}",
                "CAD_per_currency_unit",
                config.max_fx_staleness_days,
            )
    unknown = set(provenance) - set(expected)
    if unknown:
        raise DataValidationError(
            f"provenance contains unknown series: {', '.join(sorted(unknown))}"
        )

    details: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    unverified: list[str] = []
    for key, (
        series,
        expected_currency,
        expected_basis,
        configured_age,
    ) in expected.items():
        record = provenance.get(key)
        if record is None:
            missing.append(key)
            continue
        if not isinstance(record, MarketDataProvenance):
            raise DataValidationError(f"provenance for {key} has an invalid type")
        if record.currency.upper() != expected_currency.upper():
            raise DataValidationError(
                f"provenance currency for {key} must be {expected_currency}"
            )
        if record.price_basis != expected_basis:
            raise DataValidationError(
                f"provenance price basis for {key} must be {expected_basis}"
            )
        observed_at = series.last_valid_index()
        age_days: int | None = None
        stale = False
        allowed_age = min(record.max_staleness_days, configured_age)
        if observed_at is not None:
            age_days = _calendar_age_days(
                record.retrieved_at, cast(pd.Timestamp, observed_at)
            )
            if age_days < 0:
                raise DataValidationError(
                    f"provenance retrieval time for {key} precedes its latest observation"
                )
            stale = age_days > allowed_age
        else:
            unverified.append(key)
        details[key] = {
            "source": record.source,
            "retrieved_at": pd.Timestamp(record.retrieved_at),
            "currency": record.currency.upper(),
            "price_basis": record.price_basis,
            "latest_observation": cast(pd.Timestamp, observed_at)
            if observed_at is not None
            else None,
            "age_days_at_retrieval": age_days,
            "max_staleness_days": allowed_age,
            "stale": stale,
            "observation_available": observed_at is not None,
        }
    return details, missing, unverified


def _calendar_age_days(retrieved_at: pd.Timestamp, observed_at: pd.Timestamp) -> int:
    retrieved = pd.Timestamp(retrieved_at)
    observed = pd.Timestamp(observed_at)
    if (retrieved.tz is None) != (observed.tz is None):
        raise DataValidationError(
            "provenance retrieval and market-data timestamps must use compatible timezones"
        )
    if retrieved.tz is not None:
        retrieved = retrieved.tz_convert("UTC")
        observed = observed.tz_convert("UTC")
    return int((retrieved.normalize() - observed.normalize()).days)


def _cost_assumptions(config: MomentumConfig) -> dict[str, Any]:
    return {
        "commission_model": "fixed_per_order",
        "fee_per_order": config.fee_per_order,
        "slippage_bps": config.slippage_bps,
        "spread_bps_full_quote": config.spread_bps,
        "fx_fee_bps": config.fx_fee_bps,
        "fee_source": "caller_configured",
        "fx_rate_convention": "CAD_per_unit_of_local_currency",
    }


def _risk_metrics(values: pd.Series, periods_per_year: int) -> dict[str, float | None]:
    values = values.dropna()
    returns = values.pct_change(fill_method=None).dropna()
    if values.empty:
        return {
            "cagr": None,
            "annualized_volatility": None,
            "sharpe_ratio": None,
            "maximum_drawdown": None,
        }
    try:
        sharpe = _metric_scalar(
            sharpe_ratio(returns, periods_per_year=periods_per_year)
        )
    except ValueError:
        sharpe = None
    volatility = (
        _metric_scalar(annualize_volatility(returns, periods_per_year=periods_per_year))
        if len(returns) > 1
        else None
    )
    return {
        "cagr": float(cagr(values)),
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": float(maximum_drawdown(values)),
    }


def _metric_scalar(value: float | pd.Series) -> float | None:
    if isinstance(value, pd.Series):
        return float(value.iloc[0]) if len(value) == 1 else None
    return float(value)


def _frame_records(
    frame: pd.DataFrame, *, include_index: bool = False
) -> list[dict[str, Any]]:
    data = frame.copy()
    if include_index:
        data.insert(0, "timestamp", data.index)
    else:
        data = data.reset_index(drop=True)
    return _json_safe(data.to_dict(orient="records"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if value is pd.NaT or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def __getattr__(name: str) -> Any:
    if name in {
        "WalkForwardConfig",
        "WalkForwardResearchResult",
        "run_walk_forward_momentum",
    }:
        from . import tactical_walk_forward

        return getattr(tactical_walk_forward, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MarketDataProvenance",
    "MomentumConfig",
    "MomentumResearchResult",
    "run_cross_asset_momentum",
]
