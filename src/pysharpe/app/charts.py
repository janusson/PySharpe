"""Chart rendering helpers for the Streamlit app."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import streamlit as st

from pysharpe.analysis import CANADIAN_BENCHMARKS, fetch_benchmark_metrics
from pysharpe.app.analytics import (
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_SHRINKAGE_FLOOR,
    MetricResults,
    evaluate_adjusted_performance,
)
from pysharpe.app.data import SETTINGS
from pysharpe.optimization.expected_returns import shrinkage_expected_return
from pysharpe.optimization.models import (
    OptimisationPerformance,
    OptimisationResult,
    PortfolioWeights,
)
from pysharpe.optimization.tax_location import (
    AccountType,
    AssetTaxCharacteristics,
    TaxProfile,
    build_asset_characteristics_batch,
)
from pysharpe.portfolio_optimization import optimise_from_prices
from pysharpe.visualization import (
    generate_efficient_frontier,
    plot_portfolio_comparison,
)

EQUAL_WEIGHT_NAME = "Equal-Weight (1/N)"

#: Benchmark rows default to Non-Registered placement because no explicit
#: asset-location split exists for them; the assumption is surfaced in the UI.
_DEFAULT_COMPARISON_ACCOUNT: str = AccountType.NON_REG.value

try:
    import altair as alt
except ImportError:  # pragma: no cover - visual dependency
    alt = None


@dataclass(frozen=True)
class _TaxContext:
    """Tax/MER evaluation context shared by every comparison row."""

    tax_profile: TaxProfile
    asset_characteristics: dict[str, AssetTaxCharacteristics]
    account: str


def _tax_context(price_data: pd.DataFrame) -> _TaxContext:
    """Build the tax/MER evaluation context from the app settings.

    Uses the user's active ``TaxProfile`` and per-ticker MER/proxy metadata
    so the Custom Mix and benchmarks are penalized exactly like the optimizer
    pipeline (MER deduction + account-specific tax drag).
    """

    characteristics = build_asset_characteristics_batch(
        list(price_data.columns),
        proxy_map=SETTINGS.proxy_map,
        mer_by_ticker=SETTINGS.mer_by_ticker,
    )
    return _TaxContext(
        tax_profile=SETTINGS.tax_profile,
        asset_characteristics=characteristics,
        account=_DEFAULT_COMPARISON_ACCOUNT,
    )


def equal_weight_allocation(assets: Sequence[str]) -> dict[str, float]:
    """Return a strict 1/N allocation over ``assets``.

    Used as the neutral baseline for both the sidebar defaults and the
    performance comparison.  An empty asset list yields an empty mapping.
    """

    if not assets:
        return {}
    weight = 1.0 / len(assets)
    return {str(asset): weight for asset in assets}


def _normalize_user_weights(
    assets: Sequence[str], custom_weights: dict[str, float] | None
) -> dict[str, float]:
    """Re-normalize user weights over the optimizer's asset list.

    Unknown/stale tickers are dropped and missing tickers count as zero.  A
    zero-sum input falls back to a strict 1/N allocation so the pipeline
    never feeds a degenerate all-zero portfolio to the performance solver.
    """

    raw = {str(asset): float(custom_weights.get(asset, 0.0)) for asset in assets}
    total = sum(raw.values())
    if total > 0:
        return {ticker: value / total for ticker, value in raw.items()}
    return equal_weight_allocation(assets)


def _coerce_allocations(weights: object) -> dict[str, float] | None:
    """Extract a ticker→weight mapping from common weight containers."""

    if isinstance(weights, dict):
        return weights
    allocations = getattr(weights, "allocations", None)
    if isinstance(allocations, dict):
        return allocations
    return None


def _prepare_weight_chart_data(series: pd.Series) -> pd.DataFrame:
    """Shape weight allocations for charting, keeping only positive weights."""

    if series.empty:
        return pd.DataFrame(columns=["Ticker", "Weight"])

    positive_weights = series[series > 0]
    if positive_weights.empty:
        return pd.DataFrame(columns=["Ticker", "Weight"])

    chart_frame = pd.DataFrame(
        {
            "Ticker": [str(idx) for idx in positive_weights.index],
            "Weight": positive_weights.values,
        }
    )
    return chart_frame.reset_index(drop=True)


def plot_cumulative_returns(price_frame: pd.DataFrame) -> None:
    """Render a cumulative returns line chart for the provided price data."""

    if price_frame.empty:
        st.info("No price data available to plot cumulative returns.")
        return

    returns = price_frame.pct_change().dropna(how="all")
    if returns.empty:
        st.info("Not enough price history to compute cumulative returns.")
        return

    cumulative = (1 + returns).cumprod()
    try:
        st.line_chart(cumulative, height=320, width="stretch")
    except Exception as exc:  # pragma: no cover - visual guardrail
        st.warning(f"Unable to render returns chart: {exc}")


def plot_weights(weights: PortfolioWeights | dict[str, float]) -> None:
    """Display a bar chart and donut of portfolio weights.

    Accepts a :class:`PortfolioWeights` container or a plain ticker→weight
    mapping so callers can render any allocation (1/N baseline, custom mix,
    optimizer output) without extra wrappers.
    """

    allocations = _coerce_allocations(weights)
    if not allocations:
        st.info("No positive weights to display.")
        return
    series = pd.Series(allocations).sort_values(ascending=False)
    series = series[series > 0]
    if series.empty:
        st.info("No positive weights to display.")
        return

    bar_col, pie_col = st.columns(2)

    with bar_col:
        try:
            st.bar_chart(series, height=320)
        except Exception as exc:  # pragma: no cover
            st.warning(f"Unable to render weights chart: {exc}")

    chart_data = _prepare_weight_chart_data(series)
    with pie_col:
        if chart_data.empty:
            st.info("No positive weights available for pie chart.")
        elif alt is None:
            st.info("Install the 'altair' package to view the allocation pie chart.")
        else:
            try:
                donut_chart = (
                    alt.Chart(chart_data)
                    .mark_arc(innerRadius=60)
                    .encode(
                        theta=alt.Theta(field="Weight", type="quantitative"),
                        color=alt.Color(field="Ticker", type="nominal"),
                        tooltip=[
                            alt.Tooltip("Ticker:N", title="Ticker"),
                            alt.Tooltip("Weight:Q", title="Weight", format=".2%"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(donut_chart, width="stretch")
            except Exception as exc:  # pragma: no cover
                st.warning(f"Unable to render allocation pie chart: {exc}")


def plot_weight_pies(allocations: dict[str, dict[str, float]]) -> None:
    """Render one donut chart per named allocation, side by side.

    Args:
        allocations: Mapping of display name → ticker→weight mapping.  The
            first entry is expected to be the 1/N equal-weight baseline.
    """

    if not allocations:
        st.info("No allocations to display.")
        return

    cols = st.columns(len(allocations))
    for col, (name, weights) in zip(cols, allocations.items()):
        with col:
            st.markdown(f"**{name}**")
            series = pd.Series(weights).sort_values(ascending=False)
            series = series[series > 0]
            chart_data = _prepare_weight_chart_data(series)
            if chart_data.empty:
                st.info("No positive weights available for pie chart.")
            elif alt is None:
                st.info(
                    "Install the 'altair' package to view the allocation pie chart."
                )
            else:
                try:
                    donut_chart = (
                        alt.Chart(chart_data)
                        .mark_arc(innerRadius=60)
                        .encode(
                            theta=alt.Theta(field="Weight", type="quantitative"),
                            color=alt.Color(field="Ticker", type="nominal"),
                            tooltip=[
                                alt.Tooltip("Ticker:N", title="Ticker"),
                                alt.Tooltip("Weight:Q", title="Weight", format=".2%"),
                            ],
                        )
                        .properties(height=240)
                    )
                    st.altair_chart(donut_chart, width="stretch")
                except Exception as exc:  # pragma: no cover
                    st.warning(f"Unable to render allocation pie chart: {exc}")


def render_metrics_table(
    metrics_result: MetricResults,
    portfolios: dict[str, OptimisationResult] | None = None,
) -> pd.DataFrame:
    """Show metric summary table and return the raw DataFrame.

    When ``portfolios`` is provided, portfolio-level rows (named by their
    keys) are appended below the per-asset rows using the same columns.
    """

    summary = pd.DataFrame(
        {
            "expected_return": metrics_result.expected,
            "annual_volatility": metrics_result.volatility,
            "sharpe_ratio": metrics_result.sharpe,
        }
    )
    if portfolios:
        portfolio_rows = {}
        for name, result in portfolios.items():
            if result is None:
                continue
            performance = result.performance
            portfolio_rows[name] = {
                "expected_return": performance.expected_return,
                "annual_volatility": performance.volatility,
                "sharpe_ratio": performance.sharpe_ratio,
            }
        if portfolio_rows:
            summary = pd.concat([summary, pd.DataFrame(portfolio_rows).T])

    st.dataframe(
        summary.style.format(
            {
                "expected_return": "{:.4f}",
                "annual_volatility": "{:.4f}",
                "sharpe_ratio": "{:.4f}",
            }
        )
    )
    return summary


# ---------------------------------------------------------------------------
# Analysis data builders (shared between plot and comparison-table renderers)
# ---------------------------------------------------------------------------


def _build_portfolio_result(
    weights: dict[str, float],
    name: str,
    price_data: pd.DataFrame,
    *,
    expected_returns: pd.Series,
    ctx: _TaxContext,
) -> OptimisationResult:
    """Evaluate a ticker→weight mapping with shrunk, tax-adjusted returns.

    Uses the same Bayes-Stein estimator and :class:`AssetLocationEngine` drag
    model as the optimized portfolio so every comparison row is
    mathematically equivalent.
    """

    expected_return, volatility, sharpe_ratio = evaluate_adjusted_performance(
        weights,
        price_data,
        expected_returns=expected_returns,
        tax_profile=ctx.tax_profile,
        asset_characteristics=ctx.asset_characteristics,
        account=ctx.account,
    )
    performance = OptimisationPerformance(
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        start_date=str(price_data.index.min().date()),
        end_date=str(price_data.index.max().date()),
    )
    return OptimisationResult(
        name=name,
        weights=PortfolioWeights(weights),
        performance=performance,
    )


def _adjust_opt_result(
    opt_result: OptimisationResult | None,
    price_data: pd.DataFrame,
    *,
    expected_returns: pd.Series,
    ctx: _TaxContext,
) -> OptimisationResult | None:
    """Re-evaluate an optimizer result with net-of-drag expected returns.

    The optimizer's weights are kept; only the reported performance is
    recomputed so the PySharpe Optimized row reflects the same MER and
    account-specific tax drag as the Custom Mix and benchmark rows.
    """

    if opt_result is None:
        return None
    expected_return, volatility, sharpe_ratio = evaluate_adjusted_performance(
        opt_result.weights.allocations,
        price_data,
        expected_returns=expected_returns,
        tax_profile=ctx.tax_profile,
        asset_characteristics=ctx.asset_characteristics,
        account=ctx.account,
    )
    return replace(
        opt_result,
        performance=replace(
            opt_result.performance,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
        ),
    )


def _fetch_comparison_benchmarks(
    price_data: pd.DataFrame, ctx: _TaxContext
) -> pd.DataFrame:
    """Fetch benchmark metrics evaluated jointly with the asset universe."""

    start_str = str(price_data.index.min().date())
    end_str = str(price_data.index.max().date())
    return fetch_benchmark_metrics(
        list(CANADIAN_BENCHMARKS.keys()),
        start_date=start_str,
        end_date=end_str,
        reference_prices=price_data,
        tax_profile=ctx.tax_profile,
        account=ctx.account,
        risk_free_rate=DEFAULT_RISK_FREE_RATE,
    )


def _compute_frontier_data(
    price_data: pd.DataFrame,
    custom_weights: dict[str, float],
    opt_result: OptimisationResult | None = None,
) -> tuple[
    OptimisationResult,  # equal_port
    OptimisationResult,  # user_port
    OptimisationResult,  # opt_result
    pd.DataFrame,  # benchmarks_df
    np.ndarray,  # frontier_rets
    np.ndarray,  # frontier_vols
]:
    """Compute all data needed for frontier plotting and comparison.

    The first result is the strict 1/N equal-weight baseline; the second is
    the re-normalized custom mix.  If *opt_result* is provided the
    optimisation step is skipped — callers that have already optimised can
    reuse the result.
    """

    assets = list(price_data.columns)
    ctx = _tax_context(price_data)
    mu = shrinkage_expected_return(
        price_data.dropna(), shrinkage_floor=DEFAULT_SHRINKAGE_FLOOR
    )

    equal_weights = equal_weight_allocation(assets)
    user_weights = _normalize_user_weights(assets, custom_weights)
    equal_port = _build_portfolio_result(
        equal_weights, EQUAL_WEIGHT_NAME, price_data, expected_returns=mu, ctx=ctx
    )
    user_port = _build_portfolio_result(
        user_weights, "Custom Mix", price_data, expected_returns=mu, ctx=ctx
    )

    # Use caller-provided optimisation result or compute fresh, then
    # re-evaluate its performance with the net-of-drag pipeline.
    if opt_result is None:
        opt_result = optimise_from_prices(price_data, base_currency="CAD")
    opt_result = _adjust_opt_result(
        opt_result, price_data, expected_returns=mu, ctx=ctx
    )

    # Fetch Benchmarks (evaluated jointly with the asset universe)
    benchmarks_df = _fetch_comparison_benchmarks(price_data, ctx)

    # Generate Frontier Points
    frontier_rets, frontier_vols = generate_efficient_frontier(price_data)

    return (
        equal_port,
        user_port,
        opt_result,
        benchmarks_df,
        frontier_rets,
        frontier_vols,
    )


# ---------------------------------------------------------------------------
# Public rendering entry points
# ---------------------------------------------------------------------------


def run_full_analysis(
    price_data: pd.DataFrame,
    custom_weights: dict[str, float],
) -> dict:
    """Execute the full analytics pipeline and return cached results.

    Computes portfolio metrics for the 1/N equal-weight baseline and the
    custom mix, runs the Sharpe optimisation, fetches benchmarks, and
    generates efficient frontier coordinates — all in one pass so the
    results can be stored in ``st.session_state`` and reused across tabs
    without re-executing heavy solvers.

    All portfolio rows and the benchmarks are evaluated with the same
    harmonized pipeline: Bayes-Stein shrunk expected returns (benchmarks
    shrunk jointly with the asset universe) reduced by MER and
    account-specific tax drag via the :class:`AssetLocationEngine`, with
    Sharpe ratios recomputed from the adjusted returns.

    Returns a dict suitable for assignment to ``st.session_state``::

        st.session_state.opt_results = run_full_analysis(prices, weights)

    The returned dict contains:
    - ``equal_port``: :class:`OptimisationResult` for the 1/N baseline
    - ``user_port``: :class:`OptimisationResult` for the re-normalized
      custom weights (falls back to 1/N when the input is degenerate)
    - ``opt_result``: :class:`OptimisationResult` from :func:`optimise_from_prices`
      with its performance re-evaluated net of MER/tax drag
    - ``benchmarks_df``: benchmark comparison DataFrame
    - ``frontier_rets``: efficient frontier return coordinates
    - ``frontier_vols``: efficient frontier volatility coordinates
    """

    assets = list(price_data.columns)
    ctx = _tax_context(price_data)
    mu = shrinkage_expected_return(
        price_data.dropna(), shrinkage_floor=DEFAULT_SHRINKAGE_FLOOR
    )

    equal_weights = equal_weight_allocation(assets)
    user_weights = _normalize_user_weights(assets, custom_weights)
    equal_port = _build_portfolio_result(
        equal_weights, EQUAL_WEIGHT_NAME, price_data, expected_returns=mu, ctx=ctx
    )
    user_port = _build_portfolio_result(
        user_weights, "Custom Mix", price_data, expected_returns=mu, ctx=ctx
    )

    # Run the portfolio optimisation
    opt_result = optimise_from_prices(price_data, base_currency="CAD")
    opt_result = _adjust_opt_result(
        opt_result, price_data, expected_returns=mu, ctx=ctx
    )

    # Fetch Benchmarks (evaluated jointly with the asset universe)
    benchmarks_df = _fetch_comparison_benchmarks(price_data, ctx)

    # Generate Frontier Points
    frontier_rets, frontier_vols = generate_efficient_frontier(price_data)

    return {
        "opt_result": opt_result,
        "equal_port": equal_port,
        "user_port": user_port,
        "benchmarks_df": benchmarks_df,
        "frontier_rets": frontier_rets,
        "frontier_vols": frontier_vols,
    }


def render_frontier_comparison(
    price_data: pd.DataFrame, custom_weights: dict[str, float]
) -> None:
    """Render the Efficient Frontier overlay plot and comparison table.

    Convenience wrapper that calls :func:`render_frontier_plot` and
    :func:`render_performance_comparison` together.  For the refactored
    tab layout, use the individual functions directly.
    """

    render_frontier_plot(price_data, custom_weights)
    st.markdown("### Performance Comparison")
    render_performance_comparison(price_data, custom_weights)


def render_frontier_plot(
    price_data: pd.DataFrame,
    custom_weights: dict[str, float],
    opt_result: OptimisationResult | None = None,
    cached: dict | None = None,
) -> None:
    """Render only the Efficient Frontier matplotlib plot (no table).

    When *cached* is provided (e.g. from ``st.session_state.opt_results``)
    the function uses pre-computed frontier data and skips all heavy
    computation.  When *cached* is ``None`` it falls back to
    :func:`_compute_frontier_data` (which may trigger a fresh optimisation
    if *opt_result* is also ``None``).
    """

    if price_data.empty or len(price_data.columns) < 2:
        return

    try:
        if cached is not None:
            user_port = cached["user_port"]
            opt_result = cached["opt_result"]
            benchmarks_df = cached["benchmarks_df"]
            frontier_rets = cached["frontier_rets"]
            frontier_vols = cached["frontier_vols"]
        else:
            _, user_port, opt_result, benchmarks_df, frontier_rets, frontier_vols = (
                _compute_frontier_data(price_data, custom_weights, opt_result)
            )
        fig = plot_portfolio_comparison(
            frontier_returns=frontier_rets,
            frontier_vols=frontier_vols,
            user_portfolio=user_port,
            optimized_portfolio=opt_result,
            benchmarks_df=benchmarks_df,
            prices=price_data,
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating Efficient Frontier plot: {e}")


def render_performance_comparison(
    price_data: pd.DataFrame,
    custom_weights: dict[str, float],
    opt_result: OptimisationResult | None = None,
    cached: dict | None = None,
) -> None:
    """Render only the Performance Comparison markdown table.

    The 1/N equal-weight baseline is always listed first so the custom mix
    and optimizer output can be judged against the neutral reference.  When
    *cached* is provided the function uses pre-computed data and skips all
    heavy computation.
    """

    if price_data.empty or len(price_data.columns) < 2:
        return

    try:
        if cached is not None:
            equal_port = cached.get("equal_port")
            user_port = cached["user_port"]
            opt_result = cached["opt_result"]
            benchmarks_df = cached["benchmarks_df"]
        else:
            equal_port, user_port, opt_result, benchmarks_df, _, _ = (
                _compute_frontier_data(price_data, custom_weights, opt_result)
            )

        comp_data: list[dict[str, str]] = []

        def _append_portfolio(name: str, result: OptimisationResult | None) -> None:
            if result is None:
                return
            performance = result.performance
            comp_data.append(
                {
                    "Portfolio": name,
                    "Expected Return": f"{performance.expected_return:.2%}",
                    "Volatility": f"{performance.volatility:.2%}",
                    "Sharpe Ratio": f"{performance.sharpe_ratio:.2f}",
                }
            )

        _append_portfolio(EQUAL_WEIGHT_NAME, equal_port)
        _append_portfolio("Custom Mix", user_port)
        _append_portfolio("PySharpe Optimized", opt_result)

        for _, row in benchmarks_df.iterrows():
            comp_data.append(
                {
                    "Portfolio": f"Benchmark: {row['Ticker']}",
                    "Expected Return": f"{row['Annualized Return']:.2%}",
                    "Volatility": f"{row['Annualized Volatility']:.2%}",
                    "Sharpe Ratio": f"{row['Sharpe Ratio']:.2f}",
                }
            )

        st.caption(
            "All rows use Bayes-Stein shrunk expected returns (benchmarks are "
            "shrunk jointly with the asset universe) reduced by MER and "
            "account-specific tax drag via the AssetLocationEngine.  Benchmarks "
            "and the Custom Mix assume **Non-Registered** account placement with "
            "the configured TaxProfile unless an explicit asset-location split "
            f"is provided.  Sharpe ratios use the {DEFAULT_RISK_FREE_RATE:.0%} "
            "risk-free rate."
        )
        st.table(pd.DataFrame(comp_data))
    except Exception as e:
        st.error(f"Error generating Performance Comparison: {e}")


__all__ = [
    "EQUAL_WEIGHT_NAME",
    "equal_weight_allocation",
    "plot_cumulative_returns",
    "plot_weight_pies",
    "plot_weights",
    "render_metrics_table",
    "run_full_analysis",
    "render_frontier_comparison",
    "render_frontier_plot",
    "render_performance_comparison",
]
