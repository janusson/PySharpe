# Test-to-Module Map

Which test files cover which source modules. Run the right subset of tests for
your change instead of the full suite every time.

---

## `test_tax_location.py`
- `pysharpe.optimization.tax_location`
- `pysharpe.execution.rebalance` (tax-engine integration: `build_rebalance_plan` with `TaxProfile`/`AssetTaxCharacteristics`)

## `test_tax_tracker.py`
- `pysharpe.execution.tax_tracker`

## `test_2d_allocation.py`
- `pysharpe.config`
- `pysharpe.execution.allocator`
- `pysharpe.optimization.sharpe_optimizer`
- `pysharpe.optimization.tax_location`

## `test_brokerage.py`
- `pysharpe.execution.brokerage`
- `pysharpe.execution.rebalance`

## `test_metrics.py`
- `pysharpe.metrics`

## `test_optimization_base.py`
- `pysharpe.exceptions`
- `pysharpe.optimization.base`
- `pysharpe.optimization.bayesian`
- `pysharpe.optimization.sharpe_optimizer`

## `test_covariance_shrinkage.py`
- `pysharpe.optimization.estimators` (collinearity extremes: perfect / near-perfect correlation, rank-1 high-dim)
- `pysharpe.exceptions`

## `test_optimization_hrp.py`
- `pysharpe.optimization.hrp`
- `pysharpe.optimization.estimators` (shared `prepare_returns`)
- `pysharpe.exceptions`

## `test_optimization_bayesian.py`
- `pysharpe.optimization.bayesian`
- `pysharpe.optimization.estimators` (posterior eigen-clipping)
- PyPortfolioOpt `EfficientFrontier` integration (mocked trace, no MCMC)
- PyMC sampler isolated via `pytest.MonkeyPatch` — no C compilation needed

## `test_black_litterman.py`
- `pysharpe.optimization.black_litterman`
- `pysharpe.optimization.estimators` (strict-PSD hardening)
- `pysharpe.exceptions`

## `test_optimization_weights.py`
- `pysharpe.optimization.weights`
- `pysharpe.optimization` (result dataclasses: `PortfolioWeights`, `OptimisationResult`, `OptimisationPerformance`)

## `test_portfolio_optimization.py`
- `pysharpe.portfolio_optimization`
- `pysharpe.optimization.models`

## `test_fx_adjustment.py`
- `pysharpe.data.fetcher`

## `test_data_fetcher_cache.py`
- `pysharpe.data.fetcher`

## `test_fetcher.py`
- `pysharpe.data.fetcher`

## `test_collation.py`
- `pysharpe.config`
- `pysharpe.data.collation`
- `pysharpe.data.fetcher`
- Proxy-map resolution (merged from the former `test_collation_proxy.py`)

## `test_data_linkage.py`
- `pysharpe.data.linkage`
- `pysharpe.data.fetcher`
- Stitched proxy-history coverage (merged from the former `test_data_linkage_stitched.py`)

## `test_cli.py`
- `pysharpe.cli`
- `pysharpe.workflows`
- `pysharpe.optimization.models`

## `test_package_api.py`
- `pysharpe.__init__`

## `test_config.py`
- `pysharpe.config`
- `pysharpe.logging_utils`

## `test_analysis.py`
- `pysharpe.analysis.backtest`
- `pysharpe.analysis.benchmarks`
- `pysharpe.analysis.scoring`
- `pysharpe.analysis.visualization`

## `test_analysis_backtest_engine.py`
- `pysharpe.analysis.backtest_engine`
- `pysharpe.optimization.base`
- Missing-price (NaN) handling incl. fully-missing asset columns

## `test_analysis_time_series.py`
- `pysharpe.analysis.time_series`

## `test_analysis_transaction_costs.py`
- `pysharpe.analysis.backtest_engine` (spread/slippage/commission model, walk-forward costs, no-lookahead)

## `test_resampling.py`
- `pysharpe.validation.resampling` (PurgedKFold leakage guarantees, autocorrelation-decay gap sizing, regime bootstrapping)

## `test_validation_metrics.py`
- `pysharpe.validation.metrics` (DSR incl. Lo θ adjustment, effective trials)

## `test_categorization.py`
- `pysharpe.analysis.categorization`

## `test_analysis_comparison.py`
- `pysharpe.analysis.comparison`
- `pysharpe.metrics`

## `test_cash_flow_rebalance.py`
- `pysharpe.execution.cash_flow_rebalance` (multi-account contribution routing, taxable-sale guardrails)

## `test_tax_compliance.py`
- `pysharpe.guardrails.tax_compliance` (ACB tracking, superficial-loss interlock, multi-account validation)

## `test_friction.py`
- `pysharpe.validation.friction` (stress-testing execution friction)

## `test_app_streamlit.py`
- `app`
- `pysharpe.optimization.models`

## `test_app_helpers.py`
- `app`

## `test_backtest_page.py`
- `pysharpe.analysis.backtest_engine`
- `pysharpe.metrics`

## `test_visualization_frontier.py`
- `pysharpe.optimization.models`
- `pysharpe.visualization.dca`
- `pysharpe.visualization.frontier`

## `test_workflows.py`
- `pysharpe.workflows`
- `pysharpe.optimization.models`

## `test_data_collector.py`
- `pysharpe.data_collector`

## `test_ledger.py`
- `pysharpe.validation.ledger`

## `test_sample_size.py`
- `pysharpe.validation.sample_size`

---

## Quick Reference — Run These For Your Change

| Changing this module | Run |
|---------------------|-----|
| `__init__.py` | `pytest tests/test_package_api.py` |
| `analysis/backtest.py`, `analysis/benchmarks.py`, `analysis/scoring.py`, `analysis/visualization.py` | `pytest tests/test_analysis.py` |
| `analysis/backtest_engine.py` | `pytest tests/test_analysis_backtest_engine.py tests/test_analysis_transaction_costs.py` |
| `analysis/categorization.py` | `pytest tests/test_categorization.py` |
| `analysis/comparison.py` | `pytest tests/test_analysis_comparison.py` |
| `analysis/time_series.py` | `pytest tests/test_analysis_time_series.py` |
| `app.py`, `app/*.py` | `pytest tests/test_app_streamlit.py tests/test_app_helpers.py` |
| `cli.py` | `pytest tests/test_cli.py` |
| `config.py` | `pytest tests/test_config.py tests/test_2d_allocation.py` |
| `data/collation.py` | `pytest tests/test_collation.py` |
| `data/fetcher.py` | `pytest tests/test_fetcher.py tests/test_fx_adjustment.py tests/test_data_fetcher_cache.py tests/test_collation.py` |
| `data/linkage.py` | `pytest tests/test_data_linkage.py` |
| `data_collector.py` | `pytest tests/test_data_collector.py` |
| `exceptions.py` | `pytest tests/test_optimization_base.py tests/test_covariance_shrinkage.py tests/test_black_litterman.py tests/test_optimization_hrp.py` |
| `execution/allocator.py` | `pytest tests/test_2d_allocation.py` |
| `execution/brokerage.py` | `pytest tests/test_brokerage.py` |
| `execution/cash_flow_rebalance.py` | `pytest tests/test_cash_flow_rebalance.py` |
| `execution/rebalance.py` | `pytest tests/test_brokerage.py tests/test_tax_location.py` |
| `execution/tax_tracker.py` | `pytest tests/test_tax_tracker.py` |
| `guardrails/tax_compliance.py` | `pytest tests/test_tax_compliance.py` |
| `logging_utils.py` | `pytest tests/test_config.py` |
| `metrics.py` | `pytest tests/test_metrics.py tests/test_analysis_comparison.py` |
| `optimization/base.py` | `pytest tests/test_optimization_base.py tests/test_analysis_backtest_engine.py` |
| `optimization/bayesian.py` | `pytest tests/test_optimization_bayesian.py tests/test_optimization_base.py` |
| `optimization/black_litterman.py` | `pytest tests/test_black_litterman.py` |
| `optimization/estimators.py` | `pytest tests/test_covariance_shrinkage.py tests/test_optimization_hrp.py tests/test_optimization_bayesian.py tests/test_black_litterman.py` |
| `optimization/expected_returns.py` | `pytest tests/test_optimization_base.py` |
| `optimization/hrp.py` | `pytest tests/test_optimization_hrp.py` |
| `optimization/models.py` | `pytest tests/test_portfolio_optimization.py tests/test_workflows.py tests/test_cli.py tests/test_visualization_frontier.py tests/test_app_streamlit.py` |
| `optimization/sharpe_optimizer.py` | `pytest tests/test_optimization_base.py tests/test_portfolio_optimization.py` |
| `optimization/tax_location.py` | `pytest tests/test_tax_location.py tests/test_tax_tracker.py tests/test_2d_allocation.py` |
| `optimization/weights.py` | `pytest tests/test_optimization_weights.py` |
| `portfolio_optimization.py` | `pytest tests/test_portfolio_optimization.py` |
| `validation/friction.py` | `pytest tests/test_friction.py` |
| `validation/ledger.py` | `pytest tests/test_ledger.py` |
| `validation/metrics.py` | `pytest tests/test_validation_metrics.py` |
| `validation/resampling.py` | `pytest tests/test_resampling.py` |
| `validation/sample_size.py` | `pytest tests/test_sample_size.py` |
| `workflows.py` | `pytest tests/test_workflows.py` |
