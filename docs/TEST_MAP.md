# Test-to-Module Map

Which test files cover which source modules. Run the right subset of tests for
your change instead of the full suite every time.

---

## `test_tax_location.py`
- `pysharpe.optimization.tax_location`

## `test_tax_tracker.py`
- `pysharpe.execution.tax_tracker`

## `test_2d_allocation.py`
- `pysharpe.config`
- `pysharpe.execution.allocator`
- `pysharpe.optimization.sharpe_optimizer`
- `pysharpe.optimization.tax_location`

## `test_rebalance.py`
- `pysharpe.execution.rebalance`

## `test_brokerage.py`
- `pysharpe.execution.brokerage`

## `test_metrics.py`
- `pysharpe.metrics`

## `test_optimization_base.py`
- `pysharpe.exceptions`
- `pysharpe.optimization.base`
- `pysharpe.optimization.bayesian`
- `pysharpe.optimization.sharpe_optimizer`

## `test_optimization_weights.py`
- `pysharpe.optimization.weights`

## `test_optimization_models.py`
- `pysharpe.optimization.models`
- `pysharpe.optimization.weights`

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
- `pysharpe.data.collation`
- `pysharpe.data.fetcher`

## `test_collation_proxy.py`
- `pysharpe.config`
- `pysharpe.data.collation`
- `pysharpe.data.fetcher`

## `test_data_linkage.py`
- `pysharpe.data.linkage`

## `test_data_linkage_stitched.py`
- `pysharpe.data.fetcher`
- `pysharpe.data.linkage`

## `test_cli.py`
- `pysharpe.cli`
- `pysharpe.workflows`
- `pysharpe.optimization.models`

## `test_package_api.py`
- `pysharpe.__init__`

## `test_config.py`
- `pysharpe.config`

## `test_analysis.py`
- `pysharpe.analysis.backtest`
- `pysharpe.analysis.scoring`

## `test_analysis_backtest_engine.py`
- `pysharpe.analysis.backtest_engine`

## `test_analysis_benchmarks.py`
- `pysharpe.analysis.benchmarks`

## `test_analysis_time_series.py`
- `pysharpe.analysis.time_series`

## `test_analysis_transaction_costs.py`
- `pysharpe.analysis.backtest_engine`

## `test_analysis_walk_forward.py`
- `pysharpe.analysis.backtest_engine`
- `pysharpe.optimization.base`

## `test_analysis_visualization.py`
- `pysharpe.analysis.visualization`

## `test_categorization.py`
- `pysharpe.analysis.categorization`

## `test_app_streamlit.py`
- `app`
- `pysharpe.optimization.models`

## `test_app_helpers.py`
- `app`

## `test_backtest_page.py`
- `pysharpe.analysis.backtest_engine`
- `pysharpe.metrics`

## `test_dca.py`
- `pysharpe.visualization.dca`

## `test_visualization_frontier.py`
- `pysharpe.optimization.models`
- `pysharpe.visualization.frontier`

## `test_workflows.py`
- `pysharpe.workflows`
- `pysharpe.optimization.models`

## `test_data_collector.py`
- `pysharpe.data_collector`

## `test_constraints_verification.py`
- `pysharpe.portfolio_optimization`

## `test_logging_utils.py`
- `pysharpe.logging_utils`

## `test_ledger.py`
- `pysharpe.validation.ledger`

## `test_sample_size.py`
- `pysharpe.validation.sample_size`

---

## Quick Reference — Run These For Your Change

| Changing this module | Run |
|---------------------|-----|
| `optimization/tax_location.py` | `pytest tests/test_tax_location.py tests/test_tax_tracker.py tests/test_2d_allocation.py` |
| `execution/tax_tracker.py` | `pytest tests/test_tax_tracker.py` |
| `execution/allocator.py` | `pytest tests/test_2d_allocation.py` |
| `execution/rebalance.py` | `pytest tests/test_rebalance.py tests/test_brokerage.py` |
| `execution/brokerage.py` | `pytest tests/test_brokerage.py` |
| `optimization/expected_returns.py` | `pytest tests/test_optimization_base.py` |
| `optimization/sharpe_optimizer.py` | `pytest tests/test_optimization_base.py tests/test_portfolio_optimization.py` |
| `portfolio_optimization.py` | `pytest tests/test_portfolio_optimization.py tests/test_optimization_models.py` |
| `data/fetcher.py` | `pytest tests/test_fetcher.py tests/test_fx_adjustment.py tests/test_data_fetcher_cache.py tests/test_collation.py` |
| `data/collation.py` | `pytest tests/test_collation.py tests/test_collation_proxy.py` |
| `cli.py` | `pytest tests/test_cli.py` |
| `__init__.py` | `pytest tests/test_package_api.py` |
| `app.py`, `app/*.py` | `pytest tests/test_app_streamlit.py tests/test_app_helpers.py` |
| `analysis/backtest_engine.py` | `pytest tests/test_analysis_backtest_engine.py tests/test_analysis_transaction_costs.py` |
| `metrics.py` | `pytest tests/test_metrics.py` |
| `validation/ledger.py` | `pytest tests/test_ledger.py` |
| `validation/sample_size.py` | `pytest tests/test_sample_size.py` |
