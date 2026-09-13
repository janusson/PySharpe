# Test-to-Module Mapping (Quick Reference)

Run the right subset of tests for your change instead of the full suite.

## Test File → Source Module Coverage

| Test File | Modules Covered |
|-----------|----------------|
| `test_tax_location.py` | `optimization/tax_location` |
| `test_tax_tracker.py` | `execution/tax_tracker` |
| `test_2d_allocation.py` | `config`, `execution/allocator`, `optimization/sharpe_optimizer`, `optimization/tax_location` |
| `test_brokerage.py` | `execution/brokerage`, `execution/rebalance` |
| `test_cash_flow_rebalance.py` | `execution/cash_flow_rebalance` |
| `test_metrics.py` | `metrics` |
| `test_optimization_base.py` | `exceptions`, `optimization/base`, `optimization/bayesian`, `optimization/sharpe_optimizer` |
| `test_optimization_weights.py` | `optimization/weights` |
| `test_optimization_hrp.py` | `optimization/hrp`, `optimization/estimators`, `exceptions` |
| `test_optimization_bayesian.py` | `optimization/bayesian`, `optimization/estimators` |
| `test_black_litterman.py` | `optimization/black_litterman`, `optimization/estimators`, `exceptions` |
| `test_covariance_shrinkage.py` | `optimization/estimators`, `exceptions` |
| `test_portfolio_optimization.py` | `portfolio_optimization`, `optimization/models` |
| `test_fx_adjustment.py` | `data/fetcher` |
| `test_data_fetcher_cache.py` | `data/fetcher` |
| `test_fetcher.py` | `data/fetcher` |
| `test_collation.py` | `config`, `data/collation`, `data/fetcher` |
| `test_data_linkage.py` | `data/fetcher`, `data/linkage` |
| `test_cli.py` | `cli`, `workflows`, `optimization/models` |
| `test_package_api.py` | `__init__` |
| `test_config.py` | `config`, `logging_utils` |
| `test_analysis.py` | `analysis/backtest`, `analysis/benchmarks`, `analysis/scoring`, `analysis/visualization` |
| `test_analysis_backtest_engine.py` | `analysis/backtest_engine`, `optimization/base` |
| `test_analysis_time_series.py` | `analysis/time_series` |
| `test_analysis_transaction_costs.py` | `analysis/backtest_engine` |
| `test_categorization.py` | `analysis/categorization` |
| `test_analysis_comparison.py` | `analysis/comparison`, `metrics` |
| `test_app_streamlit.py` | `app`, `optimization/models` |
| `test_app_helpers.py` | `app` |
| `test_backtest_page.py` | `analysis/backtest_engine`, `metrics` |
| `test_visualization_frontier.py` | `optimization/models`, `visualization/dca`, `visualization/frontier` |
| `test_workflows.py` | `workflows`, `optimization/models` |
| `test_data_collector.py` | `data_collector` |
| `test_resampling.py` | `validation/resampling` |
| `test_ledger.py` | `validation/ledger` |
| `test_friction.py` | `validation/friction` |
| `test_validation_metrics.py` | `validation/metrics` |
| `test_sample_size.py` | `validation/sample_size` |
| `test_tax_compliance.py` | `guardrails/tax_compliance` |

## Module → Test Subset (Reverse Lookup)

| Changing this module | Run |
|---------------------|-----|
| `optimization/tax_location.py` | `pytest tests/test_tax_location.py tests/test_tax_tracker.py tests/test_2d_allocation.py` |
| `execution/tax_tracker.py` | `pytest tests/test_tax_tracker.py` |
| `execution/allocator.py` | `pytest tests/test_2d_allocation.py` |
| `execution/rebalance.py` | `pytest tests/test_brokerage.py tests/test_tax_location.py` |
| `execution/brokerage.py` | `pytest tests/test_brokerage.py` |
| `execution/cash_flow_rebalance.py` | `pytest tests/test_cash_flow_rebalance.py` |
| `guardrails/tax_compliance.py` | `pytest tests/test_tax_compliance.py` |
| `optimization/expected_returns.py` | `pytest tests/test_optimization_base.py` |
| `optimization/bayesian.py` | `pytest tests/test_optimization_bayesian.py tests/test_optimization_base.py` |
| `optimization/black_litterman.py` | `pytest tests/test_black_litterman.py` |
| `optimization/estimators.py` | `pytest tests/test_covariance_shrinkage.py tests/test_optimization_hrp.py tests/test_optimization_bayesian.py tests/test_black_litterman.py` |
| `optimization/hrp.py` | `pytest tests/test_optimization_hrp.py` |
| `optimization/models.py` | `pytest tests/test_portfolio_optimization.py tests/test_workflows.py tests/test_cli.py tests/test_visualization_frontier.py tests/test_app_streamlit.py` |
| `optimization/sharpe_optimizer.py` | `pytest tests/test_optimization_base.py tests/test_portfolio_optimization.py` |
| `optimization/weights.py` | `pytest tests/test_optimization_weights.py` |
| `portfolio_optimization.py` | `pytest tests/test_portfolio_optimization.py` |
| `data/fetcher.py` | `pytest tests/test_fetcher.py tests/test_fx_adjustment.py tests/test_data_fetcher_cache.py tests/test_collation.py` |
| `data/collation.py` | `pytest tests/test_collation.py` |
| `cli.py` | `pytest tests/test_cli.py` |
| `__init__.py` | `pytest tests/test_package_api.py` |
| `app.py`, `app/*.py` | `pytest tests/test_app_streamlit.py tests/test_app_helpers.py` |
| `analysis/backtest_engine.py` | `pytest tests/test_analysis_backtest_engine.py tests/test_analysis_transaction_costs.py` |
| `analysis/comparison.py` | `pytest tests/test_analysis_comparison.py` |
| `metrics.py` | `pytest tests/test_metrics.py tests/test_analysis_comparison.py` |
| `validation/resampling.py` | `pytest tests/test_resampling.py` |
| `validation/ledger.py` | `pytest tests/test_ledger.py` |
| `validation/friction.py` | `pytest tests/test_friction.py` |
| `validation/sample_size.py` | `pytest tests/test_sample_size.py` |
| `validation/metrics.py` | `pytest tests/test_validation_metrics.py` |
| `analysis/time_series.py` | `pytest tests/test_analysis_time_series.py` |
| `data/linkage.py` | `pytest tests/test_data_linkage.py` |
| `data_collector.py` | `pytest tests/test_data_collector.py` |
| `workflows.py` | `pytest tests/test_workflows.py` |
| `visualization/dca.py` | `pytest tests/test_visualization_frontier.py` |
| `visualization/frontier.py` | `pytest tests/test_visualization_frontier.py` |
| `logging_utils.py` | `pytest tests/test_config.py` |
| `config.py` | `pytest tests/test_config.py tests/test_2d_allocation.py` |
