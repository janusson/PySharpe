# Test-to-Module Mapping (Quick Reference)

Run the right subset of tests for your change instead of the full suite.

## Test File → Source Module Coverage

| Test File | Modules Covered |
|-----------|----------------|
| `test_tax_location.py` | `optimization/tax_location` |
| `test_tax_tracker.py` | `execution/tax_tracker` |
| `test_2d_allocation.py` | `config`, `execution/allocator`, `optimization/sharpe_optimizer`, `optimization/tax_location` |
| `test_rebalance.py` | `execution/rebalance` |
| `test_metrics.py` | `metrics` |
| `test_optimization_base.py` | `exceptions`, `optimization/base`, `optimization/bayesian`, `optimization/sharpe_optimizer` |
| `test_optimization_weights.py` | `optimization/weights` |
| `test_optimization_models.py` | `optimization/models`, `optimization/weights` |
| `test_portfolio_optimization.py` | `portfolio_optimization`, `optimization/models` |
| `test_fx_adjustment.py` | `data/fetcher` |
| `test_data_fetcher_cache.py` | `data/fetcher` |
| `test_fetcher.py` | `data/fetcher` |
| `test_collation.py` | `data/collation`, `data/fetcher` |
| `test_collation_proxy.py` | `config`, `data/collation`, `data/fetcher` |
| `test_data_linkage.py` | `data/linkage` |
| `test_data_linkage_stitched.py` | `data/fetcher`, `data/linkage` |
| `test_cli.py` | `cli`, `workflows`, `optimization/models` |
| `test_package_api.py` | `__init__` |
| `test_config.py` | `config` |
| `test_analysis.py` | `analysis/backtest`, `analysis/scoring` |
| `test_analysis_backtest_engine.py` | `analysis/backtest_engine` |
| `test_analysis_benchmarks.py` | `analysis/benchmarks` |
| `test_analysis_time_series.py` | `analysis/time_series` |
| `test_analysis_transaction_costs.py` | `analysis/backtest_engine` |
| `test_analysis_walk_forward.py` | `analysis/backtest_engine`, `optimization/base` |
| `test_analysis_visualization.py` | `analysis/visualization` |
| `test_categorization.py` | `analysis/categorization` |
| `test_app_streamlit.py` | `app`, `optimization/models` |
| `test_app_helpers.py` | `app` |
| `test_backtest_page.py` | `analysis/backtest_engine`, `metrics` |
| `test_dca.py` | `visualization/dca` |
| `test_visualization_frontier.py` | `optimization/models`, `visualization/frontier` |
| `test_workflows.py` | `workflows`, `optimization/models` |
| `test_data_collector.py` | `data_collector` |
| `test_constraints_verification.py` | `portfolio_optimization` |
| `test_logging_utils.py` | `logging_utils` |
| `test_resampling.py` | `validation/resampling` |

## Module → Test Subset (Reverse Lookup)

| Changing this module | Run |
|---------------------|-----|
| `optimization/tax_location.py` | `pytest tests/test_tax_location.py tests/test_tax_tracker.py tests/test_2d_allocation.py` |
| `execution/tax_tracker.py` | `pytest tests/test_tax_tracker.py` |
| `execution/allocator.py` | `pytest tests/test_2d_allocation.py` |
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
| `validation/resampling.py` | `pytest tests/test_resampling.py` |
