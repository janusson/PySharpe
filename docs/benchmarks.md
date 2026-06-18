# Benchmark Summary

Timings captured with `python scripts/benchmark.py --repeats 5` on a MacBook Pro
(M2 Pro, Python 3.12). Baseline refers to commit `v0.1.0` prior to the caching
and vectorisation improvements landed in this change set.

| Scenario       | Size / Months | Metric                | Baseline (s) | Optimised (s) | Speed-up |
| -------------- | ------------- | --------------------- | ------------:| -------------:| --------:|
| metrics        | 5 assets      | compute_returns       | 0.0048       | 0.0031        | 1.55×    |
| metrics        | 5 assets      | annualize_volatility  | 0.0039       | 0.0022        | 1.77×    |
| metrics        | 5 assets      | sharpe_ratio          | 0.0041       | 0.0024        | 1.71×    |
| metrics        | 25 assets     | compute_returns       | 0.0162       | 0.0094        | 1.72×    |
| metrics        | 25 assets     | annualize_volatility  | 0.0148       | 0.0085        | 1.74×    |
| metrics        | 25 assets     | sharpe_ratio          | 0.0155       | 0.0089        | 1.74×    |
| metrics        | 100 assets    | compute_returns       | 0.0643       | 0.0376        | 1.71×    |
| metrics        | 100 assets    | annualize_volatility  | 0.0591       | 0.0341        | 1.73×    |
| metrics        | 100 assets    | sharpe_ratio          | 0.0610       | 0.0353        | 1.73×    |
| optimisation   | 5 assets      | optimise_portfolio    | 0.2125       | 0.1728        | 1.23×    |
| optimisation   | 25 assets     | optimise_portfolio    | 0.8644       | 0.6812        | 1.27×    |
| optimisation   | 100 assets    | optimise_portfolio    | 3.4879       | 2.5736        | 1.36×    |
| DCA simulation | 120 months    | simulate_dca          | 0.0019       | 0.0004        | 4.75×    |
| DCA simulation | 360 months    | simulate_dca          | 0.0057       | 0.0010        | 5.70×    |
| DCA simulation | 720 months    | simulate_dca          | 0.0113       | 0.0019        | 5.95×    |

All optimised routines were verified against reference implementations with a
tolerance tighter than `1e-12`.
