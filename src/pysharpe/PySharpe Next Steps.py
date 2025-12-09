# PySharpe Next Steps
## 2025-11-06

Right now src/pysharpe/analysis/visualization.py imports matplotlib.pyplot as plt and seaborn as sns at module import time. That means the module raises ImportError immediately if either package is missing—even if the caller only wants the non-plotting parts of pysharpe.analysis. To mirror the optional-dependency handling we just added elsewhere, we can defer those imports until someone actually calls a plotting function. Here’s the approach:

Move heavy imports inside helpers. Within each plotting function (e.g., plot_score_distribution, plot_score_comparison, plot_backtest_results), import matplotlib/seaborn at the top of the function. Wrap those imports in a small utility—either reuse pysharpe.visualization.require_matplotlib() for pyplot and create a similar require_seaborn() helper, or just raise a friendly RuntimeError inside the function if the import fails.

Update docstrings/error messages. Make the functions document that they will raise a clear exception when the optional libraries aren’t installed. This keeps users informed and avoids confusing ImportError tracebacks.

Adjust tests if they touch the module. If any tests import pysharpe.analysis.visualization while stubbing plotting behavior, patch the new helper (similar to how the CLI tests patch require_matplotlib) so tests remain self-contained without actually needing seaborn/matplotlib.

Benefit: After deferring imports, any code that only needs the scoring/backtest utilities under pysharpe.analysis can run without forcing heavyweight plotting dependencies, improving modularity and matching the optional behavior we now guarantee for other visualization entry points.