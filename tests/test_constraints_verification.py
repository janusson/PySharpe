import numpy as np
import pandas as pd
import pytest

from pysharpe.portfolio_optimization import optimise_portfolio


def test_optimisation_constraints(tmp_path):
    # Setup dummy data
    dates = pd.date_range("2020-01-01", periods=300)

    np.random.seed(42)
    # A: High Risk, High Return, High MER
    # B: Low Risk, Low Return, Low MER
    # C: Noise

    # We construct returns such that they are well behaved.
    returns = pd.DataFrame(index=dates)
    returns["A"] = np.random.normal(0.001, 0.02, 300)
    returns["B"] = np.random.normal(0.0005, 0.005, 300)
    returns["C"] = np.random.normal(0.0002, 0.01, 300)

    # Force expected returns to be positive and distinct
    # Annualized: A ~ 25%, B ~ 12%, C ~ 5%

    prices = (1 + returns).cumprod() * 100
    prices.index.name = "Date"

    collated_dir = tmp_path / "collated"
    collated_dir.mkdir()
    prices.to_csv(collated_dir / "test_portfolio_collated.csv")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # MER mapping
    mer_mapping = {"A": 0.05, "B": 0.001, "C": 0.02}

    # Geo mapping
    geo_mapping = {"A": "US", "B": "CA", "C": "INT"}

    # --- Test 1: Max MER constraint ---
    # We set a max MER that forces allocation into B.
    # A (0.05) is desirable for return, but B (0.001) is needed for MER.

    max_mer = 0.015  # 1.5%

    # If solver fails, it might be due to data issues. We try/except to catch it.
    try:
        result_mer = optimise_portfolio(
            "test_portfolio",
            collated_dir=collated_dir,
            output_dir=output_dir,
            mer_mapping=mer_mapping,
            max_portfolio_mer=max_mer,
            make_plot=False,
            max_weight=1.0,
        )
        weights_mer = result_mer.weights.allocations
        port_mer = sum(weights_mer.get(t, 0) * mer_mapping[t] for t in weights_mer)

        print(f"\nTest 1 Weights: {weights_mer}")
        print(f"Portfolio MER: {port_mer}")

        assert port_mer <= max_mer + 1e-5

    except Exception as e:
        pytest.fail(f"Test 1 failed with error: {e}")

    # --- Test 2: Geo constraint ---
    # Limit US (A) to 10%
    try:
        result_geo = optimise_portfolio(
            "test_portfolio",
            collated_dir=collated_dir,
            output_dir=output_dir,
            geo_mapping=geo_mapping,
            geo_upper_bounds={"US": 0.40},
            make_plot=False,
            max_weight=1.0,
        )

        weights_geo = result_geo.weights.allocations
        print(f"\nTest 2 Weights: {weights_geo}")

        assert weights_geo.get("A", 0) <= 0.40 + 1e-5

    except Exception as e:
        pytest.fail(f"Test 2 failed with error: {e}")

    # --- Test 3: Combined Constraints ---
    try:
        result_combined = optimise_portfolio(
            "test_portfolio",
            collated_dir=collated_dir,
            output_dir=output_dir,
            mer_mapping=mer_mapping,
            max_portfolio_mer=0.01,
            geo_mapping=geo_mapping,
            geo_upper_bounds={"CA": 0.6},
            make_plot=False,
            max_weight=1.0,
        )

        weights_combined = result_combined.weights.allocations
        port_mer_combined = sum(
            weights_combined.get(t, 0) * mer_mapping[t] for t in weights_combined
        )

        print(f"\nTest 3 Weights: {weights_combined}")
        print(f"Combined MER: {port_mer_combined}")

        assert port_mer_combined <= 0.01 + 1e-5
        assert weights_combined.get("B", 0) <= 0.6 + 1e-5

    except Exception as e:
        pytest.fail(f"Test 3 failed with error: {e}")
