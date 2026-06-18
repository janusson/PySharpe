import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pysharpe.analysis.visualization import (
    plot_backtest_results,
    plot_score_comparison,
    plot_score_distribution,
)


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


def test_plot_score_distribution():
    df = pd.DataFrame({"CompositeScore": [0.1, 0.5, 0.8, 0.9, 0.5]})
    plot_score_distribution(df, title="Test Dist")

    fig = plt.gcf()
    assert fig.axes[0].get_title() == "Test Dist"


def test_plot_score_comparison():
    df = pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "GOOGL"],
            "TechScore": [0.6, 0.8, 0.7],
            "DivScore": [0.2, 0.4, 0.0],
        }
    )
    plot_score_comparison(df)

    fig = plt.gcf()
    assert "TechScore vs DivScore" in fig.axes[0].get_title()


def test_plot_backtest_results():
    values = [1000.0, 1050.0, 1100.0, 1080.0]
    plot_backtest_results(values, periods=3, initial_value=1000.0)

    fig = plt.gcf()
    assert "Backtest: Portfolio Performance" in fig.axes[0].get_title()
