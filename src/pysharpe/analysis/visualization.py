"""
Visualization utilities for security scoring analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

def plot_score_distribution(
    df: pd.DataFrame,
    score_col: str = 'CompositeScore',
    title: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot the distribution of scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing scores
    score_col : str
        Name of the score column to plot
    title : str, optional
        Custom title for the plot
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    sns.histplot(df[score_col], kde=True, bins=15)
    plt.title(title or f"Distribution of {score_col}")
    plt.xlabel(score_col)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_score_comparison(
    df: pd.DataFrame,
    x_score: str = 'TechScore',
    y_score: str = 'DivScore',
    figsize: tuple = (12, 8)
) -> None:
    """
    Create a scatter plot comparing two different scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing scores and ticker symbols
    x_score : str
        Name of the score column for x-axis
    y_score : str
        Name of the score column for y-axis
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x_score, y=y_score, hue='Ticker', s=100)
    plt.title(f"{x_score} vs {y_score}")
    plt.xlabel(x_score)
    plt.ylabel(y_score)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_backtest_results(
    portfolio_values: list,
    periods: int,
    initial_value: float,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot backtest results showing portfolio value over time.

    Parameters
    ----------
    portfolio_values : list
        List of portfolio values over time
    periods : int
        Number of periods in the backtest
    initial_value : float
        Initial portfolio value
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    plt.plot(range(periods + 1), portfolio_values, marker='o')
    plt.title("Backtest: Portfolio Performance")
    plt.xlabel("Period (months)")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    
    # Add annotations for start and end values
    plt.annotate(f"Start: ${initial_value:,.0f}", 
                (0, initial_value),
                xytext=(10, 10),
                textcoords='offset points')
    plt.annotate(f"End: ${portfolio_values[-1]:,.0f}",
                (periods, portfolio_values[-1]),
                xytext=(10, 10),
                textcoords='offset points')
    
    plt.tight_layout()