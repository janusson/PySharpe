"""Dollar-cost averaging projection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - type checking aide
    import matplotlib.pyplot as plt


def _require_matplotlib():  # pragma: no cover - optional dependency
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("matplotlib must be installed to plot DCA projections.") from exc
    return plt


@dataclass(frozen=True)
class DCAProjection:
    """Container for the simulated dollar-cost averaging series."""

    months: np.ndarray
    contributions: np.ndarray
    balances: np.ndarray

    def final_balance(self) -> float:
        return float(self.balances[-1])

    def final_contribution(self) -> float:
        return float(self.contributions[-1])


def simulate_dca(
    *,
    months: int,
    initial_investment: float,
    monthly_contribution: float,
    annual_return_rate: float,
) -> DCAProjection:
    """Simulate a dollar-cost averaging schedule.

    Parameters
    ----------
    months:
        Number of months to project. Must be greater than zero.
    initial_investment:
        Lump sum deployed at month zero.
    monthly_contribution:
        Amount invested at the end of every month.
    annual_return_rate:
        Expected annual compound rate (expressed as 0.10 for 10%).
    """

    if months <= 0:
        raise ValueError("months must be positive")

    if annual_return_rate <= -1:
        raise ValueError("annual_return_rate must be greater than -100%")

    monthly_rate = (1 + annual_return_rate) ** (1 / 12) - 1

    balances = np.zeros(months, dtype=float)
    contributions = np.zeros(months, dtype=float)
    balances[0] = initial_investment
    contributions[0] = initial_investment

    for month in range(1, months):
        contributions[month] = contributions[month - 1] + monthly_contribution
        balances[month] = (balances[month - 1] + monthly_contribution) * (1 + monthly_rate)

    month_index = np.arange(months, dtype=int)
    return DCAProjection(months=month_index, contributions=contributions, balances=balances)


def plot_dca_projection(
    projection: DCAProjection,
    *,
    ax: Optional["plt.Axes"] = None,
    show: bool = False,
    title: Optional[str] = None,
):
    """Plot the cumulative balance and contributions for a DCA projection."""

    if ax is None:
        plt = _require_matplotlib()
        _, ax = plt.subplots()
    else:
        plt = _require_matplotlib()

    ax.plot(
        projection.months,
        projection.balances,
        label="Total Balance",
        linewidth=2,
    )
    ax.plot(
        projection.months,
        projection.contributions,
        label="Total Contributions",
        linewidth=2,
        linestyle="--",
    )

    ax.set_xlabel("Months", fontweight="bold")
    ax.set_ylabel("Amount ($)", fontweight="bold")

    if title is None:
        title = "Dollar-Cost Averaging Projection"
    ax.set_title(title, fontweight="bold")

    ax.legend()
    ax.grid(True)

    if show:
        plt.show()

    return ax
