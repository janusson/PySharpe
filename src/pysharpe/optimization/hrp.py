"""Hierarchical Risk Parity (HRP) portfolio optimization.

Implements López de Prado's three‑step HRP algorithm as a non‑inversion
fallback for ill‑conditioned covariance matrices.

References
----------
López de Prado, M. (2016). "Building Diversified Portfolios that Outperform
Out of Sample." Journal of Portfolio Management, 42(4), 59‑69.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RIDGE_EPSILON: float = 1e-8
"""Ridge regularisation added to the covariance diagonal for near‑singular clusters."""

_CORRELATION_RIDGE_THRESHOLD: float = 0.999
"""Above this absolute pairwise correlation, ridge regularisation is triggered."""

# ---------------------------------------------------------------------------
# HierarchicalRiskParity
# ---------------------------------------------------------------------------


class HierarchicalRiskParity:
    """Hierarchical Risk Parity (HRP) portfolio construction.

    Builds a diversified long‑only portfolio without inverting the
    covariance matrix, making it robust to high collinearity and
    ill‑conditioned correlation structures.

    Parameters
    ----------
    returns
        ``(T, N)`` DataFrame of asset returns (rows = time, columns = tickers).
        If provided, the covariance and correlation are estimated from the
        returns.  Mutually exclusive with *cov_matrix*.
    cov_matrix
        Pre‑computed ``(N, N)`` covariance matrix.  Must be symmetric,
        positive semi‑definite.  Mutually exclusive with *returns*.

    Raises
    ------
    ValueError
        If neither (or both) *returns* and *cov_matrix* are supplied, or
        if the input matrix is not square and symmetric.

    Notes
    -----
    **Three‑step algorithm** (López de Prado, 2016):

    1. **Tree clustering** — Convert the correlation matrix to a metric
       distance ``D_{i,j} = √(½(1 − ρ_{i,j}))`` and build a hierarchical
       tree via single‑linkage clustering.
    2. **Quasi‑diagonalisation** — Reorder assets so that similar clusters
       sit along the covariance diagonal.
    3. **Recursive bisection** — Allocate capital top‑down using inverse‑
       variance weights within each cluster, splitting variance risk
       evenly between sibling branches.
    """

    def __init__(
        self,
        returns: pd.DataFrame | None = None,
        cov_matrix: pd.DataFrame | np.ndarray | None = None,
    ) -> None:
        # ------------------------------------------------------------------
        # Input validation
        # ------------------------------------------------------------------
        if returns is not None and cov_matrix is not None:
            raise ValueError("Provide either returns or cov_matrix, not both.")
        if returns is None and cov_matrix is None:
            raise ValueError("Either returns or cov_matrix must be provided.")

        # ------------------------------------------------------------------
        # Estimate or accept covariance
        # ------------------------------------------------------------------
        if returns is not None:
            self._tickers: list[str] = list(returns.columns)
            # .cov() may return a DataFrame with read‑only underlying arrays;
            # take an explicit mutable copy.
            self._cov: pd.DataFrame = returns.cov().copy()
        else:
            cov = (
                pd.DataFrame(cov_matrix).copy()
                if not isinstance(cov_matrix, pd.DataFrame)
                else cov_matrix.copy()
            )
            if cov.shape[0] != cov.shape[1]:
                raise ValueError("cov_matrix must be square.")
            self._tickers = list(cov.columns)
            self._cov = cov

        self._n_assets: int = len(self._tickers)
        if self._n_assets < 2:
            raise ValueError(
                f"At least 2 assets required for HRP, got {self._n_assets}."
            )

        # ------------------------------------------------------------------
        # Correlation matrix and near‑singularity check
        # ------------------------------------------------------------------
        diag_std: np.ndarray = np.sqrt(np.diag(self._cov.values))
        self._corr: np.ndarray = self._cov.values / np.outer(diag_std, diag_std)

        # Check for near‑perfect correlations and apply ridge if needed.
        self._ridge_applied = self._check_and_apply_ridge()

    # ------------------------------------------------------------------
    # Ridge regularisation
    # ------------------------------------------------------------------

    def _check_and_apply_ridge(self) -> bool:
        """Apply ridge to the covariance diagonal if any pair is near‑singular."""
        # Upper triangle of absolute correlations (excluding diagonal).
        upper_idx = np.triu_indices(self._n_assets, k=1)
        upper_corrs: np.ndarray = np.abs(self._corr[upper_idx])

        # Check for NaN (e.g. zero‑variance assets) or near‑perfect correlation.
        has_nan: bool = bool(np.any(np.isnan(upper_corrs)))
        max_abs_corr: float = 1.0 if has_nan else float(np.nanmax(upper_corrs))

        if has_nan or max_abs_corr > _CORRELATION_RIDGE_THRESHOLD:
            if has_nan:
                logger.warning(
                    "NaN correlation detected (likely zero‑variance asset). "
                    "Applying ridge regularisation ε = %.1e to covariance diagonal.",
                    _RIDGE_EPSILON,
                )
            else:
                logger.warning(
                    "Near‑singular correlation detected (max |ρ| = %.6f > %.3f). "
                    "Applying ridge regularisation ε = %.1e to covariance diagonal.",
                    max_abs_corr,
                    _CORRELATION_RIDGE_THRESHOLD,
                    _RIDGE_EPSILON,
                )
            # Work on a mutable copy of the covariance matrix.
            cov_values: np.ndarray = self._cov.values.copy()
            np.fill_diagonal(cov_values, cov_values.diagonal() + _RIDGE_EPSILON)
            self._cov = pd.DataFrame(
                cov_values, index=self._cov.index, columns=self._cov.columns
            )
            # Re‑compute correlation from the regularised covariance.
            diag_std_ridge = np.sqrt(np.diag(cov_values))
            self._corr = cov_values / np.outer(diag_std_ridge, diag_std_ridge)
            return True
        return False

    # ------------------------------------------------------------------
    # Step 1: Tree Clustering
    # ------------------------------------------------------------------

    def _cluster(self) -> np.ndarray:
        """Build the hierarchical tree (linkage matrix) from correlations.

        Returns
        -------
        np.ndarray
            ``(N−1, 4)`` linkage matrix from ``scipy.cluster.hierarchy.linkage``.
        """
        # D_{i,j} = √(½(1 − ρ_{i,j}))
        # Clip to [0, …) in case of floating‑point overshoot from near‑perfect
        # correlations after ridge regularisation.
        distance_condensed: np.ndarray = np.sqrt(
            np.maximum(0.0, 0.5 * (1.0 - self._corr))
        )
        # Ensure diagonal is exactly zero (may drift with floating point).
        np.fill_diagonal(distance_condensed, 0.0)

        # Compute Euclidean distances between columns of D, then cluster.
        # pdist with 'euclidean' on the correlation‑distance matrix gives
        # the distance between asset clusters.
        condensed_dist: np.ndarray = pdist(distance_condensed, metric="euclidean")
        return linkage(condensed_dist, method="single")

    # ------------------------------------------------------------------
    # Step 2: Quasi‑Diagonalisation
    # ------------------------------------------------------------------

    def _quasi_diagonalize(self, link: np.ndarray) -> list[int]:
        """Reorder asset indices so that similar clusters are adjacent.

        Parameters
        ----------
        link
            Linkage matrix from :meth:`_cluster`.

        Returns
        -------
        list[int]
            Permuted list of asset indices (leaf order).
        """
        return leaves_list(link).tolist()

    # ------------------------------------------------------------------
    # Step 3: Recursive Bisection
    # ------------------------------------------------------------------

    def _get_cluster_variance(
        self,
        cov: np.ndarray,
        cluster_indices: Sequence[int],
    ) -> float:
        """Compute the inverse‑variance‑weighted cluster variance.

        Within a cluster, each asset gets weight proportional to the
        reciprocal of its variance.  The cluster variance is then
        ``w^T · Σ_cluster · w``.

        Parameters
        ----------
        cov
            Full ``(N, N)`` covariance matrix.
        cluster_indices
            Integer indices of assets belonging to this cluster.

        Returns
        -------
        float
            Cluster variance.
        """
        sub_cov: np.ndarray = cov[np.ix_(cluster_indices, cluster_indices)]
        inv_var: np.ndarray = 1.0 / np.diag(sub_cov)

        # Normalise inverse‑variance weights to sum to 1.
        w: np.ndarray = inv_var / np.sum(inv_var)
        return float(w @ sub_cov @ w)

    def _recursive_bisection(
        self,
        cov: np.ndarray,
        sorted_indices: list[int],
    ) -> np.ndarray:
        """Allocate weights via top‑down recursive bisection.

        Parameters
        ----------
        cov
            Full ``(N, N)`` covariance matrix (quasi‑diagonalised order).
        sorted_indices
            Asset indices in quasi‑diagonalised order.

        Returns
        -------
        np.ndarray
            Weight vector of length ``N``, summing to 1.0.
        """
        n: int = len(sorted_indices)
        if n == 0:
            return np.array([], dtype=float)

        # Initialise all weights to 1 (will be scaled down through splits).
        weights: np.ndarray = np.ones(n, dtype=float)

        # Stack of (start, end) index ranges to process.
        stack: list[tuple[int, int]] = [(0, n)]

        while stack:
            start, end = stack.pop()
            length = end - start
            if length <= 1:
                continue  # Leaf node — no further splitting.

            # Split into left and right halves.
            mid: int = start + length // 2
            left_indices: list[int] = sorted_indices[start:mid]
            right_indices: list[int] = sorted_indices[mid:end]

            # Compute cluster variances.
            v_left: float = self._get_cluster_variance(cov, left_indices)
            v_right: float = self._get_cluster_variance(cov, right_indices)
            total_v: float = v_left + v_right

            if total_v <= 0.0:
                # Degenerate — equal split.
                alpha: float = 0.5
            else:
                # α = V_R / (V_L + V_R) → weight to left branch.
                # (1 − α) goes to right branch.
                # This allocates more weight to the *less* volatile cluster.
                alpha = v_right / total_v

            # Scale existing weights within each half.
            weights[start:mid] *= alpha
            weights[mid:end] *= 1.0 - alpha

            # Recurse on children.
            if length > 2:
                stack.append((start, mid))
                stack.append((mid, end))

        # Normalise to sum to exactly 1.0.
        total_weight: float = float(np.sum(weights))
        if total_weight <= 0.0:
            return np.ones_like(weights) / n
        return weights / total_weight

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(self) -> pd.Series:
        """Run the full HRP three‑step pipeline.

        Returns
        -------
        pd.Series
            Asset weights indexed by ticker, summing to 1.0.  All weights
            are non‑negative by construction (long‑only).

        Notes
        -----
        Weights are guaranteed to sum to exactly 1.0 (up to floating‑point
        precision).  No short positions or leverage are introduced.
        """
        # Step 1: Tree clustering.
        link: np.ndarray = self._cluster()

        # Step 2: Quasi‑diagonalisation (leaf order).
        sorted_indices: list[int] = self._quasi_diagonalize(link)

        # Step 3: Recursive bisection.
        cov_values: np.ndarray = self._cov.values
        weights: np.ndarray = self._recursive_bisection(cov_values, sorted_indices)

        return pd.Series(
            weights,
            index=[self._tickers[i] for i in sorted_indices],
            name="weight",
        )

    @property
    def correlation_matrix(self) -> pd.DataFrame:
        """The (possibly ridge‑regularised) correlation matrix."""
        return pd.DataFrame(self._corr, index=self._tickers, columns=self._tickers)

    @property
    def covariance_matrix(self) -> pd.DataFrame:
        """The (possibly ridge‑regularised) covariance matrix."""
        return self._cov.copy()

    @property
    def ridge_applied(self) -> bool:
        """``True`` if ridge regularisation was applied."""
        return self._ridge_applied


__all__ = ["HierarchicalRiskParity"]
