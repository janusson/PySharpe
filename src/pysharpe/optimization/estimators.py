"""Covariance estimators with shrinkage for portfolio optimization.

Provides analytical nonlinear shrinkage (Ledoit & Wolf 2017/2020) that
corrects eigenvalue dispersion without structural factor assumptions,
alongside linear shrinkage as a fallback.

References
----------
Ledoit, O. & Wolf, M. (2017). "Nonlinear Shrinkage of the Covariance Matrix
    for Portfolio Selection: Markowitz Meets Goldilocks."
Ledoit, O. & Wolf, M. (2020). "Analytical Nonlinear Shrinkage of
    Eigenvalues."
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Threshold above which a post-shrinkage condition number triggers a warning.
_CONDITION_WARN_THRESHOLD: float = 1e4


def compute_nonlinear_shrinkage(
    returns: pd.DataFrame,
    *,
    condition_warn_threshold: float = _CONDITION_WARN_THRESHOLD,
) -> pd.DataFrame:
    """Ledoit–Wolf analytical nonlinear covariance shrinkage.

    Corrects eigenvalue dispersion in sample covariance matrices using the
    asymptotic Marčenko–Pastur spectral theory.  Each sample eigenvalue is
    individually shrunk toward the population limiting distribution —
    unlike linear Ledoit–Wolf which applies a single shrinkage constant to
    every eigenvalue.

    The implementation uses a numerical Stieltjes-transform evaluation of
    the oracle shrinkage formula from Ledoit & Wolf (2017, 2020):

        dᵢ* = λᵢ / |1 − c − c·λᵢ·m(λᵢ + iη)|²

    where *c* = N/T is the concentration ratio, *λᵢ* are the sample
    eigenvalues, and *m(z)* is the Stieltjes transform computed from the
    empirical spectral distribution.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns with shape (T observations × N assets).  May be
        daily, weekly, or any frequency — the shrinkage is scale-invariant.
    condition_warn_threshold : float
        If the post-shrinkage condition number exceeds this value a warning
        is logged.  Default is ``1e4``.

    Returns
    -------
    pd.DataFrame
        Symmetric, positive-definite covariance matrix Σ_nl with the same
        ticker labels as the input columns.

    Raises
    ------
    ValueError
        If *returns* has fewer than 3 observations or 2 assets.

    Notes
    -----
    When the concentration ratio *c* ≥ 1 (more assets than observations),
    the sample covariance is rank-deficient.  Nonlinear shrinkage still
    produces a well-conditioned estimate by shrinking zero eigenvalues
    toward positive values derived from the limiting spectral density.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> returns = pd.DataFrame(
    ...     rng.normal(0, 0.01, (100, 5)),
    ...     columns=[f"A{i}" for i in range(5)],
    ... )
    >>> cov_nl = compute_nonlinear_shrinkage(returns)
    >>> cov_nl.shape
    (5, 5)
    >>> np.allclose(cov_nl, cov_nl.T)
    np.True_
    """
    _validate_input(returns)
    tickers = returns.columns.tolist()
    X = returns.values.astype(np.float64)
    T, N = X.shape
    c = N / T  # concentration ratio

    # --- Sample covariance & spectral decomposition ---------------------------
    S = np.cov(X, rowvar=False)  # shape (N, N), unbiased (ddof=1)

    eigenvalues, eigenvectors = np.linalg.eigh(S)
    # eigh returns ascending — reverse to descending for numerical stability
    idx = np.flip(np.argsort(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Guard: clip tiny negative eigenvalues that arise from floating-point
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # --- Nonlinear shrinkage of eigenvalues -----------------------------------
    d_star = _shrink_eigenvalues_nonlinear(eigenvalues, c, T)

    # --- Reconstruct cleaned covariance ---------------------------------------
    Sigma_nl = eigenvectors @ np.diag(d_star) @ eigenvectors.T

    # --- Symmetrise (repairs tiny floating-point asymmetry) -------------------
    Sigma_nl = 0.5 * (Sigma_nl + Sigma_nl.T)

    # --- Condition-number validation ------------------------------------------
    cond_before = float(np.linalg.cond(S))
    cond_after = float(np.linalg.cond(Sigma_nl))
    logger.debug(
        "Covariance condition number: %.2e → %.2e (c = N/T = %.3f)",
        cond_before,
        cond_after,
        c,
    )
    if cond_after > condition_warn_threshold:
        logger.warning(
            "Post-shrinkage condition number %.2e exceeds threshold %.1e. "
            "The matrix may still be ill-conditioned for inversion.",
            cond_after,
            condition_warn_threshold,
        )

    return pd.DataFrame(Sigma_nl, index=tickers, columns=tickers)


def compute_linear_shrinkage(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Linear Ledoit–Wolf covariance shrinkage via scikit-learn.

    This is the classical (2004) estimator that shrinks the sample
    covariance toward a structured target (identity or constant-
    correlation) using a single shrinkage intensity.  Use
    :func:`compute_nonlinear_shrinkage` for the superior eigenvalue-by-
    eigenvalue correction.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (T × N).

    Returns
    -------
    pd.DataFrame
        Linearly shrunk covariance matrix with ticker labels.
    """
    from sklearn.covariance import LedoitWolf

    _validate_input(returns)
    tickers = returns.columns.tolist()
    X = returns.values.astype(np.float64)

    lw = LedoitWolf().fit(X)
    cov = np.asarray(lw.covariance_, dtype=np.float64)

    return pd.DataFrame(cov, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_input(returns: pd.DataFrame) -> None:
    """Validate the returns DataFrame for covariance estimation."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a DataFrame, got {type(returns).__name__}")
    if returns.empty:
        raise ValueError("returns DataFrame must not be empty")
    T, N = returns.shape
    if T < 3:
        raise ValueError(
            f"Need at least 3 observations for covariance estimation; got {T}"
        )
    if N < 2:
        raise ValueError(f"Need at least 2 assets; got {N}")
    if returns.isnull().any().any():
        raise ValueError("returns must not contain NaN values")


def _shrink_eigenvalues_nonlinear(
    eigenvalues: np.ndarray,
    c: float,
    T: int,
) -> np.ndarray:
    """Apply analytical nonlinear shrinkage to sample eigenvalues.

    Uses the Hilbert-transform / kernel-density formulation from
    Ledoit & Wolf (2017, eq. 3.6) that avoids Stieltjes-transform
    singularities at the eigenvalues themselves:

        d_i* = λ_i / [(1 − c − c·λ_i·h_i)² + (π·c·λ_i·f̂(λ_i))²]

    where *h_i* is the jackknife Hilbert transform (excluding λ_i) and
    *f̂(λ_i)* is a Gaussian kernel density estimate at λ_i.

    For c ≥ 1, zero eigenvalues receive positive estimates from the
    limiting Marčenko–Pastur bulk edge.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Sample eigenvalues sorted **descending** (shape (N,)).
    c : float
        Concentration ratio N / T.
    T : int
        Number of time-series observations.

    Returns
    -------
    np.ndarray
        Shrunk eigenvalues d₁*, …, dₙ*.
    """
    N = len(eigenvalues)
    d_star = np.empty(N, dtype=np.float64)

    # --- Kernel density estimate of the eigenvalue spectrum ------------------
    if N >= 5 and np.any(eigenvalues > 1e-30):
        density, bandwidth = _kernel_density_estimate(eigenvalues)
    else:
        # Degenerate case — fall back to identity shrinkage
        density = np.full(N, 1.0, dtype=np.float64)

    # --- Shrink each eigenvalue ----------------------------------------------
    for i in range(N):
        lam = eigenvalues[i]

        if lam <= 1e-30:
            # Zero (or effectively zero) eigenvalue — assign a small positive
            # estimate derived from the bulk edge.
            if c < 1.0:
                sigma_sq = (
                    np.mean(eigenvalues[eigenvalues > 1e-30])
                    if np.any(eigenvalues > 1e-30)
                    else 1e-12
                )
                # MP lower edge: σ² · (1 − √c)²
                d_star[i] = sigma_sq * max((1.0 - np.sqrt(c)) ** 2, 1e-6)
            else:
                # c ≥ 1: all eigenvalues are in the bulk; use a tiny floor
                d_star[i] = max(np.mean(eigenvalues) * 1e-3, 1e-15)
            continue

        # --- Jackknife Hilbert transform: exclude λ_i from the average ---------
        # h_i = (1/N) · Σ_{j≠i} 1/(λ_j − λ_i)
        diff = eigenvalues - lam
        mask = np.abs(diff) > 1e-16  # exclude the i-th term
        if np.any(mask):
            with np.errstate(divide="ignore"):
                h_i = np.mean(1.0 / diff[mask])
        else:
            h_i = 0.0

        # --- Density estimate at λ_i (kernel density) ------------------------
        f_i = float(density[i] if i < len(density) else 1.0)

        # --- Nonlinear shrinkage formula (Ledoit–Wolf 2017, eq. 3.6) ---------
        # d* = λ / [(1 − c − c·λ·h)² + (π·c·λ·f̂)²]
        real_part = 1.0 - c - c * lam * h_i
        imag_part = np.pi * c * lam * f_i
        denom = real_part**2 + imag_part**2

        if denom > 1e-15:
            d_star[i] = lam / denom
        else:
            d_star[i] = lam

    # --- Post-processing ----------------------------------------------------
    # Ensure positivity
    d_star = np.maximum(d_star, 1e-15)

    # Isotonic regression: ensure monotonicity (d₁* ≥ d₂* ≥ … ≥ dₙ*)
    for i in range(N - 1, 0, -1):
        if d_star[i - 1] < d_star[i]:
            d_star[i - 1] = d_star[i]

    return d_star


def _kernel_density_estimate(
    eigenvalues: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Gaussian kernel density estimate of the eigenvalue spectrum.

    Uses Silverman's rule-of-thumb bandwidth.  Positive eigenvalues only;
    zero entries are assigned a floor density.

    Returns
    -------
    tuple[np.ndarray, float]
        (density at each eigenvalue, bandwidth used).
    """
    N = len(eigenvalues)
    sigma = float(np.std(eigenvalues))
    # Silverman's rule
    bandwidth = (
        0.9
        * min(sigma, float(np.subtract(*np.percentile(eigenvalues, [75, 25]))) / 1.34)
        * N ** (-0.2)
    )
    bandwidth = max(bandwidth, 1e-12)

    density = np.zeros(N, dtype=np.float64)

    for i in range(N):
        lam = eigenvalues[i]
        # Gaussian kernel: (1/(N·h·√(2π))) Σ exp(−(λ_i − λ_j)²/(2h²))
        z = (eigenvalues - lam) / bandwidth
        density[i] = float(
            np.mean(np.exp(-0.5 * z**2)) / (bandwidth * np.sqrt(2.0 * np.pi))
        )

    return density, bandwidth
