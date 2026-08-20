"""Covariance estimators with shrinkage for portfolio optimization.

Provides two shrinkage estimators with identical hardening guarantees:

- :func:`compute_linear_shrinkage` — classical Ledoit & Wolf (2004) linear
  shrinkage toward a constant-correlation target.
- :func:`compute_nonlinear_shrinkage` — analytical nonlinear shrinkage
  (Ledoit & Wolf 2017/2020) that corrects eigenvalue dispersion without
  structural factor assumptions.

Both estimators:

1. Handle missing observations via conservative listwise deletion (rows
   containing NaN are dropped with a logged warning — never backfilled).
2. Return matrices that are **strictly positive definite** (every eigenvalue
   is positive), enforced by :func:`ensure_strictly_psd`.
3. Raise :class:`~pysharpe.exceptions.DataValidationError` on invalid or
   degenerate inputs.

References
----------
Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for
    large-dimensional covariance matrices."  Journal of Multivariate
    Analysis, 88(2), 365-411.
Ledoit, O. & Wolf, M. (2017). "Nonlinear Shrinkage of the Covariance Matrix
    for Portfolio Selection: Markowitz Meets Goldilocks."
Ledoit, O. & Wolf, M. (2020). "Analytical Nonlinear Shrinkage of
    Eigenvalues."
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pysharpe.exceptions import DataValidationError

logger = logging.getLogger(__name__)

# Threshold above which a post-shrinkage condition number triggers a warning.
_CONDITION_WARN_THRESHOLD: float = 1e4

# Eigen-clip floors used by :func:`ensure_strictly_psd`.  The relative floor
# caps the condition number of the hardened matrix at ~1e12 while the absolute
# floor keeps fully degenerate inputs (all-zero sample spectra) strictly
# positive definite.
_PSD_RELATIVE_FLOOR: float = 1e-12
_PSD_ABSOLUTE_FLOOR: float = 1e-15


# ---------------------------------------------------------------------------
# Public data preparation / hardening helpers
# ---------------------------------------------------------------------------


def prepare_returns(
    returns: pd.DataFrame,
    *,
    min_observations: int = 3,
    min_assets: int = 2,
) -> pd.DataFrame:
    """Validate and clean a returns DataFrame for estimation.

    Missing observations are handled by **listwise deletion**: rows
    containing any NaN are dropped (with a logged warning) because partially
    observed returns cannot be safely imputed without lookahead bias.
    Assets that are entirely missing are a structural data failure and raise
    :class:`~pysharpe.exceptions.DataValidationError`.

    Args:
        returns: Asset returns with shape (T observations, N assets).
        min_observations: Minimum number of rows that must remain after
            deleting rows with missing values.
        min_assets: Minimum number of columns (assets) required.

    Returns:
        A cleaned copy of *returns* containing only finite numeric values.

    Raises:
        TypeError: If *returns* is not a pandas DataFrame.
        DataValidationError: If the data is empty, non-numeric, contains
            infinite values, has fully-missing assets, or fewer rows/assets
            remain than required.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a DataFrame, got {type(returns).__name__}")
    if returns.empty:
        raise DataValidationError("returns DataFrame must not be empty")

    # --- Numeric-only columns -----------------------------------------------
    numeric = returns.select_dtypes(include=[np.number])
    non_numeric = [str(col) for col in returns.columns if col not in numeric.columns]
    if non_numeric:
        raise DataValidationError(
            "returns must contain only numeric columns; "
            f"non-numeric columns: {', '.join(non_numeric)}"
        )

    # --- Infinite values are corrupt data ------------------------------------
    if np.isinf(numeric.values).any():
        raise DataValidationError("returns must not contain infinite values")

    # --- Missing values: conservative listwise deletion ----------------------
    n_rows, _ = returns.shape
    cleaned = numeric.dropna(axis=0, how="any")
    dropped = n_rows - len(cleaned)
    if dropped:
        logger.warning(
            "Dropped %d of %d observations with missing values before "
            "covariance estimation (listwise deletion; no backfilling).",
            dropped,
            n_rows,
        )

    # --- Fully-missing assets are a structural failure -----------------------
    fully_missing = [str(col) for col in cleaned.columns if cleaned[col].isna().all()]
    if fully_missing:
        raise DataValidationError(
            "Assets with no observed returns cannot be estimated: "
            + ", ".join(fully_missing)
        )

    T, N = cleaned.shape
    if T < min_observations:
        raise DataValidationError(
            f"Need at least {min_observations} observations for covariance "
            f"estimation; got {T} (after dropping missing rows)"
        )
    if N < min_assets:
        raise DataValidationError(f"Need at least {min_assets} assets; got {N}")

    return cleaned


def ensure_strictly_psd(
    matrix: np.ndarray | pd.DataFrame,
    *,
    relative_floor: float = _PSD_RELATIVE_FLOOR,
    absolute_floor: float = _PSD_ABSOLUTE_FLOOR,
) -> np.ndarray:
    """Symmetrise and eigen-clip a covariance matrix to strict positive definiteness.

    Eigenvalues below ``max(λ_max · relative_floor, absolute_floor)`` are
    raised to that floor and the matrix is rebuilt from its eigendecomposition.
    This guarantees every eigenvalue is strictly positive (capping the
    condition number at ~1/``relative_floor``) so downstream solvers cannot
    fail on singular or indefinite matrices.  For well-conditioned input the
    reconstruction is a no-op up to floating-point roundoff.

    Args:
        matrix: Square symmetric covariance matrix ``(n, n)``.
        relative_floor: Eigenvalue floor relative to the largest eigenvalue.
        absolute_floor: Absolute eigenvalue floor used when the spectrum is
            (numerically) all-zero.

    Returns:
        The hardened strictly positive-definite matrix as a float64
        ``np.ndarray`` (labels are dropped; callers reattach them).

    Raises:
        DataValidationError: If *matrix* is not square, contains non-finite
            values, or the hardened result fails validation.
    """
    values: np.ndarray = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise DataValidationError(
            f"matrix must be square (n, n), got shape {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise DataValidationError("matrix must contain only finite values")

    # --- Symmetrise (repairs floating-point asymmetry) -----------------------
    values = 0.5 * (values + values.T)

    eigenvalues, eigenvectors = np.linalg.eigh(values)
    lambda_max = float(eigenvalues.max()) if eigenvalues.size else 0.0
    floor = max(lambda_max * relative_floor, absolute_floor)

    clipped = np.maximum(eigenvalues, floor)
    hardened = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    hardened = 0.5 * (hardened + hardened.T)

    # --- Defensive validation (must be unreachable after clipping) -----------
    if not np.all(np.isfinite(hardened)):
        raise DataValidationError(
            "Eigen-clipping produced non-finite values; input covariance is "
            "beyond numerical repair."
        )
    if np.any(np.linalg.eigvalsh(hardened) <= 0.0):
        raise DataValidationError(
            "Failed to produce a strictly positive-definite covariance matrix."
        )
    return hardened


# ---------------------------------------------------------------------------
# Nonlinear shrinkage (Ledoit & Wolf 2017/2020)
# ---------------------------------------------------------------------------


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

    Args:
        returns: Asset returns with shape (T observations × N assets).  May
            be daily, weekly, or any frequency — the shrinkage is
            scale-invariant.  Rows with missing values are dropped
            (listwise) before estimation.
        condition_warn_threshold: If the post-shrinkage condition number
            exceeds this value a warning is logged.  Default is ``1e4``.

    Returns:
        Symmetric, **strictly positive-definite** covariance matrix Σ_nl
        with the same ticker labels as the input columns.

    Raises:
        TypeError: If *returns* is not a pandas DataFrame.
        DataValidationError: If the data fails structural validation (see
            :func:`prepare_returns`) or the hardened matrix cannot be made
            strictly positive definite.

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
    cleaned = prepare_returns(returns)
    tickers = cleaned.columns.tolist()
    X = cleaned.values.astype(np.float64)
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

    # --- Strict-PSD guarantee -------------------------------------------------
    Sigma_nl = ensure_strictly_psd(Sigma_nl)

    # --- Condition-number validation ------------------------------------------
    cond_after = float(np.linalg.cond(Sigma_nl))
    logger.debug(
        "Nonlinear shrinkage complete: c = N/T = %.3f, condition number %.2e",
        c,
        cond_after,
    )
    if cond_after > condition_warn_threshold:
        logger.warning(
            "Post-shrinkage condition number %.2e exceeds threshold %.1e. "
            "The matrix may still be ill-conditioned for inversion.",
            cond_after,
            condition_warn_threshold,
        )

    return pd.DataFrame(Sigma_nl, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# Linear shrinkage (Ledoit & Wolf 2004)
# ---------------------------------------------------------------------------


def compute_linear_shrinkage(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit–Wolf (2004) linear covariance shrinkage.

    Shrinks the sample covariance toward the constant-correlation target
    with the analytically optimal shrinkage intensity δ* ∈ [0, 1]:

        Σ_LW = δ* · F + (1 − δ*) · S

    where *F* preserves the sample variances and assigns every asset pair
    the average pairwise correlation.  Implemented directly from the paper
    (π̂, ρ̂, γ̂ estimators) so degenerate inputs — zero-variance assets,
    T < N rank deficiency — are handled explicitly rather than delegated to
    a third-party library.

    Args:
        returns: Asset returns (T × N).  Rows with missing values are
            dropped (listwise) before estimation.

    Returns:
        Symmetric, **strictly positive-definite** shrunk covariance matrix
        with ticker labels.  The diagonal exactly preserves the sample
        variances.

    Raises:
        TypeError: If *returns* is not a pandas DataFrame.
        DataValidationError: If the data fails structural validation (see
            :func:`prepare_returns`) or the hardened matrix cannot be made
            strictly positive definite.
    """
    cleaned = prepare_returns(returns)
    tickers = cleaned.columns.tolist()
    X = cleaned.values.astype(np.float64)

    Sigma_lw = _ledoit_wolf_constant_correlation(X)
    Sigma_lw = ensure_strictly_psd(Sigma_lw)

    return pd.DataFrame(Sigma_lw, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ledoit_wolf_constant_correlation(X: np.ndarray) -> np.ndarray:
    """Analytical Ledoit–Wolf (2004) shrinkage toward the constant-correlation target.

    Implements the π̂ / ρ̂ / γ̂ estimators from Ledoit & Wolf (2004) with
    vectorized inner loops (O(N²T) time, O(NT) memory).

    Args:
        X: Centered-ready returns array ``(T, N)`` (need not be centered;
            centering happens internally).

    Returns:
        Shrunk covariance ``Σ_LW = δ*·F + (1−δ*)·S`` (before any
        eigen-clipping).
    """
    T, N = X.shape
    Xc: np.ndarray = X - X.mean(axis=0, keepdims=True)  # (T, N) centered

    # --- Sample covariance (unbiased) ----------------------------------------
    S: np.ndarray = (Xc.T @ Xc) / (T - 1)  # (N, N)
    variances = np.diag(S).copy()
    std = np.sqrt(variances)

    # --- Zero-variance detection (relative to the largest variance) -----------
    scale = float(std.max()) if N else 0.0
    is_zero_var: np.ndarray = std <= _PSD_RELATIVE_FLOOR * scale

    # --- Sample correlations (pairs involving zero-variance assets → 0) -------
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rho = S / np.outer(std, std)
    rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    rho = np.clip(rho, -1.0, 1.0)
    rho[is_zero_var, :] = 0.0
    rho[:, is_zero_var] = 0.0
    np.fill_diagonal(rho, 0.0)

    # --- Constant-correlation target F ---------------------------------------
    upper_idx = np.triu_indices(N, k=1)
    r_bar = float(np.mean(rho[upper_idx])) if N > 1 else 0.0
    F: np.ndarray = np.outer(std, std) * r_bar
    np.fill_diagonal(F, variances)

    # --- π̂ : variance of the y_ijt covariance estimators ---------------------
    # y_ijt = (x_it − x̄_i)(x_jt − x̄_j); m_ij = mean_t(y_ijt) = S_ij·(T−1)/T
    m_ij = S * ((T - 1) / T)  # (N, N)
    pi_hat = 0.0
    pi_diag = 0.0

    # --- ρ̂ : sum of π̂_ii plus cross-asset θ̂ correction terms -----------------
    B: np.ndarray = (Xc**2 - variances) * Xc  # (T, N); (x_jt² − s_jj)·x_jt
    cross_hat = 0.0

    for i in range(N):
        Y_i: np.ndarray = Xc * Xc[:, i : i + 1]  # (T, N): y_ijt for fixed i
        pi_row = ((Y_i - m_ij[i : i + 1, :]) ** 2).mean(axis=0)  # (N,)
        pi_hat += float(pi_row.sum())
        pi_diag += float(pi_row[i])

        # θ̂_ii,ij for fixed i, all j: mean_t((x_it² − s_ii)·x_it·x_jt)
        A_i: np.ndarray = (Xc[:, i] ** 2 - variances[i]) * Xc[:, i]  # (T,)
        theta_i: np.ndarray = (A_i @ Xc) / T  # (N,)
        # θ̂_jj,ij for fixed i, all j: mean_t((x_jt² − s_jj)·x_jt·x_it)
        theta_j: np.ndarray = (B.T @ Xc[:, i]) / T  # (N,)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_ij = np.where(std[i] > 0.0, std / std[i], 0.0)
            ratio_ji = np.where(std > 0.0, std[i] / std, 0.0)
        contrib = 0.5 * rho[i, :] * (ratio_ij * theta_i + ratio_ji * theta_j)
        contrib[i] = 0.0  # diagonal handled by π̂_ii
        contrib = np.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
        cross_hat += float(contrib.sum())

    rho_hat = pi_diag + cross_hat

    # --- Optimal shrinkage intensity ------------------------------------------
    gamma_hat = float(np.sum((F - S) ** 2))
    if gamma_hat > 0.0:
        kappa_hat = (pi_hat - rho_hat) / gamma_hat
        delta = float(np.clip(kappa_hat / T, 0.0, 1.0))
    else:
        # Target equals the sample estimate — shrinkage is irrelevant.
        delta = 0.0

    Sigma_lw = delta * F + (1.0 - delta) * S
    return 0.5 * (Sigma_lw + Sigma_lw.T)


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

    Args:
        eigenvalues: Sample eigenvalues sorted **descending** (shape (N,)).
        c: Concentration ratio N / T.
        T: Number of time-series observations.

    Returns:
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

    Returns:
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
