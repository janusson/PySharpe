"""Black-Litterman Reverse Optimization.

Implements the Black-Litterman model for blending equilibrium implied returns
with investor views to produce posterior expected returns and covariance
matrices suitable for mean-variance optimization.

Core Functions
--------------
- :func:`compute_implied_returns` — Equilibrium implied excess returns from
  market-cap weights.
- :func:`build_views_uncertainty` — Construct a diagonal uncertainty matrix
  ``Omega`` from per-view confidence levels (0–100 %).
- :func:`blend_views` — Bayesian blending of implied returns with explicit
  investor views via the Black-Litterman master formula.

References
----------
- Black, F. & Litterman, R. (1992). "Global Portfolio Optimization."
- He, G. & Litterman, R. (1999). "The Intuition Behind Black-Litterman
  Model Portfolios."
- Idzorek, T. (2007). "A Step-by-Step Guide to the Black-Litterman Model."

Examples
--------
>>> import numpy as np
>>> rng = np.random.default_rng(42)
>>> n = 4
>>> cov = np.cov(rng.normal(0, 0.01, (500, n)).T) * 252
>>> w_mkt = np.array([0.25, 0.25, 0.25, 0.25])
>>> pi = compute_implied_returns(cov, w_mkt, risk_aversion=2.5)
>>> pi.shape
(4,)
>>> # Single view: asset 0 will outperform asset 1 by 3%
>>> P = np.array([[1, -1, 0, 0]])
>>> Q = np.array([0.03])
>>> Omega = build_views_uncertainty(cov, P, view_confidences=[70.0])
>>> er, cov_p = blend_views(pi, cov, P, Q, Omega)
>>> er.shape
(4,)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_implied_returns(
    cov_matrix: np.ndarray | pd.DataFrame,
    market_weights: np.ndarray | pd.Series,
    risk_aversion: float = 2.5,
) -> np.ndarray | pd.Series:
    """Compute equilibrium implied excess returns via reverse optimisation.

    Under the Capital Asset Pricing Model (CAPM) / Black-Litterman framework,
    the market portfolio is mean-variance efficient.  Given the covariance
    matrix and market-capitalisation weights, the implied excess returns are

        Π = δ · Σ · w_mkt

    where *δ* is the global risk-aversion scalar, *Σ* is the covariance
    matrix of asset returns, and *w_mkt* is the vector of market-cap
    (or benchmark factor) weights.

    Args:
        cov_matrix: Annualised covariance matrix of asset returns
            (``(n, n)`` numeric array or ``pd.DataFrame`` with aligned
            row/column labels).
        market_weights: Market-capitalisation or benchmark weights for
            each asset (length *n*).
        risk_aversion: Global risk-aversion coefficient *δ*.  Higher values
            increase the magnitude of implied returns.  Typical range: 2–4.

    Returns:
        Implied excess equilibrium returns Π, as a 1-D ``np.ndarray`` or
        ``pd.Series`` matching the input type of *market_weights*.

    Raises:
        ValueError: If dimensions are mismatched or inputs contain NaN.
    """
    cov: np.ndarray = np.asarray(cov_matrix, dtype=np.float64)
    w: np.ndarray = np.asarray(market_weights, dtype=np.float64).ravel()

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov_matrix must be square (n, n), got shape {cov.shape}")
    if cov.shape[0] != w.shape[0]:
        raise ValueError(
            f"cov_matrix ({cov.shape[0]} assets) and market_weights "
            f"({w.shape[0]} weights) must have matching lengths"
        )
    if np.any(np.isnan(cov)) or np.any(np.isnan(w)):
        raise ValueError("cov_matrix and market_weights must not contain NaN")

    implied: np.ndarray = risk_aversion * cov @ w

    if isinstance(market_weights, pd.Series):
        return pd.Series(implied, index=market_weights.index, name="Implied Returns")
    return implied


def build_views_uncertainty(
    cov_matrix: np.ndarray | pd.DataFrame,
    P: np.ndarray,
    view_confidences: list[float] | np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """Build the diagonal view-uncertainty matrix Ω from per-view confidence levels.

    Each view *k* has an associated confidence *c_k* ∈ (0, 100] expressed as a
    percentage.  The uncertainty (variance) of view *k* is modelled as

        ω_kk = τ · (p_k^T Σ p_k) · (1 / (c_k / 100) − 1)

    so that:
    - 100 % confidence → ω_kk ≈ 0 (the view is treated as virtually certain).
    -   0 % confidence → ω_kk → ∞ (the view is discarded; *never* pass 0 % —
      use :func:`blend_views` with a sufficiently large Ω entry instead).

    Args:
        cov_matrix: Annualised covariance matrix ``(n, n)``.
        P: The *K* × *n* pick matrix, where each row corresponds to one view.
        view_confidences: Confidence percentage per view in (0, 100].
        tau: Scalar weight on the equilibrium prior (same *τ* used in
            :func:`blend_views`).

    Returns:
        Diagonal uncertainty matrix Ω of shape ``(K, K)``, where ``K`` is
        the number of views.

    Raises:
        ValueError: If confidences are outside (0, 100] or if shapes mismatch.
    """
    cov: np.ndarray = np.asarray(cov_matrix, dtype=np.float64)
    P_arr: np.ndarray = np.asarray(P, dtype=np.float64)
    conf = np.atleast_1d(np.asarray(view_confidences, dtype=np.float64))

    if P_arr.ndim != 2:
        raise ValueError(f"P must be a 2-D array, got shape {P_arr.shape}")
    K, n = P_arr.shape
    if cov.shape != (n, n):
        raise ValueError(
            f"P has {n} columns but cov_matrix has shape {cov.shape}; "
            f"expected ({n}, {n})"
        )
    if conf.shape[0] != K:
        raise ValueError(
            f"view_confidences has {conf.shape[0]} entries but P has "
            f"{K} rows — they must match"
        )

    if np.any(conf <= 0) or np.any(conf > 100):
        raise ValueError("view_confidences must be in (0, 100]")

    # Convert percentages to fractions
    c = conf / 100.0  # shape (K,)

    omega = np.zeros((K, K), dtype=np.float64)
    for k in range(K):
        p_k = P_arr[k]  # shape (n,)
        var_view = float(p_k @ cov @ p_k)  # p_k^T Σ p_k
        omega[k, k] = tau * var_view * (1.0 / c[k] - 1.0)

    return omega


def blend_views(
    implied_returns: np.ndarray | pd.Series,
    cov_matrix: np.ndarray | pd.DataFrame,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.05,
) -> tuple[np.ndarray | pd.Series, np.ndarray | pd.DataFrame]:
    """Blend equilibrium implied returns with investor views (Black-Litterman).

    This implements the Black-Litterman *master formula* that produces
    posterior expected returns and a posterior covariance matrix:

        E(R) = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹ · [(τΣ)⁻¹ Π + Pᵀ Ω⁻¹ Q]

        Σ_p  = Σ + [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹

    where:
    - **Π**  — implied equilibrium excess returns,
    - **Σ**  — asset covariance matrix,
    - **P**  — *K* × *n* pick matrix identifying the assets involved in each view,
    - **Q**  — *K*-vector of expected return differentials for each view,
    - **Ω**  — *K* × *K* diagonal covariance matrix of view uncertainties,
    - **τ**  — scalar weight on the equilibrium prior (typically 0.01–0.10).

    Args:
        implied_returns: Equilibrium implied returns Π (length *n*).
        cov_matrix: Asset covariance matrix Σ ``(n, n)``.
        P: Pick matrix ``(K, n)``.
        Q: View-return vector ``(K,)``.
        Omega: View-uncertainty matrix ``(K, K)``, typically diagonal.
        tau: Prior-weight scalar.

    Returns:
        A ``(posterior_returns, posterior_cov)`` tuple:
        - **posterior_returns** — E(R) as 1-D ``np.ndarray`` (or ``pd.Series``
          if *implied_returns* was a Series).
        - **posterior_cov** — Σ_p as ``(n, n)`` ``np.ndarray`` (or
          ``pd.DataFrame`` if *cov_matrix* was a DataFrame).

    Raises:
        ValueError: If dimensions are inconsistent or matrices are singular.
    """
    pi: np.ndarray = np.asarray(implied_returns, dtype=np.float64).ravel()
    sigma: np.ndarray = np.asarray(cov_matrix, dtype=np.float64)
    P_arr: np.ndarray = np.asarray(P, dtype=np.float64)
    Q_arr: np.ndarray = np.asarray(Q, dtype=np.float64).ravel()
    Omega_arr: np.ndarray = np.asarray(Omega, dtype=np.float64)

    n = sigma.shape[0]

    # Validate dimensions
    if sigma.shape != (n, n):
        raise ValueError(f"cov_matrix must be ({n}, {n}), got {sigma.shape}")
    if pi.shape[0] != n:
        raise ValueError(f"implied_returns length {pi.shape[0]} != cov_matrix dim {n}")
    if P_arr.ndim != 2 or P_arr.shape[1] != n:
        raise ValueError(f"P must be (K, {n}), got {P_arr.shape}")
    if Q_arr.shape[0] != P_arr.shape[0]:
        raise ValueError(f"Q length {Q_arr.shape[0]} != P rows {P_arr.shape[0]}")
    if Omega_arr.shape != (P_arr.shape[0], P_arr.shape[0]):
        raise ValueError(
            f"Omega shape {Omega_arr.shape} != expected "
            f"({P_arr.shape[0]}, {P_arr.shape[0]})"
        )

    # --- Posterior expected returns ---
    tau_sigma_inv = np.linalg.inv(tau * sigma)  # (τΣ)⁻¹

    # Ω⁻¹ — invert the diagonal uncertainty matrix
    # Guard: allow large (effectively infinite) omega diagonal entries
    omega_inv_diag = np.zeros(Omega_arr.shape[0], dtype=np.float64)
    for k in range(Omega_arr.shape[0]):
        if Omega_arr[k, k] > 0:
            omega_inv_diag[k] = 1.0 / Omega_arr[k, k]
        else:
            # Zero uncertainty → infinite precision (numeric guard)
            omega_inv_diag[k] = 1e16

    Omega_inv = np.diag(omega_inv_diag) if Omega_arr.shape[0] > 0 else Omega_arr.copy()

    # M = (τΣ)⁻¹ + Pᵀ Ω⁻¹ P
    M = tau_sigma_inv + P_arr.T @ Omega_inv @ P_arr  # shape (n, n)

    # M⁻¹ — use solve for numerical stability when n is small
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse if M is ill-conditioned
        M_inv = np.linalg.pinv(M)

    # E(R) = M⁻¹ · [(τΣ)⁻¹ Π + Pᵀ Ω⁻¹ Q]
    rhs = tau_sigma_inv @ pi + P_arr.T @ Omega_inv @ Q_arr
    er = M_inv @ rhs  # shape (n,)

    # --- Posterior covariance ---
    sigma_p = sigma + M_inv

    # --- Preserve pandas labels if inputs have them ---
    if isinstance(implied_returns, pd.Series):
        er = pd.Series(er, index=implied_returns.index, name="BL Posterior Returns")
    if isinstance(cov_matrix, pd.DataFrame):
        sigma_p = pd.DataFrame(
            sigma_p, index=cov_matrix.index, columns=cov_matrix.columns
        )

    return er, sigma_p
