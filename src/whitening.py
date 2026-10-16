"""PCA whitening implemented from scratch (population covariance, eigh)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .preprocessing import check_matrix


@dataclass
class WhiteningResult:
    Z: np.ndarray          # (n_comp, T) whitened signals, cov = I
    K: np.ndarray          # (n_comp, C) whitening matrix, K = Lambda^-1/2 E^T
    eigvals: np.ndarray    # (C,) all eigenvalues, descending
    eigvecs: np.ndarray    # (C, C) columns = eigenvectors matching eigvals
    n_components: int

    @property
    def dewhitening(self) -> np.ndarray:
        """Pseudo-inverse of K: E Lambda^{1/2}  (C, n_comp)."""
        return np.linalg.pinv(self.K)


def covariance(Xc: np.ndarray, ddof: int = 0) -> np.ndarray:
    """C = Xc Xc^T / (T - ddof). ddof=0 is the population convention used throughout.

    If ddof=1 (sample covariance) is used for whitening, the identity-covariance check
    must use the same normalisation, otherwise Z has covariance (T-1)/T * I.
    """
    T = Xc.shape[1]
    return (Xc @ Xc.T) / (T - ddof)


def whiten(Xc: np.ndarray, n_components: int | None = None, eig_floor_rel: float = 1e-10,
           center_tol: float = 1e-6) -> WhiteningResult:
    """PCA-whiten a CENTERED matrix Xc of shape (C, T).

    C_X = E Lambda E^T, K = Lambda^{-1/2} E^T (top n_components rows), Z = K Xc.
    Raises ValueError if Xc is not centered, or if any retained eigenvalue is below
    eig_floor_rel * lambda_max (degenerate / rank-deficient input). Reduce n_components
    to drop small-variance directions instead of inflating them.
    """
    Xc = check_matrix(Xc, "Xc")
    C, T = Xc.shape
    scale = float(Xc.std())
    if scale == 0.0:
        raise ValueError("Xc is identically constant/zero; cannot whiten")
    if np.max(np.abs(Xc.mean(axis=1))) > center_tol * scale:
        raise ValueError("Xc is not centered; call preprocessing.center first")
    n = C if n_components is None else n_components
    if not (1 <= n <= C):
        raise ValueError(f"n_components must be in [1, {C}], got {n}")
    cov = covariance(Xc)
    eigvals, eigvecs = np.linalg.eigh(cov)          # ascending
    order = np.argsort(eigvals)[::-1]               # descending
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    lam_max = eigvals[0]
    floor = eig_floor_rel * lam_max
    if np.any(eigvals[:n] <= floor):
        raise ValueError(
            f"Degenerate covariance: retained eigenvalues {eigvals[:n]} include values <= "
            f"{floor:.3g} (= eig_floor_rel * lambda_max). Input is (near) rank deficient; "
            "reduce n_components or fix the data.")
    K = (eigvecs[:, :n] / np.sqrt(eigvals[:n])).T   # (n, C) = Lambda^-1/2 E^T
    return WhiteningResult(Z=K @ Xc, K=K, eigvals=eigvals, eigvecs=eigvecs, n_components=n)
