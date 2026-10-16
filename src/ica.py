"""Symmetric FastICA implemented from scratch with NumPy (no library ICA)."""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
import numpy as np


def _g_tanh(u, alpha):
    t = np.tanh(alpha * u)
    return t, alpha * (1.0 - t ** 2)


def _g_cube(u, alpha):
    return u ** 3, 3.0 * u ** 2


def _g_gauss(u, alpha):
    e = np.exp(-0.5 * u ** 2)
    return u * e, (1.0 - u ** 2) * e


NONLINEARITIES = {"tanh": _g_tanh, "cube": _g_cube, "gauss": _g_gauss}


def symmetric_decorrelation(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """W <- (W W^T)^{-1/2} W via eigendecomposition of the symmetric matrix W W^T."""
    s, U = np.linalg.eigh(W @ W.T)
    if s.min() <= eps * max(s.max(), 1.0):
        raise np.linalg.LinAlgError("W W^T is (near) singular; cannot symmetrically decorrelate")
    return (U * (1.0 / np.sqrt(s))) @ U.T @ W


@dataclass
class ICAResult:
    W: np.ndarray                 # (n_comp, n_white) rotation in the whitened space
    Y: np.ndarray                 # (n_comp, T) = W Z  (unit-variance components, RAW)
    n_iter: int
    converged: bool
    history: list = field(default_factory=list)   # convergence criterion per iteration
    nonlinearity: str = "tanh"


def fastica(Z: np.ndarray, n_components: int | None = None, max_iter: int = 500, tol: float = 1e-8,
            seed: int = 0, nonlinearity: str = "tanh", alpha: float = 1.0,
            whiteness_tol: float = 1e-3) -> ICAResult:
    """Symmetric FastICA on whitened data Z of shape (n_white, T).

    Update (per iteration):  W_new = (1/T) g(WZ) Z^T  -  diag(mean_t g'(WZ)) W,
    followed by symmetric decorrelation.  Shapes: W (n_comp, n_white), WZ (n_comp, T),
    g(WZ) Z^T -> (n_comp, n_white); mean_t g'(WZ) -> (n_comp,) scaling the rows of W.
    Convergence: max_i | |w_i_new . w_i_old| - 1 | < tol (invariant to sign flips).
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D, got {Z.shape}")
    if not np.all(np.isfinite(Z)):
        raise ValueError("Z contains NaN or infinity")
    if nonlinearity not in NONLINEARITIES:
        raise ValueError(f"Unknown nonlinearity '{nonlinearity}'; choose from {list(NONLINEARITIES)}")
    n_white, T = Z.shape
    n = n_white if n_components is None else n_components
    if not (1 <= n <= n_white):
        raise ValueError(f"n_components must be in [1, {n_white}], got {n}")
    if max_iter < 1 or tol <= 0:
        raise ValueError("max_iter must be >= 1 and tol > 0")
    cov = Z @ Z.T / T
    if np.max(np.abs(cov - np.eye(n_white))) > whiteness_tol:
        raise ValueError("Z does not have identity covariance; whiten the data before FastICA")
    g = NONLINEARITIES[nonlinearity]
    rng = np.random.default_rng(seed)
    W = symmetric_decorrelation(rng.standard_normal((n, n_white)))
    history, converged, it = [], False, 0
    for it in range(1, max_iter + 1):
        WZ = W @ Z
        gw, gp = g(WZ, alpha)
        W_new = gw @ Z.T / T - np.diag(gp.mean(axis=1)) @ W
        W_new = symmetric_decorrelation(W_new)
        lim = float(np.max(np.abs(np.abs(np.sum(W_new * W, axis=1)) - 1.0)))
        history.append(lim)
        W = W_new
        if lim < tol:
            converged = True
            break
    if not converged:
        warnings.warn(f"FastICA did not converge in {max_iter} iterations (last criterion {history[-1]:.3e})")
    return ICAResult(W=W, Y=W @ Z, n_iter=it, converged=converged, history=history,
                     nonlinearity=nonlinearity)
