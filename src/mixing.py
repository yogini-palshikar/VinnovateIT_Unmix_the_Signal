"""Instantaneous linear mixing X = A S and noise injection at a defined SNR."""
from __future__ import annotations
import numpy as np

DEFAULT_MIXING_MATRIX = np.array([[1.0, 0.5, 0.2],
                                  [0.3, 1.0, 0.4],
                                  [0.2, 0.3, 1.0]])


def validate_mixing_matrix(A: np.ndarray, n_sources: int | None = None, max_cond: float = 1e4) -> dict:
    """Verify A is finite, 2-D, has full column rank and is acceptably conditioned."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"Mixing matrix must be 2-D, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("Mixing matrix contains non-finite values")
    if n_sources is not None and A.shape[1] != n_sources:
        raise ValueError(f"Mixing matrix has {A.shape[1]} columns but there are {n_sources} sources")
    if A.shape[0] < A.shape[1]:
        raise ValueError("Underdetermined mixing (fewer mixtures than sources) is not supported by square ICA")
    rank = int(np.linalg.matrix_rank(A))
    if rank < A.shape[1]:
        raise ValueError(f"Mixing matrix is rank deficient (rank {rank} < {A.shape[1]})")
    cond = float(np.linalg.cond(A))
    if cond > max_cond:
        raise ValueError(f"Mixing matrix is ill-conditioned (cond={cond:.3g} > {max_cond:.3g})")
    det = float(np.linalg.det(A)) if A.shape[0] == A.shape[1] else float("nan")
    return {"rank": rank, "condition_number": cond, "determinant": det}


def mix_sources(S: np.ndarray, A: np.ndarray) -> np.ndarray:
    """X = A @ S. S: (n_sources, T); A: (n_mixtures, n_sources) -> X: (n_mixtures, T)."""
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"S must be (n_sources, n_samples), got {S.shape}")
    validate_mixing_matrix(A, S.shape[0])
    return np.asarray(A, dtype=np.float64) @ S


def add_noise_at_snr(X: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Add independent white Gaussian noise to each mixture channel.

    SNR_i = 10 log10( var(X_i) / var(N_i) ), so var(N_i) = var(X_i) / 10^(SNR/10).
    Variance (mean removed) is used as signal power so DC offsets do not inflate SNR.
    Returns (noisy X, measured per-channel SNR in dB).
    """
    X = np.asarray(X, dtype=np.float64)
    p_sig = X.var(axis=1, keepdims=True)
    p_noise = p_sig / 10 ** (snr_db / 10)
    N = rng.standard_normal(X.shape) * np.sqrt(p_noise)
    measured = 10 * np.log10(p_sig.ravel() / N.var(axis=1))
    return X + N, measured
