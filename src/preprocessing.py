"""Input validation and centering."""
from __future__ import annotations
import numpy as np


def check_matrix(X: np.ndarray, name: str = "X") -> np.ndarray:
    """Validate a (channels, samples) float matrix."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (channels, samples); got shape {X.shape}")
    if X.shape[1] < X.shape[0]:
        raise ValueError(f"{name} has more channels ({X.shape[0]}) than samples ({X.shape[1]}); "
                         "expected shape (channels, samples)")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or infinity")
    return X


def center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract each channel's sample mean. Returns (X_centered, mean) with mean shape (C, 1)."""
    X = check_matrix(X)
    mean = X.mean(axis=1, keepdims=True)
    return X - mean, mean


def apply_centering(X: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Apply a previously estimated mean (e.g. to new data) consistently."""
    X = check_matrix(X)
    if mean.shape != (X.shape[0], 1):
        raise ValueError(f"mean shape {mean.shape} incompatible with X {X.shape}")
    return X - mean
