"""Shared separation pipeline used by main.py, the notebook and the bonus experiments."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .preprocessing import center
from .whitening import whiten, WhiteningResult
from .ica import fastica, ICAResult


@dataclass
class SeparationRun:
    X: np.ndarray
    mean: np.ndarray
    Xc: np.ndarray
    whitening: WhiteningResult
    ica: ICAResult
    W_total: np.ndarray    # (n_comp, C) = W K, so that Y = W_total Xc
    Y: np.ndarray          # (n_comp, T) raw ICA components (= W_total Xc)


def separate(X: np.ndarray, n_components: int | None = None, seed: int = 0,
             eig_floor_rel: float = 1e-10, **ica_kwargs) -> SeparationRun:
    """Centering -> PCA whitening -> symmetric FastICA. Uses only X (blind)."""
    Xc, mean = center(X)
    wh = whiten(Xc, n_components=n_components, eig_floor_rel=eig_floor_rel)
    res = fastica(wh.Z, n_components=n_components, seed=seed, **ica_kwargs)
    W_total = res.W @ wh.K
    return SeparationRun(X=X, mean=mean, Xc=Xc, whitening=wh, ica=res, W_total=W_total, Y=W_total @ Xc)
