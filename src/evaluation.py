"""Evaluation of separated components against the clean reference sources.

All metrics are computed on mean-removed copies. Raw ICA outputs are never modified:
aligned copies are returned separately.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

EPS = 1e-12


def correlation_matrix(S: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson correlation C[i, j] between reference source i and component j. Shape (n_src, n_comp)."""
    S, Y = np.asarray(S, float), np.asarray(Y, float)
    if S.ndim != 2 or Y.ndim != 2 or S.shape[1] != Y.shape[1]:
        raise ValueError(f"Incompatible shapes S{S.shape} / Y{Y.shape}")
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(Y))):
        raise ValueError("Non-finite values in S or Y")
    Sc = S - S.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    sn = np.linalg.norm(Sc, axis=1, keepdims=True)
    yn = np.linalg.norm(Yc, axis=1, keepdims=True)
    return (Sc / np.maximum(sn, EPS)) @ (Yc / np.maximum(yn, EPS)).T


def match_components(corr: np.ndarray) -> np.ndarray:
    """Optimal one-to-one assignment maximising total |correlation|.

    Returns assign where assign[i] = index of the component matched to source i.
    """
    n_src, n_comp = corr.shape
    if n_comp < n_src:
        raise ValueError("Need at least as many components as sources")
    rows, cols = linear_sum_assignment(-np.abs(corr))
    assign = np.empty(n_src, dtype=int)
    assign[rows] = cols
    return assign


def scale_align(s_hat: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares scale (and sign) alignment: alpha = s_hat.s / s_hat.s_hat; return alpha*s_hat."""
    den = float(s_hat @ s_hat)
    alpha = float(s_hat @ s) / den if den > EPS else 0.0
    return alpha * s_hat, alpha


def si_sdr(s_hat: np.ndarray, s: np.ndarray) -> float:
    """Scale-invariant SDR (dB) for mean-removed signals.

    s_target = (<s_hat, s> / <s, s>) s ;  e = s_hat - s_target
    SI-SDR = 10 log10( ||s_target||^2 / ||e||^2 ).
    Invariant to the scale/sign of s_hat. Returns NaN for a silent reference or estimate;
    +inf is capped at 100 dB when the error energy is ~0.
    For mean-removed signals SI-SDR = 10 log10( rho^2 / (1 - rho^2) ) with rho the Pearson
    correlation, so it is a monotone re-expression of correlation, not independent evidence.
    """
    ss = float(s @ s)
    if ss < EPS or float(s_hat @ s_hat) < EPS:
        return float("nan")
    target = (float(s_hat @ s) / ss) * s
    e = s_hat - target
    ee = float(e @ e)
    if ee < EPS * ss:
        return 100.0
    return float(10 * np.log10(float(target @ target) / ee))


def evaluate_separation(S: np.ndarray, Y: np.ndarray, names: list[str] | None = None) -> dict:
    """Match, align and score components. Returns dict with per-source metrics and averages."""
    S, Y = np.asarray(S, float), np.asarray(Y, float)
    names = names or [f"source_{i+1}" for i in range(S.shape[0])]
    corr = correlation_matrix(S, Y)
    assign = match_components(corr)
    Sc = S - S.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    aligned = np.zeros_like(Sc)
    per = []
    for i, nm in enumerate(names):
        j = int(assign[i])
        a, alpha = scale_align(Yc[j], Sc[i])
        aligned[i] = a
        ref_norm = float(Sc[i] @ Sc[i])
        err = Sc[i] - a
        per.append({
            "source": nm, "component": j, "pearson": float(corr[i, j]), "abs_pearson": float(abs(corr[i, j])),
            "alpha": alpha,
            "mae": float(np.mean(np.abs(err))),
            "nmse": float((err @ err) / ref_norm) if ref_norm > EPS else float("nan"),
            "si_sdr_db": si_sdr(Yc[j], Sc[i]),
        })
    keys = ["abs_pearson", "mae", "nmse", "si_sdr_db"]
    summary = {f"mean_{k}": float(np.nanmean([p[k] for p in per])) for k in keys}
    return {"corr_matrix": corr, "assignment": assign, "aligned": aligned, "per_source": per, "summary": summary}


def amari_index(P: np.ndarray) -> float:
    """Amari performance index of the global matrix P = W_total @ A (0 = perfect, perm+scale)."""
    P = np.abs(np.asarray(P, float))
    n = P.shape[0]
    r = (P / np.maximum(P.max(axis=1, keepdims=True), EPS)).sum(axis=1) - 1
    c = (P / np.maximum(P.max(axis=0, keepdims=True), EPS)).sum(axis=0) - 1
    return float((r.sum() + c.sum()) / (2 * n * (n - 1))) if n > 1 else 0.0
