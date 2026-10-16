"""Deterministic SYNTHETIC audio generators (fallback when no real recordings are available).

These are NOT real recordings. They imitate the statistical character of speech (voiced,
syllabic, formant-shaped), music (plucked harmonic notes) and rain (sparse impulsive droplets).
"""
from __future__ import annotations
import numpy as np

_VOWELS = {"a": (730, 1090, 2440), "i": (270, 2290, 3010), "u": (300, 870, 2240),
           "e": (530, 1840, 2480), "o": (570, 840, 2410)}


def synth_speech(sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    """Speech-like signal: harmonic source with wandering pitch, formant-shaped, syllabic envelope."""
    n = int(sr * dur)
    t = np.arange(n) / sr
    smooth = np.convolve(rng.standard_normal(n), np.hanning(int(0.4 * sr)), mode="same")
    smooth /= np.abs(smooth).max()
    f0 = 115 + 18 * smooth + 4 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    formants = np.zeros((n, 3))
    env = np.zeros(n)
    pos, keys = 0, list(_VOWELS)
    while pos < n:
        syl = int(rng.uniform(0.12, 0.30) * sr)
        gap = int(rng.uniform(0.03, 0.30) * sr)
        end = min(pos + syl, n)
        formants[pos:end] = _VOWELS[keys[rng.integers(len(keys))]]
        env[pos:end] = np.hanning(end - pos) ** 0.7 * rng.uniform(0.5, 1.0)
        pos = end + gap
    k = np.ones(int(0.02 * sr)) / int(0.02 * sr)
    for j in range(3):
        formants[:, j] = np.convolve(formants[:, j], k, mode="same")
    sig = np.zeros(n)
    for h in range(1, int(4000 / 100) + 1):
        freq = h * f0
        gain = np.zeros(n)
        for j, w in enumerate((1.0, 0.6, 0.3)):
            gain += w / (1 + ((freq - formants[:, j]) / 90.0) ** 2)
        sig += (gain / h) * np.sin(h * phase)
    sig += 0.03 * rng.standard_normal(n) * (env > 0)
    return sig * env


def synth_music(sr: int, dur: float, rng: np.random.Generator, bpm: float = 104.0) -> np.ndarray:
    """Music-like signal: plucked harmonic notes (melody + bass) on a pentatonic scale."""
    n = int(sr * dur)
    out = np.zeros(n)
    step = int(sr * 60 / bpm / 2)  # eighth notes
    scale = np.array([0, 2, 4, 7, 9])
    idx = 2
    base = 261.63
    for i, s in enumerate(range(0, n, step)):
        idx = int(np.clip(idx + rng.integers(-2, 3), 0, 9))
        semis = scale[idx % 5] + 12 * (idx // 5)
        voices = [(base * 2 ** (semis / 12), 0.35, 1.0)]
        if i % 4 == 0:
            voices.append((base / 2 * 2 ** (scale[(i // 4) % 5] / 12), 0.9, 0.8))
        for f, tau, a in voices:
            L = min(int(sr * 1.2), n - s)
            tt = np.arange(L) / sr
            note = sum(np.sin(2 * np.pi * f * h * tt) / h for h in range(1, 7))
            out[s:s + L] += a * note * np.exp(-tt / tau) * (1 - np.exp(-tt / 0.004))
    return out


def synth_ambient(sr: int, dur: float, rng: np.random.Generator, rate: float = 450.0) -> np.ndarray:
    """Rain-like signal: Poisson droplets (damped high-frequency ringing) + faint low rumble."""
    n = int(sr * dur)
    out = np.zeros(n)
    L = int(0.012 * sr)
    tt = np.arange(L) / sr
    for s in np.sort(rng.integers(0, n - L, size=int(rate * dur))):
        f = rng.uniform(1800, 6500)
        amp = rng.lognormal(0.0, 0.9)
        out[s:s + L] += amp * np.sin(2 * np.pi * f * tt) * np.exp(-tt / rng.uniform(0.001, 0.004))
    rumble = np.convolve(rng.standard_normal(n), np.ones(int(sr / 120)) / (sr / 120), mode="same")
    return out + 0.15 * rumble * out.std() / max(rumble.std(), 1e-12)
