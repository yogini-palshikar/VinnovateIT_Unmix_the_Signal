import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


def laplace_sources(n_src=3, T=20000, seed=1):
    """Mathematical test signals (NOT audio): independent Laplace, uniform-amplitude-modulated sine-free noise."""
    rng = np.random.default_rng(seed)
    S = np.vstack([rng.laplace(size=T), rng.laplace(size=T) ** 3 / 10, rng.uniform(-1, 1, size=T)])[:n_src]
    return (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)
