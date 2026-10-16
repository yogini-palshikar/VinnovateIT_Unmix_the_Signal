"""Central configuration for "Unmix the Signal". All paths are project-root aware."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SOURCES_DIR = DATA_DIR / "sources"
MIXTURES_DIR = DATA_DIR / "mixtures"
OUT_DIR = ROOT / "outputs"
SEPARATED_DIR = OUT_DIR / "separated"
PLOTS_DIR = OUT_DIR / "plots"
METRICS_DIR = OUT_DIR / "metrics"

SAMPLE_RATE = 16000      # Hz, common rate for all sources
DURATION_S = 15.0        # seconds (configurable, 10-20 s recommended)
SEED = 0                 # master seed for every stochastic step

SOURCE_NAMES = ["speech", "music", "ambient"]

# Well-conditioned, diagonally dominant, non-singular mixing matrix (rows = mixtures).
MIXING_MATRIX = [[1.0, 0.5, 0.2],
                 [0.3, 1.0, 0.4],
                 [0.2, 0.3, 1.0]]

EIG_FLOOR_REL = 1e-10    # whitening: eigenvalues below this * lambda_max are rejected
FASTICA = dict(max_iter=500, tol=1e-8, nonlinearity="tanh", alpha=1.0)

SNR_LEVELS_DB = [30, 20, 10, 0]   # Bonus B
SNR_TRIALS = 3                     # noise realisations averaged per SNR level
PLAYBACK_PEAK = 0.9                # peak level of exported WAVs (listening convenience only)
