# Methodology

**Model.** Observed `X = A S`, X: (3, T), A: (3, 3), S: (3, T). `A` is known here only to generate data and to
score results; separation uses `X` alone. Assumptions: instantaneous (no delay/reverb), linear, square and
full-rank mixing, mutually independent sources, at most one Gaussian source. Real speech/music/ambient
recordings are only approximately independent.

**Pipeline** (`src/pipeline.py::separate`)
1. *Centering* (`preprocessing.center`): `Xc = X - mean(X, axis=1)`. The mean (C,1) is stored so the transform can be reapplied. Needed because whitening/ICA assume zero mean; it removes first-order structure only, so it is not whitening.
2. *PCA whitening* (`whitening.whiten`): `C = Xc Xc^T / T` (population convention), `C = E Λ E^T` via `eigh`, `K = Λ^{-1/2} E^T`, `Z = K Xc`, hence `Cov(Z) = I`. With the sample convention (`T-1`) the identity check must use the same normalisation. Retained eigenvalues `<= eig_floor_rel * λ_max` raise an error rather than being inverted (configurable, `config.EIG_FLOOR_REL`); reduce `n_components` to discard such directions.
3. *Symmetric FastICA* (`ica.fastica`): `W0` random (seeded) then `W <- (W W^T)^{-1/2} W`. Iterate `W <- (1/T) g(WZ) Z^T - diag(mean_t g'(WZ)) W`, then symmetric decorrelation. `g = tanh(αu)`. Stop when `max_i | |w_i_new·w_i_old| - 1 | < tol` (sign-invariant). Non-convergence emits a warning and `converged=False`.
4. *Reconstruction*: `Y = W Z = (W K) Xc` (raw, unit variance, arbitrary order/sign/scale).

**Why whitening helps.** After whitening, the remaining unmixing matrix is orthogonal: from `n^2` unknowns to `n(n-1)/2` rotation angles. Uncorrelated is weaker than independent, so a rotation is still undetermined by second-order statistics; ICA picks it by maximising non-Gaussianity.

**Evaluation** (`src/evaluation.py`; on mean-removed signals). `|Pearson|` matrix, optimal assignment (`linear_sum_assignment`), least-squares scale `α = ŝ·s / ŝ·ŝ` (this also fixes sign) for aligned evaluation copies, then Pearson, MAE, NMSE `= ||s-αŝ||²/||s||²`, and SI-SDR `= 10 log10(||s_target||²/||e||²)`, `s_target = (ŝ·s/s·s) s`. Caveat: for zero-mean signals NMSE `= 1-ρ²` and SI-SDR `= 10 log10(ρ²/(1-ρ²))`; these are re-expressions of correlation, not independent evidence. Amari index of `W K A` (needs ground truth) measures how close the global matrix is to permutation×scale.

**Why #mixtures >= #sources.** Square ICA inverts `A`; fewer mixtures than sources (underdetermined) is a different, harder problem needing extra assumptions (e.g. sparsity). `mixing.validate_mixing_matrix` enforces full column rank and bounds the condition number (cond(A)=2.92 here).

**Playback.** Exported WAVs are peak-normalised per file (listening convenience), signs flipped to match the reference polarity; this is not a quality improvement. Raw outputs are in `outputs/separated/raw_components.npy`. Mixtures use one common gain so relative levels persist.

**Bonus A.** PCA scores (`E^T Xc`), PCA-whitened scores (`Z`) and ICA are evaluated identically. PCA and PCA-whitened give identical metrics because whitening only rescales each principal component and the metrics are scale-invariant, which illustrates that whitening alone adds no separation. This holds for this dataset; it is not a claim about every dataset.

**Bonus B.** Input SNR per channel: `10 log10(var(X_i)/var(N_i))` with white Gaussian noise; averaged over `SNR_TRIALS` noise seeds. Output SDR/SI-SDR is measured against the clean sources. Plain FastICA has no noise model, so noise limits attainable SI-SDR (it lands slightly below the input SNR here) and 0 dB is poor.

**Bonus C.** Peak-pair landmark fingerprints with offset voting (`src/shazam_mini.py`). The 'songs' are synthetic procedurally generated pieces, because no legally usable recordings could be downloaded offline.
