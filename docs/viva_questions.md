# Viva questions and answers

Code locations: `src/preprocessing.py` (centering), `src/whitening.py`, `src/ica.py`, `src/evaluation.py`, `src/mixing.py`, `src/pipeline.py`. Key equations: `C=XcXc^T/T`, `K=Λ^{-1/2}E^T`, `Z=KXc`, `W←(WW^T)^{-1/2}W`, `W_new=g(WZ)Z^T/T − diag(mean g'(WZ))W`, `Y=WKXc`, `α=ŝ·s/ŝ·ŝ`.

**1. Why is ICA preferred over PCA for blind source separation?**  
PCA finds orthogonal directions of maximal variance using only second-order statistics; those directions are generally mixtures of the sources. ICA additionally uses higher-order statistics (non-Gaussianity) to find the rotation that makes components independent. Code: `src/whitening.py` (PCA), `src/ica.py` (ICA); demo in Bonus A (`main.py::bonus_pca_vs_ica`).

**2. Difference between uncorrelated and independent?**  
Uncorrelated: `E[xy]=E[x]E[y]`. Independent: `p(x,y)=p(x)p(y)`, which implies all cross-moments factorise. Independent implies uncorrelated, not vice versa (e.g. `x` and `x²` for symmetric `x`).

**3. Why does PCA use second-order statistics while ICA exploits higher-order structure?**  
PCA only uses the covariance. The covariance of whitened data is `I` for every rotation, so it cannot identify the rotation; higher-order statistics (here the `tanh` contrast, related to kurtosis) can.

**4. Can PCA ever separate sources?**  
Yes, when the sources are uncorrelated with distinct variances and the mixing matrix happens to be orthogonal (its principal axes align with the sources), or for Gaussian data where only decorrelation is possible/defined. Generally not.

**5. Why can PCA whitening alone not recover the sources?**  
`Z = K A S` with `K A` an orthogonal matrix `Q` (up to a rotation), not the identity. Whitening fixes scale/correlation but leaves the unknown rotation `Q`. See `docs/methodology.md`; Bonus A shows PCA/whitened scores score |r|≈0.68 versus ICA≈1.0 on this dataset.

**6. What is whitening?**  
A linear transform `Z = K Xc` giving zero mean and identity covariance. Implemented in `src/whitening.py::whiten`.

**7. How is the covariance calculated?**  
`C = Xc Xc^T / T` (population). Using `T-1` is the sample convention; the same normalisation must be used when checking `Cov(Z)=I`. `whitening.covariance`.

**8. Why eigendecomposition?**  
`C` is symmetric positive semi-definite, so `C = E Λ E^T` with orthonormal `E`; this gives `C^{-1/2}` directly as `Λ^{-1/2}E^T`. `numpy.linalg.eigh` is exact for symmetric matrices.

**9. What do eigenvalues and eigenvectors represent?**  
Eigenvectors are the principal axes; eigenvalues are the variance along each axis.

**10. Why handle near-zero eigenvalues carefully?**  
`Λ^{-1/2}` blows up noise in near-degenerate directions. `whiten` raises if a retained eigenvalue ≤ `eig_floor_rel·λ_max`; `n_components` can discard them. Tested in `tests/test_whitening.py::test_degenerate_rejected`.

**11. Why does whitening reduce free parameters?**  
The remaining unmixing matrix is orthogonal: `n(n-1)/2` angles instead of `n²` unknowns (3 vs 9 here).

**12. How can you verify whitening worked?**  
Check `Cov(Z)≈I` and mean≈0: max |Cov−I| = 4.4e-15 in the run (`metrics_main.json`), covariance heatmaps `outputs/plots/04,05`.

**13. Intuition behind maximising non-Gaussianity?**  
Mixtures of independent variables are closer to Gaussian than the originals, so the directions of extreme non-Gaussianity correspond to individual sources.

**14. What does the CLT say about mixtures?**  
A sum of independent non-Gaussian variables tends toward a Gaussian, so mixtures are more Gaussian than sources.

**15. Why are non-Gaussian sources useful?**  
Their higher-order statistics make the contrast function vary with direction, which identifies the rotation.

**16. Why is a Gaussian source problematic?**  
A Gaussian is rotation-invariant after whitening, so its direction is unidentifiable; at most one Gaussian source is allowed.

**17. Role of tanh in FastICA?**  
`g(u)=tanh(αu)` is a robust derivative of a log-cosh contrast approximating negentropy; `g'=α(1−tanh²)` appears in the Newton step. `ica.py::_g_tanh`.

**18. What is symmetric decorrelation?**  
`W ← (W W^T)^{-1/2} W` keeps all rows orthonormal simultaneously, without favouring earlier components as deflation does. `ica.py::symmetric_decorrelation`.

**19. Why is the convergence criterion sign invariant?**  
It uses `| |w_new·w_old| − 1 |`; the absolute value removes the ±1 ambiguity of each unit-norm row.

**20. Why might FastICA fail to converge?**  
Near-Gaussian sources, too few samples, ill-conditioned mixing, strong noise, too-tight tolerance, or poor initialisation. The code warns and sets `converged=False` (`tests/test_ica.py::test_nonconvergence_reported`); try other seeds or nonlinearities (`tanh`, `cube`, `gauss`).

**21. Batch vs online ICA?**  
Batch (used here) uses all samples every iteration; online/stochastic variants update from small chunks, enabling streaming at the cost of noisier convergence.

**22. Permutation, scale and sign ambiguities?**  
Swapping sources, scaling a source and `A` column inversely, or flipping signs gives the same `X`. Hence matching (`evaluation.match_components`) and scale alignment (`scale_align`).

**23. Why these audio sources?**  
Speech, music and ambient noise represent distinct, super-Gaussian-like sources. In this run they are SYNTHETIC stand-ins because no real files could be downloaded offline; real files can be placed in `data/sources/`.

**24. Why do numbers of mixtures and sources match?**  
Square ICA inverts a square `A`. Fewer mixtures is underdetermined; more mixtures can be reduced by PCA (`n_components`).

**25. Why must A have full rank?**  
A singular `A` loses information (not invertible); `validate_mixing_matrix` checks rank and cond(A)=2.92.

**26. Why retain the clean sources?**  
They are the ground truth for evaluation only; ICA never uses them.

**27. Why correlation after matching?**  
Component order is arbitrary; correlation is only meaningful for the matched pair. Absolute correlation handles sign flips; `linear_sum_assignment` prevents two sources claiming one component.

**28. Why scale alignment for evaluation?**  
ICA output scale/sign is arbitrary, so MAE/NMSE are meaningless without `α=ŝ·s/ŝ·ŝ`. Aligned copies are separate from raw outputs.

**29. What does SDR measure and limitations?**  
Here SI-SDR: energy of the part of the estimate aligned with the reference over the energy of the residual (dB). Limitations: for zero-mean signals it is a monotone function of correlation; it ignores perceptual quality and treats all error as equal; undefined for silence (NaN).

**30. Why can good correlation coexist with imperfect waveform reconstruction?**  
Correlation ignores scale and is dominated by high-energy parts; small residual interference can remain (e.g. r=0.9999 corresponds to only ≈37 dB SI-SDR since SI-SDR=10·log10(r²/(1−r²))), and quiet segments can be wrong.

**31. How would you test robustness to noise?**  
Bonus B (`main.py::bonus_snr`, `mixing.add_noise_at_snr`): sweep SNR 30/20/10/0 dB, several noise seeds, plot SI-SDR vs SNR (`outputs/plots/13_snr_robustness.png`).

**32. What changes with multiple physical devices?**  
Delays, room reverberation, different gains per microphone and clock drift give a convolutive, non-instantaneous mixture; frequency-domain ICA or similar is needed.

**33. Assumptions of instantaneous linear mixing?**  
Same-time linear combination, constant `A`, no delays/reverb, independent stationary-in-statistics sources, ≥ as many mixtures as sources, ≤ 1 Gaussian source.

**34. What if recovered sources sound distorted?**  
Check convergence/other seeds, source non-Gaussianity, conditioning, noise, clipping; then examine matched waveforms/spectrograms and consider model mismatch (reverb).

**35. Most important limitations?**  
Synthetic data in this run, ideal instantaneous model, square case, no noise model in FastICA, SI-SDR dependent on correlation, only one run per configuration (stochastic initialisation, though tests show seed robustness).
