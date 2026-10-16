"""Matplotlib figures. Every figure has title, labelled axes, units and legends where relevant."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_waveforms(sig, labels, sr, title, path, ylabel="Amplitude (a.u.)"):
    n = len(sig)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.9 * n), sharex=True, squeeze=False)
    t = np.arange(sig.shape[1]) / sr
    for ax, x, lb in zip(axes[:, 0], sig, labels):
        ax.plot(t, x, lw=0.4, label=lb)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(title)
    _save(fig, path)


def plot_cov_heatmap(C, title, path, labels=None):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(C, cmap="coolwarm", vmin=-np.abs(C).max(), vmax=np.abs(C).max())
    for (i, j), v in np.ndenumerate(C):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    n = C.shape[0]
    labels = labels or [str(i + 1) for i in range(n)]
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Covariance")
    _save(fig, path)


def plot_corr_heatmap(C, row_names, col_names, title, path):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(np.abs(C), cmap="viridis", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(C):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color="w" if abs(v) < 0.6 else "k", fontsize=9)
    ax.set_xticks(range(len(col_names)), col_names)
    ax.set_yticks(range(len(row_names)), row_names)
    ax.set_xlabel("Estimated component (raw ICA output index)")
    ax.set_ylabel("Original source")
    ax.set_title(title + "\n(colour = |r|, text = signed r)")
    fig.colorbar(im, ax=ax, label="|Pearson r|")
    _save(fig, path)


def plot_matched(S, aligned, names, sr, path, zoom_s=0.4):
    n = len(names)
    t = np.arange(S.shape[1]) / sr
    mid = S.shape[1] // 2
    z = slice(mid, mid + int(zoom_s * sr))
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.2 * n), gridspec_kw={"width_ratios": [3, 1.3]}, squeeze=False)
    for i, nm in enumerate(names):
        for k, sl in enumerate([slice(None), z]):
            ax = axes[i, k]
            ax.plot(t[sl], S[i][sl], lw=0.6, label="original (ref)", alpha=0.9)
            ax.plot(t[sl], aligned[i][sl], lw=0.6, label="ICA (scale-aligned)", alpha=0.75)
            ax.set_ylabel(f"{nm}\n(a.u.)")
            if i == n - 1:
                ax.set_xlabel("Time (s)")
            if i == 0 and k == 0:
                ax.legend(loc="upper right")
        axes[i, 1].set_title("zoom" if i == 0 else "")
    fig.suptitle("Original vs separated waveforms (evaluation copy, least-squares scale-aligned)")
    _save(fig, path)


def plot_spectrograms(S, aligned, names, sr, path):
    n = len(names)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.6 * n), squeeze=False)
    for i, nm in enumerate(names):
        for k, (x, lb) in enumerate([(S[i], "original"), (aligned[i], "ICA separated")]):
            f, t, P = spectrogram(x, sr, nperseg=512, noverlap=384)
            ax = axes[i, k]
            m = ax.pcolormesh(t, f, 10 * np.log10(P + 1e-12), shading="auto", cmap="magma", vmin=-90, vmax=0)
            ax.set_title(f"{nm} - {lb}")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time (s)")
            fig.colorbar(m, ax=ax, label="Power (dB)")
    _save(fig, path)


def plot_metrics_bar(per_source, path, title="Per-source separation quality"):
    names = [p["source"] for p in per_source]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, key, lab in zip(axes, ["abs_pearson", "nmse", "si_sdr_db"], ["|Pearson r|", "NMSE (lower is better)", "SI-SDR (dB)"]):
        vals = [p[key] for p in per_source]
        ax.bar(names, vals, color="tab:blue", label=lab)
        ax.set_ylabel(lab)
        ax.set_xlabel("Source")
        ax.legend()
    fig.suptitle(title)
    _save(fig, path)


def plot_method_comparison(rows, path):
    """rows: list of dict(method, mean_abs_pearson, mean_si_sdr_db)."""
    m = [r["method"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(m, [r["mean_abs_pearson"] for r in rows], label="mean |r|")
    axes[0].set_ylabel("Mean |Pearson r| after matching")
    axes[1].bar(m, [r["mean_si_sdr_db"] for r in rows], color="tab:orange", label="mean SI-SDR")
    axes[1].set_ylabel("Mean SI-SDR (dB)")
    for ax in axes:
        ax.set_xlabel("Method")
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
    fig.suptitle("PCA vs ICA on the same mixtures")
    _save(fig, path)


def plot_snr_curve(rows, path):
    snr = [r["input_snr_db"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].errorbar(snr, [r["mean_si_sdr_db"] for r in rows], yerr=[r["std_si_sdr_db"] for r in rows], marker="o", capsize=3, label="FastICA")
    axes[0].set_ylabel("Output mean SI-SDR (dB)")
    axes[1].errorbar(snr, [r["mean_abs_pearson"] for r in rows], yerr=[r["std_abs_pearson"] for r in rows], marker="o", capsize=3, label="FastICA", color="tab:orange")
    axes[1].set_ylabel("Output mean |Pearson r|")
    for ax in axes:
        ax.set_xlabel("Input mixture SNR (dB)")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Separation quality vs input SNR (mean +/- std over noise trials)")
    _save(fig, path)


def plot_convergence(history, path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.semilogy(np.arange(1, len(history) + 1), np.maximum(history, 1e-18), label="max_i | |w_new.w_old| - 1 |")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Convergence criterion")
    ax.set_title("FastICA convergence")
    ax.legend()
    _save(fig, path)
