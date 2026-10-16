"""Run the full "Unmix the Signal" pipeline:  python main.py

Stages: load/synthesise sources -> mix -> center -> PCA whitening -> FastICA -> evaluate ->
plots + WAVs + metrics -> Bonus A (PCA vs ICA) -> Bonus B (SNR sweep) -> Bonus C (mini-Shazam).
"""
from __future__ import annotations
import argparse, csv, json, platform, re, sys, warnings
import numpy as np
import scipy, matplotlib

import config as cfg
from src import audio_utils as au, mixing, visualization as viz
from src.evaluation import evaluate_separation, amari_index, correlation_matrix
from src.pipeline import separate
from src.whitening import covariance


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def check_wav(path, expect_sr):
    sr, = [au.read_wav(path)[1]]
    x, _ = au.read_wav(path)
    assert sr == expect_sr and np.all(np.isfinite(x)) and x.std() > 1e-4, f"bad audio file {path}"
    return {"file": str(path.relative_to(cfg.ROOT)), "sr": sr, "duration_s": round(len(x) / sr, 3),
            "min": float(x.min()), "max": float(x.max())}


def bonus_pca_vs_ica(S, run, names):
    """Bonus A: same mixtures, same matching/evaluation for 4 estimators."""
    E = run.whitening.eigvecs
    cands = {"raw mixtures (no processing)": run.X,
             "PCA (no whitening)": E.T @ run.Xc,
             "PCA-whitened (Z)": run.whitening.Z,
             "FastICA": run.Y}
    rows = []
    for m, Y in cands.items():
        ev = evaluate_separation(S, Y, names)
        rows.append({"method": m, **{f"{p['source']}_abs_pearson": p["abs_pearson"] for p in ev["per_source"]},
                     "mean_abs_pearson": ev["summary"]["mean_abs_pearson"],
                     "mean_si_sdr_db": ev["summary"]["mean_si_sdr_db"], "mean_nmse": ev["summary"]["mean_nmse"]})
    return rows


def bonus_snr(S, A, names, seed):
    """Bonus B: sweep input SNR; report output quality (SI-SDR) averaged over noise trials."""
    X = mixing.mix_sources(S, A)
    out = []
    for snr in cfg.SNR_LEVELS_DB:
        sd, pr, conv, meas_all = [], [], [], []
        for trial in range(cfg.SNR_TRIALS):
            rng = np.random.default_rng([seed, int(snr), trial])
            Xn, meas = mixing.add_noise_at_snr(X, snr, rng)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                run = separate(Xn, seed=seed, eig_floor_rel=cfg.EIG_FLOOR_REL, **cfg.FASTICA)
            ev = evaluate_separation(S, run.Y, names)
            sd.append(ev["summary"]["mean_si_sdr_db"]); pr.append(ev["summary"]["mean_abs_pearson"])
            conv.append(run.ica.converged); meas_all.append(float(meas.mean()))
        out.append({"input_snr_db": snr, "measured_input_snr_db": float(np.mean(meas_all)),
                    "mean_si_sdr_db": float(np.mean(sd)), "std_si_sdr_db": float(np.std(sd)),
                    "mean_abs_pearson": float(np.mean(pr)), "std_abs_pearson": float(np.std(pr)),
                    "converged_trials": f"{sum(conv)}/{len(conv)}"})
    return out


def bonus_shazam(S, run, ev, names, sr, seed):
    """Bonus C: synthetic 'song' library; query = ICA-separated music component vs raw mixture."""
    from src.shazam_mini import FingerprintIndex
    from src.synthetic_audio import synth_music
    idx = FingerprintIndex()
    n = S.shape[1]
    idx.add_song("main_music (synthetic)", S[names.index("music")], sr)
    for k, bpm in enumerate([84, 96, 112, 126, 140]):
        idx.add_song(f"synthetic_song_{k+1}", au.prepare_source(synth_music(sr, cfg.DURATION_S, np.random.default_rng(100 + k), bpm=bpm), sr, sr, n, "lib"), sr)
    j = names.index("music")
    seg = slice(int(4 * sr), int(9 * sr))   # 5 s excerpt
    res = {}
    for label, x in [("clean music excerpt (sanity check)", S[j][seg]),
                     ("ICA-separated music component", ev["aligned"][j][seg]),
                     ("raw mixture channel 1 (no separation)", run.X[0][seg])]:
        q = idx.query(x, sr)
        q.pop("ranking")
        res[label] = q
    return res


def update_readme(table_md):
    p = cfg.ROOT / "README.md"
    if p.is_file():
        t = p.read_text(encoding="utf-8")
        new = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                     lambda m: "<!-- RESULTS:START -->\n" + table_md + "\n<!-- RESULTS:END -->", t, flags=re.S)
        p.write_text(new, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=cfg.DURATION_S)
    ap.add_argument("--sr", type=int, default=cfg.SAMPLE_RATE)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--no-bonus", action="store_true")
    a = ap.parse_args(argv)
    cfg.DURATION_S = a.duration
    for d in (cfg.SOURCES_DIR, cfg.MIXTURES_DIR, cfg.SEPARATED_DIR, cfg.PLOTS_DIR, cfg.METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    names, sr = cfg.SOURCE_NAMES, a.sr
    # 1. sources -------------------------------------------------------------
    S, info = au.load_sources(cfg.SOURCES_DIR, names, sr, a.duration, a.seed)
    synthetic = any("SYNTHETIC" in i["origin"] for i in info)
    prefix = "synthetic_" if synthetic else ""
    for nm, x in zip(names, S):
        au.write_wav(cfg.SOURCES_DIR / f"{prefix}{nm}.wav" if synthetic else cfg.SOURCES_DIR / f"{nm}_prepared.wav", x, sr)
    # 2. mixing ----------------------------------------------------------------
    A = np.array(cfg.MIXING_MATRIX)
    A_info = mixing.validate_mixing_matrix(A, S.shape[0])
    X = mixing.mix_sources(S, A)
    gain = 0.95 / np.max(np.abs(X))        # ONE common gain keeps relative mixture levels
    for i in range(X.shape[0]):
        au.write_wav(cfg.MIXTURES_DIR / f"mixture_{i+1}.wav", X[i], sr, gain=gain)
    json.dump({"A": A, **A_info, "common_wav_gain": gain}, open(cfg.METRICS_DIR / "mixing_matrix.json", "w", encoding="utf-8"), default=_jsonable, indent=2)
    # 3-5. center / whiten / ICA ------------------------------------------------
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = separate(X, seed=a.seed, eig_floor_rel=cfg.EIG_FLOOR_REL, **cfg.FASTICA)
    if not run.ica.converged:
        print("WARNING: FastICA did not converge:", [str(c.message) for c in caught], file=sys.stderr)
    cov_before, cov_after = covariance(run.Xc), covariance(run.whitening.Z)
    whiten_err = float(np.max(np.abs(cov_after - np.eye(3))))
    # 6. evaluation --------------------------------------------------------------
    ev = evaluate_separation(S, run.Y, names)
    amari = amari_index(run.W_total @ A)
    # 7. exports --------------------------------------------------------------------
    np.save(cfg.SEPARATED_DIR / "raw_components.npy", run.Y)
    for i, nm in enumerate(names):
        j = ev["assignment"][i]
        sign = 1.0 if ev["corr_matrix"][i, j] >= 0 else -1.0   # sign flip for listening convenience only
        au.write_wav(cfg.SEPARATED_DIR / f"separated_{nm}.wav", sign * (run.Y[j] - run.Y[j].mean()), sr, peak=cfg.PLAYBACK_PEAK)
    P = cfg.PLOTS_DIR
    viz.plot_waveforms(S, names, sr, "Original clean sources", P / "01_original_sources.png")
    viz.plot_waveforms(X, [f"mixture {i+1}" for i in range(3)], sr, "Observed mixtures X = A S", P / "02_mixtures.png")
    viz.plot_waveforms(run.Xc, [f"centered mixture {i+1}" for i in range(3)], sr, "Centered mixtures", P / "03_centered_mixtures.png")
    viz.plot_cov_heatmap(cov_before, "Covariance of centered mixtures (before whitening)", P / "04_cov_before_whitening.png")
    viz.plot_cov_heatmap(cov_after, "Covariance after PCA whitening", P / "05_cov_after_whitening.png")
    viz.plot_waveforms(run.Y, [f"component {i}" for i in range(3)], sr, "Estimated independent components (raw ICA output, unit variance, arbitrary sign/order)", P / "06_estimated_components.png")
    viz.plot_matched(S, ev["aligned"], names, sr, P / "07_matched_waveforms.png")
    viz.plot_spectrograms(S, ev["aligned"], names, sr, P / "08_spectrograms.png")
    viz.plot_corr_heatmap(ev["corr_matrix"], names, [f"comp {j}" for j in range(3)], "Source-to-component correlation", P / "09_correlation_heatmap.png")
    viz.plot_metrics_bar(ev["per_source"], P / "10_metrics.png")
    viz.plot_convergence(run.ica.history, P / "11_convergence.png")
    # 8. metrics ---------------------------------------------------------------------
    rows = [{**p, "component": int(p["component"])} for p in ev["per_source"]]
    write_csv(cfg.METRICS_DIR / "metrics_main.csv", rows)
    meta = {"synthetic_sources": synthetic, "sources": info, "sample_rate": sr, "duration_s": a.duration, "seed": a.seed,
            "fastica": {**cfg.FASTICA, "converged": run.ica.converged, "n_iter": run.ica.n_iter,
                        "final_criterion": run.ica.history[-1]},
            "mixing": {**A_info}, "eigenvalues_of_covariance": run.whitening.eigvals,
            "whitening_max_abs_deviation_from_identity": whiten_err, "amari_index": amari,
            "assignment_source_to_component": ev["assignment"], "summary": ev["summary"], "per_source": ev["per_source"],
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}
    json.dump(meta, open(cfg.METRICS_DIR / "metrics_main.json", "w"), default=_jsonable, indent=2)
    # 9. bonuses --------------------------------------------------------------------------
    bonus = {}
    if not a.no_bonus:
        pv = bonus_pca_vs_ica(S, run, names)
        write_csv(cfg.METRICS_DIR / "bonus_pca_vs_ica.csv", pv)
        viz.plot_method_comparison(pv, P / "12_pca_vs_ica.png")
        sn = bonus_snr(S, A, names, a.seed)
        write_csv(cfg.METRICS_DIR / "bonus_snr.csv", sn)
        viz.plot_snr_curve(sn, P / "13_snr_robustness.png")
        sh = bonus_shazam(S, run, ev, names, sr, a.seed)
        json.dump(sh, open(cfg.METRICS_DIR / "bonus_shazam.json", "w"), default=_jsonable, indent=2)
        bonus = {"pca_vs_ica": pv, "snr": sn, "shazam": sh}
    # 10. verification of generated files ------------------------------------------------------
    wavs = [
        check_wav(p, sr)
        for folder in (cfg.MIXTURES_DIR, cfg.SEPARATED_DIR)
        for p in sorted(folder.glob("*.wav"))
    ]

    wavs += [
        check_wav(p, sr)
        for p in sorted(cfg.SOURCES_DIR.glob("*_prepared.wav"))
    ]
    for p in sorted(P.glob("*.png")):
        assert p.stat().st_size > 5000, f"plot looks empty: {p}"
    # 11. summary markdown (also injected into README) ---------------------------------------------
    L = [f"_Generated by `main.py` (seed {a.seed}, {a.duration:g} s @ {sr} Hz, "
         f"{'**SYNTHETIC sources**' if synthetic else 'real recordings'}). "
         f"FastICA ({cfg.FASTICA['nonlinearity']}): converged={run.ica.converged}, iterations={run.ica.n_iter}, "
         f"Amari index={amari:.4f}; cond(A)={A_info['condition_number']:.2f}; whitening max |Cov-I|={whiten_err:.1e}._", "",
         "| Source | Matched component | Pearson r (signed) | abs r | NMSE | MAE | SI-SDR (dB) |", "|---|---|---|---|---|---|---|"]
    for p in ev["per_source"]:
        L.append(f"| {p['source']} | {p['component']} | {p['pearson']:+.4f} | {p['abs_pearson']:.4f} | {p['nmse']:.4f} | {p['mae']:.4f} | {p['si_sdr_db']:.2f} |")
    s = ev["summary"]
    L.append(f"| **mean** | | | {s['mean_abs_pearson']:.4f} | {s['mean_nmse']:.4f} | {s['mean_mae']:.4f} | {s['mean_si_sdr_db']:.2f} |")
    if bonus:
        L += ["", "**Bonus A - PCA vs ICA (same mixtures, same matching):**", "", "| Method | mean abs r | mean SI-SDR (dB) | mean NMSE |", "|---|---|---|---|"]
        L += [f"| {r['method']} | {r['mean_abs_pearson']:.4f} | {r['mean_si_sdr_db']:.2f} | {r['mean_nmse']:.4f} |" for r in bonus["pca_vs_ica"]]
        L += ["", f"**Bonus B - input SNR sweep (mean of {cfg.SNR_TRIALS} noise trials):**", "", "| Input SNR (dB) | measured SNR (dB) | mean SI-SDR (dB) | mean abs r | converged |", "|---|---|---|---|---|"]
        L += [f"| {r['input_snr_db']} | {r['measured_input_snr_db']:.2f} | {r['mean_si_sdr_db']:.2f} +/- {r['std_si_sdr_db']:.2f} | {r['mean_abs_pearson']:.4f} | {r['converged_trials']} |" for r in bonus["snr"]]
        L += ["", "**Bonus C - mini-Shazam (synthetic 6-song library, 5 s queries):**", "", "| Query | best match | score | runner-up | n hashes |", "|---|---|---|---|---|"]
        L += [f"| {k} | {v['best_match']} | {v['best_score']} | {v['second_score']} | {v['n_query_hashes']} |" for k, v in bonus["shazam"].items()]
    md = "\n".join(L)
    (cfg.METRICS_DIR / "results_summary.md").write_text(md + "\n")
    update_readme(md)
    json.dump(wavs, open(cfg.METRICS_DIR / "audio_file_checks.json", "w"), indent=2)
    print(md)
    print(f"\nVerified {len(wavs)} WAV files and {len(list(P.glob('*.png')))} plots.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
