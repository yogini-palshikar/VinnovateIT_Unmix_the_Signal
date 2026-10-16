"""Mini audio-fingerprinting demo (Bonus C): spectral peak pairs + time-offset voting.

Educational toy only - NOT a production recogniser. Landmark-hash idea after Wang (2003).
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import spectrogram

NPERSEG, NOVERLAP = 512, 256


def peaks(x: np.ndarray, sr: int, fmax: float = 4000.0, nbhd: int = 15, floor_db: float = 50.0):
    """Return (freq_bin, time_frame) of local maxima of the log-magnitude spectrogram."""
    x = x - x.mean()
    f, t, P = spectrogram(x, sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    L = 10 * np.log10(P[f <= fmax] + 1e-12)
    mask = (L == maximum_filter(L, size=nbhd)) & (L > L.max() - floor_db)
    fi, ti = np.nonzero(mask)
    order = np.argsort(ti, kind="stable")
    return fi[order], ti[order]


def fingerprints(x: np.ndarray, sr: int, fan_out: int = 5, dt_max: int = 40, df_max: int = 40):
    """Hash pairs (anchor, target peak): h = (f1 << 20) | (f2 << 10) | dt. Returns list of (hash, t_anchor)."""
    fi, ti = peaks(x, sr)
    out = []
    for a in range(len(fi)):
        cnt = 0
        for b in range(a + 1, len(fi)):
            dt = ti[b] - ti[a]
            if dt > dt_max:
                break
            if dt < 1 or abs(int(fi[b]) - int(fi[a])) > df_max:
                continue
            out.append(((int(fi[a]) << 20) | (int(fi[b]) << 10) | int(dt), int(ti[a])))
            cnt += 1
            if cnt >= fan_out:
                break
    return out


class FingerprintIndex:
    def __init__(self):
        self.table: dict[int, list[tuple[str, int]]] = defaultdict(list)
        self.songs: list[str] = []

    def add_song(self, song_id: str, x: np.ndarray, sr: int):
        self.songs.append(song_id)
        for h, t in fingerprints(x, sr):
            self.table[h].append((song_id, t))

    def query(self, x: np.ndarray, sr: int) -> dict:
        """Vote on (song, db_time - query_time) offsets; the true song has a sharp offset peak."""
        votes: dict[tuple[str, int], int] = defaultdict(int)
        q = fingerprints(x, sr)
        for h, tq in q:
            for sid, td in self.table.get(h, ()):
                votes[(sid, td - tq)] += 1
        best_per_song = {s: 0 for s in self.songs}
        for (sid, _), c in votes.items():
            best_per_song[sid] = max(best_per_song[sid], c)
        ranked = sorted(best_per_song.items(), key=lambda kv: -kv[1])
        top, second = ranked[0][1], (ranked[1][1] if len(ranked) > 1 else 0)
        return {"best_match": ranked[0][0] if top > 0 else None, "best_score": top, "second_score": second,
                "score_ratio": float(top / max(second, 1)), "n_query_hashes": len(q), "ranking": ranked}
