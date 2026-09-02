"""
Find where the guitar plays, and split those stretches into groups that share
a tone. Two guitars with different amps in the same song end up in different
groups; the same guitar in verse and chorus ends up in one.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np
import librosa
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .dsp import to_mono, db, EPS

WIN = 2.0        # feature window, seconds
HOP = 1.0        # window hop, seconds
MIN_SECTION = 2.5
MERGE_GAP = 1.2


@dataclass
class Section:
    index: int
    start: float
    end: float
    label: int
    rms_db: float

    @property
    def duration(self):
        return self.end - self.start


@dataclass
class ToneGroup:
    id: int
    sections: List[Section] = field(default_factory=list)
    total_duration: float = 0.0
    clip_path: str = ""
    clip_start: float = 0.0
    clip_end: float = 0.0
    role: str = ""


def guitar_activity(y, sr, floor_offset_db=-26.0):
    """Boolean mask (per feature hop) of where the guitar stem is actually playing."""
    mono = to_mono(y)
    hop = int(sr * 0.05)
    rms = librosa.feature.rms(y=mono, frame_length=hop * 4, hop_length=hop)[0]
    L = db(rms + EPS)
    ref = np.percentile(L, 92)
    thresh = max(ref + floor_offset_db, np.percentile(L, 35))
    mask = L > thresh
    # Smooth: close gaps shorter than 0.4 s, drop blips shorter than 0.3 s.
    def runs(m, val):
        out, start = [], None
        for i, v in enumerate(m):
            if v == val and start is None:
                start = i
            elif v != val and start is not None:
                out.append((start, i)); start = None
        if start is not None:
            out.append((start, len(m)))
        return out
    fr = sr / hop
    for a, b in runs(mask, False):
        if (b - a) / fr < 0.4:
            mask[a:b] = True
    for a, b in runs(mask, True):
        if (b - a) / fr < 0.3:
            mask[a:b] = False
    return mask, fr, float(thresh), float(ref)


def active_regions(mask, fr, min_len=1.5):
    regions, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            regions.append((start / fr, i / fr)); start = None
    if start is not None:
        regions.append((start / fr, len(mask) / fr))
    merged = []
    for r in regions:
        if merged and r[0] - merged[-1][1] < MERGE_GAP:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(list(r) if isinstance(r, tuple) else r)
            merged[-1] = list(merged[-1])
    return [(a, b) for a, b in merged if b - a >= min_len]


def window_features(y, sr, regions):
    """One feature vector per analysis window. Tone-descriptive, pitch-blind."""
    mono = to_mono(y)
    feats, times = [], []
    for a, b in regions:
        t = a
        while t + WIN <= b + 1e-6 or (t == a and b - a >= 1.0):
            seg = mono[int(t * sr):int(min(t + WIN, b) * sr)]
            if seg.size < int(0.5 * sr):
                break
            mf = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
            v = list(mf.mean(axis=1)[1:])                     # drop MFCC0 (loudness)
            v += list(mf.std(axis=1)[1:5])
            v.append(float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr))) / 1000.0)
            v.append(float(np.mean(librosa.feature.spectral_flatness(y=seg))) * 10.0)
            v.append(float(np.mean(librosa.feature.spectral_bandwidth(y=seg, sr=sr))) / 1000.0)
            v.append(float(np.mean(librosa.feature.zero_crossing_rate(seg))) * 10.0)
            peak = float(np.max(np.abs(seg)) + EPS)
            rms = float(np.sqrt(np.mean(seg ** 2)) + EPS)
            v.append((db(peak) - db(rms)) / 10.0)
            feats.append(v)
            times.append((t, min(t + WIN, b)))
            t += HOP
    return np.array(feats, dtype=float), times


def cluster_tones(feats, max_k=5):
    """Pick the number of distinct tones by silhouette score. k=1 is a valid answer."""
    if feats.shape[0] < 4:
        return np.zeros(max(feats.shape[0], 1), dtype=int), 1, 0.0
    X = StandardScaler().fit_transform(feats)
    best_k, best_s, best_lab = 1, -1.0, np.zeros(X.shape[0], dtype=int)
    for k in range(2, min(max_k, X.shape[0] - 1) + 1):
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(X, lab)
        if s > best_s:
            best_k, best_s, best_lab = k, s, lab
    # Only accept a split if the clusters are genuinely separated.
    if best_s < 0.22:
        return np.zeros(X.shape[0], dtype=int), 1, float(max(best_s, 0.0))
    return best_lab, best_k, float(best_s)


def build_sections(times, labels, y, sr):
    mono = to_mono(y)
    raw = []
    for (a, b), lab in zip(times, labels):
        if raw and raw[-1][2] == lab and a - raw[-1][1] < 1.5:
            raw[-1][1] = b
        else:
            raw.append([a, b, int(lab)])
    out = []
    for i, (a, b, lab) in enumerate(raw):
        if b - a < MIN_SECTION:
            continue
        seg = mono[int(a * sr):int(b * sr)]
        r = float(db(np.sqrt(np.mean(seg ** 2)) + EPS)) if seg.size else -80.0
        out.append(Section(index=len(out), start=round(a, 2), end=round(b, 2),
                           label=int(lab), rms_db=round(r, 1)))
    return out


def group_and_render(sections: List[Section], y, sr, out_dir, max_clip=20.0):
    """One ToneGroup per cluster label, with a representative audio clip on disk."""
    from pathlib import Path
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for s in sections:
        groups.setdefault(s.label, ToneGroup(id=s.label)).sections.append(s)

    stereo = y if y.ndim == 2 else np.stack([y, y])
    result = []
    for gid, g in sorted(groups.items()):
        g.total_duration = round(sum(s.duration for s in g.sections), 2)
        # Representative clip: the longest section, centred, capped.
        best = max(g.sections, key=lambda s: s.duration)
        mid = (best.start + best.end) / 2
        half = min(max_clip, best.duration) / 2
        cs, ce = max(best.start, mid - half), min(best.end, mid + half)
        clip = stereo[:, int(cs * sr):int(ce * sr)]
        path = out_dir / f"tone_{gid + 1}.wav"
        sf.write(str(path), clip.T, sr)
        g.clip_path = str(path)
        g.clip_start, g.clip_end = round(cs, 2), round(ce, 2)
        result.append(g)
    result.sort(key=lambda g: g.total_duration, reverse=True)
    # Drop slivers: a few seconds of transitional material is not a "tone".
    total = sum(g.total_duration for g in result) or 1.0
    keep = [g for g in result
            if g.total_duration >= 6.0 and g.total_duration / total >= 0.05]
    dropped = [g for g in result if g not in keep]
    if not keep:
        keep = result[:1]
        dropped = result[1:]
    for g in dropped:                       # do not leave orphan clips on disk
        try:
            Path(g.clip_path).unlink(missing_ok=True)
        except Exception:
            pass
    for i, g in enumerate(keep):
        g.id = i
        new = out_dir / f"tone_{i + 1}.wav"
        if Path(g.clip_path) != new:
            Path(g.clip_path).replace(new)
            g.clip_path = str(new)
    return keep, dropped


def concat_group_audio(group: ToneGroup, y, sr, max_total=45.0):
    """All of a group's audio stitched together, for analysis (not for listening)."""
    mono = to_mono(y)
    chunks, total = [], 0.0
    for s in sorted(group.sections, key=lambda s: -s.duration):
        if total >= max_total:
            break
        take = min(s.duration, max_total - total)
        chunks.append(mono[int(s.start * sr):int((s.start + take) * sr)])
        total += take
    return np.concatenate(chunks) if chunks else np.zeros(sr)
