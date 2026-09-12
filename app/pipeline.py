"""
Full job pipeline: audio in -> stems -> guitar sections -> tone groups ->
measurements -> MG-300 MKII presets.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

from . import dsp, timefx, sections as secmod, mapper
from .separate import separate_guitar

SR = 44100


def _role(y, sr):
    """Rhythm vs lead, from note density, register and how chordal it is."""
    mono = dsp.to_mono(y)
    onset_density = len(librosa.onset.onset_detect(y=mono, sr=sr)) / (len(mono) / sr + 1e-9)
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
    # How many pitch classes are simultaneously strong -> chordal vs single-note.
    strong = (chroma > 0.65 * chroma.max(axis=0, keepdims=True)).sum(axis=0)
    polyphony = float(np.median(strong))
    try:
        f0 = librosa.yin(mono, fmin=70, fmax=1200, sr=sr)
        med_f0 = float(np.median(f0[np.isfinite(f0)]))
    except Exception:
        med_f0 = 200.0
    lead_score = (0.4 * (polyphony < 2.5) + 0.3 * (med_f0 > 260)
                  + 0.3 * (onset_density < 4.5))
    return ("lead" if lead_score >= 0.6 else "rhythm"), {
        "onset_density_hz": round(onset_density, 2),
        "polyphony": round(polyphony, 2),
        "median_f0_hz": round(med_f0, 1),
    }


def _tail_darkness(y, sr):
    """Positive dB value = decay tails are darker than the body of the note."""
    mono = dsp.to_mono(y)
    tails = timefx.find_tails(mono, sr, min_gap=0.5)
    if not tails:
        return 0.0
    full_c = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))
    cs = []
    for a, b in tails:
        seg = mono[a:b]
        if seg.size > int(0.15 * sr):
            cs.append(float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr))))
    if not cs:
        return 0.0
    return float(20 * np.log10((full_c + 1e-9) / (np.median(cs) + 1e-9)))


def _noise_floor_db(y, sr):
    mono = dsp.to_mono(y)
    hop = int(sr * 0.05)
    rms = librosa.feature.rms(y=mono, frame_length=hop * 4, hop_length=hop)[0]
    return float(dsp.db(np.percentile(rms, 5) + 1e-12))


def measure(seg, sr, tempo_bpm, stereo=None, sep_conf=0.8):
    shape = dsp.spectral_shape(seg, sr)
    dyn = dsp.dynamics(seg, sr)
    dec = dsp.sustain_decay(seg, sr)
    dist = dsp.distortion_metrics(seg, sr)
    drive, drive_parts, drive_conf = dsp.drive_score(dist, dyn, dec)
    mod = timefx.modulation_estimate(seg, sr, y_stereo=stereo)
    dly = timefx.delay_estimate(seg, sr, tempo_bpm=tempo_bpm)
    rvb = timefx.reverb_estimate(seg, sr)
    role, role_info = _role(seg, sr)
    return {
        "shape": shape, "dyn": dyn, "dec": dec, "dist": dist,
        "drive": drive, "drive_parts": drive_parts, "drive_confidence": drive_conf,
        "mod": mod, "delay": dly, "reverb": rvb,
        "role": role, "role_info": role_info,
        "tempo_bpm": tempo_bpm,
        "tail_dark": _tail_darkness(seg, sr),
        "noise_floor_db": _noise_floor_db(seg, sr),
        "separation_confidence": sep_conf,
    }


def run_job(audio_path: str, job_dir: str, progress=lambda p, msg: None,
            demucs_model: str = "htdemucs_6s", max_tones: int = 5,
            display_name: str = ""):
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    progress(5, "Loading audio")
    mix, sr = librosa.load(audio_path, sr=SR, mono=True)
    duration = len(mix) / sr
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=mix, sr=sr)[0])[0])
    progress(10, f"{duration:.0f}s, tempo about {tempo:.0f} BPM")

    progress(15, "Separating the guitar from the mix")
    guitar_path, sep = separate_guitar(audio_path, str(job_dir / "stems"),
                                       model=demucs_model,
                                       log=lambda m: progress(None, m))
    g, _ = librosa.load(guitar_path, sr=SR, mono=False)
    if g.ndim == 1:
        g = np.stack([g, g])
    gm = dsp.to_mono(g)

    stem_floor = _noise_floor_db(gm, sr)
    progress(55, "Finding where the guitar plays")
    mask, fr, thr, ref = secmod.guitar_activity(gm, sr)
    regions = secmod.active_regions(mask, fr)
    if not regions:
        raise RuntimeError("No guitar activity detected in the separated stem.")

    progress(62, f"{len(regions)} guitar region(s); comparing their tones")
    feats, times = secmod.window_features(gm, sr, regions)
    if feats.size == 0:
        raise RuntimeError("Guitar regions too short to analyse.")
    labels, k, sil = secmod.cluster_tones(feats, max_k=max_tones)
    secs = secmod.build_sections(times, labels, gm, sr)
    if not secs:
        secs = [secmod.Section(0, regions[0][0], regions[0][1], 0, -20.0)]
    groups, dropped = secmod.group_and_render(secs, g, sr, job_dir / "clips")

    if dropped:
        progress(None, f"Ignoring {len(dropped)} short transitional fragment(s) "
                       f"({sum(d.total_duration for d in dropped):.0f}s total)")
    progress(72, f"{len(groups)} distinct guitar tone(s) found; analysing each")
    results = []
    for i, grp in enumerate(groups):
        progress(72 + int(24 * i / max(len(groups), 1)),
                 f"Analysing tone {i + 1} of {len(groups)}")
        seg = secmod.concat_group_audio(grp, gm, sr)
        st = None
        s0, s1 = int(grp.clip_start * sr), int(grp.clip_end * sr)
        if g.shape[1] > s1 > s0:
            st = g[:, s0:s1]
        m = measure(seg, sr, tempo, stereo=st, sep_conf=sep["confidence"])
        m["noise_floor_db"] = stem_floor
        label = f"{'Lead' if m['role'] == 'lead' else 'Rhythm'} {i + 1}"
        preset = mapper.build_preset(m, name=label)
        results.append({
            "tone_id": i + 1,
            "name": label,
            "role": m["role"],
            "total_seconds": grp.total_duration,
            "share_pct": round(100 * grp.total_duration /
                               max(sum(x.total_duration for x in groups), 1e-9), 1),
            "sections": [asdict(s) for s in sorted(grp.sections, key=lambda s: s.start)],
            "clip": Path(grp.clip_path).name,
            "clip_range": [grp.clip_start, grp.clip_end],
            "preset": preset,
            "measurements_raw": _jsonable(m),
        })

    out = {
        "source": display_name or Path(audio_path).name,
        "duration_s": round(duration, 1),
        "tempo_bpm": round(tempo, 1),
        "separation": sep,
        "cluster": {"k": int(k), "silhouette": round(sil, 3),
                    "windows": int(feats.shape[0]),
                    "dropped_fragments": len(dropped)},
        "tones": results,
        "elapsed_s": round(time.time() - t0, 1),
        "device": mapper.CATALOG["device"],
        "catalog_version": mapper.CATALOG["catalog_version"],
    }
    (job_dir / "result.json").write_text(json.dumps(out, indent=2))
    progress(100, "Done")
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
