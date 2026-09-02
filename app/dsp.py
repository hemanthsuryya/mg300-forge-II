"""
Low-level DSP feature extraction for guitar tone analysis.

Everything in here is deterministic signal processing - no ML, no guessing.
Each function returns physically meaningful numbers (Hz, ms, dB, seconds)
so the mapping stage can be audited and argued with.
"""
from __future__ import annotations

import numpy as np
import librosa
import scipy.signal as sps

EPS = 1e-12

# Analysis bands (Hz). Chosen to line up with how guitar players actually talk
# about tone, not with octave convention.
BANDS = {
    "sub":      (20, 80),
    "low":      (80, 250),
    "lowmid":   (250, 600),
    "mid":      (600, 1600),
    "himid":    (1600, 4000),
    "presence": (4000, 8000),
    "air":      (8000, 16000),
}

# Reference tilt for an "average" mic'd electric guitar cab, in dB relative to
# the 250-600 Hz band. Used as the zero point for EQ suggestions so that a
# section is described by how it DIFFERS from a normal guitar tone.
# Zero point for tone description: the band-energy balance of a typical
# close-miked electric guitar cab, in dB relative to the 250-600 Hz band.
# A section is then described by how it DIFFERS from a normal guitar tone.
# This curve is an editable assumption - if your reference tone sits elsewhere,
# change it here and every EQ suggestion moves with it.
REFERENCE_TILT_DB = {
    "sub": -22.0, "low": -3.0, "lowmid": 0.0, "mid": -1.5,
    "himid": -5.0, "presence": -13.0, "air": -25.0,
}


def db(x):
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=float), EPS))


def to_mono(y):
    return y if y.ndim == 1 else np.mean(y, axis=0)


def band_energies(y, sr, n_fft=4096, hop=1024):
    """Mean power per band, and the full averaged magnitude spectrum."""
    y = to_mono(y)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    spec = S.mean(axis=1)
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        out[name] = float(spec[m].sum()) if m.any() else 0.0
    return out, spec, freqs


def spectral_shape(y, sr):
    """Centroid, rolloff, flatness, band ratios and a scoop/push descriptor."""
    y = to_mono(y)
    energies, spec, freqs = band_energies(y, sr)
    total = sum(energies.values()) + EPS
    ratios = {k: v / total for k, v in energies.items()}

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff85 = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

    # Mid scoop: mids relative to the mean of lows and highs. Negative = scooped.
    lo = ratios["low"] + ratios["lowmid"]
    mid = ratios["mid"]
    hi = ratios["himid"] + ratios["presence"]
    scoop = float(db(mid + EPS) - db(0.5 * (lo + hi) + EPS))

    # Tilt in dB per band relative to the reference guitar curve, then centred
    # so only the SHAPE matters. Without centring, a reference curve that is a
    # few dB off would push every EQ suggestion the same direction and rail the
    # knobs.
    ref_band = energies["lowmid"] + EPS
    tilt = {}
    for k in BANDS:
        rel = 10.0 * np.log10((energies[k] + EPS) / ref_band)
        tilt[k] = float(np.clip(rel - REFERENCE_TILT_DB[k], -18, 18))
    core = ["low", "lowmid", "mid", "himid"]
    centre = float(np.mean([tilt[k] for k in core]))
    tilt = {k: float(np.clip(v - centre, -12, 12)) for k, v in tilt.items()}

    return {
        "centroid_hz": centroid,
        "rolloff85_hz": rolloff85,
        "flatness": flatness,
        "bandwidth_hz": bandwidth,
        "band_ratios": ratios,
        "mid_scoop_db": scoop,
        "tilt_db": tilt,
    }


def envelope(y, sr, hop=256):
    """RMS envelope and its frame rate."""
    y = to_mono(y)
    rms = librosa.feature.rms(y=y, frame_length=hop * 4, hop_length=hop)[0]
    return rms, sr / hop


def dynamics(y, sr):
    """Crest factor, envelope spread, transient sharpness - the compression cues."""
    y = to_mono(y)
    peak = float(np.max(np.abs(y)) + EPS)
    rms_all = float(np.sqrt(np.mean(y ** 2)) + EPS)
    crest_db = float(db(peak) - db(rms_all))

    rms, fr = envelope(y, sr)
    active = rms[rms > np.percentile(rms, 40)]
    if active.size < 4:
        active = rms + EPS
    spread_db = float(np.percentile(db(active), 95) - np.percentile(db(active), 10))

    # Attack sharpness: median rise of the onset strength peaks.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    sharp = float(np.percentile(onset_env, 95) / (np.median(onset_env) + EPS))

    return {"crest_db": crest_db, "env_spread_db": spread_db,
            "attack_sharpness": sharp, "rms_db": float(db(rms_all))}


def sustain_decay(y, sr):
    """
    Median note decay rate in dB/s, measured over the 150 ms after each onset.
    Distorted / compressed tones decay slowly; clean picked tones decay fast.
    """
    y = to_mono(y)
    rms, fr = envelope(y, sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="frames",
                                        hop_length=256, backtrack=True)
    rates = []
    win = int(0.18 * fr)
    for o in onsets:
        # Start at the envelope PEAK after the onset, not at the onset frame -
        # onset backtracking lands before the attack, which would measure the
        # note rising and report a positive "decay".
        look = rms[o:o + int(0.10 * fr)]
        if look.size < 2:
            continue
        o = o + int(np.argmax(look))
        a, b = o + int(0.01 * fr), o + win
        if b >= len(rms) or b - a < 5:
            continue
        seg = db(rms[a:b] + EPS)
        t = np.arange(seg.size) / fr
        slope = np.polyfit(t, seg, 1)[0]
        if np.isfinite(slope):
            rates.append(slope)
    decay = float(np.median(rates)) if rates else -20.0
    density = len(onsets) / (len(y) / sr + EPS)
    return {"decay_db_per_s": decay, "onset_density_hz": float(density),
            "n_onsets": int(len(onsets))}


def polyphony(y, sr):
    """Median count of simultaneously strong pitch classes: 1 = single notes."""
    chroma = librosa.feature.chroma_cqt(y=to_mono(y), sr=sr)
    strong = (chroma > 0.65 * chroma.max(axis=0, keepdims=True)).sum(axis=0)
    return float(np.median(strong))


def valley_depth_db(y, sr):
    """
    Depth of the gaps between spectral partials, in dB (90th - 25th percentile
    of the log spectrum per frame). Saturation and intermodulation fill those
    gaps in, so this number DROPS as gain goes up. Chordal material sits lower
    than single notes, which is why drive_score weights it by polyphony.
    """
    S = np.abs(librosa.stft(to_mono(y), n_fft=4096, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    b = (freqs > 250) & (freqs < 5000)
    L = 20 * np.log10(S[b] + 1e-10)
    if L.size == 0:
        return 25.0
    return float(np.mean(np.percentile(L, 90, axis=0) - np.percentile(L, 25, axis=0)))


def hf_sustain(y, sr):
    """
    How much 2-6 kHz energy survives 175 ms into a note relative to its attack.
    Clean notes lose their highs fast; saturated notes keep buzzing.
    """
    mono = to_mono(y)
    sos = sps.butter(4, [2000 / (sr / 2), min(0.99, 6000 / (sr / 2))],
                     btype="band", output="sos")
    hi = sps.sosfilt(sos, mono)
    hop = 256
    env = librosa.feature.rms(y=hi, frame_length=1024, hop_length=hop)[0]
    onsets = librosa.onset.onset_detect(y=mono, sr=sr, hop_length=hop, units="frames")
    n = int(0.175 * sr / hop)
    ratios = [env[o + n] / (env[o] + EPS) for o in onsets
              if o + n < len(env) and env[o] > EPS]
    return float(np.median(ratios)) if ratios else 0.5


def distortion_metrics(y, sr):
    """Saturation cues that survive inside a real mix."""
    y = to_mono(y)
    energies, spec, freqs = band_energies(y, sr)
    total = sum(energies.values()) + EPS
    upper = (energies["himid"] + energies["presence"] + energies["air"]) / total
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=2048,
                                                           hop_length=512)))
    return {
        "upper_energy_ratio": float(upper),
        "zcr": zcr,
        "valley_db": valley_depth_db(y, sr),
        "hf_sustain": hf_sustain(y, sr),
        "polyphony": polyphony(y, sr),
    }


def ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo + EPS), 0.0, 1.0))


def drive_score(dist, dyn, dec):
    """
    Blend saturation cues into a single 0..1 drive estimate.

    Anchors were calibrated against a physical-model guitar rendered through a
    cabinet at five known drive settings (tests/test_detectors.py). Two cues do
    the heavy lifting and they are complementary:
      - crest factor: works on any material, saturates above 'crunch'
      - valley depth: keeps discriminating into high gain, but only on chords
      - hf sustain: keeps discriminating into high gain on single notes
    So the second and third are blended by measured polyphony.
    """
    poly = dist.get("polyphony", 2.0)
    chordal = float(np.clip((poly - 1.0) / 0.8, 0.0, 1.0))   # 1 note -> 0, ~2+ -> 1

    s_crest = 1.0 - ramp(dyn["crest_db"], 8.0, 17.0)
    s_valley = 1.0 - ramp(dist["valley_db"], 12.0, 24.0)
    s_hfsus = ramp(dist["hf_sustain"], 0.25, 0.85)
    s_decay = 1.0 - ramp(abs(dec["decay_db_per_s"]), 2.0, 25.0)

    s_density = chordal * s_valley + (1 - chordal) * s_hfsus
    score = 0.40 * s_crest + 0.45 * s_density + 0.15 * s_decay

    parts = {"crest": s_crest, "valley": s_valley, "hf_sustain": s_hfsus,
             "density_blend": s_density, "sustain": s_decay, "chordal": chordal}
    agree = [s_crest, s_density]
    conf = float(np.clip(1.0 - abs(agree[0] - agree[1]) / 0.6, 0.25, 0.95))
    return float(np.clip(score, 0, 1)), parts, conf
