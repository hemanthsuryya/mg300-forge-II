"""
Time-based and modulation effect detection.

The trick used throughout: analyse the DECAY TAILS - the stretches right after
the player stops - because that is where delay repeats and reverb tails are
exposed instead of being buried under new notes.
"""
from __future__ import annotations

import numpy as np
import librosa
import scipy.signal as sps

from .dsp import to_mono, envelope, db, EPS


# ---------------------------------------------------------------- tails

def find_tails(y, sr, min_gap=0.55, max_take=8):
    """Return (start_sample, end_sample) of decaying stretches with no new onset."""
    y = to_mono(y)
    hop = 256
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="samples",
                                        hop_length=hop, backtrack=True)
    tails = []
    bounds = list(onsets) + [len(y)]
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if (b - a) / sr >= min_gap:
            tails.append((int(a + 0.06 * sr), int(b)))
    # Prefer the longest tails - they carry the most reverb information.
    tails.sort(key=lambda t: t[1] - t[0], reverse=True)
    return tails[:max_take]


# ---------------------------------------------------------------- reverb

def _decay_fit(L, fr, skip_s=0.10, min_span=0.30):
    """
    Fit the LATE part of a dB decay curve. Fitting from the very start would
    measure the note's own attack transient instead of the room.
    Returns (rt60_seconds, |r|) or (None, 0).
    """
    n0 = int(skip_s * fr)
    if L.size - n0 < int(min_span * fr) + 4:
        return None, 0.0
    L = L[n0:]
    peak = float(np.percentile(L, 97))
    floor = float(np.percentile(L, 3))
    if peak - floor < 12:
        return None, 0.0
    usable = (L < peak - 5.0) & (L > floor + 6.0)
    # longest contiguous usable run
    best, cur, start = (0, 0), 0, None
    for i, v in enumerate(usable):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if start is not None and len(usable) - start > best[1] - best[0]:
        best = (start, len(usable))
    a, b = best
    if (b - a) < int(min_span * fr) + 3:
        return None, 0.0
    t = np.arange(b - a) / fr
    seg = L[a:b]
    slope, _ = np.polyfit(t, seg, 1)
    if slope >= -1.0:
        return None, 0.0
    r = float(np.corrcoef(t, seg)[0, 1])
    return float(-60.0 / slope), float(abs(r))


def _tail_db_envelope(x, sr, band=None, hop=128):
    if band is not None:
        sos = sps.butter(4, [band[0] / (sr / 2), min(0.99, band[1] / (sr / 2))],
                         btype="band", output="sos")
        x = sps.sosfilt(sos, x)
    rms = librosa.feature.rms(y=x, frame_length=hop * 4, hop_length=hop)[0]
    return db(rms + EPS), sr / hop


def rt60_from_tail(tail, sr, band=None):
    x = np.asarray(tail, dtype=float)
    if x.size < int(0.35 * sr):
        return None, 0.0
    L, fr = _tail_db_envelope(x, sr, band)
    return _decay_fit(L, fr)


def _diffuseness_db(y, sr, tails):
    """
    How much more noise-like the tail is than the note body. A dry note keeps
    its harmonic peaks as it decays; a reverb tail turns into diffuse noise.
    This is what separates 'long sustain' from 'actual reverb'.
    """
    mono = to_mono(y)
    onsets = librosa.onset.onset_detect(y=mono, sr=sr, units="samples")
    def flat_db(seg):
        if seg.size < 2048:
            return None
        f = float(np.mean(librosa.feature.spectral_flatness(y=seg, n_fft=2048)))
        return 10 * np.log10(f + EPS)
    body = [flat_db(mono[o:o + int(0.12 * sr)]) for o in onsets[:40]]
    body = [b for b in body if b is not None]
    late = []
    for a, b in tails:
        s = a + int(0.25 * sr)
        if b - s > 2048:
            v = flat_db(mono[s:b])
            if v is not None:
                late.append(v)
    if not body or not late:
        return 0.0
    return float(np.median(late) - np.median(body))


def reverb_estimate(y, sr):
    tails = find_tails(y, sr)
    empty = {"rt60_s": None, "rt60_hf_s": None, "confidence": 0.0, "tail_count": 0,
             "late_energy_ratio": 0.0, "diffuseness_db": 0.0}
    if not tails:
        return empty
    mono = to_mono(y)
    rts, rts_hf, qs, lates = [], [], [], []
    for a, b in tails:
        seg = mono[a:b]
        rt, q = rt60_from_tail(seg, sr)
        rt_hf, _ = rt60_from_tail(seg, sr, band=(2000, 6000))
        if rt and 0.08 < rt < 12:
            rts.append(rt); qs.append(q)
        if rt_hf and 0.08 < rt_hf < 12:
            rts_hf.append(rt_hf)
        n1, n2 = int(0.10 * sr), int(0.35 * sr)
        if seg.size > n2 + int(0.2 * sr):
            early = np.mean(seg[:n1] ** 2) + EPS
            late = np.mean(seg[n2:n2 + int(0.2 * sr)] ** 2) + EPS
            lates.append(float(late / early))
    if not rts:
        return empty
    diff = _diffuseness_db(mono, sr, tails)
    # Confidence needs BOTH a clean exponential decay and a diffuse tail.
    fit_q = float(np.median(qs))
    diffuse_q = float(np.clip(diff / 6.0, 0.0, 1.0))
    return {
        "rt60_s": float(np.median(rts)),
        "rt60_hf_s": float(np.median(rts_hf)) if rts_hf else None,
        "confidence": float(np.clip(0.55 * fit_q + 0.45 * diffuse_q, 0, 1)),
        "tail_count": len(tails),
        "late_energy_ratio": float(np.median(lates)) if lates else 0.0,
        "diffuseness_db": diff,
    }

# ---------------------------------------------------------------- delay

def strong_onsets(y, sr, hop=256, pct=80):
    """
    Onsets of NOTES ACTUALLY PLAYED, not of echoes. Echo repeats are quieter
    than the note that produced them, so a percentile threshold on onset
    strength separates the two.
    """
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if env.size == 0:
        return np.array([], dtype=int)
    delta = float(np.percentile(env, pct))
    o = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop,
                                   units="frames", delta=delta, backtrack=False)
    return (np.asarray(o) * hop).astype(int)


def decay_regions(y, sr, min_gap=0.6, max_take=8):
    """Stretches after a played note where nothing new is struck."""
    so = strong_onsets(y, sr)
    bounds = list(so) + [len(y)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if (b - a) / sr >= min_gap:
            out.append((int(a + 0.05 * sr), int(b)))
    out.sort(key=lambda t: t[1] - t[0], reverse=True)
    return out[:max_take]


def _mode_cluster(vals, tol=0.12):
    """Median of the largest tight cluster - robust against stray peak spacings."""
    v = np.sort(np.asarray(vals, dtype=float))
    best, best_n = float(np.median(v)), 0
    for x in v:
        m = np.abs(v - x) <= tol * x
        if m.sum() > best_n:
            best_n, best = int(m.sum()), float(np.median(v[m]))
    return best


def delay_estimate(y, sr, tempo_bpm=None):
    """
    Find discrete repeats in decay regions.

    Deliberately conservative: a delay is only claimed when repeats are visible
    as periodic bumps in a decay region. Whole-signal autocorrelation is NOT
    trusted on its own, because a delay synced to the tempo is mathematically
    indistinguishable from the rhythm of the playing.
    """
    y = to_mono(y)
    hop = 128
    fr = sr / hop
    regions = decay_regions(y, sr, min_gap=0.6)
    gaps, feedbacks, ac_times = [], [], []

    for a, b in regions:
        seg = y[a:b]
        if seg.size < int(0.45 * sr):
            continue
        rms = librosa.feature.rms(y=seg, frame_length=hop * 4, hop_length=hop)[0]
        L = db(rms + EPS)
        # Truncate (do not boolean-mask) at the point the tail hits the noise
        # floor - masking would splice non-adjacent frames and corrupt the
        # spacing between repeats.
        above = np.where(L > (L.max() - 45))[0]
        if above.size < 12:
            continue
        L = L[: above[-1] + 1]
        if L.size < 12:
            continue
        t = np.arange(L.size) / fr
        resid = L - np.polyval(np.polyfit(t, L, 1), t)

        pk, props = sps.find_peaks(resid, prominence=2.0, distance=int(0.04 * fr))
        if pk.size >= 2:
            g = np.diff(t[pk])
            g = g[(g > 0.04) & (g < 1.5)]
            if g.size:
                gaps.extend(g.tolist())
                h = props["prominences"]
                if h.size >= 2 and h[0] > 0:
                    feedbacks.append(float(np.clip(np.mean(h[1:]) / h[0], 0, 1)))

        # autocorrelation WITHIN the decay region (no playing here, so any
        # periodicity is the delay itself)
        r = resid - resid.mean()
        ac = librosa.autocorrelate(r)
        if ac[0] > 0:
            ac = ac / ac[0]
            lo, hi = int(0.04 * fr), min(int(1.5 * fr), ac.size - 1)
            if hi > lo + 4:
                p2, _ = sps.find_peaks(ac[lo:hi], height=0.25)
                if p2.size:
                    ac_times.append(float((p2[int(np.argmax(ac[lo:hi][p2]))] + lo) / fr))

    time_s, conf, source = None, 0.0, "none"
    if gaps:
        time_s = _mode_cluster(gaps)
        conf = float(min(0.9, 0.35 + 0.10 * len(gaps)))
        source = "tail_repeats"
        if ac_times and abs(float(np.median(ac_times)) - time_s) < 0.05:
            conf = min(0.95, conf + 0.15)
            source = "tail_repeats+autocorr"
    elif ac_times:
        time_s = float(np.median(ac_times))
        conf = 0.40
        source = "tail_autocorrelation"
    else:
        # Last resort, reported but never trusted: whole-signal periodicity is
        # confounded with the rhythm of the part.
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        ac = librosa.autocorrelate(onset_env - onset_env.mean())
        if ac.size and ac[0] > 0:
            ac = ac / ac[0]
            lo, hi = int(0.05 * fr), min(int(1.5 * fr), ac.size - 1)
            if hi > lo + 4:
                p2, _ = sps.find_peaks(ac[lo:hi], height=0.2)
                if p2.size:
                    time_s = float((p2[int(np.argmax(ac[lo:hi][p2]))] + lo) / fr)
                    conf = 0.20
                    source = "rhythm_ambiguous"
        if time_s is None:
            return {"time_s": None, "feedback": 0.0, "confidence": 0.0,
                    "division": None, "source": "none"}

    division = None
    if tempo_bpm:
        beat = 60.0 / tempo_bpm
        divs = {"1/4": beat, "dotted 1/8": beat * 0.75, "1/8": beat / 2,
                "1/8 triplet": beat / 3, "1/16": beat / 4, "1/2": beat * 2}
        name, val = min(divs.items(), key=lambda kv: abs(kv[1] - time_s))
        if abs(val - time_s) / (time_s + EPS) < 0.10:
            division = name
    fb = float(np.median(feedbacks)) if feedbacks else 0.25
    return {"time_s": time_s, "feedback": fb, "confidence": conf,
            "division": division, "source": source}


# ---------------------------------------------------------------- modulation

def _dominant_rate(series, fr, lo=0.2, hi=14.0, reject_rates=(), want_harmonicity=False):
    """
    Peak frequency, depth and prominence of a slow periodic series.

    `reject_rates` blanks out frequencies that coincide with the playing rhythm.
    `want_harmonicity` also returns how much energy sits at 2x and 3x the found
    rate: an LFO is a sine (harmonicity near 0) while a strumming pattern is a
    pulse train (harmonicity high). That is what stops steady picking from being
    reported as a tremolo.
    """
    x = np.asarray(series, dtype=float)
    nothing = (None, 0.0, 0.0, 1.0) if want_harmonicity else (None, 0.0, 0.0)
    if x.size < 24:
        return nothing
    x = x - np.mean(x)
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fr)
    m = (freqs >= lo) & (freqs <= hi)
    for r in reject_rates:
        if r > 0:
            m &= np.abs(freqs - r) > max(0.10 * r, 0.25)
    if not m.any() or spec[m].max() <= 0:
        return nothing
    i = int(np.argmax(spec[m]))
    rate = float(freqs[m][i])
    peak = float(spec[m][i])
    floor = float(np.median(spec[m]) + EPS)
    prominence = peak / floor
    depth = float(2.0 * peak / (win.sum() + EPS))
    if not want_harmonicity:
        return rate, depth, prominence

    def energy_at(f):
        band = np.abs(freqs - f) <= max(0.06 * f, 0.15)
        return float(spec[band].max()) if band.any() else 0.0
    harm = (energy_at(2 * rate) + energy_at(3 * rate)) / (peak + EPS)
    return rate, depth, prominence, float(harm)


def _active_span(mono, sr, hop=256, drop_db=32.0):
    """
    Longest contiguous stretch where the guitar is actually sounding.
    Without this, trailing silence dominates every envelope FFT and no LFO is
    ever found.
    """
    rms = librosa.feature.rms(y=mono, frame_length=hop * 4, hop_length=hop)[0]
    L = db(rms + EPS)
    keep = L > (np.percentile(L, 97) - drop_db)
    best, start = (0, 0), None
    for i, v in enumerate(keep):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if start is not None and len(keep) - start > best[1] - best[0]:
        best = (start, len(keep))
    a, b = best
    if b - a < 16:
        return 0, len(mono)
    return a * hop, min(b * hop, len(mono))


def _hp_envelope(y, sr, hop=256, window_s=2.5):
    """
    dB loudness envelope with slow drift removed (the note's own decay), so what
    survives is modulation in the 0.4-14 Hz LFO range.

    Deliberately NOT detrended per note: a tremolo creates its own amplitude
    rises, and per-note detrending would subtract the very thing being measured.
    Instead the rhythm of genuinely struck notes is estimated separately and its
    harmonics are excluded from the LFO search.
    """
    mono = to_mono(y)
    rms = librosa.feature.rms(y=mono, frame_length=hop * 4, hop_length=hop)[0]
    L = db(rms + EPS)
    L = np.maximum(L, L.max() - 60)
    fr = sr / hop
    w = max(int(window_s * fr) | 1, 5)
    from scipy.ndimage import uniform_filter1d
    resid = L - uniform_filter1d(L, size=w, mode="nearest")

    # Rate of notes that were actually STRUCK. Taken from the percussive
    # component: a pick attack has transient content, an LFO swell does not.
    play_rate = 0.0
    try:
        _, y_p = librosa.effects.hpss(mono, margin=(1.0, 3.0))
        o = librosa.onset.onset_detect(y=y_p, sr=sr, hop_length=hop, units="frames")
        play_rate = len(o) / (L.size / fr + EPS)
    except Exception:
        pass
    return resid, fr, float(play_rate)


def _db_depth_to_ratio(pp_db):
    """Peak-to-peak dB swing -> tremolo depth as a 0..1 fraction."""
    return float(np.clip(1.0 - 10 ** (-abs(pp_db) / 20.0), 0.0, 1.0))


def modulation_estimate(y, sr, y_stereo=None):
    mono = to_mono(y)
    hop = 256
    fr = sr / hop
    a, b = _active_span(mono, sr, hop)
    mono = mono[a:b]
    if mono.size < sr // 2:
        mono = to_mono(y)

    resid, fr_env, play_rate = _hp_envelope(mono, sr, hop)
    # Only reject rhythm harmonics when there IS a steady rhythm to confuse us.
    reject = ([play_rate * k for k in (1, 2, 3, 4)] if play_rate > 1.0 else [])

    # Amplitude modulation is measured on the LINEAR envelope divided by its own
    # slow average. A tremolo stays a clean sine there; strumming does not.
    lin = librosa.feature.rms(y=mono, frame_length=hop * 4, hop_length=hop)[0]
    from scipy.ndimage import uniform_filter1d
    w = max(int(2.5 * fr) | 1, 5)
    slow = uniform_filter1d(lin, size=w, mode="nearest") + EPS
    ratio = lin / slow
    am_rate, am_amp, am_prom, am_harm = _dominant_rate(
        ratio, fr, reject_rates=reject, want_harmonicity=True)
    am_depth = float(np.clip(2 * am_amp / (1 + am_amp + EPS), 0, 1))

    # spectral centroid modulation -> phaser / flanger / wah
    cent = librosa.feature.spectral_centroid(y=mono, sr=sr, hop_length=hop)[0]
    cent_n = cent / (np.mean(cent) + EPS)
    cm_rate, cm_depth, cm_prom, cm_harm = _dominant_rate(
        cent_n, fr, reject_rates=reject, want_harmonicity=True)

    # Pitch modulation -> chorus / vibrato.
    # Measured WITHIN each note: the player changing notes is melody, not
    # modulation, and subtracting each note's own median pitch is the only way
    # to stop a melody reading as a vibrato.
    pm_rate, pm_cents, pm_prom, pm_harm = None, 0.0, 0.0, 1.0
    try:
        f0 = librosa.yin(mono, fmin=70, fmax=1000, sr=sr, frame_length=2048,
                         hop_length=hop)
        voiced = np.isfinite(f0) & (f0 > 0)
        cents = np.zeros_like(f0)
        onsets = librosa.onset.onset_detect(y=mono, sr=sr, hop_length=hop,
                                            units="frames", backtrack=True)
        bounds = [0] + [int(o) for o in onsets] + [f0.size]
        ok = np.zeros(f0.size, dtype=bool)
        skip = int(0.05 * fr)            # ignore the pitch-unstable attack
        for a, b in zip(bounds[:-1], bounds[1:]):
            a = min(a + skip, b)
            seg_v = voiced[a:b]
            if (b - a) < 12 or seg_v.sum() < 10:
                continue
            seg = f0[a:b].copy()
            med = float(np.median(seg[seg_v]))
            if med <= 0:
                continue
            c = 1200 * np.log2(np.maximum(seg, 1e-6) / med)
            c[~seg_v] = 0.0
            # Also remove any slide/bend within the note - that is playing too.
            t = np.arange(c.size)
            if c.size > 8:
                c = c - np.polyval(np.polyfit(t, c, 1), t)
            cents[a:b] = np.clip(c, -120, 120)
            ok[a:b] = True
        if ok.sum() > 48:
            series = sps.medfilt(cents, 5)
            pm_rate, pm_depth, pm_prom, pm_harm = _dominant_rate(
                series, fr, reject_rates=reject, want_harmonicity=True)
            pm_cents = float(pm_depth)
    except Exception:
        pass

    # comb filtering -> flanger / short-delay colouration
    n_fft = 4096
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=1024))
    logS = np.log(S + EPS)
    ceps = np.abs(np.fft.rfft(logS, axis=0))
    df = sr / n_fft                       # Hz per spectral bin
    quef = np.arange(ceps.shape[0]) / (logS.shape[0] * df)   # seconds
    band = (quef > 0.0005) & (quef < 0.015)                  # 0.5-15 ms combs
    comb_strength, comb_ms = 0.0, None
    comb_sweep_rate, comb_sweep_prom, comb_sweep_harm = None, 0.0, 1.0
    if band.any():
        cb = ceps[band]
        cm = cb.mean(axis=1)
        i = int(np.argmax(cm))
        comb_strength = float(cm[i] / (np.median(cm) + EPS))
        comb_ms = float(quef[band][i] * 1000)
        # A guitar's own harmonic series ALSO looks like a comb - the pitch
        # period lands in the same quefrency range. What separates a flanger is
        # that its comb SWEEPS. Track the peak quefrency per frame and test
        # whether it moves periodically and smoothly.
        track = quef[band][np.argmax(cb, axis=0)] * 1000.0     # ms per frame
        fr_cep = sr / 1024
        if track.size > 24:
            comb_sweep_rate, _, comb_sweep_prom, comb_sweep_harm = _dominant_rate(
                track / (np.mean(track) + EPS), fr_cep, lo=0.1, hi=6.0,
                reject_rates=reject, want_harmonicity=True)

    stereo = {"correlation": None, "side_ratio": None}
    if y_stereo is not None and np.ndim(y_stereo) == 2 and np.shape(y_stereo)[0] == 2:
        l, r = y_stereo[0], y_stereo[1]
        n = min(l.size, r.size)
        if n > sr:
            mid = (l[:n] + r[:n]) / 2
            side = (l[:n] - r[:n]) / 2
            stereo = {"correlation": float(np.corrcoef(l[:n], r[:n])[0, 1]),
                      "side_ratio": float((np.mean(side ** 2) + EPS) /
                                          (np.mean(mid ** 2) + EPS))}

    return {
        "am_rate_hz": am_rate, "am_depth": float(am_depth),
        "am_harmonicity": float(am_harm), "am_prominence": float(am_prom),
        "cm_rate_hz": cm_rate, "cm_depth": float(cm_depth),
        "cm_prominence": float(cm_prom), "cm_harmonicity": float(cm_harm),
        "pm_rate_hz": pm_rate, "pm_depth_cents": float(pm_cents),
        "pm_prominence": float(pm_prom), "pm_harmonicity": float(pm_harm),
        "comb_strength": comb_strength, "comb_ms": comb_ms,
        "comb_sweep_rate_hz": comb_sweep_rate,
        "comb_sweep_prominence": float(comb_sweep_prom),
        "comb_sweep_harmonicity": float(comb_sweep_harm),
        "play_rate_hz": play_rate,
        "stereo": stereo,
    }
