"""
Realistic synthetic guitar for calibrating the detectors.

Karplus-Strong plucked strings (full harmonic series, natural decay) ->
optional saturation -> cabinet simulation. This behaves like a real mic'd
guitar in the ways that matter here: harmonic density, crest factor,
intermodulation between strings, and high-frequency rolloff.
"""
import numpy as np
import scipy.signal as sps

SR = 44100
rng = np.random.default_rng(7)

E_CHORD = [82.41, 123.47, 164.81, 207.65, 246.94, 329.63]   # E major
POWER5 = [82.41, 123.47, 164.81]                            # E5
LEAD_NOTES = [329.63, 392.00, 440.00, 493.88, 587.33]


def ks_pluck(f0, dur, sr=SR, damping=0.996, bright=0.5, seed=None):
    """Karplus-Strong string: comb resonator excited by a filtered noise burst."""
    r = np.random.default_rng(seed)
    N = max(int(round(sr / f0)), 4)
    n = int(dur * sr)
    exc = np.zeros(n)
    burst = r.uniform(-1, 1, N)
    # Pick position / hardness: lowpass the burst for a warmer attack.
    k = max(1, int((1 - bright) * 12))
    burst = np.convolve(burst, np.ones(k) / k, mode="same")
    exc[:N] = burst
    a = np.zeros(N + 2)
    a[0] = 1.0
    a[N] = -damping * 0.5
    a[N + 1] = -damping * 0.5
    y = sps.lfilter([1.0], a, exc)
    y *= np.minimum(1.0, np.arange(n) / (0.002 * sr))
    return y / (np.max(np.abs(y)) + 1e-9)


def chord(freqs, dur, sr=SR, strum=0.012, damping=0.996, bright=0.5, seed=0):
    n = int(dur * sr)
    out = np.zeros(n)
    for i, f in enumerate(freqs):
        y = ks_pluck(f, dur, sr, damping, bright, seed=seed * 100 + i)
        off = int(i * strum * sr)
        out[off:] += y[:n - off] * (0.9 ** i)
    return out / (np.max(np.abs(out)) + 1e-9)


def cabinet(y, sr=SR, low_hz=95, high_hz=5200, presence_hz=2600, presence_db=4.0):
    """4x12-ish response: highpass, steep top rolloff, upper-mid resonance."""
    sos_hp = sps.butter(2, low_hz / (sr / 2), btype="high", output="sos")
    sos_lp = sps.butter(6, high_hz / (sr / 2), btype="low", output="sos")
    y = sps.sosfilt(sos_lp, sps.sosfilt(sos_hp, y))
    w0 = presence_hz / (sr / 2)
    b, a = sps.iirpeak(w0, Q=1.2)
    y = sps.lfilter(b, a, y) * (10 ** (presence_db / 40)) * 0.5 + y * 0.5
    return y / (np.max(np.abs(y)) + 1e-9)


def saturate(y, drive=1.0, asym=0.0):
    x = np.tanh(drive * y + asym * (drive * y) ** 2 * 0.15)
    return x / (np.max(np.abs(x)) + 1e-9)


def rig(y, drive=1.0, sr=SR, **cab):
    """Full amp-ish chain: drive then cab. drive=1 is clean, 30 is metal."""
    return cabinet(saturate(y, drive), sr=sr, **cab)


def part(kind="rhythm", bars=6, bpm=120, sr=SR, tail=2.5, damping=0.996, seed=0):
    """A guitar part with musical spacing and a silent tail for decay analysis."""
    beat = 60.0 / bpm
    parts = []
    if kind == "rhythm":
        for b in range(bars):
            for i in range(4):
                parts.append(chord(POWER5 if i % 2 else E_CHORD, beat,
                                   sr=sr, damping=damping, seed=seed + b * 4 + i))
    else:
        for b in range(bars):
            for i in range(4):
                f = LEAD_NOTES[(b * 4 + i) % len(LEAD_NOTES)]
                parts.append(ks_pluck(f, beat, sr=sr, damping=0.9985,
                                      bright=0.7, seed=seed + b * 4 + i))
    y = np.concatenate(parts)
    return np.concatenate([y, np.zeros(int(tail * sr))])


def sustained(f0=196.0, dur=3.0, sr=SR, tail=1.0, damping=0.9995):
    """One long note - the clean case for modulation detection."""
    y = ks_pluck(f0, dur, sr, damping=damping, bright=0.6, seed=3)
    return np.concatenate([y, np.zeros(int(tail * sr))])


def tremolo(y, rate=5.0, depth=0.7, sr=SR):
    t = np.arange(len(y)) / sr
    return y * (1 - depth / 2 + depth / 2 * np.sin(2 * np.pi * rate * t))


def chorus(y, rate=0.9, depth_ms=6.0, sr=SR, mix=0.5):
    t = np.arange(len(y)) / sr
    d = (depth_ms / 1000.0) * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))
    idx = np.clip(np.arange(len(y)) - d * sr, 0, len(y) - 1)
    wet = np.interp(idx, np.arange(len(y)), y)
    return (1 - mix) * y + mix * wet


def phaser(y, rate=0.5, sr=SR, stages=4, mix=0.5):
    t = np.arange(len(y)) / sr
    lfo = 400 + 1400 * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))
    out = y.copy()
    block = 1024
    for i in range(0, len(y) - block, block):
        f = np.mean(lfo[i:i + block])
        b, a = sps.iirnotch(f / (sr / 2), Q=1.0)
        out[i:i + block] = sps.lfilter(b, a, y[i:i + block])
    return (1 - mix) * y + mix * out


def add_delay(y, time_s=0.375, feedback=0.45, mix=0.45, sr=SR, repeats=8):
    out = y.copy()
    n = int(time_s * sr)
    g = mix
    for r in range(1, repeats + 1):
        s = n * r
        if s >= len(y):
            break
        d = np.zeros_like(y)
        d[s:] = y[:len(y) - s] * g
        out += d
        g *= feedback
    return out / (np.max(np.abs(out)) + 1e-9)


def add_reverb(y, rt60=1.8, sr=SR, mix=0.35, seed=0, damping=1.6):
    r = np.random.default_rng(seed)
    n = int(rt60 * 1.4 * sr)
    t = np.arange(n) / sr
    ir = r.standard_normal(n) * 10 ** (-3 * t / rt60)
    # High frequencies decay faster in a real room.
    hf = sps.sosfilt(sps.butter(2, 2500 / (sr / 2), btype="high", output="sos"), ir)
    ir = ir - hf * (1 - 10 ** (-3 * t / (rt60 / damping))) * 0.5
    ir[:int(0.015 * sr)] *= np.linspace(0, 1, int(0.015 * sr))
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = sps.fftconvolve(y, ir)[:len(y)]
    wet /= (np.max(np.abs(wet)) + 1e-9)
    out = (1 - mix) * y + mix * wet
    return out / (np.max(np.abs(out)) + 1e-9)
