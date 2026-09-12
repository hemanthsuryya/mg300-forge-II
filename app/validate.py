"""
Validate a preset against the real thing.

Play or reamp through the MG-300, record what comes out, and this measures the
capture with the SAME pipeline that measured the song. The answer is a per-axis
difference in the units the axes are defined in, plus a specific instruction per
block - "Bass is 14 too low", not "sounds thin".

Why this is worth more than listening: the ear adapts within seconds, and A/B
comparison across a browser tab is hopeless. The measurement does not adapt.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from . import perceptual
from .pipeline import measure, SR

# How close each axis has to be before that dimension counts as matched.
# Wider tolerance where the underlying measurement is less certain.
TOLERANCE = {
    "grit": 8.0, "body": 10.0, "bite": 10.0, "honk": 10.0, "squash": 12.0,
    "space": 15.0, "echo": 15.0, "swirl": 20.0, "wah": 25.0,
}

# What to reach for when an axis is off, in the order worth trying.
FIXES = {
    "grit":   ("AMP Gain, then the drive pedal's Gain",
               "more gain / hotter pedal", "less gain / back the pedal off"),
    "body":   ("AMP Bass, then EQ 100-220 Hz",
               "raise Bass", "lower Bass, or pick a tighter cab"),
    "bite":   ("AMP Treble and Presence, then EQ 2.6-6.4 kHz",
               "raise Treble/Presence", "lower Treble/Presence, or a darker cab"),
    "honk":   ("AMP Middle, and the drive pedal's voicing",
               "raise Middle, or a mid-humped pedal like T Screamer",
               "lower Middle, or a scooped pedal"),
    "squash": ("COMP Sustain",
               "more compression", "less compression, or switch it off"),
    "space":  ("RVB Level, then Decay",
               "more reverb level", "less reverb level"),
    "echo":   ("DLY E.Level, then F.Back",
               "more repeats", "fewer repeats"),
    "swirl":  ("MOD Depth / Intensity",
               "more modulation depth", "less modulation depth"),
    "wah":    ("EFX Touch Wah Sense",
               "more wah sensitivity", "less wah, or switch it off"),
}


def _grade(score):
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "close"
    if score >= 55:
        return "in the neighbourhood"
    return "not there yet"


def compare_capture(reference_measurements: dict, capture_path: str,
                    target_axes: dict | None = None,
                    tempo_bpm: float = 120.0) -> dict:
    """
    reference_measurements: what the song's guitar measured
    capture_path:           audio recorded out of the MG-300
    target_axes:            the axis positions actually dialled in (from the
                            radar). Defaults to the analysed positions.
    """
    y, sr = librosa.load(capture_path, sr=SR, mono=False)
    if y.ndim == 1:
        stereo = np.stack([y, y])
        mono = y
    else:
        stereo = y
        mono = np.mean(y, axis=0)
    if mono.size < sr:
        raise ValueError("Capture is under a second long - record more of it.")

    cap_m = measure(mono, sr, tempo_bpm, stereo=stereo, sep_conf=1.0)

    # Space and Echo are measured from decay tails. A capture that stops dead
    # after the last note has no tails, and both would silently read zero -
    # which looks like "your reverb is missing" when it is really "your
    # recording is too short".
    warnings = []
    from .timefx import find_tails
    tails = find_tails(mono, sr)
    longest = max(((b - a) / sr for a, b in tails), default=0.0)
    if longest < 1.5:
        warnings.append(
            f"The longest decay tail in this capture is {longest:.1f}s. Reverb and "
            f"Delay need a tail to measure - stop playing and let at least 3 seconds "
            f"of decay record before you stop the take, then re-capture.")
    cap_axes = perceptual.to_axes(cap_m)
    ref_axes = dict(target_axes) if target_axes else perceptual.to_axes(reference_measurements)

    rows, weighted, wsum = [], 0.0, 0.0
    for a in perceptual.AXES:
        k = a["id"]
        want = float(ref_axes.get(k, 0.0))
        got = float(cap_axes.get(k, 0.0))
        diff = got - want
        tol = TOLERANCE[k]
        within = abs(diff) <= tol
        # 100 at zero error, 0 at three tolerances out.
        axis_score = float(np.clip(100 * (1 - abs(diff) / (3 * tol)), 0, 100))
        where, up, down = FIXES[k]
        rows.append({
            "axis": k, "label": a["label"], "target": round(want, 1),
            "captured": round(got, 1), "diff": round(diff, 1),
            "tolerance": tol, "within": within, "score": round(axis_score),
            "fix": None if within else
                   f"{'Too much' if diff > 0 else 'Too little'} - {down if diff > 0 else up}. "
                   f"Reach for: {where}.",
        })
        w = 1.6 if k in ("grit", "body", "bite", "honk") else 1.0
        weighted += axis_score * w
        wsum += w

    overall = float(weighted / max(wsum, 1e-9))
    worst = sorted([r for r in rows if not r["within"]],
                   key=lambda r: -abs(r["diff"]))
    if warnings:
        # Do not score dimensions the capture cannot support.
        for r in rows:
            if r["axis"] in ("space", "echo"):
                r["within"] = True
                r["fix"] = "not measurable in this capture - see the warning above"
        rows_for_score = [r for r in rows if r["axis"] not in ("space", "echo")]
        weighted = sum(r["score"] * (1.6 if r["axis"] in ("grit", "body", "bite", "honk") else 1.0)
                       for r in rows_for_score)
        wsum = sum(1.6 if r["axis"] in ("grit", "body", "bite", "honk") else 1.0
                   for r in rows_for_score)
        overall = float(weighted / max(wsum, 1e-9))
        worst = sorted([r for r in rows_for_score if not r["within"]],
                       key=lambda r: -abs(r["diff"]))

    return {
        "warnings": warnings,
        "overall_score": round(overall, 1),
        "grade": _grade(overall),
        "axes_matched": sum(1 for r in rows if r["within"]),
        "axes_total": len(rows),
        "rows": rows,
        "priority_fixes": [r["fix"] for r in worst[:3]],
        "capture": Path(capture_path).name,
        "capture_axes": cap_axes,
        "target_axes": ref_axes,
        "note": "Levels are normalised out, so overall loudness differences do "
                "not count against the score. Record the capture with the same "
                "playing you are comparing against - a different performance "
                "moves Compression and Drive on its own.",
    }
