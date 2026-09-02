"""
The perceptual layer: nine musician-language axes over the measurements.

Design rule that makes this trustworthy: an axis does NOT edit a knob. It edits
the MEASUREMENT the knob was derived from, and then the ordinary mapper re-runs.
Drag Grit up and the chain re-solves the way it would have if the song had
actually been that saturated - the amp can change, the drive pedal can change,
the compressor can drop out because the amp is now doing the squashing.

Consequence worth knowing: forward(measurements) -> axes -> inverse -> mapper
reproduces the original preset EXACTLY when nothing is dragged, because inverse
applies deltas, not absolutes. tests/test_perceptual.py pins that down.
"""
from __future__ import annotations

import copy
import math

import numpy as np

AXES = [
    {"id": "grit",   "label": "Grit",   "hint": "raspiness, saturation",
     "from": "crest factor + spectral valley depth", "drives": "AMP Gain, drive pedal"},
    {"id": "body",   "label": "Body",   "hint": "low-end weight",
     "from": "80-250 Hz balance", "drives": "AMP Bass, EQ lows, cab"},
    {"id": "bite",   "label": "Bite",   "hint": "presence, attack edge",
     "from": "1.6-8 kHz balance + centroid", "drives": "AMP Treble/Presence, cab"},
    {"id": "honk",   "label": "Honk",   "hint": "scooped to mid-forward",
     "from": "mid scoop in dB", "drives": "AMP Middle, pedal voicing"},
    {"id": "squash", "label": "Squash", "hint": "compression, sustain",
     "from": "crest factor + envelope spread", "drives": "COMP"},
    {"id": "space",  "label": "Space",  "hint": "wet to dry",
     "from": "RT60 + late tail energy", "drives": "RVB"},
    {"id": "echo",   "label": "Echo",   "hint": "delay amount",
     "from": "repeat level + feedback", "drives": "DLY"},
    {"id": "swirl",  "label": "Swirl",  "hint": "chorus, phaser, tremolo",
     "from": "modulation depth", "drives": "MOD"},
    {"id": "wah",    "label": "Wah",    "hint": "envelope filter",
     "from": "not measured - taste control", "drives": "EFX slot (Touch Wah)"},
]
AXIS_IDS = [a["id"] for a in AXES]

# Wah and the drive pedal are the same physical block on this unit.
CONFLICTS = [{"axes": ["wah"], "with": "EFX drive pedal",
              "note": "Touch Wah and an overdrive share the single EFX slot. "
                      "Raising Wah above 10 takes that slot."}]


def _c(x, lo=0.0, hi=100.0):
    return float(np.clip(x, lo, hi))


def _ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo + 1e-12), 0, 1))


# --------------------------------------------------------------- forward

def to_axes(m: dict) -> dict:
    """
    Measurements -> the nine axes, 0-100. This is where the radar starts.

    For the three effect axes the enablement question is answered by ASKING THE
    MAPPER, not by re-implementing its thresholds here. Otherwise the radar can
    show Echo at 95 while the chain shows DLY off - which happened, because the
    mapper rejects repeats that do not decay and this function did not know.
    """
    from . import mapper                      # local import: avoids a cycle
    tilt = m["shape"]["tilt_db"]
    crest = m["dyn"]["crest_db"]

    rv = m["reverb"]
    if mapper.reverb_block(m).get("enabled") and rv.get("rt60_s"):
        space = 100 * (0.45 * _ramp(rv["rt60_s"], 0.3, 3.5)
                       + 0.55 * _ramp(rv.get("late_energy_ratio", 0), 0.01, 0.35))
    else:
        space = 0.0

    d = m["delay"]
    if mapper.delay_block(m).get("enabled") and d.get("time_s"):
        echo = 100 * (0.5 * _ramp(d.get("confidence", 0), 0.4, 0.95)
                      + 0.5 * _ramp(d.get("feedback", 0.25), 0.05, 0.7))
    else:
        echo = 0.0

    mod = m["mod"]
    if mapper.mod_block(m).get("enabled"):
        swirl = 100 * max(
            _ramp(mod.get("am_depth", 0) if mod.get("am_harmonicity", 1) < 0.55 else 0,
                  0.1, 0.8),
            _ramp(mod.get("cm_depth", 0), 0.08, 0.35),
            _ramp(mod.get("pm_depth_cents", 0), 4.0, 40.0),
        )
    else:
        swirl = 0.0

    return {
        "grit":   _c(m["drive"] * 100),
        "body":   _c(50 + 3.5 * tilt["low"] + 1.5 * tilt["sub"]),
        "bite":   _c(50 + 3.5 * tilt["himid"] + 1.5 * tilt["presence"]),
        "honk":   _c(50 + 4.0 * m["shape"]["mid_scoop_db"]),
        "squash": _c(100 - 100 * _ramp(crest, 6.0, 18.0)),
        "space":  _c(space),
        "echo":   _c(echo),
        "swirl":  _c(swirl),
        "wah":    _c(100 * m.get("wah_amount", 0.0)),
    }


# --------------------------------------------------------------- inverse

def apply_axes(m: dict, axes: dict) -> tuple[dict, list[str]]:
    """
    Apply the user's axis positions by perturbing the measurements.

    Only DELTAS from the analysed position are applied, so an untouched axis
    changes nothing at all and the original preset is reproduced bit for bit.
    Returns the perturbed measurements and the list of axes the user moved.
    """
    m = copy.deepcopy(m)
    base = to_axes(m)
    moved = [k for k in AXIS_IDS
             if k in axes and abs(float(axes[k]) - base[k]) > 0.5]
    if not moved:
        return m, []

    def delta(k):
        return float(axes[k]) - base[k] if k in axes else 0.0

    tilt = m["shape"]["tilt_db"]

    # --- Grit: the drive estimate itself
    if "grit" in moved:
        m["drive"] = float(np.clip(float(axes["grit"]) / 100.0, 0, 1))
        m["drive_confidence"] = min(m.get("drive_confidence", 0.5), 0.95)

    # --- Body / Bite / Honk: spectral tilt
    if "body" in moved:
        d = delta("body")
        tilt["low"] += d / 3.5
        tilt["sub"] += d / 3.5 * 0.6
    if "bite" in moved:
        d = delta("bite")
        tilt["himid"] += d / 3.5
        tilt["presence"] += d / 3.5 * 0.6
        tilt["air"] += d / 3.5 * 0.3
        m["shape"]["centroid_hz"] = float(m["shape"]["centroid_hz"] * 2 ** (d / 120.0))
    if "honk" in moved:
        d = delta("honk")
        m["shape"]["mid_scoop_db"] += d / 4.0
        tilt["mid"] += d / 4.0
        tilt["lowmid"] += d / 4.0 * 0.5
    for k in tilt:
        tilt[k] = float(np.clip(tilt[k], -14, 14))

    # --- Squash: crest factor is what the compressor rule reads
    if "squash" in moved:
        target = float(axes["squash"])
        m["dyn"]["crest_db"] = float(6.0 + (1 - target / 100.0) * 12.0)
        m["dyn"]["env_spread_db"] = float(np.clip(
            22.0 - 0.16 * target, 4.0, 26.0))

    # --- Space: synthesise the reverb measurements the mapper reads
    if "space" in moved:
        v = float(axes["space"]) / 100.0
        if v < 0.06:
            m["reverb"] = {**m["reverb"], "rt60_s": None, "confidence": 0.0,
                           "late_energy_ratio": 0.0}
        else:
            keep_rt = m["reverb"].get("rt60_s")
            rt = keep_rt if (keep_rt and "space" not in moved) else 0.35 + v * 3.2
            m["reverb"] = {**m["reverb"],
                           "rt60_s": float(rt),
                           "rt60_hf_s": float(rt / 1.3),
                           "late_energy_ratio": float(0.01 + v * 0.34),
                           "diffuseness_db": 8.0,
                           "confidence": 0.9}

    # --- Echo: same idea for delay
    if "echo" in moved:
        v = float(axes["echo"]) / 100.0
        if v < 0.06:
            m["delay"] = {**m["delay"], "time_s": None, "confidence": 0.0}
        else:
            t = m["delay"].get("time_s")
            if not t:
                bpm = m.get("tempo_bpm") or 120.0
                t = 60.0 / bpm / 2.0           # an eighth note is the safe default
                div = "1/8"
            else:
                div = m["delay"].get("division")
            m["delay"] = {**m["delay"], "time_s": float(t), "division": div,
                          "feedback": float(np.clip(0.05 + v * 0.6, 0, 0.8)),
                          "confidence": 0.9, "source": "user"}

    # --- Swirl: move the modulation measurements across or below the mapper's
    #     gate, PRESERVING whatever character was detected. Scaling the wrong
    #     feature would silently turn a phaser into a chorus.
    if "swirl" in moved:
        from . import mapper                              # local: avoids a cycle
        v = float(axes["swirl"]) / 100.0
        mod = dict(m["mod"])
        kind_by_model = {x["name"]: x["kind"] for x in mapper.CATALOG["mods"]}
        cur = mapper.mod_block(m)
        kind = kind_by_model.get(cur["model"]) if cur.get("enabled") else None

        if v < 0.06:
            mod.update({"cm_depth": 0.0, "pm_depth_cents": 0.0, "am_depth": 0.0,
                        "comb_sweep_prominence": 0.0})
        elif kind == "tremolo":
            mod["am_depth"] = float(np.clip(0.16 + v * 0.75, 0, 0.98))
            mod["am_prominence"] = max(mod.get("am_prominence", 0), 12.0)
            mod["am_harmonicity"] = min(mod.get("am_harmonicity", 0.2), 0.3)
        elif kind in ("phaser", "flanger"):
            mod["cm_depth"] = float(0.11 + v * 0.22)
            mod["cm_prominence"] = max(mod.get("cm_prominence", 0), 8.0)
            mod["pm_depth_cents"] = min(mod.get("pm_depth_cents", 0.0), 6.0)
        else:
            # chorus/vibrato, or nothing detected - chorus is the safe default
            mod["pm_depth_cents"] = float(19.0 + v * 24.0)   # stays under vibrato
            mod["pm_prominence"] = max(mod.get("pm_prominence", 0), 10.0)
            mod["pm_harmonicity"] = min(mod.get("pm_harmonicity", 0.2), 0.3)
            if kind is None:
                # Nothing was detected, so the measured pitch rate is noise, not
                # an LFO. Use a musical default instead of an unreliable number.
                mod["pm_rate_hz"] = 1.1
            else:
                mod["pm_rate_hz"] = mod.get("pm_rate_hz") or 1.1
            mod["cm_depth"] = float(max(mod.get("cm_depth", 0.0), 0.08 + v * 0.10))
        m["mod"] = mod

    # --- Wah: a taste control, not a measurement
    if "wah" in moved:
        m["wah_amount"] = float(axes["wah"]) / 100.0

    m["user_moved_axes"] = moved
    return m, moved
