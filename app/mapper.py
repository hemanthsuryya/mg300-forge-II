"""
Tone measurements -> MG-300 MKII preset.

Every decision here is a rule with a stated reason, and every block carries a
confidence. Nothing is a black box: if a choice looks wrong you can read the
reason, change the number, and re-run.

Knob values are 0-100 to match the MG's display. Where a real unit is involved
(delay time, reverb decay, LFO rate) the physical target is given as well,
because the knob-to-value curve differs per model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CATALOG = json.loads((Path(__file__).parent / "catalog" / "mg300mk2.json").read_text())


def clamp(x, lo=0, hi=100):
    return int(round(float(np.clip(x, lo, hi))))


def brightness_from_centroid(hz):
    if not hz or hz <= 0:
        return 0.5
    return float(np.clip((math.log(hz) - math.log(700)) / (math.log(4000) - math.log(700)), 0, 1))


def mid_category(scoop_db):
    if scoop_db <= -4.0:
        return "scoop"
    if scoop_db <= -1.5:
        return "scoop_soft"
    if scoop_db >= 2.5:
        return "push"
    return "flat"


MID_AFFINITY = {
    ("scoop", "scoop"): 1.0, ("scoop", "scoop_soft"): 0.7, ("scoop", "flat"): 0.4, ("scoop", "push"): 0.1,
    ("scoop_soft", "scoop"): 0.7, ("scoop_soft", "scoop_soft"): 1.0, ("scoop_soft", "flat"): 0.75, ("scoop_soft", "push"): 0.35,
    ("flat", "scoop"): 0.4, ("flat", "scoop_soft"): 0.75, ("flat", "flat"): 1.0, ("flat", "push"): 0.7,
    ("push", "scoop"): 0.1, ("push", "scoop_soft"): 0.35, ("push", "flat"): 0.7, ("push", "push"): 1.0,
}


# ------------------------------------------------------------------ amp

def choose_amp(m):
    drive = m["drive"] * 100.0
    bright = brightness_from_centroid(m["shape"]["centroid_hz"])
    midcat = mid_category(m["shape"]["mid_scoop_db"])
    low = m["shape"]["band_ratios"]["low"] + m["shape"]["band_ratios"]["sub"]
    low_norm = float(np.clip(low / 0.45, 0, 1))

    scored = []
    for amp in CATALOG["amps"]:
        if amp["family"] == "acoustic" and drive > 25:
            continue
        w0, w1 = amp["gain_window"]
        if w0 <= drive <= w1:
            gain_fit = 1.0
        else:
            d = (w0 - drive) if drive < w0 else (drive - w1)
            gain_fit = float(np.clip(1.0 - d / 28.0, 0, 1))
        bright_fit = float(np.clip(1.0 - abs(bright - amp["brightness"]) / 0.45, 0, 1))
        mid_fit = MID_AFFINITY[(midcat, amp["mid_shape"])]
        low_fit = float(np.clip(1.0 - abs(low_norm - amp["low_weight"]) / 0.5, 0, 1))
        score = 0.46 * gain_fit + 0.24 * bright_fit + 0.20 * mid_fit + 0.10 * low_fit
        scored.append((score, amp))
    scored.sort(key=lambda t: -t[0])
    best, alts = scored[0], [a for _, a in scored[1:4]]
    margin = best[0] - (scored[1][0] if len(scored) > 1 else 0)
    return best[1], best[0], margin, alts, {"brightness": bright, "mid_cat": midcat,
                                            "low_norm": low_norm, "drive_pct": drive}


def amp_block(m, amp, ctx):
    drive = ctx["drive_pct"]
    w0, w1 = amp["gain_window"]
    span = max(w1 - w0, 1)
    gain = clamp(100 * (drive - w0) / span, 5, 98)

    tilt = m["shape"]["tilt_db"]
    # The amp already has its own voicing baked in; only ask the knobs for the
    # difference between what we measured and what this amp naturally does.
    bright_bias = (amp["brightness"] - 0.62) * 14.0     # dB-ish
    low_bias = (amp["low_weight"] - 0.55) * 14.0
    mid_bias = {"scoop": -4.0, "scoop_soft": -2.0, "flat": 0.0, "push": 3.5}[amp["mid_shape"]]

    bass = clamp(50 + 2.2 * ((tilt["low"] + tilt["sub"] * 0.3) - low_bias), 5, 95)
    middle = clamp(50 + 2.4 * ((tilt["lowmid"] * 0.4 + tilt["mid"] * 0.6) - mid_bias), 5, 95)
    treble = clamp(50 + 2.2 * (tilt["himid"] - bright_bias), 5, 95)
    presence = clamp(50 + 2.0 * (tilt["presence"] * 0.7 - bright_bias * 0.8), 5, 95)

    return {
        "module": "AMP",
        "enabled": True,
        "model": amp["name"],
        "based_on": amp["ref"],
        "params": {"Gain": gain, "Bass": bass, "Middle": middle,
                   "Treble": treble, "Presence": presence, "Master": 65},
        "why": f"{amp['voice']}. Measured drive {drive:.0f}/100 sits in this amp's "
               f"{w0}-{w1} window; tone stack set from the measured spectral tilt.",
    }


# ------------------------------------------------------------------ drive pedal

def efx_block(m, amp, ctx):
    wah = m.get("wah_amount", 0.0)
    if wah > 0.10:
        p = next(e for e in CATALOG["efx"] if e["id"] == "touch_wah")
        return {"module": "EFX", "model": p["name"], "enabled": True,
                "params": {"Type": 50, "Wow": clamp(25 + 70 * wah),
                           "Sense": clamp(30 + 60 * wah)},
                "why": "You asked for wah. Note this takes the single EFX slot, "
                       "so any drive pedal is displaced - the amp's own Gain now "
                       "has to supply all the dirt.",
                "user_set": True}
    drive = ctx["drive_pct"]
    w1 = amp["gain_window"][1]
    headroom_short = drive - w1
    scoop = m["shape"]["mid_scoop_db"]

    if headroom_short <= 2 and drive < 55:
        # Amp alone covers it. Offer a boost only for lead roles.
        if m.get("role") == "lead" and drive > 25:
            p = next(e for e in CATALOG["efx"] if e["id"] == "katana_boost")
            return {"module": "EFX", "model": p["name"], "enabled": True,
                    "params": {"Volume": 62, "Boost": 40},
                    "why": "Lead section: clean level lift into the amp, no extra dirt."}
        return {"module": "EFX", "model": "(off)", "enabled": False, "params": {},
                "why": "The amp's own gain covers the measured saturation - no pedal needed."}

    need = max(headroom_short, 0) + (10 if drive > 70 else 0)
    if scoop <= -4.0:
        family = ["muff_fuzz", "eat_dist", "dist_one"]
    elif scoop >= 2.5:
        family = ["t_scream", "red_dirt", "st_singer"]
    else:
        family = ["crunch", "blues_dry", "dist_one", "rc_boost"]
    pool = [e for e in CATALOG["efx"] if e["id"] in family]
    pick = min(pool, key=lambda e: abs(e["gain_add"] - max(need, 8)))

    gain_knob = clamp(30 + 2.2 * need, 15, 92)
    tone_knob = clamp(50 + 3.0 * m["shape"]["tilt_db"]["himid"])
    params = {}
    for p in pick["params"]:
        pl = p.lower()
        if "gain" in pl or "sustain" in pl or "fuzz" in pl or "sensitivity" in pl:
            params[p] = gain_knob
        elif "tone" in pl or "filter" in pl or "treble" in pl:
            params[p] = tone_knob
        elif "bass" in pl:
            params[p] = clamp(50 + 3.0 * m["shape"]["tilt_db"]["low"])
        else:
            params[p] = 60
    return {"module": "EFX", "model": pick["name"], "enabled": True, "params": params,
            "why": f"Measured drive ({drive:.0f}) exceeds the amp's natural range; "
                   f"{'mid-humped' if scoop >= 2.5 else 'scooped' if scoop <= -4 else 'flat'} "
                   f"voicing matches the section's midrange."}


# ------------------------------------------------------------------ compressor

def comp_block(m):
    crest = m["dyn"]["crest_db"]
    spread = m["dyn"]["env_spread_db"]
    sharp = m["dyn"]["attack_sharpness"]
    drive = m["drive"]

    # Heavy distortion compresses on its own - don't double up.
    if drive > 0.6:
        return {"module": "COMP", "model": "(off)", "enabled": False, "params": {},
                "why": "High-gain amp is already doing the compressing."}
    if crest > 13.5 and spread > 13:
        return {"module": "COMP", "model": "(off)", "enabled": False, "params": {},
                "why": f"Dynamics are open (crest {crest:.1f} dB) - no compression audible."}

    squash = float(np.clip((14.0 - crest) / 7.0, 0, 1))
    if sharp > 6.5 and drive < 0.25:
        model, params = "K Comp", {"Level": 55, "Sustain": clamp(30 + 55 * squash),
                                   "Clipping": clamp(20 + 30 * squash)}
    elif squash > 0.65:
        model, params = "Rose Comp", {"Level": 55, "Sustain": clamp(35 + 55 * squash)}
    else:
        model, params = "Studio Comp", {"Level": 55, "Threshold": clamp(70 - 40 * squash),
                                        "Ratio": clamp(25 + 45 * squash), "Release": 45}
    return {"module": "COMP", "model": model, "enabled": True, "params": params,
            "why": f"Crest factor {crest:.1f} dB and {spread:.1f} dB envelope spread "
                   f"indicate {'strong' if squash > 0.6 else 'light'} compression."}


# ------------------------------------------------------------------ modulation

def rate_to_knob(hz, lo=0.1, hi=10.0):
    return clamp(100 * (math.log(max(hz, lo)) - math.log(lo)) / (math.log(hi) - math.log(lo)))


def mod_block(m):
    """
    Decide whether a modulation effect is on, and which one.

    Gate first, classify second. The gate is the DEPTH of spectral-centroid
    movement, which separates cleanly in testing (dry material sits at 0.02-0.07,
    modulated material at 0.13+) where prominence and rate do not. Type is then
    read from what is moving: pitch wobble means a modulated delay line
    (chorus/vibrato), a sweeping comb means a flanger, a filter sweep with no
    pitch shift means a phaser.

    Type identification is the least certain part of this whole app, and the
    confidences returned here say so.
    """
    mod = m["mod"]
    strict = 1.0 if m.get("separation_confidence", 0.8) >= 0.6 else 1.6
    am_r, am_d, am_p = mod["am_rate_hz"], mod["am_depth"], mod["am_prominence"]
    am_h = mod.get("am_harmonicity", 1.0)
    cm_r, cm_d, cm_p = mod["cm_rate_hz"], mod["cm_depth"], mod["cm_prominence"]
    pm_r, pm_c, pm_p = mod["pm_rate_hz"], mod["pm_depth_cents"], mod["pm_prominence"]
    comb = mod["comb_strength"]
    comb_ms = mod.get("comb_ms")
    sweep_r = mod.get("comb_sweep_rate_hz")
    sweep_p = mod.get("comb_sweep_prominence", 0.0)
    sweep_h = mod.get("comb_sweep_harmonicity", 1.0)
    st = mod["stereo"]

    off = {"module": "MOD", "model": "(off)", "enabled": False, "params": {},
           "why": "No periodic amplitude, pitch or filter movement above what "
                  "ordinary playing produces. If you can hear movement on the "
                  "record, raise Swirl.",
           "confidence": 0.55}
    wide = (st.get("side_ratio") or 0) > 0.25

    # 1. Tremolo: loudness modulates as a clean sine. Checked first because it
    #    is the one modulation type that is measured reliably.
    if (am_p > 8.0 * strict and am_d > 0.15 and am_h < 0.55
            and (cm_p < 3.0 or am_d > 2 * cm_d)):
        return {"module": "MOD", "model": "Tremolo", "enabled": True,
                "params": {"Rate": rate_to_knob(am_r), "Depth": clamp(am_d * 110)},
                "target": {"rate_hz": round(am_r, 2), "depth_pct": clamp(am_d * 110)},
                "why": f"Loudness modulates as a clean sine at {am_r:.2f} Hz, "
                       f"{am_d*100:.0f}% depth (harmonic content {am_h:.2f}, so it "
                       f"is an LFO rather than picking).",
                "confidence": float(np.clip(am_p / 40, 0.35, 0.9))}

    # 2. Pitch is wobbling -> a modulated delay line (chorus / vibrato).
    #    Threshold sits just above what ordinary playing produces after
    #    per-note pitch detrending. This is the closest call the app makes and
    #    the confidence says so - the Swirl axis is there to overrule it.
    if pm_c > 18.0 * strict and cm_d > 0.07 and pm_r:
        if pm_c > 45:
            return {"module": "MOD", "model": "Vibrator", "enabled": True,
                    "params": {"Rate": rate_to_knob(pm_r), "Depth": clamp(pm_c * 1.6),
                               "Rise Time": 40},
                    "target": {"rate_hz": round(pm_r, 2), "depth_cents": round(pm_c, 1)},
                    "why": f"Deep pitch modulation, {pm_c:.0f} cents at {pm_r:.2f} Hz - "
                           f"too deep for chorus.",
                    "confidence": 0.45}
        model = "St. Chorus" if wide else "CE-2"
        params = ({"Rate": rate_to_knob(pm_r), "Width": clamp(20 + pm_c * 1.8),
                   "Intensity": 55} if wide else
                  {"Rate": rate_to_knob(pm_r), "Depth": clamp(20 + pm_c * 1.8)})
        return {"module": "MOD", "model": model, "enabled": True, "params": params,
                "target": {"rate_hz": round(pm_r, 2), "depth_cents": round(pm_c, 1)},
                "why": f"Pitch wobbles {pm_c:.1f} cents at {pm_r:.2f} Hz"
                       + (" and the image is wide - stereo chorus."
                          if wide else " - chorus.")
                       + " Chorus is the least certain call in this app; if the "
                         "record sounds dry, pull Swirl to zero.",
                "confidence": 0.4}

    # 3. Filter movement with no pitch shift -> phaser, or flanger if the comb
    #    that is moving sits in flanger territory (under ~6 ms).
    if cm_d > 0.105 * strict and pm_c < 8.0:
        if (comb > 3.0 * strict and sweep_p > 4.0 and sweep_r
                and comb_ms is not None and comb_ms < 6.0):
            return {"module": "MOD", "model": "Flanger", "enabled": True,
                    "params": {"Rate": rate_to_knob(sweep_r),
                               "Width": clamp(40 + 200 * cm_d),
                               "Feedback": clamp(35 + 25 * (comb - 3)), "Level": 60},
                    "target": {"rate_hz": round(sweep_r, 2), "comb_ms": round(comb_ms, 2)},
                    "why": f"Comb notches around {comb_ms:.1f} ms sweep at "
                           f"{sweep_r:.2f} Hz - a static harmonic series would not move.",
                    "confidence": 0.45}
        rate = cm_r or 1.0
        model = "Phase 100" if cm_d > 0.16 else "Phase 90"
        params = {"Speed": rate_to_knob(rate)}
        if model == "Phase 100":
            params["Intensity"] = clamp(35 + 220 * cm_d)
        return {"module": "MOD", "model": model, "enabled": True, "params": params,
                "target": {"rate_hz": round(rate, 2)},
                "why": f"The spectral centre sweeps {cm_d*100:.0f}% at about "
                       f"{rate:.2f} Hz with no pitch shift. The measured rate can land "
                       f"on a harmonic of the true LFO - halve it if it sounds twice "
                       f"as fast as the record.",
                "confidence": 0.4}

    # 4. Wide and static with no sweep: doubling rather than an LFO effect.
    if (st.get("side_ratio") or 0) > 0.45 and comb > 2.0:
        return {"module": "MOD", "model": "Detune", "enabled": True,
                "params": {"Shift-L": 45, "Shift-R": 55, "Mix": 45},
                "why": "Very wide, non-periodic stereo image - doubling or detune "
                       "rather than a swept effect.",
                "confidence": 0.4}
    return off


# ------------------------------------------------------------------ delay

def delay_block(m):
    d = m["delay"]
    need = 0.45 if m.get("separation_confidence", 0.8) >= 0.6 else 0.65
    if not d["time_s"] or d["confidence"] < need:
        return {"module": "DLY", "model": "(off)", "enabled": False, "params": {},
                "why": "No decaying repeats found in the decay regions.", "confidence": 0.5}
    if d["feedback"] > 0.85:
        # Echoes always get quieter. Repeats that hold their level are the
        # rhythm section bleeding through, not a delay.
        return {"module": "DLY", "model": "(off)", "enabled": False, "params": {},
                "why": f"Periodicity at {d['time_s']*1000:.0f} ms does not decay "
                       f"(level ratio {d['feedback']:.2f}) - that is the song's "
                       f"rhythm, not an echo.", "confidence": 0.6}

    ms = d["time_s"] * 1000.0
    dark = m.get("tail_dark", 0.0)   # >0 means repeats are darker than the dry signal
    stereo_wide = (m["mod"]["stereo"].get("side_ratio") or 0) > 0.35

    if ms < 140 and d["feedback"] < 0.35:
        model = "Tape Echo"; why_extra = "short slapback"
    elif stereo_wide:
        model = "Pan Delay"; why_extra = "repeats spread across the stereo field"
    elif dark > 2.0:
        model = "Analog Delay"; why_extra = "repeats are darker than the dry signal"
    elif m["mod"]["comb_strength"] > 2.8:
        model = "Mod Delay"; why_extra = "repeats carry modulation"
    else:
        model = "Digital Delay"; why_extra = "repeats keep the dry signal's brightness"

    time_knob = clamp(100 * ms / 1000.0)
    fb = clamp(d["feedback"] * 110)
    lvl = clamp(25 + 55 * d["confidence"])
    names = {e["id"]: e for e in CATALOG["delays"]}
    key = {"Tape Echo": "tape_echo", "Pan Delay": "pan_delay", "Analog Delay": "analog_delay",
           "Mod Delay": "mod_delay", "Digital Delay": "digi_delay"}[model]
    params = {}
    for p in names[key]["params"]:
        pl = p.lower()
        if "time" in pl or pl == "rate":
            params[p] = time_knob
        elif "back" in pl or "intensity" in pl:
            params[p] = fb
        elif "mod" in pl:
            params[p] = 35
        else:
            params[p] = lvl
    div = f" ({d['division']} at {m.get('tempo_bpm', 0):.0f} BPM)" if d["division"] else ""
    return {"module": "DLY", "model": model, "enabled": True, "params": params,
            "target": {"time_ms": round(ms), "feedback_pct": round(d["feedback"] * 100),
                       "division": d["division"]},
            "why": f"Repeats every {ms:.0f} ms{div}; {why_extra}.",
            "confidence": float(d["confidence"])}


# ------------------------------------------------------------------ reverb

def reverb_block(m):
    rv = m["reverb"]
    rt = rv["rt60_s"]
    need = 0.5 if m.get("separation_confidence", 0.8) >= 0.6 else 0.7
    if not rt or rv["late_energy_ratio"] < 0.02 or rv["confidence"] < need:
        return {"module": "RVB", "model": "(off)", "enabled": False, "params": {},
                "why": "Decay tails die too fast for an audible reverb.", "confidence": 0.5}

    hf = rv["rt60_hf_s"] or rt
    damping = rt / max(hf, 1e-3)         # >1.3 means the highs die first
    if rt < 0.9:
        model = "Room"
    elif rt < 1.9 and damping < 1.35 and m["drive"] < 0.45:
        model = "Spring"
    elif rt < 2.3:
        model = "Plate"
    elif rt < 4.0:
        model = "Hall"
    else:
        model = "Shimmer"

    lo, hi = next(r for r in CATALOG["reverbs"] if r["name"] == model)["typical_rt60"]
    decay = clamp(100 * (rt - lo) / max(hi - lo, 0.1), 10, 95)
    level = clamp(18 + 160 * rv["late_energy_ratio"], 8, 85)
    params = {"Decay": decay, "Pre Delay": clamp(20 + 30 * min(rt, 2) / 2), "Level": level}
    if model == "Shimmer":
        params["Mix"] = level
    return {"module": "RVB", "model": model, "enabled": True, "params": params,
            "target": {"rt60_s": round(rt, 2), "hf_damping": round(damping, 2)},
            "why": f"Measured RT60 {rt:.2f} s"
                   + (", highs damped first" if damping > 1.3 else ", even decay")
                   + f" - {model.lower()} territory.",
            "confidence": float(rv["confidence"])}


# ------------------------------------------------------------------ EQ + gate

def eq_block(m, amp_blk):
    """Residual correction after the amp tone stack has done what it can."""
    tilt = m["shape"]["tilt_db"]
    used = {"Bass": amp_blk["params"]["Bass"], "Middle": amp_blk["params"]["Middle"],
            "Treble": amp_blk["params"]["Treble"], "Presence": amp_blk["params"]["Presence"]}
    residual = {}
    # If the amp knob is already pinned, the EQ has to take up the slack.
    def slack(knob, want_db):
        if knob >= 95:
            return max(want_db - 14, 0)
        if knob <= 5:
            return min(want_db + 14, 0)
        return 0.0
    residual["100 Hz"] = slack(used["Bass"], tilt["low"]) + tilt["sub"] * 0.25
    residual["220 Hz"] = tilt["low"] * 0.2
    residual["500 Hz"] = tilt["lowmid"] * 0.35
    residual["1.2 kHz"] = slack(used["Middle"], tilt["mid"]) + tilt["mid"] * 0.2
    residual["2.6 kHz"] = slack(used["Treble"], tilt["himid"]) + tilt["himid"] * 0.15
    residual["6.4 kHz"] = slack(used["Presence"], tilt["presence"]) + tilt["air"] * 0.15
    residual = {k: round(float(np.clip(v, -9, 9)), 1) for k, v in residual.items()}

    if max(abs(v) for v in residual.values()) < 2.0:
        return {"module": "EQ", "model": "(off)", "enabled": False, "params": {},
                "why": "The amp tone stack already lands on the measured curve."}
    return {"module": "EQ", "model": "6-Band EQ", "enabled": True,
            "params": {k: f"{v:+.1f} dB" for k, v in residual.items()},
            "why": "Residual correction the amp's four knobs can't reach."}


def gate_block(m):
    nf = m.get("noise_floor_db", -70.0)
    play = m["dyn"]["rms_db"]
    margin = play - nf
    if m["drive"] < 0.35 and margin > 45:
        return {"module": "NG", "model": "Noise Gate", "enabled": False, "params": {},
                "why": "Clean tone with a low noise floor - gate not needed."}
    thr = clamp(25 + (60 - min(margin, 60)) * 1.1, 15, 85)
    return {"module": "NG", "model": "Noise Gate", "enabled": True,
            "params": {"Threshold": thr, "Decay": clamp(40 + 30 * m["drive"])},
            "why": f"{'High gain' if m['drive'] > 0.5 else 'Moderate gain'} with "
                   f"{margin:.0f} dB between playing level and noise floor."}


def ir_block(amp, m):
    cab = amp["default_cab"]
    bright = brightness_from_centroid(m["shape"]["centroid_hz"])
    note = ""
    if bright > 0.8 and cab in ("V412", "1960"):
        cab, note = "GB412", " (brighter cab chosen - the section is very top-heavy)"
    elif bright < 0.3 and cab in ("A212", "JZ120"):
        cab, note = "DR112", " (darker cab chosen - the section is very warm)"
    return {"module": "IR", "model": cab, "enabled": True, "params": {"Level": 60},
            "why": f"Standard pairing for {amp['name']}{note}."}


# ------------------------------------------------------------------ assemble

def build_preset(m, name="Preset"):
    amp, score, margin, alts, ctx = choose_amp(m)
    a = amp_block(m, amp, ctx)
    blocks = {
        "NG": gate_block(m),
        "COMP": comp_block(m),
        "EFX": efx_block(m, amp, ctx),
        "AMP": a,
        "IR": ir_block(amp, m),
        "EQ": eq_block(m, a),
        "MOD": mod_block(m),
        "DLY": delay_block(m),
        "RVB": reverb_block(m),
    }
    amp_conf = float(np.clip(0.35 + score * 0.4 + min(margin, 0.2) * 1.5, 0, 0.95))
    overall = float(np.mean([
        amp_conf,
        m.get("drive_confidence", 0.5),
        blocks["MOD"].get("confidence", 0.5),
        blocks["DLY"].get("confidence", 0.5),
        blocks["RVB"].get("confidence", 0.5),
        m.get("separation_confidence", 0.5),
    ]))
    moved = m.get("user_moved_axes") or []
    AXIS_BLOCKS = {"grit": ("AMP", "EFX"), "body": ("AMP", "EQ"), "bite": ("AMP", "EQ", "IR"),
                   "honk": ("AMP", "EQ", "EFX"), "squash": ("COMP",), "space": ("RVB",),
                   "echo": ("DLY",), "swirl": ("MOD",), "wah": ("EFX",)}
    touched = sorted({b for ax in moved for b in AXIS_BLOCKS.get(ax, ())})
    for blk in blocks.values():
        if blk["module"] in touched:
            blk["user_influenced"] = True
    return {
        "name": name[:24],
        "device": CATALOG["device"],
        "user_moved_axes": moved,
        "user_influenced_blocks": touched,
        "chain": [blocks[k] for k in CATALOG["chain_order"]],
        "amp_alternatives": [{"model": x["name"], "ref": x["ref"]} for x in alts],
        "confidence": {"amp": round(amp_conf, 2), "drive": round(m.get("drive_confidence", 0.5), 2),
                       "overall": round(overall, 2)},
        "measurements": {
            "drive_0_100": round(m["drive"] * 100, 1),
            "centroid_hz": round(m["shape"]["centroid_hz"]),
            "mid_scoop_db": round(m["shape"]["mid_scoop_db"], 1),
            "crest_db": round(m["dyn"]["crest_db"], 1),
            "decay_db_per_s": round(m["dec"]["decay_db_per_s"], 1),
            "rt60_s": m["reverb"]["rt60_s"] and round(m["reverb"]["rt60_s"], 2),
            "delay_ms": m["delay"]["time_s"] and round(m["delay"]["time_s"] * 1000),
            "tempo_bpm": round(m.get("tempo_bpm", 0)),
            "tilt_db": {k: round(v, 1) for k, v in m["shape"]["tilt_db"].items()},
        },
    }
