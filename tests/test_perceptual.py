"""
Tests for the radar axes and the hardware-validation scoring.

The cross-coupling section mirrors tests T4-T12 in docs/VALIDATION.md: each axis
must move its own blocks and leave the others alone. Running it in software
first means that when a real capture shows coupling, the pedal or the rig is the
suspect - not the mapping.

Run: python3 tests/test_perceptual.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import numpy as np
import soundfile as sf

from app import perceptual, mapper, validate
from app.pipeline import measure
from tests import synth as S
from tests.synth import SR

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name} | {detail}")
    print(("  pass  " if cond else "  FAIL  ") + name + (" | " + detail if detail else ""))


def blocks(p):
    return {b["module"]: (b["model"], tuple(sorted((b.get("params") or {}).items())))
            for b in p["chain"]}


def preset_for(m, axes=None):
    mm, _ = perceptual.apply_axes(m, axes or {})
    return mapper.build_preset(mm, "t")


MATERIAL = {
    "clean chords": S.rig(S.part("rhythm", bars=3, tail=2.0), drive=1.5),
    "crunch chords": S.rig(S.part("rhythm", bars=3, tail=2.0), drive=8.0),
    "metal chords": S.rig(S.part("rhythm", bars=3, tail=2.0), drive=40.0),
    "clean lead": S.rig(S.part("lead", bars=3, tail=2.0), drive=2.0),
}
M = {k: measure(v, SR, 120.0) for k, v in MATERIAL.items()}

print("\n== untouched radar must reproduce the analysed preset exactly ==")
for name, m in M.items():
    base = mapper.build_preset(m, "t")
    ax = perceptual.to_axes(m)
    again = preset_for(m, ax)
    check(f"{name}: identity", blocks(base) == blocks(again),
          "chain identical" if blocks(base) == blocks(again) else "CHAIN DRIFTED")

print("\n== the radar must never contradict the chain ==")
for name, m in M.items():
    ax = perceptual.to_axes(m)
    p = mapper.build_preset(m, "t")
    on = {b["module"]: (b.get("enabled", True) and b["model"] != "(off)")
          for b in p["chain"]}
    ok = all([(ax["space"] > 0) == on["RVB"], (ax["echo"] > 0) == on["DLY"],
              (ax["swirl"] > 0) == on["MOD"]])
    check(f"{name}: effect axes agree with blocks", ok,
          f"space {ax['space']:.0f}/RVB {on['RVB']}, echo {ax['echo']:.0f}/DLY "
          f"{on['DLY']}, swirl {ax['swirl']:.0f}/MOD {on['MOD']}")

print("\n== each axis must move in the direction it claims ==")
m = M["crunch chords"]
ax = perceptual.to_axes(m)


def amp_of(p):
    return next(b for b in p["chain"] if b["module"] == "AMP")


lo = amp_of(preset_for(m, {**ax, "grit": max(ax["grit"] - 40, 5)}))
hi = amp_of(preset_for(m, {**ax, "grit": min(ax["grit"] + 40, 98)}))
fam = {a["name"]: a["family"] for a in mapper.CATALOG["amps"]}
order = {"clean": 0, "acoustic": 0, "crunch": 1, "hi_gain": 2, "metal": 3}
check("grit up -> hotter amp family or more gain",
      order[fam[hi["model"]]] > order[fam[lo["model"]]]
      or hi["params"]["Gain"] > lo["params"]["Gain"],
      f"{lo['model']} g{lo['params']['Gain']} -> {hi['model']} g{hi['params']['Gain']}")

for axis, knob in (("body", "Bass"), ("bite", "Treble"), ("honk", "Middle")):
    a_lo = amp_of(preset_for(m, {**ax, axis: max(ax[axis] - 35, 2)}))
    a_hi = amp_of(preset_for(m, {**ax, axis: min(ax[axis] + 35, 98)}))
    check(f"{axis} up -> AMP {knob} up",
          a_hi["params"][knob] > a_lo["params"][knob],
          f"{a_lo['params'][knob]} -> {a_hi['params'][knob]}")

print("\n== effect axes must switch their block on and off ==")
for axis, module in (("space", "RVB"), ("echo", "DLY"), ("swirl", "MOD")):
    for name, mm in M.items():
        on = preset_for(mm, {**perceptual.to_axes(mm), axis: 75})
        off = preset_for(mm, {**perceptual.to_axes(mm), axis: 0})
        b_on = next(b for b in on["chain"] if b["module"] == module)
        b_off = next(b for b in off["chain"] if b["module"] == module)
        ok = b_on.get("enabled") and b_on["model"] != "(off)" and \
            (not b_off.get("enabled") or b_off["model"] == "(off)")
        check(f"{axis}=75/0 toggles {module} ({name})", ok,
              f"on={b_on['model']} off={b_off['model']}")

print("\n== wah takes the shared EFX slot, and says so ==")
w = preset_for(M["crunch chords"], {**ax, "wah": 80})
efx = next(b for b in w["chain"] if b["module"] == "EFX")
check("wah -> Touch Wah in EFX", efx["model"] == "Touch Wah", efx["model"])
check("wah conflict is explained", "displaced" in efx.get("why", "").lower(),
      efx.get("why", "")[:60])

print("\n== cross-coupling: an axis must not disturb unrelated blocks (T4-T12) ==")
# Which modules each axis is allowed to touch. Anything else moving is a bug.
# EFX appears for body/bite on purpose: when a drive pedal is in the chain its
# own Tone and Bass controls are part of the tone stack, so the mapper sets them
# from the same spectral tilt. That is intended behaviour, not cross-talk.
ALLOWED = {"grit": {"AMP", "EFX", "COMP", "NG", "EQ", "IR"},
           "body": {"AMP", "EQ", "IR", "EFX"}, "bite": {"AMP", "EQ", "IR", "EFX"},
           "honk": {"AMP", "EQ", "EFX"}, "squash": {"COMP"},
           "space": {"RVB"}, "echo": {"DLY"}, "swirl": {"MOD"}, "wah": {"EFX"}}
base_blocks = blocks(preset_for(M["crunch chords"], ax))
for axis, allowed in ALLOWED.items():
    moved_to = min(ax[axis] + 40, 95) if ax[axis] < 50 else max(ax[axis] - 40, 5)
    nb = blocks(preset_for(M["crunch chords"], {**ax, axis: moved_to}))
    changed = {k for k in base_blocks if base_blocks[k] != nb[k]}
    stray = changed - allowed
    check(f"{axis} touches only its own blocks", not stray,
          f"changed {sorted(changed) or 'nothing'}"
          + (f"; STRAY {sorted(stray)}" if stray else ""))

print("\n== validation scoring, rehearsed in software (no hardware needed) ==")
os.makedirs("/tmp/vtest", exist_ok=True)
ref_audio = MATERIAL["crunch chords"]
ref_m = M["crunch chords"]
ref_axes = perceptual.to_axes(ref_m)

cases = [
    ("identical capture", ref_audio, None, None),
    ("too dark", S.rig(S.part("rhythm", bars=3, tail=2.0), drive=8.0,
                       high_hz=2600, presence_db=0.0), "bite", "-"),
    ("too clean", S.rig(S.part("rhythm", bars=3, tail=2.0), drive=1.0), "grit", "-"),
    ("too saturated", S.rig(S.part("rhythm", bars=3, tail=2.0), drive=60.0), "grit", "+"),
    # Long tail on purpose: reverb is measured from decay, so a capture that
    # stops before the tail finishes cannot show it (validate.py warns about
    # exactly this, and docs/VALIDATION.md tells you to leave the silence in).
    ("drenched in reverb",
     S.add_reverb(S.rig(S.part("rhythm", bars=3, tail=5.0), drive=8.0),
                  rt60=2.4, mix=0.6), "space", "+"),
]
scores = {}
for label, aud, axis, sign in cases:
    path = f"/tmp/vtest/{label.replace(' ', '_')}.wav"
    sf.write(path, aud, SR)
    r = validate.compare_capture(ref_m, path, target_axes=ref_axes, tempo_bpm=120.0)
    scores[label] = r["overall_score"]
    if axis is None:
        check("identical capture scores high", r["overall_score"] >= 92,
              f"score {r['overall_score']}, {r['axes_matched']}/{r['axes_total']} axes matched")
    else:
        row = next(x for x in r["rows"] if x["axis"] == axis)
        want_dir = (row["diff"] > 0) if sign == "+" else (row["diff"] < 0)
        check(f"{label}: flags {axis} {sign}", (not row["within"]) and want_dir,
              f"{axis} diff {row['diff']:+.1f} (tol {row['tolerance']}), "
              f"score {r['overall_score']}")

check("a wrong capture always scores below an identical one",
      all(v < scores["identical capture"] - 5
          for k, v in scores.items() if k != "identical capture"),
      ", ".join(f"{k} {v}" for k, v in scores.items()))

r = validate.compare_capture(
    ref_m, "/tmp/vtest/too_dark.wav", target_axes=ref_axes, tempo_bpm=120.0)
check("a failing capture comes with instructions", bool(r["priority_fixes"]),
      str(r["priority_fixes"][0])[:80])

print("\n== a truncated capture must be flagged, not silently scored ==")
short = MATERIAL["crunch chords"][:int(6.5 * SR)]
sf.write("/tmp/vtest/truncated.wav", short, SR)
rt = validate.compare_capture(ref_m, "/tmp/vtest/truncated.wav",
                              target_axes=ref_axes, tempo_bpm=120.0)
check("truncated capture is warned about", bool(rt.get("warnings")),
      (rt.get("warnings") or ["none"])[0][:90])

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED: " + f)
sys.exit(1 if FAIL else 0)
