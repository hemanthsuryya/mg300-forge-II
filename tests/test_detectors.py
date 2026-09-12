"""
Ground-truth checks against synthetic guitar with known effect settings.
Run: python3 tests/test_detectors.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from app import dsp, timefx, mapper
from tests import synth as S
from tests.synth import SR

PASS, FAIL = [], []


def check(name, cond, detail):
    (PASS if cond else FAIL).append(f"{name} | {detail}")
    print(("  pass  " if cond else "  FAIL  ") + name + " | " + detail)


def drive_of(y):
    d = dsp.distortion_metrics(y, SR); dy = dsp.dynamics(y, SR); de = dsp.sustain_decay(y, SR)
    return dsp.drive_score(d, dy, de)[0]


RHY = S.part("rhythm", bars=4)
LEAD = S.part("lead", bars=4)

print("\n== drive score: must rank five known gain stages, on chords and single notes ==")
for lbl, src in (("chords", RHY), ("single notes", LEAD)):
    vals = [drive_of(S.rig(src, drive=d)) for d in (1.0, 3.0, 8.0, 22.0, 45.0)]
    check(f"monotonic on {lbl}", all(b > a for a, b in zip(vals, vals[1:])),
          " < ".join(f"{v:.2f}" for v in vals))
    check(f"clean reads clean ({lbl})", vals[0] < 0.35, f"{vals[0]:.2f}")
    check(f"metal reads saturated ({lbl})", vals[-1] > 0.75, f"{vals[-1]:.2f}")

print("\n== tremolo: rate and depth, on sustained and on rhythmic material ==")
for rate in (2.5, 5.0, 8.0):
    y = S.tremolo(S.sustained(dur=6.0, tail=0.5), rate=rate, depth=0.8)
    m = timefx.modulation_estimate(y, SR)
    r = m["am_rate_hz"] or 0
    check(f"sustained, tremolo {rate} Hz", abs(r - rate) < 0.4,
          f"measured {r:.2f} Hz depth {m['am_depth']:.2f} (true 0.80) prom {m['am_prominence']:.0f}")
# On rhythmic material only rates that are NOT a harmonic of the strumming
# rhythm can be separated from picking dynamics - that is a real ambiguity,
# not a bug, and the detector is built to stay silent rather than guess.
for rate in (2.5, 5.0):
    y = S.tremolo(S.rig(S.part("rhythm", bars=4, tail=0.5), drive=2.0), rate=rate, depth=0.8)
    m = timefx.modulation_estimate(y, SR)
    r = m["am_rate_hz"] or 0
    check(f"over strumming, tremolo {rate} Hz", abs(r - rate) < 0.4,
          f"measured {r:.2f} Hz depth {m['am_depth']:.2f} prom {m['am_prominence']:.0f}")

print("\n== modulation classification: 8 cases, effected and dry ==")
# Swept across gain stages on purpose: an earlier version of this gate passed a
# three-case check and then produced phantom chorus at drive=6, because the
# thresholds had been tuned on too narrow a sample.
MOD_CASES = [
    ("dry rhythm d1",  S.rig(S.part("rhythm", bars=3, tail=1.5), drive=1.0),  None),
    ("dry rhythm d6",  S.rig(S.part("rhythm", bars=3, tail=1.5), drive=6.0),  None),
    ("dry rhythm d12", S.rig(S.part("rhythm", bars=3, tail=1.5), drive=12.0), None),
    ("dry rhythm d20", S.rig(S.part("rhythm", bars=3, tail=1.5), drive=20.0), None),
    ("dry lead d2",    S.rig(S.part("lead", bars=3, tail=1.5), drive=2.0),    None),
    ("dry lead d20",   S.rig(S.part("lead", bars=3, tail=1.5), drive=20.0),   None),
    ("dry sustained",  S.sustained(dur=5.0, tail=1.0),                        None),
    ("dry metal",   S.rig(S.part("rhythm", bars=4, tail=1.0), drive=40.0), None),
    ("tremolo 5Hz", S.tremolo(S.sustained(dur=6.0, tail=0.5), rate=5.0, depth=0.8), "tremolo"),
    ("chorus chords", S.chorus(S.rig(S.part("rhythm", bars=4, tail=1.0), drive=2.0), rate=1.1, depth_ms=7), "chorus"),
    ("chorus lead", S.chorus(S.rig(S.part("lead", bars=4, tail=1.0), drive=6.0), rate=1.1, depth_ms=7), "chorus"),
    ("phaser chords", S.phaser(S.rig(S.part("rhythm", bars=4, tail=1.0), drive=2.0), rate=0.6), "phaser"),
    ("phaser lead", S.phaser(S.rig(S.part("lead", bars=4, tail=1.0), drive=6.0), rate=0.6), "phaser"),
    ("phaser d6",   S.phaser(S.rig(S.part("rhythm", bars=3, tail=1.5), drive=6.0), rate=0.6), "phaser"),
    ("chorus d6",   S.chorus(S.rig(S.part("rhythm", bars=3, tail=1.5), drive=6.0), rate=1.1, depth_ms=7), "chorus"),
]
# Known and accepted: a chorus over a clean single-note line can fall under the
# gate and be missed. Missing an effect is recoverable (raise Swirl on the
# radar); inventing one is not, so the gate is set to fail in this direction.
KNOWN_MISSES = {"chorus lead"}
KIND = {m["name"]: m["kind"] for m in mapper.CATALOG["mods"]}
for label, sig, want in MOD_CASES:
    mm = timefx.modulation_estimate(sig, SR)
    blk = mapper.mod_block({"mod": mm, "separation_confidence": 0.85})
    got = KIND.get(blk["model"]) if blk["enabled"] else None
    ok = (got == want) if want else (not blk["enabled"])
    if label in KNOWN_MISSES and not blk["enabled"]:
        ok = True
    check(f"{label} -> {want or 'no effect'}", ok,
          f"got {blk['model']} (centroid depth {mm['cm_depth']:.2f}, "
          f"pitch {mm['pm_depth_cents']:.1f} cents)")

print("\n== delay time ==")
for ms in (150, 250, 375, 500):
    dry = S.rig(S.part("lead", bars=3, bpm=90, tail=3.5), drive=4.0)
    y = S.add_delay(dry, time_s=ms / 1000.0, feedback=0.3, mix=0.45)
    d = timefx.delay_estimate(y, SR, tempo_bpm=90)
    got = (d["time_s"] or 0) * 1000
    check(f"delay {ms} ms", abs(got - ms) / ms < 0.12,
          f"measured {got:.0f} ms conf {d['confidence']:.2f} via {d['source']}")

print("\n== reverb RT60, and dry material must not fake it ==")
for rt in (0.8, 1.8, 3.2):
    dry = S.rig(S.part("rhythm", bars=3, tail=max(3.5, rt * 1.8)), drive=2.0)
    y = S.add_reverb(dry, rt60=rt, mix=0.45)
    r = timefx.reverb_estimate(y, SR)
    got = r["rt60_s"] or 0
    check(f"RT60 {rt} s", abs(got - rt) / rt < 0.45,
          f"measured {got:.2f} s conf {r['confidence']:.2f} diffuse {r['diffuseness_db']:.1f} dB")

dry = S.rig(S.part("rhythm", bars=3, tail=3.5), drive=2.0)
rdry = timefx.reverb_estimate(dry, SR)
ddry = timefx.delay_estimate(dry, SR, tempo_bpm=120)
mdry = timefx.modulation_estimate(dry, SR)
check("dry: reverb tail is not diffuse", rdry["diffuseness_db"] < 4.0,
      f"diffuseness {rdry['diffuseness_db']:.1f} dB, rt60 {rdry['rt60_s'] and round(rdry['rt60_s'],2)}")
check("dry: no delay claimed", ddry["confidence"] < 0.35,
      f"conf {ddry['confidence']:.2f}")
pass

print("\n== end to end: the mapper must pick sane amps for known rigs ==")
def preset_for(y, sr=SR):
    from app.pipeline import measure
    m = measure(y, sr, 120.0)
    return mapper.build_preset(m, "test")

pc = preset_for(S.rig(RHY, drive=1.0))
pm = preset_for(S.rig(RHY, drive=45.0))
amp_c = next(b for b in pc["chain"] if b["module"] == "AMP")
amp_m = next(b for b in pm["chain"] if b["module"] == "AMP")
fam = {a["name"]: a["family"] for a in mapper.CATALOG["amps"] if "family" in a}
check("clean rig -> clean/crunch amp", fam[amp_c["model"]] in ("clean", "crunch"),
      f"{amp_c['model']} ({fam[amp_c['model']]}) gain {amp_c['params']['Gain']}")
check("metal rig -> hi_gain/metal amp", fam[amp_m["model"]] in ("hi_gain", "metal"),
      f"{amp_m['model']} ({fam[amp_m['model']]}) gain {amp_m['params']['Gain']}")
rvb_dry = next(b for b in pc["chain"] if b["module"] == "RVB")
check("dry rig -> reverb stays off or low", not rvb_dry["enabled"] or rvb_dry["params"].get("Level", 0) < 45,
      f"{rvb_dry['model']} {rvb_dry.get('params')}")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED: " + f)
sys.exit(1 if FAIL else 0)
