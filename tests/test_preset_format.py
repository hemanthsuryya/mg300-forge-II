"""
Tests for the QuickTone preset encoder.

The strongest check here is reproduction: encoding the baseline with one control
changed must give the same bytes QuickTone itself wrote when that control was
changed by hand. Those exports live in samples/reference_presets/ and are not
committed, so those checks skip when the files are absent.

Run: python3 tests/test_preset_format.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

from app import preset_format as pf

PASS, FAIL = [], []
REF = Path(__file__).resolve().parent.parent / "samples" / "reference_presets"
TEMPLATE = pf.TEMPLATE.read_bytes()


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name} | {detail}")
    print(("  pass  " if cond else "  FAIL  ") + name + (" | " + detail if detail else ""))


def body(data, scene=0):
    """Scene bytes excluding the name field."""
    b = scene * pf.SCENE_SIZE
    return data[b + pf.BODY_START:b + pf.NAME_OFF] + data[b + pf.NAME_OFF + pf.NAME_LEN:b + pf.SCENE_SIZE]


print("template")
t = pf.decode(TEMPLATE)
check("template name", t["name"] == "pan studio mid g", t["name"])
expect = {"AMP": (15, True), "EFX": (7, True), "COMP": (1, False), "MOD": (5, True),
          "DLY": (1, False), "RVB": (2, True), "IR": (13, True)}
for m, (idx, on) in expect.items():
    got = t["blocks"][m]
    check(f"template {m} matches QuickTone screenshot", (got["menu_index"], got["enabled"]) == (idx, on),
          f"got {got['menu_index']} {'on' if got['enabled'] else 'off'}")

print("reproduce QuickTone's own exports")
single = {"cal_01_gain100": ("AMP", "Gain"), "cal_02_bass100": ("AMP", "Bass"),
          "cal_03_mid100": ("AMP", "Middle"), "cal_04_treble100": ("AMP", "Treble"),
          "cal_05_master100": ("AMP", "Master"), "cal_06_presence100": ("AMP", "Presence"),
          "cal_08_rvb": ("RVB", "Level")}
for fn, (m, p) in single.items():
    f = REF / f"{fn}.mg300MK2patch"
    if not f.is_file():
        print(f"  skip  {fn} (not present)")
        continue
    ours = pf.encode({m: {"params": {p: 100}}}, "x")
    check(f"{fn} byte-identical in S1", body(ours) == body(f.read_bytes()))

selectors = {"cal_22_amp_first": ("AMP", 1, True), "cal_23_amp_last": ("AMP", 30, True),
             "cal_20_mod_ce1": ("MOD", 1, True), "cal_21_mod_sch1": ("MOD", 14, True),
             "cal_09_dly": ("DLY", 1, True), "cal_07_comp": ("COMP", 2, True),
             "cal_11_efx": ("EFX", 1, True)}
for fn, (m, idx, on) in selectors.items():
    hits = list(REF.glob(f"*{fn}.mg300MK2patch"))
    if not hits:
        print(f"  skip  {fn} (not present)")
        continue
    got = pf.decode(hits[0].read_bytes())["blocks"][m]
    check(f"{fn} decodes {m} as menu {idx}", (got["menu_index"], got["enabled"]) == (idx, on),
          f"got {got['menu_index']} {'on' if got['enabled'] else 'off'}")
off = REF / "cal_12_allblocksoff.mg300MK2patch"
if off.is_file():
    blocks = pf.decode(off.read_bytes())["blocks"]
    check("cal_12 decodes every block as bypassed", not any(b["enabled"] for b in blocks.values()))

print("encoding")
settings = {"AMP": {"menu_index": 10, "params": {"Gain": 70, "Master": 40, "Bass": 20,
                                                  "Middle": 60, "Treble": 80, "Presence": 10}},
            "IR": {"menu_index": 10}, "COMP": {"enabled": True, "menu_index": 2},
            "DLY": {"enabled": True, "menu_index": 4}, "MOD": {"enabled": False},
            "RVB": {"menu_index": 3, "params": {"Level": 90}}}
data = pf.encode(settings, "FORGE TEST 1")
check("size unchanged", len(data) == pf.FILE_SIZE)
check("region after the scenes untouched",
      data[pf.N_SCENES * pf.SCENE_SIZE:] == TEMPLATE[pf.N_SCENES * pf.SCENE_SIZE:])
for s in range(pf.N_SCENES):
    d = pf.decode(data, s)
    ok = (d["name"] == "FORGE TEST 1" and d["blocks"]["AMP"]["menu_index"] == 10
          and d["blocks"]["AMP"]["params"]["Treble"] == 80 and not d["blocks"]["MOD"]["enabled"]
          and d["blocks"]["MOD"]["menu_index"] == 5 and d["blocks"]["RVB"]["params"]["Level"] == 90)
    check(f"round trip S{s + 1}", ok)

changed = {i for i in range(pf.BODY_START, pf.SCENE_SIZE) if data[i] != TEMPLATE[i]}
allowed = ({pf.SELECTOR[m] for m in settings} | set(pf.KNOBS["AMP"].values())
           | {pf.KNOBS["RVB"]["Level"]} | set(range(pf.NAME_OFF, pf.NAME_OFF + pf.NAME_LEN)))
check("only the intended bytes change in S1", changed <= allowed,
      f"unexpected: {sorted(hex(i) for i in changed - allowed)}")

check("knob values clamp to 0-100",
      pf.decode(pf.encode({"AMP": {"params": {"Gain": 140, "Bass": -5}}}, "x"))["blocks"]["AMP"]["params"]
      == {**t["blocks"]["AMP"]["params"], "Gain": 100, "Bass": 0})
long_name = pf.decode(pf.encode({}, "A NAME LONGER THAN SIXTEEN"))["name"]
check("name truncated to 16 characters", long_name == "A NAME LONGER TH", long_name)
try:
    pf.encode({"AMP": {"menu_index": 64}}, "x")
    check("menu index 64 rejected", False)
except ValueError:
    check("menu index 64 rejected", True)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
