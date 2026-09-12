"""
Catalog consistency tests.

Every model the mapper can choose has to exist on the pedal under the name
QuickTone shows and at its menu position - otherwise the settings sheet names
something you cannot find, and a preset file selects the wrong model.

Run: python3 tests/test_catalog.py
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAT = json.loads((ROOT / "app" / "catalog" / "mg300mk2.json").read_text())
MAPPER = (ROOT / "app" / "mapper.py").read_text()

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name} | {detail}")
    print(("  pass  " if cond else "  FAIL  ") + name + (" | " + detail if detail else ""))


print("menus match QuickTone")
SIZES = {"amps": 30, "cabs": 36, "efx": 14, "comps": 2, "mods": 14,
         "delays": 7, "reverbs": 5, "eqs": 2}
for kind, n in SIZES.items():
    idx = sorted(e["menu_index"] for e in CAT[kind] if "menu_index" in e)
    check(f"{kind}: menu positions 1-{n}", idx == list(range(1, n + 1)), f"got {len(idx)}")
    for field in ("id", "name"):
        vals = [e[field] for e in CAT[kind]]
        check(f"{kind}: {field}s unique", len(vals) == len(set(vals)))

print("mapper references")
used = []
for kind, args in re.findall(r'entry\("(\w+)", ([^)]*)\)', MAPPER):
    used += [(kind, i) for i in re.findall(r'"(\w+)"', args)]
used += [("efx", i) for lst in re.findall(r"family = \[([^\]]*)\]", MAPPER)
         for i in re.findall(r'"(\w+)"', lst)]
check("found the mapper's model references", len(used) >= 20, f"{len(used)} found")
for kind, mid in used:
    e = next((x for x in CAT[kind] if x["id"] == mid), None)
    check(f"{kind}/{mid} is on the pedal", e is not None and "menu_index" in e)

cabs = {e["name"] for e in CAT["cabs"]}
for a in CAT["amps"]:
    if "gain_window" in a:
        check(f"{a['name']} default cab exists", a["default_cab"] in cabs, a["default_cab"])
for name in set(re.findall(r'"([A-Z0-9]+ \d{3})"', MAPPER)):
    check(f"cab '{name}' used by the mapper exists", name in cabs)

print("unprofiled entries are never auto-picked")
for kind in ("amps", "efx", "comps"):
    for e in CAT[kind]:
        if "note" in e and "gain_window" not in e and "gain_add" not in e and "params" not in e:
            check(f"{kind}/{e['id']} not referenced by the mapper", (kind, e["id"]) not in used)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
