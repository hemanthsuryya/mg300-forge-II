"""
Write QuickTone preset files (.mg300MK2patch) for the NUX MG-300 MKII.

The layout was pinned by exporting one preset from QuickTone with a single
control changed per file and diffing against the baseline
(samples/reference_presets/cal_*.mg300MK2patch). Each export differed in exactly
the byte for the control that moved, so there is no checksum over the data.

What is known, and therefore written:
  - 3 scenes (S1-S3) of 126 bytes at 0x00, 0x7E, 0xFC; name at +0x6D, 16 bytes
  - block selectors: the menu position (1-based) as shown in QuickTone, with
    0x40 set when the block is bypassed
  - AMP Gain/Presence/Master/Bass/Middle/Treble and RVB Level, 0-100 as displayed

Everything else - other effect knobs, EQ bands, the gate, the product/version
header and the embedded-IR region - is copied from a genuine QuickTone export
(the template). An encoded file is therefore always a file QuickTone has already
accepted, with only the bytes listed above changed.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "catalog" / "mg300mk2_template.mg300MK2patch"
CATALOG = json.loads((HERE / "catalog" / "mg300mk2.json").read_text())

FILE_SIZE = 8654
SCENE_SIZE = 0x7E
N_SCENES = 3
BODY_START = 0x04          # bytes 0-3 of each scene are left as the template has them
NAME_OFF, NAME_LEN = 0x6D, 16
BYPASS = 0x40

# Model selector per block, keyed by the mapper's module names.
SELECTOR = {"COMP": 0x05, "EFX": 0x06, "AMP": 0x07, "MOD": 0x0A,
            "DLY": 0x0B, "RVB": 0x0C, "IR": 0x0D}

KNOBS = {
    "AMP": {"Gain": 0x20, "Presence": 0x21, "Master": 0x22,
            "Bass": 0x23, "Middle": 0x24, "Treble": 0x25},
    "RVB": {"Level": 0x4E},
}

# Where each module's models live in the catalog.
CATALOG_KEY = {"COMP": "comps", "EFX": "efx", "AMP": "amps", "MOD": "mods",
               "DLY": "delays", "RVB": "reverbs", "IR": "cabs"}


def _knob(v) -> int:
    return int(round(min(max(float(v), 0.0), 100.0)))


def _name(name: str) -> bytes:
    raw = name.encode("ascii", errors="replace")[:NAME_LEN]
    return raw.ljust(NAME_LEN, b"\0")


def encode(blocks: dict, name: str, template: bytes | None = None) -> bytes:
    """
    blocks: {module: {"enabled": bool, "menu_index": int | None, "params": {..}}}

    menu_index is the number QuickTone shows next to the model in its dropdown.
    None keeps the template's model and only applies the on/off state. Modules
    and params this layout does not cover are ignored; the template's bytes stay.
    The same settings go into all three scenes, so S1/S2/S3 sound identical.
    """
    buf = bytearray(TEMPLATE.read_bytes() if template is None else template)
    if len(buf) != FILE_SIZE:
        raise ValueError(f"Template is {len(buf)} bytes, expected {FILE_SIZE}.")

    scene0 = bytes(buf[BODY_START:SCENE_SIZE])
    for s in range(1, N_SCENES):
        base = s * SCENE_SIZE
        buf[base + BODY_START:base + SCENE_SIZE] = scene0

    for s in range(N_SCENES):
        base = s * SCENE_SIZE
        for module, blk in blocks.items():
            off = SELECTOR.get(module)
            if off is not None:
                idx = blk.get("menu_index")
                if idx is None:
                    idx = buf[base + off] & ~BYPASS & 0xFF
                elif not 1 <= int(idx) < BYPASS:
                    raise ValueError(f"{module} menu_index {idx} is out of range.")
                buf[base + off] = int(idx) | (0 if blk.get("enabled", True) else BYPASS)
            for param, value in (blk.get("params") or {}).items():
                koff = KNOBS.get(module, {}).get(param)
                if koff is not None:
                    buf[base + koff] = _knob(value)
        buf[base + NAME_OFF:base + NAME_OFF + NAME_LEN] = _name(name)
    return bytes(buf)


def decode(data: bytes, scene: int = 0) -> dict:
    """Read back the fields this module knows about, for one scene."""
    if len(data) != FILE_SIZE:
        raise ValueError(f"File is {len(data)} bytes, expected {FILE_SIZE}.")
    base = scene * SCENE_SIZE
    raw_name = data[base + NAME_OFF:base + NAME_OFF + NAME_LEN]
    return {
        "name": raw_name.split(b"\0")[0].decode("ascii", errors="replace"),
        "blocks": {
            m: {"menu_index": data[base + off] & ~BYPASS & 0xFF,
                "enabled": not data[base + off] & BYPASS,
                "params": {p: data[base + k] for p, k in KNOBS.get(m, {}).items()}}
            for m, off in SELECTOR.items()
        },
    }


def menu_index(module: str, model: str) -> int | None:
    """The QuickTone menu position for a catalog model name, if the catalog has one."""
    for entry in CATALOG.get(CATALOG_KEY.get(module, ""), []):
        if isinstance(entry, dict) and entry.get("name") == model:
            return entry.get("menu_index")
    return None


def from_preset(preset: dict) -> tuple[bytes, list[str]]:
    """
    Encode a preset dict from mapper.build_preset. Returns the file bytes and a
    list of notes about blocks that could not be written as analysed.
    """
    blocks, notes = {}, []
    for blk in preset["chain"]:
        module = blk["module"]
        if module not in SELECTOR:
            continue
        if not blk.get("enabled"):
            blocks[module] = {"enabled": False}
            continue
        idx = menu_index(module, blk["model"])
        if idx is None:
            notes.append(f"{module}: '{blk['model']}' has no QuickTone menu position "
                         f"in the catalog - left as the template has it.")
            continue
        blocks[module] = {"enabled": True, "menu_index": idx,
                          "params": blk.get("params") or {}}
    return encode(blocks, preset["name"]), notes


def write(preset: dict, path: str | Path) -> list[str]:
    data, notes = from_preset(preset)
    Path(path).write_bytes(data)
    return notes
