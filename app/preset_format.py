"""
Placeholder for writing real QuickTone preset files.

The MG-300 MKII's preset format is proprietary and undocumented; NUX does not
publish a spec and QuickTone does not export anything human-readable. Guessing a
byte layout would produce files that either fail to import or - worse - import
as something that sounds nothing like the analysis.

So this module deliberately does nothing yet. To make it real, export one preset
from QuickTone and drop it in `samples/reference_presets/`, ideally a few that
differ by exactly one parameter (for example the same preset saved with Gain at
0, 50 and 100). Diffing those pins down where each field lives, and this module
then becomes a straightforward struct writer:

    build_preset(...) -> dict            # what the analyser already produces
    encode(preset_dict) -> bytes         # to be written here
    write(preset_dict, path)             # to be written here

Until then the deliverable is the settings sheet, which contains every value
needed to dial the preset in by hand or in QuickTone.
"""
from __future__ import annotations

from pathlib import Path

SUPPORTED = False
REFERENCE_DIR = Path(__file__).resolve().parent.parent / "samples" / "reference_presets"


def encode(preset: dict) -> bytes:
    raise NotImplementedError(
        "The QuickTone preset byte layout has not been reverse-engineered yet. "
        f"Put a few exported reference presets in {REFERENCE_DIR} first."
    )


def reference_presets_found() -> list[Path]:
    if not REFERENCE_DIR.is_dir():
        return []
    return sorted(p for p in REFERENCE_DIR.iterdir() if p.is_file())
