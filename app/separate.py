"""
Stem separation. Demucs when available, graceful fallback when not.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import scipy.signal as sps


def demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
        return True
    except Exception:
        return shutil.which("demucs") is not None


def separate_guitar(audio_path: str, out_dir: str, model: str = "htdemucs_6s",
                    log=print):
    """
    Returns (guitar_path, info). htdemucs_6s produces a dedicated guitar stem.
    If Demucs is missing or fails, falls back to a band-limited mid-extraction
    of the full mix and flags lower confidence.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if demucs_available():
        try:
            log(f"Separating stems with Demucs ({model}) - this is the slow part.")
            cmd = [sys.executable, "-m", "demucs", "-n", model,
                   "-o", str(out_dir), audio_path]
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                           timeout=60 * 60)
            stem_dir = out_dir / model / Path(audio_path).stem
            guitar = stem_dir / "guitar.wav"
            other = stem_dir / "other.wav"
            if guitar.exists():
                return str(guitar), {"method": f"demucs:{model}",
                                     "stem": "guitar", "confidence": 0.85,
                                     "stem_dir": str(stem_dir)}
            if other.exists():
                return str(other), {"method": f"demucs:{model}",
                                    "stem": "other", "confidence": 0.55,
                                    "stem_dir": str(stem_dir)}
        except subprocess.CalledProcessError as e:
            log(f"Demucs failed: {(e.stderr or '')[-400:]}")
        except Exception as e:
            log(f"Demucs unavailable at runtime: {e}")

    log("Falling back to band-limited mix analysis (install demucs for real "
        "separation - tone estimates will be less reliable).")
    y, sr = librosa.load(audio_path, sr=44100, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    # Side channel emphasis: guitars are usually panned wider than lead vocal,
    # bass and kick, which sit dead centre.
    mid = (y[0] + y[1]) / 2
    side = (y[0] - y[1]) / 2
    est = 0.55 * side + 0.45 * mid
    sos = sps.butter(4, [110 / (sr / 2), 7000 / (sr / 2)], btype="band", output="sos")
    est = sps.sosfilt(sos, est)
    out = out_dir / "guitar_fallback.wav"
    sf.write(str(out), np.stack([est, est], axis=1), sr)
    return str(out), {"method": "fallback:band+side", "stem": "approximate",
                      "confidence": 0.35, "stem_dir": str(out_dir)}
