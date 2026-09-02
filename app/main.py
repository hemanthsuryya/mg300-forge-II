"""
Tone Chaser - local web app.

Drops a song in, comes back with MG-300 MKII preset recipes.

Run:  python3 -m uvicorn app.main:app --port 8000
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from pathlib import Path

import datetime as dt
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from starlette.middleware.sessions import SessionMiddleware

from .pipeline import run_job
from .report import render_html_report, render_text_sheet
from . import perceptual, mapper, validate as validate_mod
from .db import init_db, session_scope
from .models import Job, User
from . import auth as auth_mod
from .auth import current_user, require_user

ROOT = Path(__file__).resolve().parent.parent
# Overridable so Docker can mount a volume somewhere else.
JOBS = Path(os.environ.get("JOBS_DIR", ROOT / "jobs"))
WEB = ROOT / "web"
JOBS.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_db()
    _startup_banner()
    yield


def _startup_banner() -> None:
    """
    Say what auth is actually configured.

    "google": true only means the variables are non-empty, not that Google will
    accept them, so print the client id and the callback URL - the two things
    an invalid_client or redirect_uri_mismatch comes down to.
    """
    from .db import DATABASE_URL
    where = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
    print(f"Tone Chaser · database {where}")
    if auth_mod.google_enabled():
        cid = auth_mod.GOOGLE_CLIENT_ID
        base = auth_mod.BASE_URL or "http://127.0.0.1:8000"
        print(f"Tone Chaser · Google sign-in ON, client {cid}")
        print(f"Tone Chaser · this exact URI must be authorised in the Google "
              f"console: {base}/auth/google/callback")
        if not cid.endswith(".apps.googleusercontent.com"):
            print("Tone Chaser · WARNING: that does not look like a Google "
                  "client id (they end in .apps.googleusercontent.com)")
    else:
        print("Tone Chaser · Google sign-in OFF (set GOOGLE_CLIENT_ID and "
              "GOOGLE_CLIENT_SECRET to enable), email/password accounts only")


app = FastAPI(title="Tone Chaser", lifespan=_lifespan)

# A generated secret is fine for a single local process, but it changes on every
# restart (logging everyone out) and differs per worker, so a real deployment
# must set SESSION_SECRET.
_secret = os.environ.get("SESSION_SECRET", "").strip()
if not _secret:
    _secret = secrets.token_urlsafe(32)
    print("SESSION_SECRET is not set - using a random one. Sessions will not "
          "survive a restart. Set SESSION_SECRET in production.")
app.add_middleware(
    SessionMiddleware,
    secret_key=_secret,
    session_cookie="tonechaser_session",
    max_age=60 * 60 * 24 * 14,
    same_site="lax",
    https_only=os.environ.get("COOKIE_SECURE", "0") in ("1", "true", "yes"),
)
app.include_router(auth_mod.router)

_state: dict[str, dict] = {}
_lock = threading.Lock()

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff"}


def _set(job_id, **kw):
    with _lock:
        _state.setdefault(job_id, {}).update(kw)


def _append_log(job_id, msg):
    with _lock:
        _state.setdefault(job_id, {}).setdefault("log", []).append(msg)


def _progress(job_id):
    def cb(pct, msg):
        if pct is not None:
            _set(job_id, progress=pct)
        if msg:
            _append_log(job_id, msg)
            _set(job_id, status_text=msg)
    return cb


def _ytdlp() -> str | None:
    """
    Locate yt-dlp. run.sh execs .venv/bin/python without activating the venv,
    so PATH does not contain .venv/bin - look beside the interpreter too.
    """
    found = shutil.which("yt-dlp")
    if found:
        return found
    local = Path(sys.executable).parent / "yt-dlp"
    return str(local) if local.is_file() else None


def _fetch_url(url: str, dest: Path, log) -> Path:
    exe = _ytdlp()
    if not exe:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")
    log(None, "Downloading audio from the URL")
    out = dest / "source.%(ext)s"
    cmd = [exe, "-x", "--audio-format", "wav", "--audio-quality", "0",
           "-o", str(out), "--no-playlist", url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"Download failed: {(r.stderr or '')[-500:]}")
    files = [p for p in dest.iterdir() if p.suffix.lower() in AUDIO_EXT]
    if not files:
        raise RuntimeError("Download produced no audio file.")
    return max(files, key=lambda p: p.stat().st_size)


def _owned_job(job_id: str, user: User) -> Job:
    """
    Resolve a job the caller is allowed to see.

    Job ids are short and guessable, so every job route goes through here.
    A job belonging to someone else is a 404, not a 403 - there is no reason to
    confirm that an id exists to a user who cannot have it.
    """
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None or job.user_id != user.id:
            raise HTTPException(404, "Unknown job")
        return job


def _finish_job(job_id: str, **fields) -> None:
    """Record the outcome on the job row; never let bookkeeping kill the run."""
    try:
        with session_scope() as s:
            job = s.get(Job, job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)
    except Exception as e:  # pragma: no cover
        print(f"warning: could not update job {job_id}: {e}")


def _work(job_id: str, audio: Path | None, url: str | None, model: str, max_tones: int):
    d = JOBS / job_id
    cb = _progress(job_id)
    try:
        _set(job_id, state="running", progress=1)
        _finish_job(job_id, state="running", progress=1)
        if audio is None:
            audio = _fetch_url(url, d, cb)
            _set(job_id, source_name=audio.name)
        result = run_job(str(audio), str(d), progress=cb,
                         demucs_model=model, max_tones=max_tones)
        (d / "report.html").write_text(render_html_report(result), encoding="utf-8")
        (d / "settings.txt").write_text(render_text_sheet(result), encoding="utf-8")
        (d / "presets.json").write_text(json.dumps(
            {"device": result["device"], "source": result["source"],
             "presets": [t["preset"] for t in result["tones"]]}, indent=2))
        _set(job_id, state="done", progress=100, result=result)
        _finish_job(job_id, state="done", progress=100,
                    tone_count=len(result.get("tones", [])),
                    tempo_bpm=int(result.get("tempo_bpm") or 0),
                    finished_at=dt.datetime.now(dt.timezone.utc))
    except Exception as e:
        _append_log(job_id, f"ERROR: {e}")
        _set(job_id, state="error", error=str(e),
             traceback=traceback.format_exc()[-2000:])
        _finish_job(job_id, state="error", error=str(e)[:2000],
                    finished_at=dt.datetime.now(dt.timezone.utc))


@app.post("/api/analyze")
async def analyze(file: UploadFile | None = File(None), url: str = Form(""),
                  model: str = Form("htdemucs_6s"), max_tones: int = Form(5),
                  user: User = Depends(require_user)):
    job_id = uuid.uuid4().hex[:12]
    d = JOBS / job_id
    d.mkdir(parents=True, exist_ok=True)
    audio = None
    if file is not None and file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in AUDIO_EXT:
            raise HTTPException(400, f"Unsupported audio type '{ext}'")
        audio = d / f"source{ext}"
        with audio.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        name = file.filename
    elif url.strip():
        name = url.strip()
    else:
        raise HTTPException(400, "Provide an audio file or a URL.")
    with session_scope() as s:
        s.add(Job(id=job_id, user_id=user.id, source_name=name[:500],
                  state="queued", progress=0, from_url=audio is None))
    _set(job_id, state="queued", progress=0, log=[], source_name=name,
         status_text="Queued")
    threading.Thread(target=_work, args=(job_id, audio, url.strip(), model,
                                         int(max_tones)), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
async def job(job_id: str, user: User = Depends(require_user)):
    row = _owned_job(job_id, user)
    with _lock:
        st = dict(_state.get(job_id, {}))
    if not st:
        # Live progress lives in memory, so a job from before a restart has
        # none. Rebuild what we can from the row and the result on disk.
        st = row.public()
        st["state"] = row.state
        rp = JOBS / job_id / "result.json"
        if row.state == "done" and rp.is_file():
            st["result"] = json.loads(rp.read_text())
        st["log"] = []
    st["log"] = st.get("log", [])[-40:]
    return JSONResponse(st)


@app.get("/api/jobs")
async def list_jobs(user: User = Depends(require_user), limit: int = 30):
    with session_scope() as s:
        rows = s.scalars(
            select(Job).where(Job.user_id == user.id)
            .order_by(Job.created_at.desc()).limit(max(1, min(limit, 100)))
        ).all()
        return {"jobs": [r.public() for r in rows]}


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str, user: User = Depends(require_user)):
    _owned_job(job_id, user)
    shutil.rmtree(JOBS / job_id, ignore_errors=True)
    with session_scope() as s:
        row = s.get(Job, job_id)
        if row is not None:
            s.delete(row)
    with _lock:
        _state.pop(job_id, None)
    return {"deleted": job_id}


@app.get("/api/job/{job_id}/clip/{name}")
async def clip(job_id: str, name: str, user: User = Depends(require_user)):
    _owned_job(job_id, user)
    p = (JOBS / job_id / "clips" / name).resolve()
    if not p.is_file() or JOBS.resolve() not in p.parents:
        raise HTTPException(404, "No such clip")
    return FileResponse(p, media_type="audio/wav")


@app.get("/api/job/{job_id}/download/{what}")
async def download(job_id: str, what: str, user: User = Depends(require_user)):
    _owned_job(job_id, user)
    names = {"report": ("report.html", "text/html"),
             "sheet": ("settings.txt", "text/plain"),
             "presets": ("presets.json", "application/json"),
             "raw": ("result.json", "application/json")}
    if what not in names:
        raise HTTPException(404, "Unknown download")
    fn, mt = names[what]
    p = JOBS / job_id / fn
    if not p.is_file():
        raise HTTPException(404, "Not ready")
    return FileResponse(p, media_type=mt, filename=f"tonechaser_{job_id}_{fn}")


def _tone_or_404(job_id: str, tone_id: int, user: User):
    _owned_job(job_id, user)
    with _lock:
        st = dict(_state.get(job_id, {}))
    res = st.get("result")
    if not res:
        p = JOBS / job_id / "result.json"
        if not p.is_file():
            raise HTTPException(404, "Unknown or unfinished job")
        res = json.loads(p.read_text())
    for t in res["tones"]:
        if t["tone_id"] == tone_id:
            return res, t
    raise HTTPException(404, "Unknown tone")


@app.get("/api/axes")
async def axes():
    return {"axes": perceptual.AXES, "conflicts": perceptual.CONFLICTS}


@app.post("/api/job/{job_id}/tone/{tone_id}/adjust")
async def adjust(job_id: str, tone_id: int, payload: dict,
                 user: User = Depends(require_user)):
    """
    Re-derive a preset from dragged axis positions.

    The axes perturb the MEASUREMENTS and the ordinary mapper re-runs, so the
    whole chain re-solves rather than a single knob being nudged.
    """
    res, tone = _tone_or_404(job_id, tone_id, user)
    m = tone["measurements_raw"]
    wanted = {k: float(v) for k, v in (payload.get("axes") or {}).items()
              if k in perceptual.AXIS_IDS}
    m2, moved = perceptual.apply_axes(m, wanted)
    preset = mapper.build_preset(m2, name=tone["name"])
    return {"preset": preset, "axes": perceptual.to_axes(m2), "moved": moved,
            "baseline_axes": perceptual.to_axes(m)}


@app.post("/api/job/{job_id}/tone/{tone_id}/validate")
async def validate_capture(job_id: str, tone_id: int,
                           file: UploadFile = File(...),
                           axes_json: str = Form(""),
                           user: User = Depends(require_user)):
    """
    Compare a capture of the pedal against the tone it was supposed to match.

    Upload what came back out of the MG-300 (reamped through it, or played
    through it). Same measurement pipeline runs on both sides, so the answer is
    a per-axis difference and a per-block instruction, not a vibe.
    """
    res, tone = _tone_or_404(job_id, tone_id, user)
    d = JOBS / job_id / "captures"
    d.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "capture.wav").suffix.lower() or ".wav"
    if ext not in AUDIO_EXT:
        raise HTTPException(400, f"Unsupported audio type '{ext}'")
    dest = d / f"tone_{tone_id}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    target_axes = json.loads(axes_json) if axes_json.strip() else None
    try:
        report = validate_mod.compare_capture(
            reference_measurements=tone["measurements_raw"],
            capture_path=str(dest),
            target_axes=target_axes,
            tempo_bpm=res.get("tempo_bpm", 120.0),
        )
    except Exception as e:
        raise HTTPException(400, f"Could not analyse the capture: {e}")
    (JOBS / job_id / f"validation_tone_{tone_id}.json").write_text(
        json.dumps(report, indent=2))
    return report


@app.get("/api/health")
async def health(request: Request):
    from .separate import demucs_available
    u = current_user(request)
    return {"demucs": demucs_available(), "yt_dlp": bool(_ytdlp()),
            "google": auth_mod.google_enabled(),
            "user": u.public() if u else None}


@app.get("/")
async def index(request: Request):
    if current_user(request) is None:
        return RedirectResponse("/login", status_code=303)
    return HTMLResponse((WEB / "index.html").read_text(encoding="utf-8"))


app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
