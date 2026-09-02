"""
Authentication: Google sign-in, plus email/password accounts.

Google is the headline option, but password accounts are deliberately kept:
until an operator has registered an OAuth client with Google there would
otherwise be no way to get in at all, which would make a fresh deployment
unusable rather than merely un-Googled.

The session is a signed cookie holding nothing but the user id.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import secrets

import bcrypt
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select

from .db import session_scope
from .models import User

router = APIRouter()

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
# Set this when the app sits behind a proxy or a real domain, so the callback
# URL Google is sent matches the one registered in the Google console.
BASE_URL = os.environ.get("BASE_URL", "").strip().rstrip("/")
ALLOW_REGISTRATION = os.environ.get("ALLOW_REGISTRATION", "1") not in ("0", "false", "no")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")
MIN_PASSWORD = 8

_oauth = None


def google_enabled() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)


def _google():
    """Lazily build the Authlib client so the app imports without Authlib set up."""
    global _oauth
    if _oauth is None:
        from authlib.integrations.starlette_client import OAuth
        o = OAuth()
        o.register(
            name="google",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )
        _oauth = o
    return _oauth.google


# --------------------------------------------------------------------------
# passwords
# --------------------------------------------------------------------------

def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(pw: str, hashed: str | None) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(pw.encode("utf-8"), hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------
# session / current user
# --------------------------------------------------------------------------

def _login(request: Request, user: User) -> None:
    request.session.clear()
    request.session["uid"] = user.id


def current_user(request: Request) -> User | None:
    uid = request.session.get("uid")
    if not uid:
        return None
    with session_scope() as s:
        return s.get(User, uid)


def require_user(request: Request) -> User:
    """API dependency: 401 rather than a redirect, so fetch() can react."""
    u = current_user(request)
    if u is None:
        raise HTTPException(401, "Sign in to continue.")
    return u


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------

def _js(value) -> str:
    """
    Encode a value as a JavaScript literal safe to paste inside a <script>.

    json.dumps on its own is not enough: it escapes quotes but leaves "<" and
    ">" alone, so a message containing "</script>" would close the block early
    and whatever followed would run. Some of these messages carry provider
    exception text, so that is reachable, not theoretical.
    """
    return (json.dumps(value)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))


def _login_page(request: Request, error: str = "", notice: str = "",
                mode: str = "signin") -> HTMLResponse:
    from .main import WEB  # local import avoids a circular import at module load
    html = (WEB / "login.html").read_text(encoding="utf-8")
    # Messages are encoded into the page's script: some of them carry provider
    # exception text, which must not be able to break out of the string
    # literal - or out of the <script> block - it lands in.
    repl = {
        "{{ERROR_JSON}}": _js(error),
        "{{NOTICE_JSON}}": _js(notice),
        "{{GOOGLE}}": "1" if google_enabled() else "",
        "{{ALLOW_REGISTRATION}}": "1" if ALLOW_REGISTRATION else "",
        "{{MODE}}": "signup" if mode == "signup" else "signin",
    }
    for k, v in repl.items():
        html = html.replace(k, v)
    return HTMLResponse(html)


@router.get("/login")
async def login_page(request: Request, error: str = "", notice: str = "",
                     mode: str = "signin"):
    if current_user(request) is not None:
        return RedirectResponse("/", status_code=303)
    return _login_page(request, error=error, notice=notice, mode=mode)


@router.post("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login?notice=Signed+out.", status_code=303)


# --------------------------------------------------------------------------
# email + password
# --------------------------------------------------------------------------

@router.post("/auth/register")
async def register(request: Request, email: str = Form(""), password: str = Form(""),
                   name: str = Form("")):
    if not ALLOW_REGISTRATION:
        return _login_page(request, error="Registration is closed on this server.")
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        return _login_page(request, error="That does not look like an email address.",
                           mode="signup")
    if len(password) < MIN_PASSWORD:
        return _login_page(request,
                           error=f"Use at least {MIN_PASSWORD} characters for the password.",
                           mode="signup")
    with session_scope() as s:
        existing = s.scalar(select(User).where(User.email == email))
        if existing is not None:
            # Don't reveal much, but do help the common real case.
            if existing.google_sub and not existing.password_hash:
                return _login_page(request, mode="signin",
                                   error="That email already signs in with Google. "
                                         "Use the Google button.")
            return _login_page(request, mode="signin",
                               error="That email is already registered. Sign in instead.")
        user = User(email=email, name=name.strip() or None,
                    password_hash=hash_password(password),
                    last_login_at=dt.datetime.now(dt.timezone.utc))
        s.add(user)
        s.flush()
        _login(request, user)
    return RedirectResponse("/", status_code=303)


@router.post("/auth/login")
async def login_local(request: Request, email: str = Form(""), password: str = Form("")):
    email = email.strip().lower()
    with session_scope() as s:
        user = s.scalar(select(User).where(User.email == email))
        ok = user is not None and verify_password(password, user.password_hash)
        if not ok:
            # Constant-ish work whether or not the user exists.
            if user is None:
                bcrypt.checkpw(b"x", bcrypt.hashpw(b"x", bcrypt.gensalt()))
            if user is not None and user.google_sub and not user.password_hash:
                return _login_page(request,
                                   error="That account signs in with Google. "
                                         "Use the Google button.")
            return _login_page(request, error="Wrong email or password.")
        user.last_login_at = dt.datetime.now(dt.timezone.utc)
        _login(request, user)
    return RedirectResponse("/", status_code=303)


# --------------------------------------------------------------------------
# Google
# --------------------------------------------------------------------------

@router.get("/auth/google")
async def google_start(request: Request):
    if not google_enabled():
        return _login_page(request,
                           error="Google sign-in is not configured on this server.")
    redirect_uri = f"{BASE_URL}/auth/google/callback" if BASE_URL else \
        str(request.url_for("google_callback"))
    # nonce is required for OpenID Connect id_token validation
    nonce = secrets.token_urlsafe(16)
    request.session["g_nonce"] = nonce
    return await _google().authorize_redirect(request, redirect_uri, nonce=nonce)


@router.get("/auth/google/callback", name="google_callback")
async def google_callback(request: Request):
    if not google_enabled():
        return RedirectResponse("/login?error=Google+sign-in+is+not+configured.",
                                status_code=303)
    try:
        token = await _google().authorize_access_token(request)
    except Exception as e:
        return _login_page(request, error=f"Google sign-in failed: {e}")

    nonce = request.session.pop("g_nonce", None)
    info = token.get("userinfo")
    if not info:
        try:
            info = await _google().parse_id_token(token, nonce=nonce)
        except Exception as e:
            return _login_page(request, error=f"Could not read the Google profile: {e}")

    sub = str(info.get("sub") or "")
    email = (info.get("email") or "").strip().lower()
    if not sub or not email:
        return _login_page(request, error="Google did not return an email address.")
    if info.get("email_verified") is False:
        return _login_page(request,
                           error="That Google account's email is not verified.")

    name = info.get("name") or None
    picture = info.get("picture") or None

    with session_scope() as s:
        user = s.scalar(select(User).where(User.google_sub == sub))
        if user is None:
            # Same email arriving via Google: link it rather than making a
            # second account the person cannot tell apart from the first.
            user = s.scalar(select(User).where(User.email == email))
            if user is None:
                user = User(email=email, name=name, picture=picture, google_sub=sub)
                s.add(user)
            else:
                user.google_sub = sub
        user.name = user.name or name
        user.picture = picture or user.picture
        user.last_login_at = dt.datetime.now(dt.timezone.utc)
        s.flush()
        _login(request, user)
    return RedirectResponse("/", status_code=303)


# --------------------------------------------------------------------------
# who am I
# --------------------------------------------------------------------------

@router.get("/api/me")
async def me(user: User = Depends(require_user)):
    return user.public()


@router.get("/api/auth/config")
async def auth_config():
    return {"google": google_enabled(), "registration": ALLOW_REGISTRATION}
