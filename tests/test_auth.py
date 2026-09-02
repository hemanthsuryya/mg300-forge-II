"""
Auth and job-ownership checks.

The interesting property is not "login works" but that a signed-in user cannot
reach somebody else's job. Job ids are 12 hex characters and appear in URLs, so
every job route has to be scoped to the owner.

Run: python3 tests/test_auth.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Point the app at a throwaway database and jobs tree before it is imported.
_TMP = tempfile.mkdtemp(prefix="tonechaser-auth-test-")
os.environ["DATA_DIR"] = _TMP
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"
os.environ["JOBS_DIR"] = f"{_TMP}/jobs"
os.environ["SESSION_SECRET"] = "test-secret-not-for-real-use"
os.environ.pop("GOOGLE_CLIENT_ID", None)
os.environ.pop("GOOGLE_CLIENT_SECRET", None)

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402
from app.db import init_db, session_scope  # noqa: E402
from app.models import Job, User  # noqa: E402

init_db()  # TestClient is used without its lifespan, so create tables here

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  pass  " if cond else "  FAIL  ") + name + (" | " + detail if detail else ""))


def client():
    return TestClient(app, follow_redirects=False)


print("\n== the app must be closed to anonymous visitors ==")
c = client()
check("index redirects to the login page", c.get("/").status_code == 303,
      c.get("/").headers.get("location", ""))
check("login page renders", c.get("/login").status_code == 200)
r = c.get("/api/jobs")
check("job list is 401 when signed out", r.status_code == 401)
r = c.post("/api/analyze", files={"file": ("x.wav", b"RIFF", "audio/wav")})
check("analyze is 401 when signed out", r.status_code == 401, str(r.status_code))

print("\n== registration and sign-in ==")
c = client()
r = c.post("/auth/register", data={"email": "alice@example.com",
                                   "password": "correct horse battery",
                                   "name": "Alice"})
check("register redirects into the app", r.status_code == 303 and
      r.headers.get("location") == "/", r.headers.get("location", ""))
r = c.get("/api/me")
check("session is live after registering", r.status_code == 200 and
      r.json()["email"] == "alice@example.com", r.text[:80])
check("registered account reports a password", r.json()["has_password"] is True)
check("registered account is not google-linked", r.json()["google_linked"] is False)

r = c.get("/")
check("index serves the app once signed in", r.status_code == 200 and
      "Tone Chaser" in r.text)

# short password
c2 = client()
r = c2.post("/auth/register", data={"email": "b@example.com", "password": "short"})
check("short password is rejected", r.status_code == 200 and "at least 8" in r.text)
check("no session from a rejected registration",
      c2.get("/api/me").status_code == 401)

# duplicate email
c3 = client()
r = c3.post("/auth/register", data={"email": "alice@example.com", "password": "another one"})
check("duplicate email is refused", r.status_code == 200 and "already registered" in r.text)

# bad email
c4 = client()
r = c4.post("/auth/register", data={"email": "not-an-email", "password": "long enough pw"})
check("malformed email is refused", r.status_code == 200 and "email address" in r.text)

print("\n== sign out and back in ==")
r = c.post("/auth/logout")
check("logout redirects to login", r.status_code == 303)
check("session is gone after logout", c.get("/api/me").status_code == 401)

r = c.post("/auth/login", data={"email": "alice@example.com", "password": "wrong"})
check("wrong password is refused", r.status_code == 200 and "Wrong email or password" in r.text)
check("no session from a failed sign-in", c.get("/api/me").status_code == 401)

r = c.post("/auth/login", data={"email": "ALICE@example.com",
                                "password": "correct horse battery"})
check("sign-in works and email is case-insensitive",
      r.status_code == 303 and c.get("/api/me").status_code == 200)

print("\n== google button appears only when configured ==")
cfg = c.get("/api/auth/config").json()
check("google reported off when unconfigured", cfg["google"] is False, str(cfg))
check("login page hides the google button", '"{{GOOGLE}}"' not in c.get("/login").text)
import app.auth as auth_mod  # noqa: E402
auth_mod.GOOGLE_CLIENT_ID = "x"; auth_mod.GOOGLE_CLIENT_SECRET = "y"
check("google reported on once configured",
      client().get("/api/auth/config").json()["google"] is True)
auth_mod.GOOGLE_CLIENT_ID = ""; auth_mod.GOOGLE_CLIENT_SECRET = ""
r = client().get("/auth/google")
check("google route explains itself when unconfigured",
      r.status_code == 200 and "not configured" in r.text)

print("\n== a server-rendered message cannot break out of the script block ==")
# The messages land inside a JS string literal. json.dumps escapes the quotes
# but not "</script>", which would close the block and run whatever followed.
# Some of these messages carry provider exception text, so this is reachable.
import app.auth as _a  # noqa: E402
PAYLOAD = '</script><script>window.__pwned=1</script>'
page = _a._login_page(None, error=PAYLOAD).body.decode()
check("no raw </script> from a message", "</script><script>" not in page,
      page[page.find("errText"):][:90] if "errText" in page else "")
check("angle brackets are unicode-escaped", "\\u003c/script" in page)
check("the message still round-trips to the browser",
      "window.__pwned" in page, "text is present but inert")
import json as _json, re as _re  # noqa: E402
for probe in ['"', "'", "\\", "</SCRIPT >", "\u2028", "&", "a\nb", "</script >"]:
    msg = f"x{probe}y"
    pg = _a._login_page(None, error=msg).body.decode()
    lit = _re.search(r'const errText\s*=\s*\((".*?")\)\s*\|\|', pg)
    # The literal must parse back to exactly what went in - that is the real
    # property: one intact JS string, nothing leaking out of it.
    ok = bool(lit) and _json.loads(lit.group(1)) == msg
    check(f"message with {probe!r} round-trips as one literal", ok,
          (lit.group(1)[:52] if lit else "no literal found"))
    check(f"message with {probe!r} cannot close the block",
          "</script" not in pg.split("</script>")[0].lower()
          or "\\u003c/script" in pg)

print("\n== a job belongs to one account and nobody else ==")
bob = client()
bob.post("/auth/register", data={"email": "bob@example.com", "password": "bobs long password"})
with session_scope() as s:
    alice_id = s.query(User).filter(User.email == "alice@example.com").one().id
    bob_id = s.query(User).filter(User.email == "bob@example.com").one().id
    s.add(Job(id="a" * 12, user_id=alice_id, source_name="alice.wav", state="done"))
    s.add(Job(id="b" * 12, user_id=bob_id, source_name="bob.wav", state="done"))

check("owner can read their own job", c.get("/api/job/" + "a" * 12).status_code == 200)
for path, label in [
    ("/api/job/{j}", "status"),
    ("/api/job/{j}/download/report", "download"),
    ("/api/job/{j}/clip/tone_1.wav", "clip"),
]:
    r = bob.get(path.format(j="a" * 12))
    check(f"other user gets 404 on {label}", r.status_code == 404, str(r.status_code))
r = bob.post(f"/api/job/{'a' * 12}/tone/1/adjust", json={"axes": {}})
check("other user gets 404 on adjust", r.status_code == 404, str(r.status_code))
r = bob.post(f"/api/job/{'a' * 12}/tone/1/validate",
             files={"file": ("c.wav", b"RIFF", "audio/wav")})
check("other user gets 404 on validate", r.status_code == 404, str(r.status_code))
r = bob.delete("/api/job/" + "a" * 12)
check("other user cannot delete someone's job", r.status_code == 404, str(r.status_code))

print("\n== the job list shows only your own ==")
al = c.get("/api/jobs").json()["jobs"]
bl = bob.get("/api/jobs").json()["jobs"]
check("alice sees exactly her job",
      [j["job_id"] for j in al] == ["a" * 12], str([j["job_id"] for j in al]))
check("bob sees exactly his job",
      [j["job_id"] for j in bl] == ["b" * 12], str([j["job_id"] for j in bl]))
check("job list carries the source name", al[0]["source_name"] == "alice.wav")

print("\n== unknown ids look the same as forbidden ones ==")
check("unknown job id is 404", c.get("/api/job/" + "f" * 12).status_code == 404)

print("\n== a user can delete their own job ==")
check("owner deletes their job", c.delete("/api/job/" + "a" * 12).status_code == 200)
check("deleted job is gone", c.get("/api/job/" + "a" * 12).status_code == 404)
check("bob's job survived alice's delete", bob.get("/api/job/" + "b" * 12).status_code == 200)

shutil.rmtree(_TMP, ignore_errors=True)
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
