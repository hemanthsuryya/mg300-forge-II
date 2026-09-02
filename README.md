# Tone Chaser

Drop in a song. It isolates the guitar, finds the sections where the guitar is
playing, groups those sections by tone, records an excerpt of each one, measures
what that tone is actually doing, and writes a NUX MG-300 MKII preset recipe for
each distinct tone.

Runs entirely on your machine. Nothing is uploaded anywhere.

---

## Run it with Docker (recommended)

```bash
cp .env.example .env
# put a secret in it:  openssl rand -base64 32
docker compose up -d --build
```

Then open <http://127.0.0.1:8000> and create an account.

That brings up two containers: the app and Postgres. Three named volumes keep
state across restarts and rebuilds — `db-data` (accounts and job records),
`job-data` (uploads, stems, clips, reports) and `model-cache` (the Demucs
weights, ~250 MB, downloaded on the first analysis rather than baked into the
image).

Torch is installed from the CPU-only wheel index, which keeps the image around
2 GB instead of 6+ GB. Analysis is CPU-bound either way.

`SESSION_SECRET` is the one required setting; compose refuses to start without
it, rather than booting with a random secret that would silently sign everyone
out on the next restart.

```bash
docker compose logs -f app      # watch it work
docker compose down             # stop, keep data
docker compose down -v          # stop and delete accounts, jobs and weights
```

## Or run it directly

```bash
./run.sh
```

First run creates a virtualenv and installs everything (a few minutes; Demucs
pulls in PyTorch, around 2 GB). Then open <http://127.0.0.1:8000>.

With no `DATABASE_URL` set it uses a SQLite file in `data/`, so there is no
database server to install. Set `SESSION_SECRET` to keep sessions across
restarts.

To skip Demucs for a quick trial, install everything else and the app falls back
to analysing the mix directly. It will say so, and it lowers its own confidence
scores, because tone estimates taken through drums and vocals are worth less.

Requires Python 3.10+, `ffmpeg` on PATH (`brew install ffmpeg`). `run.sh` picks
the newest qualifying interpreter it can find, since a bare `python3` is often
older than 3.10.

---

## Accounts and sign-in

The app is behind a login. Two ways in:

- **Google** — one button, no password. Off until you configure it (below).
- **Email and password** — always available, so a fresh deployment is usable
  before you have an OAuth client, and stays usable if Google is unreachable.

Signing in with Google using an email that already has a password account
**links** the two rather than creating a second account you cannot tell apart.

Each analysis belongs to the account that started it. Job ids are short and sit
in URLs, so every job route is scoped to its owner; somebody else's job is a
404, not a 403. Your past analyses are listed on the front page and can be
reopened or deleted.

Once your own account exists, set `ALLOW_REGISTRATION=0` to close signups.

### Turning on Google sign-in

1. Google Cloud Console → **APIs & Services** → **Credentials** →
   **Create credentials** → **OAuth client ID** → **Web application**.
2. Under *Authorised redirect URIs* add exactly:
   - `http://127.0.0.1:8000/auth/google/callback` for local use
   - `https://your-domain/auth/google/callback` in production
3. Put the client id and secret in `.env`:

   ```
   GOOGLE_CLIENT_ID=...apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=...
   ```

4. `docker compose up -d` again.

The redirect URI has to match what Google has on file character for character.
Behind a proxy or a real domain, set `BASE_URL=https://your-domain` so the app
sends that instead of guessing from the request.

### When Google refuses

| What Google says | What it means |
|---|---|
| **401: invalid_client** — "The OAuth client was not found" | The client id reaching Google is not a real one. Either `GOOGLE_CLIENT_ID` is a placeholder, the variable never made it into the process, or the client was deleted. Check with `curl -s localhost:8000/api/auth/config` — `{"google":true}` only means *something* is set, not that it is valid. |
| **401: invalid_client** — "Unauthorized" | The id is real but `GOOGLE_CLIENT_SECRET` is wrong. |
| **400: redirect_uri_mismatch** | The callback the app sent is not on the client's authorised list. The error page shows the URI it tried; paste that exact string into the Google console, or set `BASE_URL`. |
| **403: access_denied** | The client is in "Testing" mode and the account is not on the test-user list. Add it, or publish the client. |
| Button does not appear at all | Both `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` must be non-empty. Restart after editing `.env` — the values are read at startup. |

The credentials are read once when the process starts, so `docker compose up -d`
again (or restart `run.sh`) after changing them.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SESSION_SECRET` | *required in compose* | Signs the session cookie. Changing it logs everyone out. |
| `DATABASE_URL` | SQLite in `data/` | e.g. `postgresql+psycopg://user:pw@host/db` |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | empty | Enables the Google button when both are set. |
| `BASE_URL` | guessed from request | Public origin, for the OAuth callback behind a proxy. |
| `COOKIE_SECURE` | `0` | Set to `1` when serving over HTTPS. |
| `ALLOW_REGISTRATION` | `1` | `0` closes signups. |
| `JOBS_DIR` / `DATA_DIR` | `jobs/`, `data/` | Where audio and the SQLite file live. |
| `PORT` | `8000` | Published port. |

**Serving this on the open internet:** put it behind TLS and set
`COOKIE_SECURE=1`. Uploading audio and running Demucs is expensive and
unauthenticated users cannot do it, but any registered user can, so close
registration or keep it on a private network.

---

## What it does, in order

1. **Separate** — Demucs `htdemucs_6s`, which has a dedicated guitar stem
   (most models only give you a catch-all "other").
2. **Find the guitar** — an adaptive level threshold on the guitar stem,
   with short gaps closed and blips discarded.
3. **Group by tone** — every 2 s window gets a timbre fingerprint (MFCCs plus
   centroid, flatness, bandwidth, zero-crossing rate, crest factor).
   Agglomerative clustering, with the number of tones chosen by silhouette
   score. One tone is a valid answer; it will not invent a second one to look
   busy. Fragments under 6 s or 5 % of the guitar time are dropped as
   transitions.
4. **Record** — a representative excerpt of each tone group is written to
   `jobs/<id>/clips/tone_N.wav` and is playable in the browser.
5. **Measure** — see below.
6. **Map** — measurements become MG-300 MKII blocks, each with a stated reason
   and a confidence.

## What it measures, and how

| Quantity | Method |
|---|---|
| **Drive** (0-100) | Crest factor, plus spectral valley depth on chords / high-frequency sustain on single notes, plus note decay rate. Calibrated against a physical string model rendered through a cab at five known gain settings. |
| **Tone shape** | Band energies in seven bands, expressed as dB deviation from a reference mic'd-cab curve, then centred so only the shape matters. |
| **Compression** | Crest factor, envelope spread, attack sharpness. Skipped when the amp is already saturating. |
| **Delay** | Periodic bumps in decay regions, i.e. after a struck note where nothing new is played. Cross-checked with autocorrelation inside the same region. **Repeats that do not get quieter are rejected** as the song's rhythm rather than an echo. |
| **Reverb** | RT60 from a fit to the late part of the decay, plus a *diffuseness* test: a reverb tail turns noise-like, a long sustaining note stays harmonic. Both are required. |
| **Modulation** | Amplitude modulation for tremolo, spectral-centroid sweep for phaser/wah, comb structure via cepstrum for flanger, pitch wobble for chorus/vibrato, stereo width for detune. Every one of these must be **sinusoidal** — harmonic content in the modulation spectrum means it is strumming, not an LFO. |
| **Tempo / division** | Beat tracking on the full mix; delay times are reported in ms and as a note division. |

Whole-mix autocorrelation is deliberately never trusted for delay: a delay
synced to the tempo is mathematically indistinguishable from the rhythm of the
part. The app would rather say nothing than guess.

## Tweaking it: the radar

Every tone card carries a nine-axis radar. It starts where the analysis put it,
and you drag it from there.

| Axis | In plain terms | Measured from | Moves |
|---|---|---|---|
| **Grit** | raspiness, saturation | crest factor + spectral valley depth | AMP Gain, drive pedal choice |
| **Body** | low-end weight | 80-250 Hz balance | AMP Bass, EQ lows, cab |
| **Bite** | presence, attack edge | 1.6-8 kHz balance + centroid | AMP Treble/Presence, cab |
| **Honk** | scooped to mid-forward | mid scoop in dB | AMP Middle, pedal voicing |
| **Squash** | compression, sustain | crest factor + envelope spread | COMP |
| **Space** | wet to dry | RT60 + late tail energy | RVB |
| **Echo** | delay amount | repeat level + feedback | DLY |
| **Swirl** | chorus, phaser, tremolo | modulation depth | MOD |
| **Wah** | envelope filter | not measured - a taste control | EFX slot |

**An axis does not turn a knob. It edits the measurement the knob came from, and
the whole chain re-solves.** Pull Grit down on a metal tone and you don't just
get less gain - the amp becomes a Super Rvb, the drive pedal disappears, and the
compressor switches on because the amp is no longer doing the squashing. Every
block that moved gets outlined and tagged.

Consequences worth knowing:

- Leave the radar alone and you get the analysed preset byte for byte. The axes
  apply *deltas*, so an untouched axis changes nothing. Tested.
- The radar can never contradict the chain. Whether Space, Echo and Swirl read
  above zero is answered by asking the mapper, not by re-implementing its
  thresholds - an earlier version showed Echo at 95 next to a DLY block that was
  correctly switched off.
- **Wah and the drive pedal are the same physical slot.** Raise Wah past 10 and
  the overdrive is displaced; the UI says so rather than quietly dropping it.
- Drag with the mouse, or tab to a point and use arrow keys (shift for bigger
  steps, Home to reset that axis).

## Validating against the actual pedal

The app can tell you what it measured. Only the MG-300 can tell you whether the
preset gets there. Dial the preset in, reamp the excerpt through the pedal over
USB, record the output, and drop it into the **Validate** panel: the capture goes
through the *same* measurement pipeline, and you get a per-axis difference, a
match score, and an instruction per block - "Bass is 14 too low", not "sounds
thin". The captured tone is drawn on the same radar in green, so the shape of the
mismatch is visible at a glance.

**[docs/VALIDATION.md](docs/VALIDATION.md)** has the wiring, and 16 test cases in
three tiers: prove the loop is honest (null loop, level independence,
repeatability), prove each axis moves the right thing and nothing else, then the
real question - does it match the record, and does it converge when you follow
the fixes.

## What you get

- **In the browser** — one card per tone, the recorded excerpt, a timeline of
  where that tone appears, every block with its settings, and the reason each
  choice was made.
- **HTML sheet** — the same thing, printable, self-contained.
- **Dial-in text** — plain text, one line per block, for reading off a phone
  while you set the pedal up.
- **presets.json** — machine-readable, ready to become real preset files (below).
- **Raw measurements** — every number the decisions were made from.

Knob values are 0-100 to match the MG's display. Where a physical quantity was
measured (delay ms, RT60, LFO Hz) the **target** is printed too — set the knob
until the unit shows that value, because the knob-to-value curve differs per
model.

---

## The one thing this cannot do yet

Write a file QuickTone will import. NUX does not publish the preset format and
QuickTone exports nothing readable. Guessing the byte layout would produce files
that fail to import, or import as something that sounds nothing like the
analysis — which is worse than no file at all.

**To unlock it:** export a few presets from QuickTone into
`samples/reference_presets/`. Most useful is the *same* preset saved three times
with one parameter changed — Gain at 0, then 50, then 100. Diffing those pins
down where each field lives. `app/preset_format.py` is the stub waiting for it.

Until then the settings sheet contains every value needed to enter the preset by
hand, which takes about a minute per preset in QuickTone.

---

## Honest limits

- **Separation is the ceiling.** Demucs guitar stems are good, not perfect.
  Doubled guitars panned hard, or a guitar sitting under a loud synth, will
  bleed. The confidence score already accounts for this; the raw measurements
  let you check.
- **Two guitars playing simultaneously get analysed as one tone.** Clustering
  splits by *time*, not by source. Overlapping parts average together.
- **Drive on solo single notes is less certain than on chords** — fewer partials
  means fewer places for saturation to show up. Reflected in the confidence.
- **Modulation locked to the strumming rhythm** (a tremolo at exactly twice the
  strum rate) is not separable from picking dynamics. The app stays silent
  rather than guessing.
- **Long, high-feedback delays** in dense playing are approximate; 100-500 ms
  measures well.
- **The amp catalogue in `app/catalog/mg300mk2.json` is an assumption.** Model
  names follow NUX's MG-series naming; verify against your firmware's list and
  edit that file if anything differs. Voicing descriptors (brightness, mid
  shape, gain window) are hand-authored judgements — tune them and the amp
  choices move with them. Nothing is hard-coded.

## Tests

```bash
python3 tests/test_detectors.py
```

```bash
python3 tests/test_perceptual.py
```

```bash
python3 tests/test_auth.py
```

136 checks in total. The first suite synthesises guitar with **known** effect
settings and checks the detectors recover them: five gain stages ranked
correctly, tremolo rates within 0.4 Hz, delay times within 12 %, RT60 within
45 %, and — the part that matters most — dry signals must not produce phantom
effects, swept across every gain stage rather than a couple of samples.

The second suite covers the radar and the validation maths: untouched axes
reproduce the preset exactly, each axis moves in the direction it claims, no
axis disturbs a block it has no business touching (the software rehearsal of
hardware tests T4-T12), and a deliberately wrong capture is scored down and
diagnosed correctly — so the scoring is proven before anything is plugged in.

The third suite is about access, not audio. The interesting property is not that
login works but that a signed-in user cannot reach somebody else's analysis:
every job route — status, clips, downloads, radar adjust, validate, delete — is
checked from a second account and has to answer 404. It runs against a
throwaway SQLite database and needs no server.

## Layout

```
app/dsp.py            spectra, dynamics, drive scoring
app/timefx.py         delay, reverb, modulation detection
app/sections.py       guitar activity, tone clustering, clip rendering
app/mapper.py         measurements -> MG-300 MKII blocks (all the rules live here)
app/perceptual.py     the nine radar axes, forward and inverse
app/validate.py       compare a capture of the pedal against the target
app/catalog/          the device model list - edit freely
app/pipeline.py       orchestration
app/report.py         HTML and text output
app/main.py           web server, job ownership
app/auth.py           Google OAuth + password accounts, sessions
app/db.py             engine and session handling
app/models.py         User and Job tables
web/index.html        the whole UI
web/login.html        the login / sign-up page
Dockerfile            CPU-only torch image
docker-compose.yml    app + Postgres + volumes
.env.example          every setting, commented
docs/VALIDATION.md    hardware wiring + 16 test cases
tests/                synthetic ground truth
```
