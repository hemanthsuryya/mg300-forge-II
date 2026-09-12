# Validating a preset against the actual pedal

The app can tell you what it *measured*. Only the MG-300 can tell you whether the
preset actually gets there. This is the loop that closes that gap.

The principle: **capture the pedal's output and run it through the same
measurement pipeline that measured the song.** Both sides get reduced to the same
nine axes, and the difference is reported as numbers with a specific instruction
per block. Ears adapt within seconds and cannot A/B across a browser tab; the
measurement does not adapt.

---

## Rig it up

The MG-300 MKII is a USB-C audio interface with a ReAmp mode, so no extra
hardware is needed.

```
   [ app: tone_N.wav ]  --USB out-->  [ MG-300 MKII, ReAmp mode ]  --USB in-->  [ capture.wav ]
                                        preset dialled in
```

1. USB-C from the pedal to the Mac. It appears as an audio interface.
2. Set the pedal's USB routing to **ReAmp** (audio from the computer goes through
   the modelling chain and comes back).
3. In any DAW (or `sox`/`ffmpeg`), play the excerpt `jobs/<id>/clips/tone_N.wav`
   out to the pedal and record the return to `capture.wav`.
4. **Let the tail record.** Keep recording for at least 3 seconds after the last
   note. Reverb and Delay are measured from decay tails; a take that stops dead has
   none, and both would read zero — which looks like "the reverb is missing" when
   it really means "the recording is too short". The app detects this and warns
   rather than scoring you on it, but re-capturing costs you a round.
5. Drop `capture.wav` into the **Validate against the pedal** panel on that tone's
   card.

**Set the input level once and leave it.** Absolute loudness is normalised out of
the score, but clipping the pedal's input is not — it manufactures Drive that
isn't in the preset.

### If you'd rather play it than reamp

Also fine, and better for judging feel. Record yourself playing the same part
through the preset. Be aware that a different performance moves **Compression** and
**Drive** on its own — that's playing dynamics, not the preset. Reamping the app's
own excerpt removes that variable entirely, which is why it's the default.

---

## The test cases

Run them in order. Each isolates one thing, so a failure points somewhere
specific instead of "it sounds off". T1–T3 need no song at all — they check the
measurement chain itself before you trust it to judge tone.

### Tier 1 — prove the loop is honest

| # | Test | Do this | Pass |
|---|---|---|---|
| **T1** | Null loop | Bypass every block on the MG (or set the preset to a flat clean with no FX), reamp `tone_N.wav` through it, validate against **itself** rather than the song | Drive/Low End/Brightness/Mids within tolerance. Anything else means the interface, level or routing is colouring the signal, and every later result is suspect |
| **T2** | Level independence | Reamp the same preset twice at input levels 6 dB apart | Both scores within 3 points of each other. Proves loudness is genuinely normalised out |
| **T3** | Repeatability | Reamp the identical setup twice, changing nothing | Scores within 2 points. That spread is your noise floor — no later difference smaller than it means anything |

### Tier 2 — prove each axis actually moves the right thing

Dial the preset, then change **one** control on the pedal and re-capture. Each
test says which axis must move and which must not. This is what catches a
mapping that's pointing at the wrong knob.

| # | Change on the pedal | Must move | Must stay put | Pass |
|---|---|---|---|---|
| **T4** | AMP Gain +25 | Drive up | Low End, Brightness, Mids within tolerance | Drive rises ≥10 |
| **T5** | AMP Bass +25 | Low End up | Drive, Brightness | Low End rises ≥10 |
| **T6** | AMP Treble +25 | Brightness up | Low End, Drive | Brightness rises ≥10 |
| **T7** | AMP Middle +25 | Mids up | Low End, Brightness | Mids rises ≥10 |
| **T8** | COMP on, Sustain 80 | Compression up | Low End, Brightness, Mids | Compression rises ≥12 |
| **T9** | RVB Level 0 → 70 | Reverb up | everything else | Reverb rises ≥20 |
| **T10** | DLY E.Level 0 → 70 | Delay up | Low End, Brightness | Delay rises ≥20 |
| **T11** | MOD Depth 0 → 80 | Modulation up | Drive, Low End | Modulation rises ≥20 |
| **T12** | EFX Touch Wah on | Wah/Brightness move | Low End | Something moves — Wah is a taste axis, not a measured one, so this is a smoke test only |

Any test where the "must stay put" column moves more than its tolerance is a
**cross-coupling bug worth reporting** — tell me which one and I'll fix the
mapping.

### Tier 3 — the real question

| # | Test | Pass |
|---|---|---|
| **T13** | Dial the analysed preset, reamp, validate against the song | Score ≥75 ("close"). Below that, work the `priority_fixes` list top-down and re-capture |
| **T14** | Convergence | Apply the top fix, re-capture. Score must improve. Three rounds should reach ≥85 |
| **T15** | Radar honesty | Drag one axis by +30, dial the pedal to the new preset, re-capture | The captured value for that axis moves in the same direction by a comparable amount. If it doesn't, the axis is lying about what it controls |
| **T16** | Blind listen | Loop the excerpt against your capture with the app hidden | Your ears and the score should agree. If the score says 90 and it sounds wrong, the tolerances are wrong — that's useful, tell me what you heard |

---

## Reading the result

`overall_score` weights Drive, Low End, Brightness and Mids at 1.6× because they are the
tone; effects are seasoning. Per-axis tolerances live in `TOLERANCE` in
`app/validate.py` and are deliberately wider where the underlying measurement is
weaker (Modulation 20, Wah 25, Drive only 8).

The captured tone is drawn on the same radar as a green dashed polygon, so the
shape of the mismatch is visible at a glance: a green shape pulled in at one
spoke is a single knob; one that's uniformly smaller is a level or input-gain
problem, not a tone problem.

Every validation run is saved to `jobs/<id>/validation_tone_N.json`, so a
sequence of captures is a record of whether you actually converged.

---

## What would make this much stronger

If you can get a preset exported out of QuickTone (see the README), the loop can
close completely: the app writes the preset, you load it, capture, and it
adjusts its own parameters and re-writes — no hand-dialling between rounds. Right
now step "dial it in" is manual, which is the slow part of every test above.
