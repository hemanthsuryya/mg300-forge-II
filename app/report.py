"""Human-readable outputs: a printable HTML card set and a plain dial-in sheet."""
from __future__ import annotations

import html
import json

CSS = """
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--ink:#e7e9ee;--dim:#98a0b3;
--hot:#ff8a3d;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:0}
.sub{color:var(--dim);font-size:13px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:18px;margin:18px 0}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:center}
.chain{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;margin-top:14px}
.blk{border:1px solid var(--line);border-radius:10px;padding:12px;background:#12151c}
.blk.off{opacity:.42}
.blk h3{margin:0 0 2px;font-size:12px;letter-spacing:.12em;color:var(--dim);text-transform:uppercase}
.blk .model{font-size:15px;font-weight:600;color:var(--hot)}
.blk.off .model{color:var(--dim)}
.kv{display:flex;justify-content:space-between;font-size:13px;padding:2px 0;border-top:1px dotted #232833}
.kv span:last-child{font-variant-numeric:tabular-nums;color:#fff}
.why{color:var(--dim);font-size:12px;margin-top:8px;line-height:1.45}
.badge{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;
border:1px solid var(--line);color:var(--dim)}
.bar{height:6px;background:#242936;border-radius:99px;overflow:hidden;width:120px}
.bar i{display:block;height:100%;background:var(--hot)}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--dim);font-weight:500}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
"""


def _conf_color(c):
    return "var(--ok)" if c >= 0.66 else ("var(--warn)" if c >= 0.4 else "var(--bad)")


def _block_html(b):
    off = not b.get("enabled", True) or b.get("model") == "(off)"
    rows = "".join(
        f'<div class="kv"><span>{html.escape(str(k))}</span><span>{html.escape(str(v))}</span></div>'
        for k, v in (b.get("params") or {}).items())
    tgt = b.get("target")
    tgt_html = ""
    if tgt:
        bits = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in tgt.items() if v is not None)
        tgt_html = f'<div class="why"><b>Target:</b> {html.escape(bits)}</div>'
    return (f'<div class="blk{" off" if off else ""}"><h3>{html.escape(b["module"])}</h3>'
            f'<div class="model">{html.escape(str(b.get("model", "-")))}</div>'
            + (f'<div class="sub">{html.escape(b["based_on"])}</div>' if b.get("based_on") else "")
            + rows + tgt_html
            + f'<div class="why">{html.escape(b.get("why", ""))}</div></div>')


def render_html_report(result: dict) -> str:
    parts = [f"<!doctype html><meta charset='utf-8'><title>Tone Chaser - MG-300 MKII presets - "
             f"{html.escape(result['source'])}</title><style>{CSS}</style><div class='wrap'>"]
    parts.append(f"<h1>MG-300 MKII preset sheet</h1><div class='sub'>"
                 f"{html.escape(result['source'])} &middot; {result['duration_s']}s &middot; "
                 f"{result['tempo_bpm']} BPM &middot; separated with "
                 f"{html.escape(result['separation']['method'])} &middot; "
                 f"{len(result['tones'])} distinct guitar tone(s)</div>")
    for t in result["tones"]:
        p = t["preset"]
        c = p["confidence"]["overall"]
        secs = ", ".join(f"{s['start']:.0f}-{s['end']:.0f}s" for s in t["sections"][:10])
        more = "" if len(t["sections"]) <= 10 else f" +{len(t['sections']) - 10} more"
        parts.append(
            f"<div class='card'><div class='row' style='justify-content:space-between'>"
            f"<div><h2>Tone {t['tone_id']} &middot; {html.escape(t['name'])}</h2>"
            f"<div class='sub'>{t['total_seconds']:.0f}s of guitar ({t['share_pct']}%) "
            f"&middot; {html.escape(t['role'])}</div></div>"
            f"<div class='row'><span class='badge'>confidence</span>"
            f"<div class='bar'><i style='width:{c*100:.0f}%;background:{_conf_color(c)}'></i></div>"
            f"<span class='mono'>{c:.2f}</span></div></div>"
            f"<div class='sub' style='margin-top:8px'>Heard at: {html.escape(secs)}{more}</div>"
            f"<div class='chain'>{''.join(_block_html(b) for b in p['chain'])}</div>"
            f"<div class='why' style='margin-top:12px'>Runner-up amps: "
            + ", ".join(html.escape(f"{a['model']} ({a['ref']})") for a in p["amp_alternatives"])
            + "</div>"
            + "<details style='margin-top:10px'><summary class='sub'>Measurements</summary>"
            + "<table>" + "".join(
                f"<tr><th>{html.escape(k)}</th><td class='mono'>{html.escape(json.dumps(v))}</td></tr>"
                for k, v in p["measurements"].items()) + "</table></details>"
            + "</div>")
    parts.append("<div class='card'><h2>How to read this</h2><div class='why'>"
                 "Knob values are 0-100 to match the MG's display. Where a physical "
                 "quantity was measured (delay time in ms, reverb RT60, LFO rate in Hz) "
                 "the target is printed too - dial the knob until the unit shows that "
                 "value rather than trusting the knob number, because each model's "
                 "knob-to-value curve differs. Confidence below 0.4 means treat that "
                 "block as a starting point, not an answer."
                 "</div></div></div>")
    return "".join(parts)


def render_text_sheet(result: dict) -> str:
    L = [f"MG-300 MKII preset sheet", f"Source : {result['source']}",
         f"Length : {result['duration_s']}s at {result['tempo_bpm']} BPM",
         f"Stems  : {result['separation']['method']}",
         f"Tones  : {len(result['tones'])}", ""]
    for t in result["tones"]:
        p = t["preset"]
        L += ["=" * 66,
              f"TONE {t['tone_id']}: {t['name']}  ({t['total_seconds']:.0f}s, "
              f"{t['share_pct']}% of the guitar, confidence {p['confidence']['overall']:.2f})",
              "  Heard at: " + ", ".join(f"{s['start']:.0f}-{s['end']:.0f}s"
                                         for s in t["sections"][:12]),
              "-" * 66]
        for b in p["chain"]:
            if not b.get("enabled", True) or b.get("model") == "(off)":
                L.append(f"  {b['module']:<5} OFF")
                continue
            ps = "  ".join(f"{k} {v}" for k, v in (b.get("params") or {}).items())
            L.append(f"  {b['module']:<5} {b['model']:<16} {ps}")
            if b.get("target"):
                L.append(f"        target: " + ", ".join(
                    f"{k.replace('_', ' ')} {v}" for k, v in b["target"].items() if v is not None))
            L.append(f"        why: {b['why']}")
        L += ["  Runner-up amps: " + ", ".join(f"{a['model']} ({a['ref']})"
                                               for a in p["amp_alternatives"]), ""]
    return "\n".join(L)
