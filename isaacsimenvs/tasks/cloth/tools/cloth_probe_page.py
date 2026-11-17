"""Build the cloth-realism page from cloth_probe.py outputs: 3D playback panels plus result tables.

    python isaacsimenvs/tasks/cloth/tools/cloth_probe_page.py --spec page_spec.json --out page.html

The spec is {"title", "header" (html), "footer" (html), "sections": [{"panel": {...}} | {"table": {...}}]}.
A panel is {"title", "lede", "json", "npz", "select": [sheet keys], "cols", "labels": {key: text},
"stride": frame stride, "jumps": [[label, phase prefix]]}; a table is {"title", "lede", "html"} (a
pre-rendered <table>). Frames are embedded as int16 at 0.1 mm.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

FAB_COLOR = dict(denim="#3e5f8a", towel="#c9a46a", cotton="#d9d4c5", big_napkin="#b85c5c", napkin="#d88d7a",
                 silk="#9b7bb5", chiffon="#7fb6a8")


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def panel_data(p: dict) -> dict:
    """``p["extra"]`` = [{"json", "npz"}, ...]: more probe runs of the same scenario whose sheets ``select``
    may name too (looked up by key across all runs; the first run sets the timeline)."""
    runs = []
    for src in [dict(json=p["json"], npz=p["npz"])] + list(p.get("extra", [])):
        m = json.loads(Path(src["json"]).read_text())
        ks = [s["key"] + (f"__c{int(round(s['line'][0] * 1000))}mm_a{int(round(s['line'][1]))}" if s.get("line") else "")
              for s in m["sheets"]]
        runs.append((m, np.load(src["npz"]), ks))
    meta, npz, _ = runs[0]

    def find(k):
        for m, z, ks in runs:
            if k in ks:
                return m, z, ks.index(k)
        raise KeyError(k)

    sel = p["select"]
    stride = int(p.get("stride", 1))
    times = np.asarray(npz["times"])[::stride]
    cols = int(p.get("cols", len(sel)))
    sheets = []
    # Layout: columns left-to-right, rows front-to-back, each cell as big as its largest sheet + gap.
    sizes = [find(k)[0]["sheets"][find(k)[2]]["size"] for k in sel]
    cell = max(sizes) + 0.05
    rows = (len(sel) + cols - 1) // cols
    for i, k in enumerate(sel):
        m, z, j = find(k)
        s = m["sheets"][j]
        fr = z[f"s{j}_frames"][::stride][: len(times)].astype(np.float32)  # (T, N, 3), sheet-local
        if len(fr) < len(times):  # pad a shorter run with its last frame
            fr = np.concatenate([fr, np.repeat(fr[-1:], len(times) - len(fr), 0)])
        faces = z[f"s{j}_faces"].astype(np.uint16 if fr.shape[1] < 65535 else np.uint32)
        q = np.clip(np.round(fr * 1e4), -32767, 32767).astype(np.int16)
        c, r = i % cols, i // cols
        sheets.append(dict(
            label=p.get("labels", {}).get(k, k), color=FAB_COLOR.get(s["fab"], "#999999"),
            x=(c - (cols - 1) / 2) * cell, z=(r - (rows - 1) / 2) * cell * 0.9,
            n=int(fr.shape[1]), faces=b64(faces), wide=bool(faces.dtype == np.uint32), frames=b64(q),
            mesh=s["mesh"], edge_mm=round(1e3 * s["edge_length"], 1), size_cm=round(100 * s["size"]),
        ))
    return dict(title=p["title"], lede=p.get("lede", ""), times=[float(t) for t in times],
                phases=meta["phases"], sheets=sheets, extent=[cols * cell, rows * cell],
                jumps=p.get("jumps", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    sections = []
    for sec in spec["sections"]:
        if "panel" in sec:
            sections.append(dict(kind="panel", **panel_data(sec["panel"])))
        else:
            sections.append(dict(kind="table", **sec["table"]))
    tmpl = (Path(__file__).with_name("cloth_probe_page.html")).read_text()
    html = (tmpl.replace("/*TITLE*/", spec["title"])
            .replace("/*HEADER*/", spec["header"])
            .replace("/*SECTIONS*/", json.dumps(sections, separators=(",", ":")))
            .replace("/*FOOTER*/", spec.get("footer", "")))
    Path(a.out).write_text(html)
    print(f"wrote {a.out} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
