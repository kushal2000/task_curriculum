"""Viser playback of fabric_cantilever.py: each fabric preset's sheet hanging off a clamp, the
exact elastica for its literature bending length as a red line, old vs tuned edge_ke and current vs
density-scaled edge_kd side by side. No GPU needed (replays the saved .npz).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import viser

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = [
    ("old_ke|kd_now", "100·B ke, kd 1e-3 (eval 238829)", (150, 150, 160)),
    ("old_ke|kd_scaled", "100·B ke, kd scaled", (120, 170, 220)),
    ("tuned_ke|kd_now", "tuned ke, kd 1e-3", (240, 180, 90)),
    ("tuned_ke|kd_scaled", "tuned ke, kd scaled", (90, 200, 120)),
]
DEFAULT_ON = {"old_ke|kd_now", "tuned_ke|kd_scaled"}
DX, DY = 0.18, 0.16  # variant spacing along x, fabric spacing along y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="docs/results/fabric_cantilever")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    d = np.load(ROOT / a.data / "cantilever.npz")
    summary = json.loads((ROOT / a.data / "summary.json").read_text())
    frames, times, faces = d["frames"], d["times"], d["faces"]
    labels = [s.split("|", 1) for s in d["labels"]]
    clamp_x = float(d["clamp_x"])
    fabrics = list(summary["fabrics"])

    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=1.6, height=1.4, cell_size=0.05, position=(0.25, 0.4, -0.12))

    slot = {}  # sheet index -> (fabric idx, variant idx)
    for k, (fab, var) in enumerate(labels):
        slot[k] = (fabrics.index(fab), [v[0] for v in VARIANTS].index(var))

    def offset(fi, vi):
        return np.array([vi * DX, fi * DY, 0.0])

    # Static: clamp blocks, elastica targets, labels.
    targets = {}
    for fi, fab in enumerate(fabrics):
        ex = d[f"elastica_{fab}"]
        f = summary["fabrics"][fab]
        server.scene.add_label(
            f"/labels/{fab}", f"{fab}  c {100 * f['c']:.1f} cm  target {f['theta_target']:.0f}°",
            position=tuple(offset(fi, -1) + np.array([0.08, 0, 0.02])),
        )
        for vi in range(len(VARIANTS)):
            o = offset(fi, vi)
            server.scene.add_box(
                f"/clamp/{fab}/{vi}", dimensions=(clamp_x + 0.05, 0.11, 0.01), color=(90, 70, 60),
                position=tuple(o + np.array([(clamp_x - 0.05) / 2, 0, -0.0095])),
            )
            pts = np.stack([ex[0] + clamp_x, np.zeros_like(ex[0]), ex[1]], -1)
            segs = []
            for y in (-0.05, 0.0, 0.05):
                p = pts + np.array([0, y, 0.001]) + o
                segs.append(np.stack([p[:-1], p[1:]], 1))
            targets[(fab, vi)] = server.scene.add_line_segments(
                f"/target/{fab}/{vi}", points=np.concatenate(segs).astype(np.float32),
                colors=(220, 40, 40), line_width=2.5,
            )
    for vi, (_, desc, _) in enumerate(VARIANTS):
        server.scene.add_label(f"/labels/var{vi}", desc, position=tuple(offset(-1, vi) + np.array([0, 0.04, 0])))

    with server.gui.add_folder("Playback"):
        t_slider = server.gui.add_slider("t [s]", min=0.0, max=float(times[-1]), step=float(times[1] - times[0]), initial_value=float(times[-1]))
        play = server.gui.add_checkbox("play", False)
        speed = server.gui.add_slider("speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
    with server.gui.add_folder("Show"):
        show = {v: server.gui.add_checkbox(desc, v in DEFAULT_ON) for v, desc, _ in VARIANTS}
        show_target = server.gui.add_checkbox("elastica target (red)", True)

    rows = ["| fabric | c cm | target ° | eval 238829 ° | tuned ° | ke tuned (×B) | kd scaled | drop t90 eval / tuned s |",
            "|---|---|---|---|---|---|---|---|"]
    for fab in fabrics:
        f = summary["fabrics"][fab]
        rows.append(
            f"| {fab} | {100 * f['c']:.1f} | {f['theta_target']:.0f} | {f['old_ke/kd_now']['theta_final']:.0f} "
            f"| {f['tuned_ke/kd_scaled']['theta_final']:.0f} | {f['ke']:.1e} ({f['ke_over_B']:.0f}) | {f['kd_scaled']:.1e} "
            f"| {f['old_ke/kd_now']['t90_s']:.2f} / {f['tuned_ke/kd_scaled']['t90_s']:.2f} |"
        )
    with server.gui.add_folder("Results"):
        server.gui.add_markdown("\n".join(rows))

    meshes = {}

    def draw(ti):
        for k, (fi, vi) in slot.items():
            name = f"/sheet/{k}"
            v = VARIANTS[vi][0]
            if not show[v].value:
                if name in meshes:
                    meshes.pop(name).remove()
                continue
            verts = frames[ti, k] + offset(fi, vi)
            meshes[name] = server.scene.add_mesh_simple(
                name, vertices=verts.astype(np.float32), faces=faces, color=VARIANTS[vi][2],
                side="double", flat_shading=False,
            )
        for fab in fabrics:
            for vi in range(len(VARIANTS)):
                targets[(fab, vi)].visible = show_target.value and show[VARIANTS[vi][0]].value

    last = None
    t_wall = time.time()
    while True:
        now = time.time()
        if play.value:
            t = t_slider.value + (now - t_wall) * speed.value
            t_slider.value = float(t if t <= times[-1] else 0.0)
        t_wall = now
        ti = int(np.searchsorted(times, t_slider.value - 1e-9))
        key = (ti, tuple(c.value for c in show.values()), show_target.value)
        if key != last:
            draw(min(ti, len(times) - 1))
            last = key
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
