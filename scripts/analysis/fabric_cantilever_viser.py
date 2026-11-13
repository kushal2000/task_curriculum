"""Viser playback of fabric_cantilever.py: each fabric preset's sheet hanging off a clamp, the
exact elastica for its literature bending length as a red line, old vs tuned edge_ke and current vs
density-scaled edge_kd side by side. No GPU needed (replays the saved .npz).

`View` is also used by fabric_results_viser.py, which shows every probe in one server.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import viser

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fabric_textures import add_fabric_mesh  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = "docs/results/fabric_cantilever"
VARIANTS = [
    ("old_ke|kd_now", "100·B ke, kd 1e-3", (150, 150, 160)),
    ("old_ke|kd_scaled", "100·B ke, kd scaled", (120, 170, 220)),
    ("tuned_ke|kd_now", "tuned ke, kd 1e-3", (240, 180, 90)),
    ("tuned_ke|kd_scaled", "tuned ke, kd scaled", (90, 200, 120)),
]
DEFAULT_ON = {"old_ke|kd_now", "tuned_ke|kd_scaled"}
DX, DY = 0.18, 0.16  # variant spacing along x, fabric spacing along y


class View:
    """The cantilever scene under scene prefix `prefix`, its controls in one GUI folder."""

    title = "Cantilever"

    def __init__(self, server: viser.ViserServer, data: str = DEFAULT_DATA, prefix: str = "", title: str | None = None):
        self.server, self.p = server, prefix
        self.title = title or type(self).title
        d = np.load(ROOT / data / "cantilever.npz")
        summary = json.loads((ROOT / data / "summary.json").read_text())
        self.frames, self.times, self.faces = d["frames"], d["times"], d["faces"]
        labels = [s.split("|", 1) for s in d["labels"]]
        clamp_x = float(d["clamp_x"])
        self.fabrics = fabrics = list(summary["fabrics"])
        p = prefix

        self.static = [server.scene.add_grid(f"{p}/ground", width=1.6, height=1.4, cell_size=0.05,
                                             position=(0.25, 0.4, -0.12))]
        self.slot = {}  # sheet index -> (fabric idx, variant idx)
        for k, (fab, var) in enumerate(labels):
            self.slot[k] = (fabrics.index(fab), [v[0] for v in VARIANTS].index(var))

        # Static: clamp blocks, elastica targets, labels.
        self.targets = {}
        for fi, fab in enumerate(fabrics):
            ex = d[f"elastica_{fab}"]
            f = summary["fabrics"][fab]
            self.static.append(server.scene.add_label(
                f"{p}/labels/{fab}", f"{fab}  c {100 * f['c']:.1f} cm  target {f['theta_target']:.0f}°",
                position=tuple(self.offset(fi, -1) + np.array([0.08, 0, 0.02])),
            ))
            for vi in range(len(VARIANTS)):
                o = self.offset(fi, vi)
                self.static.append(server.scene.add_box(
                    f"{p}/clamp/{fab}/{vi}", dimensions=(clamp_x + 0.05, 0.11, 0.01), color=(90, 70, 60),
                    position=tuple(o + np.array([(clamp_x - 0.05) / 2, 0, -0.0095])),
                ))
                pts = np.stack([ex[0] + clamp_x, np.zeros_like(ex[0]), ex[1]], -1)
                segs = []
                for y in (-0.05, 0.0, 0.05):
                    q = pts + np.array([0, y, 0.001]) + o
                    segs.append(np.stack([q[:-1], q[1:]], 1))
                self.targets[(fab, vi)] = server.scene.add_line_segments(
                    f"{p}/target/{fab}/{vi}", points=np.concatenate(segs).astype(np.float32),
                    colors=(220, 40, 40), line_width=2.5,
                )
        for vi, (_, desc, _) in enumerate(VARIANTS):
            self.static.append(server.scene.add_label(
                f"{p}/labels/var{vi}", desc, position=tuple(self.offset(-1, vi) + np.array([0, 0.04, 0]))))

        times = self.times
        self.folder = server.gui.add_folder(self.title)
        with self.folder:
            with server.gui.add_folder("Playback"):
                self.t_slider = server.gui.add_slider("t [s]", min=0.0, max=float(times[-1]),
                                                      step=float(times[1] - times[0]), initial_value=float(times[-1]))
                self.play = server.gui.add_checkbox("play", False)
                self.speed = server.gui.add_slider("speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
            with server.gui.add_folder("Show"):
                self.show = {v: server.gui.add_checkbox(desc, v in DEFAULT_ON) for v, desc, _ in VARIANTS}
                self.show_target = server.gui.add_checkbox("elastica target (red)", True)
                self.textured = server.gui.add_checkbox("fabric textures", True)

            rows = ["| fabric | c cm | target ° | 100·B, kd 1e-3 ° | tuned ° | ke tuned (×B) | kd scaled | drop t90 100·B / tuned s |",
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

        # Oblique view over the fabric rows (y) and variant columns (x).
        self.camera = dict(position=(0.27, -0.75, 0.55), look_at=(0.27, 0.40, -0.03), fov=0.75)
        self.meshes = {}
        self.active, self.last, self.t_wall = True, None, time.time()

    def offset(self, fi, vi):
        return np.array([vi * DX, fi * DY, 0.0])

    def set_visible(self, on: bool):
        self.active = on
        self.folder.visible = on
        for h in self.static:
            h.visible = on
        if not on:
            for h in self.targets.values():
                h.visible = False
            for h in self.meshes.values():
                h.remove()
            self.meshes.clear()
        self.last, self.t_wall = None, time.time()

    def draw(self, ti):
        for k, (fi, vi) in self.slot.items():
            name = f"{self.p}/sheet/{k}"
            if not self.show[VARIANTS[vi][0]].value:
                if name in self.meshes:
                    self.meshes.pop(name).remove()
                continue
            verts = self.frames[ti, k] + self.offset(fi, vi)
            self.meshes[name] = add_fabric_mesh(self.server, name, verts, self.faces, self.fabrics[fi],
                                                VARIANTS[vi][2], textured=self.textured.value)
        for fab in self.fabrics:
            for vi in range(len(VARIANTS)):
                self.targets[(fab, vi)].visible = self.show_target.value and self.show[VARIANTS[vi][0]].value

    def tick(self):
        if not self.active:
            return
        times, now = self.times, time.time()
        if self.play.value:
            t = self.t_slider.value + (now - self.t_wall) * self.speed.value
            self.t_slider.value = float(t if t <= times[-1] else 0.0)
        self.t_wall = now
        ti = min(int(np.searchsorted(times, self.t_slider.value - 1e-9)), len(times) - 1)
        key = (ti, tuple(c.value for c in self.show.values()), self.show_target.value, self.textured.value)
        if key != self.last:
            self.draw(ti)
            self.last = key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    view = View(server, a.data)
    for k, v in view.camera.items():
        setattr(server.initial_camera, k, v)
    while True:
        view.tick()
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
