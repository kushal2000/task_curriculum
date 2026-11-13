"""Viser lineup of fabric_droop_lineup.py: each sheet clamped on a block, 5 cm flap drooping under
gravity, name + droop angle above it, all in one row seen from the side. No GPU needed.

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
DEFAULT_DATA = "docs/results/fabric_droop"
DX = 0.17  # sheet spacing along x
COLORS = [(150, 150, 160), (235, 170, 190), (240, 210, 120), (120, 180, 230), (230, 150, 90), (120, 200, 140), (70, 100, 170)]


class View:
    """The droop lineup under scene prefix `prefix`, its controls in one GUI folder."""

    title = "Droop lineup"

    def __init__(self, server: viser.ViserServer, data: str = DEFAULT_DATA, prefix: str = "", title: str | None = None):
        self.server, self.p = server, prefix
        self.title = title or type(self).title
        d = np.load(ROOT / data / "droop.npz")
        angles = json.loads((ROOT / data / "angles.json").read_text())["droop_deg"]
        self.frames, self.times, self.faces = d["frames"], d["times"], d["faces"]
        self.names = names = [str(n) for n in d["names"]]
        clamp_x = float(d["clamp_x"])
        self.n = n = len(names)

        mid = DX * (n - 1) / 2
        self.camera = dict(position=(mid, -0.95, 0.02), look_at=(mid, 0.0, -0.03), fov=0.6)

        self.offs = [np.array([k * DX, 0.0, 0.0]) for k in range(n)]
        self.static = []
        for k, name in enumerate(names):
            o = self.offs[k]
            self.static.append(server.scene.add_box(
                f"{prefix}/clamp/{k}", dimensions=(clamp_x + 0.05, 0.11, 0.012), color=(90, 70, 60),
                position=tuple(o + np.array([(clamp_x - 0.05) / 2, 0, -0.0068])),
            ))
            self.static.append(server.scene.add_label(
                f"{prefix}/label/{k}", f"{name}  {angles[name]:.0f}°", position=tuple(o + np.array([0.0, 0.0, 0.05]))))

        times = self.times
        self.folder = server.gui.add_folder(self.title)
        with self.folder:
            with server.gui.add_folder("Playback"):
                self.t_slider = server.gui.add_slider("t [s]", min=0.0, max=float(times[-1]),
                                                      step=float(times[1] - times[0]), initial_value=float(times[-1]))
                self.play = server.gui.add_checkbox("play", False)
                self.textured = server.gui.add_checkbox("fabric textures", True)

        self.meshes = []
        self.active, self.last, self.t_wall = True, None, time.time()

    def set_visible(self, on: bool):
        self.active = on
        self.folder.visible = on
        for h in self.static:
            h.visible = on
        if not on:
            for h in self.meshes:
                h.remove()
            self.meshes.clear()
        self.last, self.t_wall = None, time.time()

    def draw(self, ti):
        self.meshes = [
            add_fabric_mesh(self.server, f"{self.p}/sheet/{k}", self.frames[ti, k] + self.offs[k], self.faces,
                            self.names[k], COLORS[k % len(COLORS)], textured=self.textured.value)
            for k in range(self.n)
        ]

    def tick(self):
        if not self.active:
            return
        times, now = self.times, time.time()
        if self.play.value:
            t = self.t_slider.value + (now - self.t_wall)
            self.t_slider.value = float(t if t <= times[-1] else 0.0)
        self.t_wall = now
        ti = min(int(np.searchsorted(times, self.t_slider.value - 1e-9)), len(times) - 1)
        if (ti, self.textured.value) != self.last:
            self.draw(ti)
            self.last = (ti, self.textured.value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--port", type=int, default=8091)
    a = ap.parse_args()
    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(control_layout="collapsible")
    view = View(server, a.data)
    for k, v in view.camera.items():
        setattr(server.initial_camera, k, v)
    while True:
        view.tick()
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
