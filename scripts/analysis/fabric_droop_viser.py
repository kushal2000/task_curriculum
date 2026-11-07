"""Viser lineup of fabric_droop_lineup.py: each sheet clamped on a block, 5 cm flap drooping under
gravity, name + droop angle above it, all in one row seen from the side. No GPU needed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import viser

ROOT = Path(__file__).resolve().parents[2]
DX = 0.17  # sheet spacing along x
COLORS = [(150, 150, 160), (235, 170, 190), (240, 210, 120), (120, 180, 230), (230, 150, 90), (120, 200, 140), (70, 100, 170)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="docs/results/fabric_droop")
    ap.add_argument("--port", type=int, default=8091)
    a = ap.parse_args()
    d = np.load(ROOT / a.data / "droop.npz")
    angles = json.loads((ROOT / a.data / "angles.json").read_text())["droop_deg"]
    frames, times, faces, names = d["frames"], d["times"], d["faces"], [str(n) for n in d["names"]]
    clamp_x = float(d["clamp_x"])
    n = len(names)

    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(control_layout="collapsible")
    mid = DX * (n - 1) / 2
    server.initial_camera.position = (mid, -0.95, 0.02)
    server.initial_camera.look_at = (mid, 0.0, -0.03)
    server.initial_camera.fov = 0.6

    offs = [np.array([k * DX, 0.0, 0.0]) for k in range(n)]
    for k, name in enumerate(names):
        o = offs[k]
        server.scene.add_box(
            f"/clamp/{k}", dimensions=(clamp_x + 0.05, 0.11, 0.012), color=(90, 70, 60),
            position=tuple(o + np.array([(clamp_x - 0.05) / 2, 0, -0.0068])),
        )
        server.scene.add_label(f"/label/{k}", f"{name}  {angles[name]:.0f}°", position=tuple(o + np.array([0.0, 0.0, 0.05])))

    with server.gui.add_folder("Playback"):
        t_slider = server.gui.add_slider("t [s]", min=0.0, max=float(times[-1]), step=float(times[1] - times[0]), initial_value=float(times[-1]))
        play = server.gui.add_checkbox("play", False)

    def draw(ti):
        for k in range(n):
            server.scene.add_mesh_simple(
                f"/sheet/{k}", vertices=(frames[ti, k] + offs[k]).astype(np.float32), faces=faces,
                color=COLORS[k % len(COLORS)], side="double", flat_shading=False,
            )

    last, t_wall = None, time.time()
    while True:
        now = time.time()
        if play.value:
            t = t_slider.value + (now - t_wall)
            t_slider.value = float(t if t <= times[-1] else 0.0)
        t_wall = now
        ti = min(int(np.searchsorted(times, t_slider.value - 1e-9)), len(times) - 1)
        if ti != last:
            draw(ti)
            last = ti
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
